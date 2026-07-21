from __future__ import annotations

import json
import os
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from deepseek import DeepSeekClient

# Optional: protect the API with a key
API_KEY = os.environ.get("API_KEY", "")

app = FastAPI(title="DeepSeek OpenAI-compatible API")

# Single shared client — one session per process
_client: Optional[DeepSeekClient] = None
_chat_session_id: Optional[str] = None
_parent_message_id: Optional[str] = None


def get_client() -> DeepSeekClient:
    global _client, _chat_session_id, _parent_message_id
    if _client is None:
        _client = DeepSeekClient()
        _chat_session_id, _parent_message_id = _client.create_session()
    return _client


def reset_session() -> None:
    global _chat_session_id, _parent_message_id
    client = get_client()
    _chat_session_id, _parent_message_id = client.create_session()


# ── Models ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ── Auth helper ───────────────────────────────────────────────────────────────

def check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "DeepSeek OpenAI-compatible API"}


@app.get("/v1/models")
def list_models(request: Request):
    check_auth(request)
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "created": 0, "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "created": 0, "owned_by": "deepseek"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    global _chat_session_id, _parent_message_id

    check_auth(request)

    thinking_enabled = "reason" in body.model.lower()
    messages = [m.model_dump() for m in body.messages]

    client = get_client()

    if body.stream:
        return StreamingResponse(
            _stream_response(client, messages, thinking_enabled),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_response(client, messages, thinking_enabled)


async def _stream_response(
    client: DeepSeekClient,
    messages: list[dict],
    thinking_enabled: bool,
) -> AsyncGenerator[str, None]:
    global _chat_session_id, _parent_message_id

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # Opening chunk
    yield _sse(completion_id, created, delta={"role": "assistant", "content": ""})

    try:
        for delta, new_parent_id in client.stream_message(
            messages=messages,
            chat_session_id=_chat_session_id,
            parent_message_id=_parent_message_id,
            thinking_enabled=thinking_enabled,
        ):
            if new_parent_id is not None:
                _parent_message_id = new_parent_id
            if delta:
                yield _sse(completion_id, created, delta={"content": delta})

    except RuntimeError as exc:
        err_msg = str(exc)
        # Try session refresh on auth errors
        if any(code in err_msg for code in ("401", "403", "40001", "40004", "session")):
            try:
                reset_session()
            except Exception:
                pass
        yield _sse(completion_id, created, delta={"content": f"\n\n[Error: {err_msg}]"})

    # Final chunk
    yield _sse(completion_id, created, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _non_stream_response(
    client: DeepSeekClient,
    messages: list[dict],
    thinking_enabled: bool,
) -> JSONResponse:
    global _chat_session_id, _parent_message_id

    full_text = ""
    try:
        for delta, new_parent_id in client.stream_message(
            messages=messages,
            chat_session_id=_chat_session_id,
            parent_message_id=_parent_message_id,
            thinking_enabled=thinking_enabled,
        ):
            if new_parent_id is not None:
                _parent_message_id = new_parent_id
            if delta:
                full_text += delta
    except RuntimeError as exc:
        err_msg = str(exc)
        if any(code in err_msg for code in ("401", "403", "40001", "40004", "session")):
            try:
                reset_session()
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=err_msg)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def _sse(completion_id: str, created: int, delta: dict = {}, finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"
