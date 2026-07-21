from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from deepseek import DeepSeekClient

API_KEY = os.environ.get("API_KEY", "")

app = FastAPI(title="DeepSeek OpenAI-compatible API")

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
    _chat_session_id, _parent_message_id = get_client().create_session()


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("type", "")
                if t == "text":
                    parts.append(block.get("text", ""))
                elif t == "tool_result":
                    # flatten nested content inside tool results
                    parts.append(_flatten_content(block.get("content", "")))
                elif t not in ("image", "image_url"):
                    # unknown block — try to grab any text-ish value
                    for key in ("text", "content", "value"):
                        v = block.get(key)
                        if isinstance(v, str) and v:
                            parts.append(v)
                            break
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models(request: Request):
    _check_auth(request)
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "created": 0, "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "created": 0, "owned_by": "deepseek"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _chat_session_id, _parent_message_id

    _check_auth(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Log truncated body to Railway logs for debugging
    logging.warning("BODY: %s", json.dumps(body)[:1000])

    model = body.get("model", "deepseek-chat") or "deepseek-chat"
    stream = bool(body.get("stream", False))
    thinking_enabled = "reason" in str(model).lower()

    raw_messages = body.get("messages") or []
    messages = [
        {"role": str(m.get("role", "user")), "content": _flatten_content(m.get("content", ""))}
        for m in raw_messages
        if isinstance(m, dict)
    ]

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    client = get_client()

    if stream:
        return StreamingResponse(
            _stream_response(client, messages, thinking_enabled),
            media_type="text/event-stream",
        )
    return await _non_stream_response(client, messages, thinking_enabled)


async def _stream_response(
    client: DeepSeekClient,
    messages: list[dict],
    thinking_enabled: bool,
) -> AsyncGenerator[str, None]:
    global _chat_session_id, _parent_message_id

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    ts = int(time.time())

    yield _sse(cid, ts, delta={"role": "assistant", "content": ""})

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
                yield _sse(cid, ts, delta={"content": delta})

    except RuntimeError as exc:
        msg = str(exc)
        if any(c in msg for c in ("401", "403", "40001", "40004", "session")):
            try:
                reset_session()
            except Exception:
                pass
        yield _sse(cid, ts, delta={"content": f"\n\n[Error: {msg}]"})

    yield _sse(cid, ts, finish_reason="stop")
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
        msg = str(exc)
        if any(c in msg for c in ("401", "403", "40001", "40004", "session")):
            try:
                reset_session()
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=msg)

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    return JSONResponse({
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def _sse(cid: str, ts: int, delta: dict = {}, finish_reason: Optional[str] = None) -> str:
    return f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"
