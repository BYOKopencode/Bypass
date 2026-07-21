from __future__ import annotations

import base64
import copy
import ctypes
import json
import os
import re
import tempfile
import traceback
from typing import Any, Dict, Generator, Iterable, Optional

import numpy as np
import requests
import wasmtime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WASM_PATH = os.path.join(SCRIPT_DIR, "sha3_wasm_bg.wasm")


def _resolve_log_path() -> str:
    candidate = os.path.join(SCRIPT_DIR, "deepseek_sse_debug.log")
    try:
        with open(candidate, "a"):
            pass
        return candidate
    except OSError:
        return os.path.join(tempfile.gettempdir(), "deepseek_sse_debug.log")


DEBUG_LOG_PATH = _resolve_log_path()

AUTH = os.environ.get("DEEPSEEK_AUTH", "Bearer esIZOMVvzMue5aeShauZj2gow4vD0HBo5+KGYIXqEShF+B3Tl3WfBwc8GPGK1Edb")

COOKIES = {
    "ds_session_id": os.environ.get("DS_SESSION_ID", "9062f7743f474947a5fee673cdd1d293"),
    "aws-waf-token": os.environ.get("DS_WAF_TOKEN", "0cd45ed5-1511-400b-96ff-08160f1a9871:BQoAf0M3yIUWAAAA:DdHPjA5yOqDlP9YJ9TdRAs7wQ5sstgOM4/cYTFY87IVkHyYpsqoRQpbGgWH76m1CNR169GiHQcCPGfRveMlH8vKAq/ljJ5LdOMgbKOGFztqAsDvell0D74Fw3bVmGmPOPSDLkoptnP69ZKL0OD/643Rbq//YqDI5moftFXjc9p0RZ8CMPaubPg0IIYNJ"),
    ".thumbcache_6b2e5483f9d858d7c661c5e276b6a6ae": os.environ.get("DS_THUMBCACHE", "BunzYkXXcB/G1PRNvJm0lvEqvmnxoJpSHLmqFtM4qG47OsoMkMI/2HYpYHXe1jUvDpGyCrZdWNqnYhMgTh0kSA%3D%3D"),
    "smidV2": os.environ.get("DS_SMID", "20260717133259a420c0ef619c6c2017e8009a71b4e9830044ff149e8ba2470"),
}


class DeepSeekHash:
    def __init__(self) -> None:
        self.instance = None
        self.memory = None
        self.store = None

    def init(self, wasm_path: str) -> "DeepSeekHash":
        if not os.path.exists(wasm_path):
            raise FileNotFoundError(
                f"WASM file not found: {wasm_path}\n"
                "Place sha3_wasm_bg.wasm in the same directory as this script."
            )
        engine = wasmtime.Engine()
        with open(wasm_path, "rb") as f:
            module = wasmtime.Module(engine, f.read())
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]
        return self

    def _write(self, text: str) -> tuple[int, int]:
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](self.store, length, 1)
        memory_view = self.memory.data_ptr(self.store)
        dest = ctypes.cast(memory_view, ctypes.c_void_p).value + ptr
        ctypes.memmove(dest, encoded, length)
        return ptr, length

    def solve(self, challenge: str, salt: str, difficulty: float, expire_at: int) -> Optional[int]:
        prefix = f"{salt}_{expire_at}_"
        stack_ptr = self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, -16)
        try:
            challenge_ptr, challenge_len = self._write(challenge)
            prefix_ptr, prefix_len = self._write(prefix)
            self.instance.exports(self.store)["wasm_solve"](
                self.store, stack_ptr,
                challenge_ptr, challenge_len,
                prefix_ptr, prefix_len,
                float(difficulty),
            )
            memory_view = self.memory.data_ptr(self.store)
            success = int.from_bytes(bytes(memory_view[stack_ptr:stack_ptr + 4]), "little", signed=True)
            if success == 0:
                return None
            answer = np.frombuffer(bytes(memory_view[stack_ptr + 8:stack_ptr + 16]), dtype=np.float64)[0]
            return int(answer)
        finally:
            self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, 16)


class DeepSeekStreamState:
    _CONTENT_PATH = re.compile(r"^(?:response/)?fragments/(-?\d+)/content$", re.IGNORECASE)
    _FRAGMENT_PATH = re.compile(r"^(?:response/)?fragments/(-?\d+)$", re.IGNORECASE)
    _FIELD_PATH = re.compile(r"^(?:response/)?fragments/(-?\d+)/(\w+)$", re.IGNORECASE)

    def __init__(self) -> None:
        self.fragments: list[dict[str, Any]] = []
        self.direct_text = ""
        self.message_id: Optional[str] = None
        self.server_error: Optional[str] = None

    @staticmethod
    def _walk(value: Any) -> Iterable[dict[str, Any]]:
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from DeepSeekStreamState._walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from DeepSeekStreamState._walk(child)

    def _capture_message_id(self, event: Any) -> None:
        for item in self._walk(event):
            for key in ("response_message_id", "message_id", "id"):
                value = item.get(key)
                if not isinstance(value, str) or not value:
                    continue
                if key == "id" and not ("message" in item or "fragments" in item or item.get("role") == "assistant"):
                    continue
                self.message_id = value
                return

    def _capture_error(self, event: Any) -> None:
        for item in self._walk(event):
            status = str(item.get("status", "")).upper()
            code = item.get("code")
            error = item.get("error")
            if isinstance(error, str) and error:
                self.server_error = error
                return
            if isinstance(error, dict):
                message = error.get("message") or error.get("msg")
                if isinstance(message, str) and message:
                    self.server_error = message
                    return
            if status in {"FAILED", "ERROR", "REJECTED"}:
                message = item.get("message") or item.get("msg") or item.get("detail")
                if isinstance(message, str) and message:
                    self.server_error = message
                elif code is not None:
                    self.server_error = f"DeepSeek stream failed with code {code}"

    def _resolve_index(self, raw_index: str) -> int:
        index = int(raw_index)
        if index < 0:
            return max(0, len(self.fragments) + index)
        return index

    def _grow(self, index: int) -> None:
        while len(self.fragments) <= index:
            self.fragments.append({"type": "RESPONSE", "content": ""})

    @staticmethod
    def _snapshot_from(event: Any) -> Optional[list[dict[str, Any]]]:
        possible = []
        if isinstance(event, dict):
            possible.extend([
                event.get("fragments"),
                event.get("response", {}).get("fragments") if isinstance(event.get("response"), dict) else None,
            ])
            value = event.get("v")
            if isinstance(value, dict):
                possible.extend([
                    value.get("fragments"),
                    value.get("response", {}).get("fragments") if isinstance(value.get("response"), dict) else None,
                ])
        for candidate in possible:
            if isinstance(candidate, list):
                return [copy.deepcopy(f) for f in candidate if isinstance(f, dict)]
        return None

    def _feed_openai_shape(self, event: dict[str, Any]) -> None:
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    self.direct_text += content
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and len(content) >= len(self.direct_text):
                    self.direct_text = content
            text = choice.get("text")
            if isinstance(text, str):
                self.direct_text += text

    def feed(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        self._capture_message_id(event)
        self._capture_error(event)
        self._feed_openai_shape(event)
        operation = str(event.get("o", "")).upper()
        value = event.get("v")
        if operation == "BATCH" and isinstance(value, list):
            for child in value:
                self.feed(child)
            return
        snapshot = self._snapshot_from(event)
        if snapshot is not None:
            self.fragments = snapshot
        path = str(event.get("p", "")).strip("/")
        if re.fullmatch(r"(?:response/)?fragments", path, re.IGNORECASE):
            if operation == "APPEND" and isinstance(value, list):
                self.fragments.extend(copy.deepcopy(f) for f in value if isinstance(f, dict))
            elif operation in {"SET", "REPLACE", ""} and isinstance(value, list):
                self.fragments = [copy.deepcopy(f) for f in value if isinstance(f, dict)]
            return
        match = self._CONTENT_PATH.fullmatch(path)
        if match and isinstance(value, str):
            index = self._resolve_index(match.group(1))
            self._grow(index)
            if operation in {"SET", "REPLACE"}:
                self.fragments[index]["content"] = value
            else:
                self.fragments[index]["content"] = str(self.fragments[index].get("content", "")) + value
            return
        match = self._FRAGMENT_PATH.fullmatch(path)
        if match and isinstance(value, dict):
            index = self._resolve_index(match.group(1))
            self._grow(index)
            if operation in {"SET", "REPLACE"}:
                self.fragments[index] = copy.deepcopy(value)
            else:
                self.fragments[index].update(copy.deepcopy(value))
            return
        match = self._FIELD_PATH.fullmatch(path)
        if match and isinstance(value, (str, int, float, bool)):
            index = self._resolve_index(match.group(1))
            key = match.group(2)
            self._grow(index)
            self.fragments[index][key] = value
            return
        if not path and isinstance(value, str) and self.fragments:
            last = self.fragments[-1]
            last["content"] = str(last.get("content", "")) + value

    def visible_text(self) -> str:
        if self.fragments:
            ignored = {"THINK", "THINKING", "REASONING", "TIP", "SEARCH", "STATUS", "TOOL"}
            pieces = [
                f.get("content", "")
                for f in self.fragments
                if isinstance(f.get("content"), str)
                and f.get("content")
                and str(f.get("type", "RESPONSE")).upper() not in ignored
            ]
            text = "".join(pieces)
            if text:
                return text
        return self.direct_text


class DeepSeekClient:
    def __init__(self) -> None:
        self.hasher = DeepSeekHash().init(WASM_PATH)
        self.session = requests.Session()
        self.session.cookies.update(COOKIES)
        self.base_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-platform": "web",
            "x-client-version": "2.2.0",
            "x-app-version": "20241129.1",
            "x-client-locale": "en_US",
            "x-client-timezone-offset": "19800",
            "authorization": AUTH,
            "content-type": "application/json",
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

    @staticmethod
    def _biz_data(response: requests.Response) -> Any:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"DeepSeek returned invalid JSON: {response.text[:500]}") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            return data.get("biz_data")
        raise RuntimeError(f"DeepSeek response missing expected 'data' structure: {response.text[:500]}")

    def create_session(self) -> tuple[str, None]:
        response = self.session.post(
            "https://chat.deepseek.com/api/v0/chat_session/create",
            headers=self.base_headers,
            json={"character_id": None},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Unable to create chat session: {response.status_code} - {response.text[:500]}")
        biz_data = self._biz_data(response)
        session_id = None
        if isinstance(biz_data, dict):
            chat_session = biz_data.get("chat_session")
            if isinstance(chat_session, dict):
                session_id = chat_session.get("id")
            if not session_id:
                session_id = biz_data.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(f"DeepSeek did not return a chat session ID. Response: {response.text[:500]}")
        return session_id, None

    def _pow_header(self) -> Dict[str, str]:
        response = self.session.post(
            "https://chat.deepseek.com/api/v0/chat/create_pow_challenge",
            headers=self.base_headers,
            json={"target_path": "/api/v0/chat/completion"},
            timeout=30,
        )
        response.raise_for_status()
        biz_data = self._biz_data(response)
        try:
            challenge = biz_data["challenge"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected PoW response: {response.text[:500]}") from exc
        answer = self.hasher.solve(
            challenge["challenge"], challenge["salt"],
            challenge["difficulty"], challenge["expire_at"],
        )
        if answer is None:
            raise RuntimeError("Failed to solve the proof-of-work challenge.")
        payload = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }
        token = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode("ascii")
        return {**self.base_headers, "x-ds-pow-response": token}

    def stream_message(
        self,
        messages: list[dict],
        chat_session_id: str,
        parent_message_id: Optional[str],
        thinking_enabled: bool = False,
        search_enabled: bool = False,
    ) -> Generator[tuple[str, Optional[str]], None, None]:
        """
        Yields (delta_text, new_parent_message_id) tuples.
        new_parent_message_id is only set on the final yield.
        """
        # Build prompt from messages list (OpenAI format)
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
            if isinstance(m.get("content"), str)
        )
        # If last message is just user, send it directly
        if messages and messages[-1].get("role") == "user":
            prompt = messages[-1]["content"]

        payload = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "model_type": "default",
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "action": None,
            "preempt": False,
        }

        response = self.session.post(
            "https://chat.deepseek.com/api/v0/chat/completion",
            headers=self._pow_header(),
            json=payload,
            timeout=120,
            stream=True,
        )

        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek returned {response.status_code}: {response.text[:500]}")

        response.encoding = "utf-8"
        state = DeepSeekStreamState()
        emitted_len = 0
        data_lines: list[str] = []

        def handle_raw(raw: str) -> Optional[str]:
            nonlocal emitted_len
            raw = raw.strip()
            if not raw or raw == "[DONE]":
                return None
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                return None
            state.feed(event)
            current = state.visible_text()
            if len(current) > emitted_len:
                delta = current[emitted_len:]
                emitted_len = len(current)
                return delta
            return None

        for raw_line in response.iter_lines(chunk_size=8192, decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
            if line == "":
                if data_lines:
                    delta = handle_raw("\n".join(data_lines))
                    data_lines.clear()
                    if delta:
                        yield delta, None
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.lstrip().startswith(("{", "[")):
                if data_lines:
                    delta = handle_raw("\n".join(data_lines))
                    data_lines.clear()
                    if delta:
                        yield delta, None
                delta = handle_raw(line.strip())
                if delta:
                    yield delta, None

        if data_lines:
            delta = handle_raw("\n".join(data_lines))
            if delta:
                yield delta, None

        if state.server_error:
            raise RuntimeError(f"DeepSeek stream error: {state.server_error}")

        yield "", state.message_id
