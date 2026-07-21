# DeepSeek OpenAI-compatible API

Wraps DeepSeek's private browser API as an OpenAI-compatible `/v1/chat/completions` endpoint.

## Files

```
deepseek-api/
├── deepseek.py          # Core client
├── main.py              # FastAPI app
├── requirements.txt
├── Procfile
└── sha3_wasm_bg.wasm    # ← you must add this
```

## Deploy on Railway

1. Push this folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Set environment variables (optional but recommended):
   - `API_KEY` — key callers must send as `Authorization: Bearer <key>`
   - `DEEPSEEK_AUTH` — your DeepSeek bearer token
   - `DS_SESSION_ID` — ds_session_id cookie
   - `DS_WAF_TOKEN` — aws-waf-token cookie
   - `DS_THUMBCACHE` — thumbcache cookie
   - `DS_SMID` — smidV2 cookie
4. Railway auto-detects the Procfile and runs it

**Important:** make sure `sha3_wasm_bg.wasm` is committed to the repo alongside the Python files.

## Usage

```python
import openai

client = openai.OpenAI(
    base_url="https://your-app.up.railway.app/v1",
    api_key="your-api-key",  # or anything if API_KEY not set
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# With thinking (R1-style)
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Solve this step by step..."}],
)
```

## Notes

- The single DeepSeek session is shared across all requests — concurrent requests may interleave
- When cookies/auth expire, update the env vars and redeploy
- `deepseek-reasoner` model name enables thinking mode
