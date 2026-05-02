"""
Smoke-test the LLM endpoint used by rag.chat.

Expected behavior (success)
----------------------------
**Default** (OpenAI-compatible API — Ollama and vLLM both support this):

1. ``GET http://localhost:11434/v1/models`` → **200**
2. ``POST http://localhost:11434/v1/chat/completions`` → **200**

**Native Ollama only** (set ``OLLAMA_USE_NATIVE=1``):

1. ``GET .../api/tags`` → **200**
2. ``POST .../api/chat`` → **200**

Common failures
---------------
- **404 on chat** — usually **model not installed**. Run ``ollama pull <VLLM_MODEL>``.
- **405 on GET /api/chat** — normal; chat endpoints expect **POST**.
"""

import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chat.agent import _openai_chat_url_and_ollama_origin, _prefer_ollama_native_api
from rag.config import VLLM_BASE_URL, VLLM_MODEL, VLLM_TIMEOUT_S


def _ollama_installed_models(tags_json: dict) -> list[str]:
    out = []
    for m in tags_json.get("models") or []:
        n = m.get("name")
        if n:
            out.append(n)
    return out


def main():
    _, ollama_origin = _openai_chat_url_and_ollama_origin()

    with httpx.Client(timeout=VLLM_TIMEOUT_S) as c:
        if _prefer_ollama_native_api():
            r = c.get(f"{ollama_origin}/api/tags")
            r.raise_for_status()
            tags = r.json()
            print("OK: GET /api/tags ->", r.status_code)
            installed = _ollama_installed_models(tags)
            print("   Installed models:", ", ".join(installed) if installed else "(none)")
            if installed and VLLM_MODEL not in installed:
                print(
                    f"   WARNING: VLLM_MODEL={VLLM_MODEL!r} not in list. "
                    f"Fix: ollama pull {VLLM_MODEL}  OR  set VLLM_MODEL to one of the names above."
                )

            r2 = c.post(
                f"{ollama_origin}/api/chat",
                json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": "Say OK in one word."}],
                    "stream": False,
                    "options": {"num_predict": 16, "temperature": 0},
                },
            )
            if r2.status_code != 200:
                print("FAIL: POST /api/chat ->", r2.status_code, r2.text[:500])
                if r2.status_code == 404:
                    print(
                        "\nMost likely: model is not pulled. Example:\n"
                        f"  ollama pull {VLLM_MODEL}\n"
                        "Or use an exact name from 'Installed models' above."
                    )
                raise SystemExit(1)

            data = r2.json()
            txt = (data.get("message") or {}).get("content") or ""
            print("OK: POST /api/chat:", repr(txt[:120]))
            print(json.dumps(data, indent=2)[:500])
            return

        openai_base = VLLM_BASE_URL.rstrip("/")
        if not openai_base.endswith("/v1"):
            openai_base = f"{openai_base}/v1"
        r = c.get(f"{openai_base}/models")
        r.raise_for_status()
        print("OK: GET /v1/models ->", r.status_code)

        body = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Say OK in one word."}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        }
        r2 = c.post(f"{openai_base}/chat/completions", json=body)
        if r2.status_code != 200:
            print("FAIL:", r2.status_code, r2.text[:500])
            raise SystemExit(1)
        data = r2.json()
        txt = data["choices"][0]["message"]["content"]
        print("OK: POST /v1/chat/completions:", repr(txt[:120]))
        print(json.dumps(data, indent=2)[:500])


if __name__ == "__main__":
    main()
