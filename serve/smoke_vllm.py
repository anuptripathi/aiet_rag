"""Verify an OpenAI-compatible vLLM endpoint responds."""

import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import VLLM_BASE_URL, VLLM_MODEL


def main():
    base = VLLM_BASE_URL.rstrip("/")
    with httpx.Client(timeout=30.0) as c:
        r = c.get(f"{base}/models")
        r.raise_for_status()
        print("OK: GET /models ->", r.status_code)

        body = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Say OK in one word."}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        }
        r2 = c.post(f"{base}/chat/completions", json=body)
        r2.raise_for_status()
        data = r2.json()
        txt = data["choices"][0]["message"]["content"]
        print("OK: chat completion:", repr(txt[:120]))
        print(json.dumps(data, indent=2)[:500])


if __name__ == "__main__":
    main()
