"""
Tool-calling loop: the model may request retrieval via XML tags, up to MAX_TOOL_HOPS,
then answers with citations grounded in returned chunks.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from rag.config import (
    MAX_TOOL_HOPS,
    RAG_LLM_MAX_TOKENS,
    RAG_TOOL_CONTEXT_CHARS,
    RETRIEVE_TOP_K,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_TIMEOUT_S,
)


def _openai_chat_url_and_ollama_origin() -> tuple[str, str]:
    """
    Normalize OpenAI-compatible URL and Ollama HTTP root for native /api/chat fallback.

    VLLM_BASE_URL may be ``http://host:11434`` or ``http://host:11434/v1``.
    OpenAI chat completions live at ``.../v1/chat/completions``.
    """
    b = VLLM_BASE_URL.rstrip("/")
    if b.endswith("/v1"):
        openai_root = b
        ollama_origin = b[:-3]
    else:
        openai_root = f"{b}/v1"
        ollama_origin = b
    return f"{openai_root}/chat/completions", ollama_origin.rstrip("/")


def _prefer_ollama_native_api() -> bool:
    """
    Use only POST .../api/chat (no OpenAI shim).

    Default is **False**: use OpenAI-compatible ``POST .../v1/chat/completions`` first (Ollama
    exposes this at ``http://localhost:11434/v1/...`` alongside ``GET /v1/models``).

    Set ``OLLAMA_USE_NATIVE=1`` if your server has no ``/v1`` routes; on 404 we still fall back
    to ``/api/chat`` when the URL contains port 11434.
    """
    explicit = os.environ.get("OLLAMA_USE_NATIVE", "").strip().lower()
    if explicit in ("1", "true", "yes"):
        return True
    return False

SYSTEM_PROMPT = (
    "You are a 3GPP / 5G technical assistant. Answer using the corpus retrieved for you.\n"
    "When you need specification text, request retrieval by outputting these tags exactly:\n"
    "<tool_call>retrieve</tool_call>\n"
    "<tool_query>your concise search query in English</tool_query>\n\n"
    f"You may call retrieval up to {MAX_TOOL_HOPS} times. After you have enough context, "
    "answer clearly and cite specs (e.g. TS 38.331) when chunk metadata includes a spec id.\n"
    "If retrieval returns nothing useful, say so honestly."
)


def _retrieve_dispatch(query: str, **kwargs: Any) -> dict[str, Any]:
    mode = os.environ.get("RAG_RETRIEVE_MODE", "local").lower()
    if mode == "http":
        from rag.retrieve.client import RetrieveClient

        return RetrieveClient().retrieve(query, **kwargs)
    from rag.retrieve.pipeline import retrieve

    return retrieve(query, **kwargs)


def _chat_completion(messages: list[dict[str, str]]) -> str:
    openai_url, ollama_origin = _openai_chat_url_and_ollama_origin()
    body_openai: dict[str, Any] = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    if RAG_LLM_MAX_TOKENS > 0:
        body_openai["max_tokens"] = RAG_LLM_MAX_TOKENS

    ollama_opts: dict[str, Any] = {"temperature": 0.2}
    if RAG_LLM_MAX_TOKENS > 0:
        ollama_opts["num_predict"] = RAG_LLM_MAX_TOKENS

    body_ollama = {
        "model": VLLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": ollama_opts,
    }

    with httpx.Client(timeout=VLLM_TIMEOUT_S) as c:
        if _prefer_ollama_native_api():
            r = c.post(f"{ollama_origin}/api/chat", json=body_ollama)
            r.raise_for_status()
            data = r.json()
            return (data.get("message") or {}).get("content") or ""

        r = c.post(openai_url, json=body_openai)
        if r.status_code == 404 and (
            "11434" in openai_url
            or os.environ.get("OLLAMA_FALLBACK_NATIVE", "").lower() in ("1", "true", "yes")
        ):
            r2 = c.post(f"{ollama_origin}/api/chat", json=body_ollama)
            r2.raise_for_status()
            data = r2.json()
            return (data.get("message") or {}).get("content") or ""

        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"] or ""


_TOOL_RE = re.compile(
    r"<tool_call>\s*retrieve\s*</tool_call>\s*"
    r"<tool_query>\s*(.*?)\s*</tool_query>",
    re.IGNORECASE | re.DOTALL,
)


def run_agent(user_question: str) -> dict[str, Any]:
    """
    Run retrieval-augmented generation with up to MAX_TOOL_HOPS retrieve calls.
    Returns assistant text plus transcript for debugging.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]
    transcript: list[dict[str, Any]] = []
    hops = 0

    while True:
        assistant_text = _chat_completion(messages)
        transcript.append({"role": "assistant", "content": assistant_text})

        m = _TOOL_RE.search(assistant_text)
        if not m:
            return {
                "answer": assistant_text,
                "transcript": transcript,
                "tool_hops": hops,
            }
        if hops >= MAX_TOOL_HOPS:
            return {
                "answer": assistant_text,
                "transcript": transcript,
                "tool_hops": hops,
            }

        rq = m.group(1).strip()
        result = _retrieve_dispatch(rq, top_k=RETRIEVE_TOP_K)
        hops += 1
        tool_msg = json.dumps(result, ensure_ascii=False)[:RAG_TOOL_CONTEXT_CHARS]
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append(
            {
                "role": "user",
                "content": f"[retrieve result for query: {rq}]\n{tool_msg}",
            }
        )
        transcript.append(
            {
                "role": "tool",
                "retrieve_query": rq,
                "preview": tool_msg[:2000],
            }
        )


def generate_answer_only(user_question: str, contexts: list[dict[str, Any]]) -> str:
    """Single-shot generation from pre-fetched hits (no tool loop)."""
    ctx = json.dumps(contexts, ensure_ascii=False)[:RAG_TOOL_CONTEXT_CHARS]
    messages = [
        {
            "role": "system",
            "content": "Answer using only the provided JSON context. Cite spec ids when present.",
        },
        {
            "role": "user",
            "content": f"Question: {user_question}\n\nContext:\n{ctx}",
        },
    ]
    return _chat_completion(messages)
