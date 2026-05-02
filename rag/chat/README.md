# Chat REPL

End-to-end **retrieve → generate** using an OpenAI-compatible **vLLM** server.

## Prerequisites

1. Qdrant running with your ingested `3gpp_docs` collection (or hybrid collection if you used `HYBRID_INDEX=1`).
2. An **OpenAI-compatible** chat API: default config targets **Ollama** at `VLLM_BASE_URL` (`http://localhost:11434/v1`) with `llama3.2:1b` — see `serve/README-serve.md` and `serve/docker-compose.ollama-cpu.yml` for CPU / 16 GB RAM setups.
3. Environment variables as needed:
   - `VLLM_MODEL` — must match the tag your server reports (e.g. `llama3.2:1b`).
   - `RAG_RETRIEVE_MODE=local` (default) calls retrieval in-process; use `http` if you run `python -m rag.retrieve.service` separately and set `RETRIEVE_BASE_URL`.

## Usage

```bash
# Interactive
python -m rag.chat.tui

# One-shot
python -m rag.chat.tui --single "What is RRC connection setup?"
```

The agent prompts the LLM to emit `<tool_call>retrieve</tool_call>` / `<tool_query>...</tool_query>` when it needs 3GPP text, then merges retrieval JSON into the conversation (up to `MAX_TOOL_HOPS`, default 3).

## Speed (CPU / large Qdrant indexes)

Slowness is usually **LLM decoding on CPU**, plus **cross-encoder rerank** and **extra retrieval passes** — not Qdrant scanning millions of points (ANN search stays fast).

Quick wins:

- **`python -m rag.chat.tui --fast --single "..."`** — sets `RAG_FAST=1`, skips rerank / glossary / parent-expand / second retrieval pass, smaller candidate pool.
- **`USE_RERANKER=0`** — avoids loading a second transformer for reranking (large win on RAM and latency).
- **`MAX_TOOL_HOPS=1`** — one retrieve round instead of up to three.
- **`RETRIEVE_TOP_K=3`** — fewer chunks → shorter prompts → faster generation.
- **`RAG_LLM_MAX_TOKENS=512`** — shorter answers, much faster on CPU Ollama.
- **`RAG_TOOL_CONTEXT_CHARS=8000`** — smaller retrieve JSON in the prompt.

Example (PowerShell):

```powershell
$env:RAG_FAST="1"; $env:USE_RERANKER="0"; $env:MAX_TOOL_HOPS="1"
$env:RAG_LLM_MAX_TOKENS="512"; $env:RETRIEVE_TOP_K="3"
python -m rag.chat.tui --single "What is RRC connection setup?"
```
