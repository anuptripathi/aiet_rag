# Smoke tests & infra

## LLM on a 16 GB RAM laptop (no GPU) — **recommended**

Use **[Ollama](https://ollama.com)** in Docker with a **tiny instruct model**. Generation is **CPU-only**, so answers are slow but fine for local demos; **retrieval quality** still drives your RAG.

### 1. Start Ollama

From the repo root:

```powershell
docker compose -f serve/docker-compose.ollama-cpu.yml up -d
```

### 2. Pull a small model (once)

```powershell
docker compose -f serve/docker-compose.ollama-cpu.yml exec ollama ollama pull llama3.2:1b
```

**Alternatives** if `1b` is too weak or you have a bit more RAM headroom (still CPU):

- `llama3.2:3b` — better answers, heavier (may be tight on 16 GB with Qdrant + embeddings running).

### 3. Wire env vars (defaults in `rag/config.py` already match Ollama)

```powershell
$env:VLLM_BASE_URL = "http://localhost:11434/v1"
$env:VLLM_MODEL = "llama3.2:1b"
python serve/smoke_vllm.py
python -m rag.chat.tui --single "What is RRC?"
```

**Low-RAM tips:** Run Qdrant and retrieval **without** loading extra models at the same time if you see swapping. You can set `USE_RERANKER=0` to skip the cross-encoder reranker in `rag/retrieve/pipeline.py` and save RAM.

**Ollama URLs:** Current Ollama exposes OpenAI-compatible **`GET /v1/models`** and **`POST /v1/chat/completions`** under `http://localhost:11434/v1/...`. The chat agent uses those **by default** (same as vLLM). Use **`OLLAMA_USE_NATIVE=1`** only if you must force **`POST /api/chat`** with no `/v1` shim.

---

## Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Ingest your corpus (`chunks.jsonl` → indexer). Default collection: `3gpp_docs`.

```bash
python serve/smoke_qdrant.py
```

---

## Optional: Google Gemma 4 + vLLM (NVIDIA GPU only)

Gemma 4 is **not** aimed at 16 GB RAM / CPU-only machines. If you have a **suitable GPU**, see `docker-compose.gemma4.yml` and the [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html).

---

## Optional: Generic vLLM (GPU)

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000
```

Set `VLLM_BASE_URL=http://localhost:8000/v1` and the matching `VLLM_MODEL`.

```bash
python serve/smoke_vllm.py
```

---

## Retrieval API (optional)

```bash
python -m rag.retrieve.service
```

Health: `GET http://127.0.0.1:8088/healthz`  
Retrieve: `POST http://127.0.0.1:8088/retrieve` with JSON `{"query":"...", "top_k":5}`
