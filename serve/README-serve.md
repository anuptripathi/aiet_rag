# Smoke tests & infra

## Qdrant

Run a local Qdrant (Docker example):

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Build your collection with ingest (`chunks.jsonl` → indexer). Default collection name: `3gpp_docs`.

Smoke check:

```bash
python serve/smoke_qdrant.py
```

## vLLM (OpenAI-compatible API)

Start vLLM so it exposes `VLLM_BASE_URL` (default `http://localhost:8000/v1`). Set `VLLM_MODEL` to the served model id.

Example (adjust GPU / model paths):

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000
```

Smoke check:

```bash
python serve/smoke_vllm.py
```

## Retrieval API (optional)

```bash
python -m rag.retrieve.service
```

Health: `GET http://127.0.0.1:8088/healthz`  
Retrieve: `POST http://127.0.0.1:8088/retrieve` with JSON body `{"query":"...", "top_k":5}`
