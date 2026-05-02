## change env

```bash
.venv\Scripts\activate
```

## basic useful command

```bash
# -m tells Python → “this is a package module”
# Now rag is recognized as a package
# Make sure you have:
# rag/__init__.py
# rag/ingest/__init__.py
# Even if empty:
# touch rag/__init__.py
# touch rag/ingest/__init__.py

python -m rag.ingest.chunker
python -m rag/ingest/indexer.py

# see count in db.
python .\test\data-verify.py

# last few data in db.
#simple
python .\test\data-tail.py -n 5
#with collection name
python .\test\data-tail.py --collection 3gpp_docs -n 5
#with batch (faster if collection is large)
python .\test\data-tail.py -n 5 --batch 1000
```

## required isntallations

```bash
#create ollma container
docker compose -f serve/docker-compose.ollama-cpu.yml up -d
# take model pull in the above container.
docker compose -f serve/docker-compose.ollama-cpu.yml exec ollama ollama pull llama3.2:1b
python serve/smoke_vllm.py
python -m rag.chat.tui --single "What is RRC connection setup?"
```

## run retrieval faster

```bash
$env:RAG_LLM_MAX_TOKENS="512"
$env:MAX_TOOL_HOPS="1"
$env:RETRIEVE_TOP_K="3"
python -m rag.chat.tui --fast --single "What is RRC connection setup?"

#Or set USE_RERANKER=0 globally if you do not use --fast.
```
