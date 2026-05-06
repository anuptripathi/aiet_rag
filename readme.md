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
python -m rag.ingest.indexer

## More customized commands
#run chunker with different values for no. of chunk words and overlap words
$env:CHUNK_TARGET_WORDS="180";
$env:CHUNK_OVERLAP_WORDS="36";
python -m rag.ingest.chunker

$env:QDRANT_RECREATE = "1" # or '0'
python -m rag.ingest.indexer

# see count in db.
python .\test\data-verify.py

# last few data in db.
#simple
python .\test\data-tail.py -n 5
#with collection name
python .\test\data-tail.py --collection 3gpp_docs2 -n 5
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
#with fast enabled, glossary, parents, ref search is off
if _rag_fast_env():
    use_glossary = False
    expand_parents = False
    cross_ref_second_pass = False

```

## run evaluation

```bash
python -m test.mcq_eval --input "c:\Users\Anup Tripathi\Downloads\Sampled_3GPP_TR_Questions.json" --fast

# Run only first 20
python -m test.mcq_eval --input "c:\Users\Anup Tripathi\Downloads\Sampled_3GPP_TR_Questions.json" --fast --limit 20

# Add a small delay between questions
python -m test.mcq_eval --input "c:\Users\Anup Tripathi\Downloads\Sampled_3GPP_TR_Questions.json" --fast --sleep 0.2

# Write output somewhere else
python -m test.mcq_eval --input "c:\Users\Anup Tripathi\Downloads\Sampled_3GPP_TR_Questions.json" --fast --out "..\\rag\\data\\mcq_results.jsonl" --limit 2
```

## Split chunk if required (not so useful)

```bash
# you can split chunks
python -m rag.ingest.chunk_splitter
# custom:
python -m rag.ingest.chunk_splitter -i ".\rag\data\chunks.jsonl" -o ".\rag\data" -m 20000 --prefix chunks
# separate folder for shards:
python -m rag.ingest.chunk_splitter -o ".\rag\data\shards"

#run indexer to truncate existing collection.
$env:QDRANT_RECREATE="1"; $env:HYBRID_INDEX="1"; python -m rag.ingest.indexer

#you can index chunks form shards
$root = "C:\Users\Anup Tripathi\Documents\code\aiet\rag\data\shards"
$env:HYBRID_INDEX = "1"

$env:CHUNKS_JSONL = "$root\chunks1.jsonl"
$env:FORCE_CHUNK_UPSERT="1"
$env:QDRANT_RECREATE = "1"
python -m rag.ingest.indexer

$env:QDRANT_RECREATE = "0"
$env:FORCE_CHUNK_UPSERT= "0"
$env:CHUNKS_JSONL = "$root\chunks2.jsonl"
python -m rag.ingest.indexer
# … repeat for chunks3, …
```

## other useful commands

```bash
#clear .env on windows
Remove-Item Env:CHUNKS_JSONL
Remove-Item Env:QDRANT_RECREATE
Remove-Item Env:HYBRID_INDEX
Remove-Item Env:FORCE_CHUNK_UPSERT

#check what's in .env
Get-ChildItem Env:CHUNKS_JSONL,Env:QDRANT_RECREATE,Env:HYBRID_INDEX,Env:FORCE_CHUNK_UPSERT

# curl command windos for delete or get collectoin
Invoke-RestMethod -Method Delete -Uri "http://localhost:6333/collections/3gpp_docs"
Invoke-RestMethod -Method Get -Uri "http://localhost:6333/collections"
```
