"""Single source of truth for paths, model IDs, and tunables."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# --- Corpus paths (ingest) ---
DATA_DIR = os.path.join(PROJECT_ROOT, "TSpec-LLM", "3GPP-subset")
_DEFAULT_CHUNKS_JSONL = os.path.join(BASE_DIR, "data", "chunks.jsonl")
# Written by chunker; default file for indexer when env unset.
CHUNKS_JSONL = os.path.abspath(_DEFAULT_CHUNKS_JSONL)
# Indexer input only: set CHUNKS_JSONL env to a shard (chunk_splitter output).
INGEST_CHUNKS_JSONL = os.path.abspath(
    os.environ.get("CHUNKS_JSONL", CHUNKS_JSONL)
)
CLAUSES_JSONL = os.path.join(BASE_DIR, "data", "clauses.jsonl")

# Chunking (ingest/chunker): word windows with overlap; ~200 words ≈ typical embedding budget
CHUNK_TARGET_WORDS = int(os.environ.get("CHUNK_TARGET_WORDS", "200"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", "40"))

# TR 21.905 glossary source (place file here or override via env)
TR_21905_PATH = os.environ.get(
    "TR_21905_PATH",
    os.path.join(PROJECT_ROOT, "data", "tr_21_905.txt"),
)
GLOSSARY_JSON = os.path.join(PROJECT_ROOT, "data", "glossary.json")

# --- Qdrant ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "3gpp_docs2")

# When True, indexer will recreate the collection.
QDRANT_RECREATE = os.environ.get("QDRANT_RECREATE", "0") == "1"
# When True, indexer upserts every chunk (re-embeds) even if point id already exists.
FORCE_CHUNK_UPSERT = os.environ.get("FORCE_CHUNK_UPSERT", "0") == "1"
# When True, indexer creates dense + sparse (BM25) named vectors; requires re-ingest.
HYBRID_INDEX = os.environ.get("HYBRID_INDEX", "0") == "1"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# --- Dense embeddings (BGE) ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") #BAAI/bge-base-en-v1.5
# Must match the embedding model output dimension (e.g. bge-base=768, bge-small=384).
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "384")) #768

# --- Sparse / BM25 (FastEmbed, must match hybrid indexer) ---
SPARSE_EMBEDDING_MODEL = os.environ.get("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")

# --- Retrieval service (FastAPI) ---
RETRIEVE_BIND_HOST = os.environ.get("RETRIEVE_BIND_HOST", "0.0.0.0")
RETRIEVE_PORT = int(os.environ.get("RETRIEVE_PORT", "8088"))
RETRIEVE_BASE_URL = os.environ.get(
    "RETRIEVE_BASE_URL", f"http://127.0.0.1:{RETRIEVE_PORT}"
)

# Fusion + rerank
RRF_K = int(os.environ.get("RRF_K", "60"))
PREFETCH_LIMIT = int(os.environ.get("PREFETCH_LIMIT", "40"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "40"))
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
USE_RERANKER = os.environ.get("USE_RERANKER", "1") == "1"
MAX_PARENT_EXPAND = int(os.environ.get("MAX_PARENT_EXPAND", "12"))

# Latency (CPU / small laptops): set RAG_FAST=1 or tune env vars below
RAG_FAST = os.environ.get("RAG_FAST", "0") == "1"
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "5"))
# Cap LLM output tokens (0 = server default). Helps a lot on CPU Ollama.
RAG_LLM_MAX_TOKENS = int(os.environ.get("RAG_LLM_MAX_TOKENS", "0"))
# Truncate JSON passed back into the chat after retrieve (characters)
RAG_TOOL_CONTEXT_CHARS = int(os.environ.get("RAG_TOOL_CONTEXT_CHARS", "24000"))

# --- LLM: OpenAI-compatible API (Ollama, vLLM, LM Studio, etc.) ---
# Defaults match serve/docker-compose.ollama-cpu.yml (CPU-friendly).
# Ollama: ``http://localhost:11434`` or ``http://localhost:11434/v1`` (OpenAI routes under /v1).
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11434").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama3.2:1b")
VLLM_TIMEOUT_S = float(os.environ.get("VLLM_TIMEOUT_S", "300"))

# --- Agent loop ---
MAX_TOOL_HOPS = int(os.environ.get("MAX_TOOL_HOPS", "3"))
# "simple" = always similarity-search then answer (no LLM tool XML).
# "tool" = legacy loop where the model emits <tool_call>retrieve</tool_call>.
RAG_AGENT_MODE = os.environ.get("RAG_AGENT_MODE", "simple").strip().lower()
