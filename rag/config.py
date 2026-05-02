"""Single source of truth for paths, model IDs, and tunables."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# --- Corpus paths (ingest) ---
DATA_DIR = os.path.join(PROJECT_ROOT, "TSpec-LLM", "3GPP-clean")
CHUNKS_JSONL = os.path.join(BASE_DIR, "data", "chunks.jsonl")
CLAUSES_JSONL = os.path.join(BASE_DIR, "data", "clauses.jsonl")

# TR 21.905 glossary source (place file here or override via env)
TR_21905_PATH = os.environ.get(
    "TR_21905_PATH",
    os.path.join(PROJECT_ROOT, "data", "tr_21_905.txt"),
)
GLOSSARY_JSON = os.path.join(BASE_DIR, "data", "glossary.json")

# --- Qdrant ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "3gpp_docs")

# When True, indexer creates dense + sparse (BM25) named vectors; requires re-ingest.
HYBRID_INDEX = os.environ.get("HYBRID_INDEX", "0") == "1"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# --- Dense embeddings (BGE) ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
VECTOR_SIZE = 768

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

# --- vLLM / OpenAI-compatible generation ---
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
VLLM_TIMEOUT_S = float(os.environ.get("VLLM_TIMEOUT_S", "120"))

# --- Agent loop ---
MAX_TOOL_HOPS = int(os.environ.get("MAX_TOOL_HOPS", "3"))
