import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "../TSpec-LLM/3GPP-clean")

CHUNKS_JSONL = os.path.join(BASE_DIR, "data/chunks.jsonl")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "3gpp_docs"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768