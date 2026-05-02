"""Verify Qdrant is up, collection exists, and a query returns points."""

import os
import sys

# Allow `python serve/smoke_qdrant.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_HOST, QDRANT_PORT


def main():
    c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    col = c.get_collection(COLLECTION_NAME)
    n = col.points_count
    print(f"OK: collection {COLLECTION_NAME!r} has {n} points")

    m = SentenceTransformer(EMBEDDING_MODEL)
    v = m.encode("RRC connection setup", normalize_embeddings=True).tolist()
    r = c.query_points(collection_name=COLLECTION_NAME, query=v, limit=2)
    assert r.points, "query returned no points"
    print(f"OK: query returned {len(r.points)} hit(s), top score {r.points[0].score}")


if __name__ == "__main__":
    main()
