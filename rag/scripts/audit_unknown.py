"""Scan Qdrant for points missing spec/domain metadata or placeholder values."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_client import QdrantClient
from rag.config import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    bad = 0
    total = 0
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        for rec in records:
            total += 1
            pl = rec.payload or {}
            spec = pl.get("spec")
            dom = pl.get("domain")
            if spec is None or spec == "" or spec.lower() == "unknown":
                bad += 1
                print(f"id={rec.id} chunk_id={pl.get('chunk_id')} missing/bad spec={spec!r}")
            if dom is None or dom == "":
                bad += 1
                print(f"id={rec.id} chunk_id={pl.get('chunk_id')} missing domain")
        if offset is None:
            break

    print(f"\nScanned {total} points; flagged rows printed above (may double-count payload issues).")


if __name__ == "__main__":
    main()
