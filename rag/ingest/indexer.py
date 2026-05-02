import json
import os
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rag.config import *
from rag.ingest.embeddings import embed_batch

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection(recreate: bool = False):
    if recreate:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        return

    try:
        client.get_collection(COLLECTION_NAME)
        return
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def load_chunks():
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def _point_id(chunk_id: str) -> str:
    # Qdrant point IDs must be int or UUID.
    # uuid5 gives a deterministic UUID for a given chunk_id, enabling idempotent reruns.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

def _flush_batch(batch):
    if not batch:
        return 0, 0

    ids = [_point_id(rec["id"]) for rec in batch]

    existing = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids,
        with_payload=False,
        with_vectors=False,
    )
    existing_ids = {str(p.id) for p in existing}

    to_insert = [rec for rec in batch if _point_id(rec["id"]) not in existing_ids]
    if not to_insert:
        return 0, len(batch)

    texts = [rec["text"] for rec in to_insert]
    vectors = embed_batch(texts)

    points = [
        PointStruct(
            id=_point_id(rec["id"]),
            vector=vec,
            payload={
                "chunk_id": rec["id"],
                "text": rec["text"],
                **rec["metadata"],
            },
        )
        for rec, vec in zip(to_insert, vectors)
    ]

    client.upsert(COLLECTION_NAME, points)
    return len(to_insert), (len(batch) - len(to_insert))

def index():
    batch = []
    inserted = 0
    skipped = 0

    for record in tqdm(load_chunks()):
        batch.append(record)

        if len(batch) == 64:
            ins, skip = _flush_batch(batch)
            inserted += ins
            skipped += skip
            batch = []

    if batch:
        ins, skip = _flush_batch(batch)
        inserted += ins
        skipped += skip

    print(f"✅ Inserted: {inserted} | ⏭️ Skipped (already present): {skipped}")

if __name__ == "__main__":
    ensure_collection(recreate=os.getenv("QDRANT_RECREATE", "0") == "1")
    index()