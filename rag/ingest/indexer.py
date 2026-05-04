import json
import os
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
)
from qdrant_client.http import models as qm

from rag.config import (
    CHUNKS_JSONL,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    FORCE_CHUNK_UPSERT,
    HYBRID_INDEX,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_RECREATE,
    SPARSE_EMBEDDING_MODEL,
    SPARSE_VECTOR_NAME,
    VECTOR_SIZE,
)
from rag.ingest.embeddings import embed_batch

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _sparse_embed_batch(texts: list[str]) -> list[SparseVector]:
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(SPARSE_EMBEDDING_MODEL)
    out: list[SparseVector] = []
    for emb in model.embed(texts):
        idx = emb.indices
        val = emb.values
        if hasattr(idx, "tolist"):
            idx = idx.tolist()
        if hasattr(val, "tolist"):
            val = val.tolist()
        out.append(
            SparseVector(
                indices=[int(i) for i in idx],
                values=[float(x) for x in val],
            )
        )
    return out


def ensure_collection(recreate: bool = False):
    if not HYBRID_INDEX:
        if recreate:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                ),
            )
            return

        try:
            client.get_collection(COLLECTION_NAME)
            return
        except Exception:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                ),
            )
        return

    # Hybrid: named dense + sparse BM25
    if recreate:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    modifier=qm.Modifier.IDF
                )
            },
        )
        return

    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    modifier=qm.Modifier.IDF
                )
            },
        )


def load_chunks():
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _flush_batch(batch):
    if not batch:
        return 0, 0

    ids = [_point_id(rec["id"]) for rec in batch]

    if FORCE_CHUNK_UPSERT:
        to_insert = batch
    else:
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
    dense_vecs = embed_batch(texts)

    if HYBRID_INDEX:
        sparse_vecs = _sparse_embed_batch(texts)
        points = [
            PointStruct(
                id=_point_id(rec["id"]),
                vector={
                    DENSE_VECTOR_NAME: dvec,
                    SPARSE_VECTOR_NAME: svec,
                },
                payload={
                    "chunk_id": rec["id"],
                    "text": rec["text"],
                    **rec["metadata"],
                },
            )
            for rec, dvec, svec in zip(to_insert, dense_vecs, sparse_vecs)
        ]
    else:
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
            for rec, vec in zip(to_insert, dense_vecs)
        ]

    client.upsert(COLLECTION_NAME, points)
    return len(to_insert), (len(batch) - len(to_insert))


def index():
    if HYBRID_INDEX:
        try:
            import fastembed  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HYBRID_INDEX=1 requires `pip install fastembed` for BM25 sparse vectors."
            ) from e

    if FORCE_CHUNK_UPSERT:
        print("⚙️  FORCE_CHUNK_UPSERT=1: upserting all rows (re-embed even if ids exist).")

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
    ensure_collection(recreate=QDRANT_RECREATE)
    index()
