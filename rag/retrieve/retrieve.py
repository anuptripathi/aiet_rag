from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rag.config import *

# ---------------------------
# Init
# ---------------------------

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

model = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------
# Embed query
# ---------------------------

def embed_query(query):
    return model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


# ---------------------------
# Search Qdrant
# ---------------------------

def search(query, top_k=5):

    query_vector = embed_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    formatted = []

    for r in response.points:

        payload = r.payload

        formatted.append({
            "score": r.score,
            "text": payload.get("text"),
            "spec": payload.get("spec"),
            "domain": payload.get("domain"),
            "chunk_id": payload.get("chunk_id")
        })

    return formatted


# ---------------------------
# CLI Test
# ---------------------------

if __name__ == "__main__":

    query = input("🔍 Query: ")

    results = search(query)

    print("\n=====================\n")

    for i, r in enumerate(results):

        print(f"Result #{i+1}")
        print(f"Score: {r['score']}")
        print(f"Spec: {r['spec']}")
        print(f"Domain: {r['domain']}")
        print()
        print(r["text"][:1000])
        print("\n-------------------\n")