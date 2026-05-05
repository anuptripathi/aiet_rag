from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_batch(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()