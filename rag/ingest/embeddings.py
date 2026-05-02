from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_batch(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()