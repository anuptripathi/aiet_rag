from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

print(client.count("3gpp_docs2"))