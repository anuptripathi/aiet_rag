import argparse
from collections import deque

from qdrant_client import QdrantClient


def main():
    parser = argparse.ArgumentParser(
        description="Print last N points encountered while scrolling Qdrant collection."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", default="3gpp_docs")
    parser.add_argument("-n", type=int, default=5, help="Number of records to print")
    parser.add_argument("--batch", type=int, default=256, help="Scroll batch size")
    args = parser.parse_args()

    client = QdrantClient(args.host, port=args.port)

    tail = deque(maxlen=max(args.n, 1))
    next_offset = None
    scanned = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=args.collection,
            limit=args.batch,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for p in points:
            tail.append(p)
            scanned += 1

        if next_offset is None:
            break

    print(f"scanned={scanned} tail={len(tail)}")
    for i, p in enumerate(tail, start=max(scanned - len(tail) + 1, 1)):
        payload = p.payload or {}
        text = payload.get("text") or ""
        text_preview = (text[:160] + "…") if len(text) > 160 else text
        chunk_id = payload.get("chunk_id") or payload.get("id")
        spec = payload.get("spec")
        domain = payload.get("domain")
        path = payload.get("path")

        print("-" * 80)
        print(f"#{i} point_id={p.id}")
        if chunk_id:
            print(f"  chunk_id={chunk_id}")
        if spec or domain:
            print(f"  spec={spec} domain={domain}")
        if path:
            print(f"  path={path}")
        if text_preview:
            print(f"  text={text_preview}")


if __name__ == "__main__":
    main()

