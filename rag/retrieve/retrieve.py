from rag.retrieve.pipeline import legacy_search as search


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