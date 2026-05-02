"""Simple terminal REPL for the RAG agent."""

import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Telecom RAG terminal chat")
    p.add_argument(
        "--single",
        type=str,
        default="",
        help="Ask one question and exit (non-interactive)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Set RAG_FAST=1 and USE_RERANKER=0 for lower latency (see rag/config.py)",
    )
    args = p.parse_args()

    if args.fast:
        os.environ["RAG_FAST"] = "1"
        os.environ["USE_RERANKER"] = "0"

    # Import after env so rag.config sees RAG_FAST / USE_RERANKER when loaded.
    from rag.chat.agent import run_agent

    if args.single:
        out = run_agent(args.single)
        print(out["answer"])
        return

    print("Telecom RAG — type a question (empty line to quit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        out = run_agent(q)
        print("\nAssistant:\n", out["answer"])


if __name__ == "__main__":
    main()
