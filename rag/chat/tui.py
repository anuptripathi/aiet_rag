"""Simple terminal REPL for the RAG agent."""

import argparse

from rag.chat.agent import run_agent


def main():
    p = argparse.ArgumentParser(description="Telecom RAG terminal chat")
    p.add_argument(
        "--single",
        type=str,
        default="",
        help="Ask one question and exit (non-interactive)",
    )
    args = p.parse_args()

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
