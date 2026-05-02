"""
End-to-end single-question runner: retrieve (local pipeline) + optional generate.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from rag.chat.agent import generate_answer_only
from rag.retrieve.pipeline import retrieve


def run_one(
    question: str,
    *,
    top_k: int = 5,
    generate: bool = False,
) -> dict[str, Any]:
    r = retrieve(question, top_k=top_k)
    out: dict[str, Any] = {"question": question, "retrieve": r}
    if generate:
        out["answer"] = generate_answer_only(question, r.get("hits") or [])
    return out


def main():
    p = argparse.ArgumentParser(description="Run one RAG question end-to-end")
    p.add_argument("question", nargs="?", default="", help="Question text")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--generate", action="store_true", help="Call vLLM after retrieval")
    p.add_argument("--out", type=str, default="", help="Write JSON result to path")
    args = p.parse_args()
    q = args.question.strip() or os.environ.get("RAG_EVAL_QUESTION", "").strip()
    if not q:
        q = input("Question: ").strip()
    if not q:
        raise SystemExit("No question provided.")
    result = run_one(q, top_k=args.top_k, generate=args.generate)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
