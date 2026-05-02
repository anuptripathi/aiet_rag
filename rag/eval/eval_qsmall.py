"""
Placeholder Q-small-style benchmark.

Place a JSONL file at `rag/data/qsmall.jsonl` where each line is like:
{"question": "...", "gold_chunk_ids": ["..."], "gold_specs": ["38.331"]}

Then run:
  python -m rag.eval.eval_qsmall
"""

from __future__ import annotations

import json
import os
from typing import Any

from rag.config import BASE_DIR
from rag.retrieve.pipeline import retrieve

QS_PATH = os.path.join(BASE_DIR, "data", "qsmall.jsonl")


def recall_at_k(retrieved_ids: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved_ids[:k])
    hit = sum(1 for g in gold if g in top)
    return hit / len(gold)


def run_benchmark(path: str | None = None) -> dict[str, Any]:
    path = path or QS_PATH
    if not os.path.isfile(path):
        return {
            "error": f"No benchmark file at {path}",
            "hint": "Create JSONL with question + gold_chunk_ids per line.",
        }

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    k = 5
    scores = []
    for row in rows:
        q = row["question"]
        gold = row.get("gold_chunk_ids") or []
        out = retrieve(q, top_k=k, expand_parents=False, cross_ref_second_pass=False)
        ids = [h.get("chunk_id") for h in out.get("hits") or [] if h.get("chunk_id")]
        scores.append(recall_at_k(ids, gold, k))

    avg = sum(scores) / len(scores) if scores else 0.0
    return {"n": len(rows), f"recall@{k}": avg, "per_question": scores}


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=QS_PATH)
    args = p.parse_args()
    r = run_benchmark(args.path)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
