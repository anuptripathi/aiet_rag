"""
Evaluate RAG performance on MCQ-style question sets.

Input format (your file):
{
  "question_1": {
    "question": "...",
    "option_1": "...",
    "option_2": "...",
    "option_3": "...",
    "option_4": "...",
    "answer": "option_2: ...",
    ...
  },
  ...
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable


_OPT_RE = re.compile(r"\boption_(?P<n>[1-4])\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b(?P<l>[ABCD])\b")


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _truth_option_id(answer_field: str) -> str | None:
    """
    Extract "option_#" from ground-truth answer field like:
      "option_2: Specular reflection on the ground"
    """
    if not answer_field:
        return None
    m = _OPT_RE.search(answer_field)
    if not m:
        return None
    return f"option_{m.group('n')}"


def _option_map(item: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(1, 5):
        k = f"option_{i}"
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = _normalize_ws(v)
    return out


def _prompt_mcq(question: str, options: dict[str, str]) -> str:
    """
    Force a stable output so we can score reliably.
    """
    lines = [
        "You are evaluating a multiple-choice question using retrieved 3GPP context.",
        "Pick exactly one correct option.",
        "",
        f"Question: {question.strip()}",
        "",
        "Choices:",
        f"option_1: {options.get('option_1','')}",
        f"option_2: {options.get('option_2','')}",
        f"option_3: {options.get('option_3','')}",
        f"option_4: {options.get('option_4','')}",
        "",
        "Output format (must follow exactly):",
        "final_option: option_#",
        "final_text: <copy the option text>",
        "reason: <1-2 sentences with spec id if present>",
    ]
    return "\n".join(lines)


def _extract_predicted_option(answer_text: str, options: dict[str, str]) -> str | None:
    """
    Best-effort extraction of predicted option from model output.
    Prefers explicit "option_#" then falls back to A/B/C/D, then option-text match.
    """
    t = answer_text or ""

    # 1) option_# mentioned anywhere
    m = _OPT_RE.search(t)
    if m:
        return f"option_{m.group('n')}"

    # 2) A/B/C/D (common MCQ style)
    m2 = _LETTER_RE.search(t.strip().upper())
    if m2:
        letter = m2.group("l")
        return {"A": "option_1", "B": "option_2", "C": "option_3", "D": "option_4"}[letter]

    # 3) try match by exact option text substring
    low = t.lower()
    scored: list[tuple[int, str]] = []
    for opt_id, opt_text in options.items():
        ot = opt_text.strip()
        if not ot:
            continue
        # simple containment check; score by length so longer matches win
        if ot.lower() in low:
            scored.append((len(ot), opt_id))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    return None


@dataclass
class EvalRow:
    qid: str
    question: str
    truth_option: str | None
    pred_option: str | None
    correct: bool
    model_answer: str
    category: str | None = None
    difficulty: str | None = None
    tool_hops: int | None = None
    seconds: float | None = None


def _iter_questions(payload: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    # Preserve file order (Python 3.7+ dicts are ordered)
    for k, v in payload.items():
        if isinstance(v, dict) and isinstance(k, str) and k.lower().startswith("question_"):
            yield k, v


def main() -> int:
    ap = argparse.ArgumentParser(description="MCQ evaluator for rag.chat.agent.run_agent")
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to MCQ JSON file (e.g. Sampled_3GPP_TR_Questions.json)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="mcq_results.jsonl",
        help="Output JSONL path (default: mcq_results.jsonl)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max questions to run (0=all)")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Set RAG_FAST=1 and USE_RERANKER=0 for lower latency",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between questions (helps avoid overloading local LLM server)",
    )
    args = ap.parse_args()

    if args.fast:
        os.environ["RAG_FAST"] = "1"
        os.environ["USE_RERANKER"] = "0"

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Import after env vars so rag.config sees them.
    from rag.chat.agent import run_agent

    rows: list[EvalRow] = []
    total = 0
    correct = 0

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as w:
        for qid, item in _iter_questions(payload):
            if args.limit and total >= args.limit:
                break
            total += 1

            q = str(item.get("question") or "").strip()
            options = _option_map(item)
            truth = _truth_option_id(str(item.get("answer") or ""))

            prompt = _prompt_mcq(q, options)
            t0 = time.time()
            try:
                out = run_agent(prompt)
                ans = str(out.get("answer") or "")
                hops = int(out.get("tool_hops") or 0)
            except Exception as e:
                ans = f"[EVAL_ERROR] {type(e).__name__}: {e}"
                hops = None
            dt = time.time() - t0

            pred = _extract_predicted_option(ans, options)
            is_ok = bool(truth and pred and truth.lower() == pred.lower())
            if is_ok:
                correct += 1

            row = EvalRow(
                qid=qid,
                question=q,
                truth_option=truth,
                pred_option=pred,
                correct=is_ok,
                model_answer=ans,
                category=item.get("category"),
                difficulty=item.get("difficulty"),
                tool_hops=hops,
                seconds=dt,
            )
            rows.append(row)

            w.write(
                json.dumps(
                    {
                        "qid": row.qid,
                        "question": row.question,
                        "truth_option": row.truth_option,
                        "pred_option": row.pred_option,
                        "correct": row.correct,
                        "category": row.category,
                        "difficulty": row.difficulty,
                        "tool_hops": row.tool_hops,
                        "seconds": row.seconds,
                        "model_answer": row.model_answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            w.flush()

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

    acc = (correct / total) if total else 0.0
    print(f"Ran {total} questions")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

