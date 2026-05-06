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
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar


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


def _question_sort_key(k: str) -> tuple[int, str]:
    """
    Sort numerically by embedded question id when possible:
      question_2 < question_10 (lexicographic order would not).
    """
    m = re.match(r"^question_(\d+)", k, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), k
    return 10**9, k


def _iter_questions(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for k, v in payload.items():
        if isinstance(v, dict) and isinstance(k, str) and k.lower().startswith("question_"):
            items.append((k, v))
    items.sort(key=lambda kv: _question_sort_key(kv[0]))
    return items


T = TypeVar("T")


def _run_with_timeout(fn: Callable[[], T], seconds: float) -> T:
    """
    Run a blocking call in a worker thread with a hard timeout (works on Windows).
    """
    if seconds <= 0:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        return fut.result(timeout=seconds)


def _append_text(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _atomic_json(path: str, obj: Any) -> None:
    """Best-effort atomic JSON write for crash breadcrumbs."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".mcq_stage_", suffix=".tmp", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise


def _run_agent_subprocess(prompt: str, timeout_s: float) -> dict[str, Any]:
    """
    Run `run_agent` in a separate interpreter process.
    Survives many native crashes in the worker (parent keeps going).
    """
    cmd = [sys.executable, "-m", "test.mcq_eval_one"]
    body = json.dumps({"prompt": prompt}, ensure_ascii=False)
    # Default generous timeout when caller passes 0 (parent eval has no timeout).
    to = timeout_s if timeout_s > 0 else float(os.environ.get("MCQ_EVAL_TIMEOUT", "3600") or "3600")
    r = subprocess.run(
        cmd,
        input=body,
        capture_output=True,
        text=True,
        timeout=to,
        env=os.environ.copy(),
    )
    if r.returncode != 0:
        err = (r.stderr or "").strip()
        out_snip = (r.stdout or "").strip()[:4000]
        return {
            "answer": (
                f"[EVAL_ERROR] subprocess_exit={r.returncode} "
                f"stderr={err[:2000]} stdout_head={out_snip[:2000]}"
            ),
            "tool_hops": 0,
        }
    try:
        parsed: Any = json.loads((r.stdout or "").strip() or "{}")
        if isinstance(parsed, dict):
            return parsed
        return {"answer": f"[EVAL_ERROR] bad_json_stdout: {parsed!r}", "tool_hops": 0}
    except json.JSONDecodeError as e:
        return {
            "answer": (
                f"[EVAL_ERROR] json_decode: {e}; stdout={(r.stdout or '')[:2000]!r} "
                f"stderr={(r.stderr or '')[:2000]!r}"
            ),
            "tool_hops": 0,
        }


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
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("MCQ_EVAL_TIMEOUT", "0") or "0"),
        help="Per-question timeout in seconds (0 = no timeout). Can also set MCQ_EVAL_TIMEOUT.",
    )
    ap.add_argument(
        "--meta-first-line",
        action="store_true",
        help="Write a first JSONL record with run metadata so the output file is never empty while waiting.",
    )
    ap.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "Run each question in a fresh subprocess (python -m test.mcq_eval_one). "
            "Slower (reloads imports/models per question) but avoids one native crash "
            "stopping the whole batch."
        ),
    )
    ap.add_argument(
        "--progress-log",
        type=str,
        default="",
        help="Append-only progress log path (default: <out>.progress.log)",
    )
    args = ap.parse_args()

    if args.fast:
        os.environ["RAG_FAST"] = "1"
        os.environ["USE_RERANKER"] = "0"

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        print("ERROR: input JSON must be an object at the top level (dict).", file=sys.stderr)
        return 2

    # Import after env vars so rag.config sees them.
    from rag.chat.agent import run_agent

    rows: list[EvalRow] = []
    written = 0
    correct = 0

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    progress_log = os.path.abspath(args.progress_log or f"{out_path}.progress.log")
    stage_path = f"{out_path}.stage.json"

    planned = _iter_questions(payload)
    planned_n = len(planned)
    max_n = planned_n if not args.limit else min(planned_n, args.limit)
    print(f"[mcq_eval] input={os.path.abspath(args.input)}", flush=True)
    print(f"[mcq_eval] matched_question_keys={planned_n} will_run={max_n}", flush=True)
    print(f"[mcq_eval] output={out_path}", flush=True)
    print(f"[mcq_eval] progress_log={progress_log}", flush=True)
    print(f"[mcq_eval] stage_file={stage_path}", flush=True)
    if args.isolate:
        print(
            "[mcq_eval] isolate=1 (each question runs in a subprocess; expect slower runs)",
            flush=True,
        )
    if args.timeout and args.timeout > 0:
        print(f"[mcq_eval] per_question_timeout_s={args.timeout}", flush=True)

    exit_code = 0
    try:
        with open(out_path, "w", encoding="utf-8") as w:
            if args.meta_first_line:
                w.write(
                    json.dumps(
                        {
                            "event": "run_start",
                            "input": os.path.abspath(args.input),
                            "matched_question_keys": planned_n,
                            "will_run": max_n,
                            "limit": args.limit,
                            "timeout_s": args.timeout,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                w.flush()

            _append_text(
                progress_log,
                f"RUN_START limit={args.limit} isolate={int(bool(args.isolate))} "
                f"matched={planned_n}",
            )

            for idx, (qid, item) in enumerate(planned, start=1):
                if args.limit and written >= args.limit:
                    break

                try:
                    q = str(item.get("question") or "").strip()
                    options = _option_map(item)
                    truth = _truth_option_id(str(item.get("answer") or ""))

                    prompt = _prompt_mcq(q, options)
                    t0 = time.time()
                    print(f"[mcq_eval] ({idx}/{max_n}) {qid} START", flush=True)
                    _append_text(progress_log, f"START {idx}/{max_n} {qid}")
                    _atomic_json(
                        stage_path,
                        {
                            "phase": "before_run_agent",
                            "qid": qid,
                            "idx": idx,
                            "isolate": bool(args.isolate),
                        },
                    )
                    try:
                        if args.isolate:
                            out = _run_agent_subprocess(prompt, float(args.timeout))
                        else:

                            def _call() -> dict[str, Any]:
                                return run_agent(prompt)

                            if args.timeout and args.timeout > 0:
                                out = _run_with_timeout(_call, args.timeout)
                            else:
                                out = _call()
                        ans = str(out.get("answer") or "")
                        hops = int(out.get("tool_hops") or 0)
                    except subprocess.TimeoutExpired as e:
                        ans = f"[EVAL_ERROR] subprocess.TimeoutExpired: {e}"
                        hops = None
                    except FuturesTimeout:
                        ans = f"[EVAL_ERROR] TimeoutError: exceeded {args.timeout}s"
                        hops = None
                    except Exception as e:
                        ans = f"[EVAL_ERROR] {type(e).__name__}: {e}"
                        hops = None
                    dt = time.time() - t0

                    pred = _extract_predicted_option(ans, options)
                    is_ok = bool(truth and pred and truth.lower() == pred.lower())
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

                    try:
                        payload_line = json.dumps(
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
                    except TypeError as e:
                        payload_line = json.dumps(
                            {
                                "qid": row.qid,
                                "truth_option": row.truth_option,
                                "pred_option": row.pred_option,
                                "correct": row.correct,
                                "seconds": row.seconds,
                                "model_answer": "[EVAL_ERROR] json_encode: "
                                f"{type(e).__name__}: {e}",
                            },
                            ensure_ascii=False,
                        )

                    w.write(payload_line + "\n")
                    w.flush()
                    written += 1
                    if is_ok:
                        correct += 1

                    try:
                        os.unlink(stage_path)
                    except OSError:
                        pass
                    print(
                        f"[mcq_eval] ({idx}/{max_n}) {qid} DONE "
                        f"written={written} sec={dt:.2f}",
                        flush=True,
                    )
                    _append_text(
                        progress_log,
                        f"DONE {idx}/{max_n} {qid} written_total={written} sec={dt:.2f}",
                    )

                    if args.sleep and args.sleep > 0:
                        time.sleep(args.sleep)
                except KeyboardInterrupt:
                    raise
                except BaseException as e:
                    # Something escaped the inner handlers (e.g. MemoryError). Record and continue.
                    err = f"[ITER_FATAL] {type(e).__name__}: {e}"
                    print(f"[mcq_eval] ERROR on {qid}: {err}", file=sys.stderr, flush=True)
                    try:
                        w.write(
                            json.dumps(
                                {
                                    "qid": qid,
                                    "truth_option": None,
                                    "pred_option": None,
                                    "correct": False,
                                    "seconds": None,
                                    "model_answer": err,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        w.flush()
                        written += 1
                    except Exception as wexc:
                        print(
                            f"[mcq_eval] could not write error row: {wexc!r}",
                            file=sys.stderr,
                            flush=True,
                        )
    except KeyboardInterrupt:
        print(
            "\n[mcq_eval] interrupted — partial results were flushed to disk "
            f"(written_rows={written}, correct={correct})",
            flush=True,
        )
        exit_code = 130
    except Exception as e:
        print(f"[mcq_eval] fatal error (aborting run): {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        acc = (correct / written) if written else 0.0
        print(
            f"[mcq_eval] SUMMARY rows_written={written} correct={correct} "
            f"accuracy={acc:.3f} output={out_path}",
            flush=True,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

