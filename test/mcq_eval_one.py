"""
Run a single `rag.chat.agent.run_agent` call (stdin JSON -> stdout JSON).

Used by `python -m test.mcq_eval --isolate` so one bad/native crash does not
kill the whole evaluation loop.

Stdin format:
  {"prompt": "<full user prompt string>"}

Stdout: JSON object compatible with `run_agent` return value.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    data = json.load(sys.stdin)
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        json.dump({"answer": "[EVAL_ERROR] missing prompt", "tool_hops": 0}, sys.stdout)
        print()
        return 2

    from rag.chat.agent import run_agent

    out = run_agent(prompt)
    json.dump(out, sys.stdout, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
