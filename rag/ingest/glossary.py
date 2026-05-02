"""
Parse TR 21.905 (or similar) into acronym → expansion.

Supports plain text / markdown-ish exports: TSV, multi-space columns, and simple tables.
If the source file is missing, writes an empty glossary {} so retrieval can still run.
"""

import json
import os
import re
from typing import Any

from rag.config import GLOSSARY_JSON, TR_21905_PATH


def _norm_acronym(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # Common in 3GPP: "LTE" or "LTE \t ..."
    return s


def _parse_line_acronym_expansion(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Markdown table row: | ABBR | Full text |
    if "|" in line:
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) >= 2 and re.match(r"^[A-Za-z0-9][A-Za-z0-9\-]*$", parts[0]):
            if parts[0].lower() in ("abbreviation", "acronym", "term"):
                return None
            return parts[0], " ".join(parts[1:])
    # Tab-separated
    if "\t" in line:
        left, _, right = line.partition("\t")
        left, right = left.strip(), right.strip()
        if left and right and len(left) <= 32:
            return left, right
    # Two-column by 2+ spaces
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9\-]{1,31})\s{2,}(.+)$", line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def parse_glossary_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_line_acronym_expansion(line)
            if not parsed:
                continue
            abbr, expansion = parsed
            abbr = _norm_acronym(abbr)
            if abbr and expansion:
                out[abbr] = expansion.strip()
    return out


def load_or_build_glossary(
    source_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    source_path = source_path or TR_21905_PATH
    output_path = output_path or GLOSSARY_JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.isfile(source_path):
        meta = {
            "source": source_path,
            "entries": 0,
            "warning": "Source file not found; place TR 21.905 export at TR_21905_PATH "
            "or set env TR_21905_PATH.",
        }
        payload = {"meta": meta, "glossary": {}}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"⚠️  No glossary source at {source_path}; wrote empty {output_path}")
        return payload

    glossary = parse_glossary_file(source_path)
    meta = {"source": os.path.abspath(source_path), "entries": len(glossary)}
    payload = {"meta": meta, "glossary": glossary}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Glossary: {len(glossary)} entries → {output_path}")
    return payload


if __name__ == "__main__":
    load_or_build_glossary()
