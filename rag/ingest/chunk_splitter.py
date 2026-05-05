"""
Split a large chunks.jsonl into smaller shards (default max 20_000 lines each).

Example:
  python -m rag.ingest.chunk_splitter
  python -m rag.ingest.chunk_splitter --input rag/data/chunks.jsonl --output-dir rag/data/shards

Then ingest shards (only recreate collection on the first run):
  $env:QDRANT_RECREATE="1";  python -m rag.ingest.indexer   # first shard only
  $env:CHUNKS_JSONL="...\\rag\\data\\chunks1.jsonl"; python -m rag.ingest.indexer
  $env:CHUNKS_JSONL="...\\rag\\data\\chunks2.jsonl"; python -m rag.ingest.indexer
"""

from __future__ import annotations

import argparse
import os

from rag.config import BASE_DIR


def split_jsonl(
    input_path: str,
    output_dir: str,
    *,
    max_records: int,
    prefix: str,
) -> list[tuple[str, int]]:
    """Write shards to ``{prefix}{n}.jsonl``. Returns list of (path, line_count)."""
    os.makedirs(output_dir, exist_ok=True)

    results: list[tuple[str, int]] = []
    part = 0
    count_in_part = 0
    out_file = None

    try:
        with open(input_path, "r", encoding="utf-8") as inp:
            for line in inp:
                if out_file is None or count_in_part >= max_records:
                    if out_file is not None:
                        out_file.close()
                    part += 1
                    path = os.path.join(output_dir, f"{prefix}{part}.jsonl")
                    out_file = open(path, "w", encoding="utf-8")
                    results.append((path, 0))
                    count_in_part = 0
                assert out_file is not None
                out_file.write(line)
                count_in_part += 1
                p, c = results[-1]
                results[-1] = (p, c + 1)
    finally:
        if out_file is not None:
            out_file.close()

    return results


def main() -> None:
    default_input = os.path.join(BASE_DIR, "data", "chunks.jsonl")
    default_out = os.path.join(BASE_DIR, "data")

    parser = argparse.ArgumentParser(
        description="Split chunks.jsonl into shards of at most N records "
        "(for incremental indexer runs)."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=default_input,
        help=f"Source JSONL (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=default_out,
        help=f"Directory for chunks1.jsonl, chunks2.jsonl, … (default: {default_out})",
    )
    parser.add_argument(
        "--max-records",
        "-m",
        type=int,
        default=20_000,
        help="Maximum lines per output file (default: 20000).",
    )
    parser.add_argument(
        "--prefix",
        default="chunks",
        help="Output base name: {prefix}1.jsonl, {prefix}2.jsonl, … (default: chunks).",
    )
    args = parser.parse_args()

    if args.max_records < 1:
        raise SystemExit("--max-records must be >= 1")

    src = os.path.abspath(args.input)
    if not os.path.isfile(src):
        raise SystemExit(f"Input not found: {src}")

    out_dir = os.path.abspath(args.output_dir)
    results = split_jsonl(
        src,
        out_dir,
        max_records=args.max_records,
        prefix=args.prefix,
    )

    total_lines = sum(n for _, n in results)
    print(f"✅ Split {total_lines} records from {src}")
    print(f"📁 Output directory: {out_dir}")
    for path, n in results:
        print(f"   {os.path.basename(path)}  ({n} lines)")


if __name__ == "__main__":
    main()
