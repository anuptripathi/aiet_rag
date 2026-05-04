import json
import os
import re
from tqdm import tqdm

from rag.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_TARGET_WORDS,
    CHUNKS_JSONL,
    DATA_DIR,
)

_HEADING_MD = re.compile(r"^#{1,6}\s+(.+)$")
_CLAUSE_LINE = re.compile(r"^(\d+(?:\.\d+)+)(?:\s+(.*))?$")


def read_md_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                yield os.path.join(root, file)


def split_by_sections(text):
    if not text or not isinstance(text, str):
        return []

    pattern = r"\n(?=(?:#+\s)|(?:\d+\.\d+(?:\.\d+)*))"
    sections = re.split(pattern, text)

    cleaned = []
    for sec in sections:
        if sec and isinstance(sec, str):
            sec = sec.strip()
            if sec:
                cleaned.append(sec)

    return cleaned


def split_heading_body(section: str) -> tuple[str, str, str | None]:
    """
    Returns (section_heading, body, embed_prefix).

    section_heading: stored in metadata; may be the full first line for clauses.
    embed_prefix: short text prepended to every chunk for retrieval; if None,
    use section_heading when non-empty. For a clause whose body is the tail of
    the same line, embed_prefix is the clause id only to avoid duplicating the title.
    """
    section = section.strip()
    if not section:
        return "", "", None

    lines = section.split("\n", 1)
    first = lines[0].strip()
    rest = lines[1].strip() if len(lines) > 1 else ""

    m = _HEADING_MD.match(first)
    if m:
        heading = m.group(1).strip()
        if rest:
            return heading, rest, None
        return "", section, None

    m2 = _CLAUSE_LINE.match(first)
    if m2:
        tail = (m2.group(2) or "").strip()
        clause_id = m2.group(1).strip()
        if rest:
            return first.strip(), rest, None
        if tail:
            return first.strip(), tail, clause_id

    return "", section, None


def chunk_text_words(
    text: str,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    """Split into overlapping word windows (whitespace-delimited tokens)."""
    if not text or not isinstance(text, str):
        return []

    words = re.findall(r"\S+", text)
    if not words:
        return []

    max_words = max(32, max_words)
    overlap_words = min(max(0, overlap_words), max_words - 1)
    step = max(1, max_words - overlap_words)

    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        piece = words[start : start + max_words]
        if piece:
            chunks.append(" ".join(piece))
        start += step

    return chunks


def extract_metadata(filepath):
    filename = os.path.basename(filepath)
    spec = filename.replace(".md", "")

    if spec.startswith("38"):
        domain = "RAN"
    elif spec.startswith("23"):
        domain = "Core"
    elif spec.startswith("24"):
        domain = "NAS"
    else:
        domain = "Other"

    return {
        "spec": spec,
        "domain": domain,
        "path": filepath,
    }


def process():
    os.makedirs(os.path.dirname(CHUNKS_JSONL), exist_ok=True)

    files = list(read_md_files(DATA_DIR))

    print(f"📂 Found {len(files)} markdown files")
    print(
        f"⚙️  chunk words={CHUNK_TARGET_WORDS} overlap={CHUNK_OVERLAP_WORDS} "
        f"(override with CHUNK_TARGET_WORDS / CHUNK_OVERLAP_WORDS)"
    )

    total_chunks = 0

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as out:

        for filepath in tqdm(files):

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"❌ Failed to read {filepath}: {e}")
                continue

            if not text or not isinstance(text, str):
                continue

            sections = split_by_sections(text)

            if not sections:
                sections = [text]

            meta = extract_metadata(filepath)

            for sec_id, section in enumerate(sections):

                if not section or not isinstance(section, str):
                    continue

                heading, body, embed_prefix = split_heading_body(section)
                if not body:
                    continue

                pieces = chunk_text_words(
                    body,
                    CHUNK_TARGET_WORDS,
                    CHUNK_OVERLAP_WORDS,
                )

                if not pieces:
                    continue

                parent_id = f"{meta['spec']}_{sec_id}"
                prefix_label = embed_prefix if embed_prefix is not None else heading
                prefix = f"{prefix_label}\n\n" if prefix_label else ""

                for i, piece in enumerate(pieces):

                    chunk_text = prefix + piece if prefix else piece
                    if not chunk_text.strip():
                        continue

                    record = {
                        "id": f"{meta['spec']}_{sec_id}_{i}",
                        "text": chunk_text,
                        "metadata": {
                            **meta,
                            "section_id": sec_id,
                            "parent_id": parent_id,
                            **({"section_heading": heading} if heading else {}),
                        },
                    }

                    out.write(json.dumps(record) + "\n")
                    total_chunks += 1

    print(f"\n✅ Chunking complete")
    print(f"📦 Total chunks created: {total_chunks}")
    print(f"📁 Output file: {CHUNKS_JSONL}")


if __name__ == "__main__":
    process()
