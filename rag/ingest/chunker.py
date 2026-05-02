import os
import re
import json
from tqdm import tqdm
from rag.config import DATA_DIR, CHUNKS_JSONL


# ---------------------------
# Read all markdown files
# ---------------------------
def read_md_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                yield os.path.join(root, file)


# ---------------------------
# Split into sections (telecom-aware)
# ---------------------------
def split_by_sections(text):
    if not text or not isinstance(text, str):
        return []

    # Split on:
    # - Markdown headings (#, ##, ###)
    # - Clause numbers like 1.1, 2.3.4
    pattern = r'\n(?=(?:#+\s)|(?:\d+\.\d+(?:\.\d+)*))'

    sections = re.split(pattern, text)

    # Clean sections
    cleaned = []
    for sec in sections:
        if sec and isinstance(sec, str):
            sec = sec.strip()
            if sec:
                cleaned.append(sec)

    return cleaned


# ---------------------------
# Chunk text safely
# ---------------------------
def chunk_text(text, max_len=500):
    if not text or not isinstance(text, str):
        return []

    chunks = []

    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len]

        if chunk and chunk.strip():
            chunks.append(chunk.strip())

    return chunks


# ---------------------------
# Extract metadata (telecom-specific)
# ---------------------------
def extract_metadata(filepath):
    filename = os.path.basename(filepath)
    spec = filename.replace(".md", "")

    # Domain classification (basic for now)
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
        "path": filepath
    }


# ---------------------------
# Main processing pipeline
# ---------------------------
def process():
    os.makedirs(os.path.dirname(CHUNKS_JSONL), exist_ok=True)

    files = list(read_md_files(DATA_DIR))

    print(f"📂 Found {len(files)} markdown files")

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
                # fallback: treat whole doc as one section
                sections = [text]

            meta = extract_metadata(filepath)

            for sec_id, section in enumerate(sections):

                if not section or not isinstance(section, str):
                    continue

                chunks = chunk_text(section)

                if not chunks:
                    continue

                for i, chunk in enumerate(chunks):

                    if not chunk.strip():
                        continue

                    record = {
                        "id": f"{meta['spec']}_{sec_id}_{i}",
                        "text": chunk,
                        "metadata": meta
                    }

                    out.write(json.dumps(record) + "\n")
                    total_chunks += 1

    print(f"\n✅ Chunking complete")
    print(f"📦 Total chunks created: {total_chunks}")
    print(f"📁 Output file: {CHUNKS_JSONL}")


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    process()