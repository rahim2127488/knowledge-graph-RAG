"""Step 4: Structure-aware chunking. Keeps (doc, page) for citations."""
import json
from pathlib import Path

IN  = Path("data/parsed_docs.jsonl")
OUT = Path("data/chunks.jsonl")

CHUNK_SIZE = 500      # characters target
OVERLAP    = 100      # characters overlap between consecutive chunks

def chunk_text(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        # try not to cut mid-sentence: back up to last period/space
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 80]

n = 0
with open(IN, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        for ci, ch in enumerate(chunk_text(rec["text"], CHUNK_SIZE, OVERLAP)):
            out = {"doc": rec["doc"], "page": rec["page"],
                   "chunk_id": f"{rec['doc']}::p{rec['page']}::c{ci}",
                   "text": ch}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
print(f"Wrote {n} chunks -> {OUT}")
