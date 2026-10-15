"""Step 3: Parse all PDFs in data/raw/ into one JSONL of {doc, page, text}."""
import fitz  # PyMuPDF
import json, os
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/parsed_docs.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def parse_pdf(path: Path):
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = "\n".join(line for line in text.splitlines() if line.strip())
        if len(text) < 50:   # skip near-empty pages (figures/title pages)
            continue
        yield {"doc": path.name, "page": i + 1, "text": text}
    doc.close()

with open(OUT, "w", encoding="utf-8") as f:
    n = 0
    for pdf in sorted(RAW.glob("*.pdf")):
        for rec in parse_pdf(pdf):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
print(f"Parsed {n} pages from {len(list(RAW.glob('*.pdf')))} PDFs -> {OUT}")
