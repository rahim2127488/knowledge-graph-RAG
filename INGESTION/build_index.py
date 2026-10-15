"""Step 6a (FIXED): Embed chunks into ChromaDB - local MiniLM + COSINE distance.
The cosine metric gives good contrast (relevant chunks ~0.2-0.4, irrelevant ~1.0+).
Run:  python 03_build_index.py    (a few minutes, fully local, one-time)"""
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CHUNKS = Path("data/chunks.jsonl")
DB_DIR = "chroma_db"

ef = embedding_functions.DefaultEmbeddingFunction()
chroma = chromadb.PersistentClient(path=DB_DIR)
try:
    chroma.delete_collection("aircraft_docs")
except Exception:
    pass
col = chroma.get_or_create_collection(
    "aircraft_docs",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},   # <-- THE FIX
)

recs = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
BATCH = 100
for i in range(0, len(recs), BATCH):
    batch = recs[i:i+BATCH]
    col.add(ids=[r["chunk_id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[{"doc": r["doc"], "page": r["page"]} for r in batch])
    print(f"embedded {i+len(batch)}/{len(recs)}")
print("Indexed with COSINE distance - retrieval contrast restored")
