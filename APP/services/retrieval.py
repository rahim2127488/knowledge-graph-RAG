import os
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

TOP_K = 5
CANDIDATES = 30
STOPWORDS = {
    "when", "what", "which", "where", "how", "why", "does", "did",
    "the", "and", "for", "with", "from", "that", "this", "have",
    "has", "was", "were", "are", "is",
}


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    doc: str
    page: int


@lru_cache(maxsize=1)
def get_collection():
    """Open the prebuilt collection. Run 03_build_index.py if it is missing."""
    db_path = os.getenv("CHROMA_DB_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "aircraft_docs")
    client = chromadb.PersistentClient(path=db_path)
    try:
        return client.get_collection(
            collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not open ChromaDB collection '{collection_name}' in '{db_path}'. "
            "Build the index first with: python 03_build_index.py"
        ) from exc


def _keywords(question: str) -> list[str]:
    return [
        word.strip("?.,()!\"' :;").lower()
        for word in question.split()
        if len(word.strip("?.,()!\"' :;")) > 4
        and word.strip("?.,()!\"' :;").lower() not in STOPWORDS
    ]


def _keyword_score(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def retrieve(question: str, k: int = TOP_K) -> list[RetrievedChunk]:
    """Preserve the original pipeline: vector top-30, keyword rerank, best 5."""
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=CANDIDATES)
    docs = result["documents"][0]
    metadatas = result["metadatas"][0]
    if not docs:
        return []

    keywords = _keywords(question)
    pairs = list(zip(docs, metadatas))
    ranked = sorted(
        pairs,
        key=lambda pair: _keyword_score(pair[0], keywords),
        reverse=True,
    )

    # Match the original behavior: if no keywords match the top ranked chunk,
    # fall back to the original vector-search order.
    selected = pairs[:k] if _keyword_score(ranked[0][0], keywords) == 0 else ranked[:k]
    return [
        RetrievedChunk(text=text, doc=metadata["doc"], page=int(metadata["page"]))
        for text, metadata in selected
    ]
