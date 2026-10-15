"""FINAL answer pipeline v5: hybrid retrieval + local Llama, exact-citation prompt.
Usage: python 04_ollama.py plus your question in quotes."""
import sys
import httpx
import chromadb
from chromadb.utils import embedding_functions

TOP_K = 5
CANDIDATES = 30
ANSWER_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

ef = embedding_functions.DefaultEmbeddingFunction()
col = chromadb.PersistentClient(path="chroma_db").get_collection(
    "aircraft_docs", embedding_function=ef)

PROMPT = """You are a technical assistant for aircraft documentation.
Answer the question using ONLY the context below. If the answer is not in
the context, say: I don't know based on the provided documents.
Every context section starts with a source label in square brackets.
Cite your answer by copying that label EXACTLY at the END, for example:
The ASPIRE programme was launched in February 2008. [airbus_fast_47.pdf, page 5]
Always write one or two complete sentences FIRST, then the citation. Never answer with only the citation.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""

STOPWORDS = {"when", "what", "which", "where", "how", "why", "does", "did",
             "the", "and", "for", "with", "from", "that", "this", "have",
             "has", "was", "were", "are", "is"}

def retrieve(question, k=TOP_K):
    wide = col.query(query_texts=[question], n_results=CANDIDATES)
    docs, metas = wide["documents"][0], wide["metadatas"][0]
    keywords = [w.strip("?.,()").lower() for w in question.split()
                if len(w.strip("?.,()")) > 4 and w.strip("?.,()").lower() not in STOPWORDS]
    def kw_score(d):
        dl = d.lower()
        return sum(1 for kw in keywords if kw in dl)
    scored = sorted(zip(docs, metas), key=lambda x: kw_score(x[0]), reverse=True)
    if kw_score(scored[0][0]) == 0:
        return list(zip(docs, metas))[:k]
    return scored[:k]

def answer(question: str):
    pairs = retrieve(question)
    context = "\n\n".join(
        f"[{m['doc']}, page {m['page']}]\n{d}" for d, m in pairs)
    r = httpx.post(OLLAMA_URL, json={
        "model": ANSWER_MODEL,
        "messages": [{"role": "user",
                      "content": PROMPT.format(context=context, question=question)}],
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the procedure for engine start?"
    print(answer(q))
