import os

import ollama

ANSWER_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

PROMPT_TEMPLATE = """You are a technical assistant for aircraft documentation.
Answer the question using ONLY the context below. If the answer is not in
context, say: I don't know based on the provided documents.
Every context section starts with a source label in square brackets.
Cite your answer by copying that label EXACTLY at the END, for example:
The ASPIRE programme was launched in February 2008. [airbus_fast_47.pdf, page 5]
Always write one or two complete sentences FIRST, then the citation. Never answer with only the citation.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def generate_answer(question: str, context: str) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=ANSWER_MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(context=context, question=question),
            }
        ],
        options={"temperature": 0},
    )
    return response.message.content