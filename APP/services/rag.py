from APP.services.retrieval import retrieve
from APP.schemas import AnswerResponse, Source
from APP.services.llm import generate_answer


def answer_question(question: str) -> AnswerResponse:
    chunks = retrieve(question)
    if not chunks:
        return AnswerResponse(
            question=question,
            answer="I don't know based on the provided documents.",
            sources=[],
        )

    context = "\n\n".join(
        f"[{chunk.doc}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
    )
    answer = generate_answer(question, context)
    seen: set[tuple[str, int]] = set()
    sources: list[Source] = []
    for chunk in chunks:
        key = (chunk.doc, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(doc=chunk.doc, page=chunk.page))
    return AnswerResponse(question=question, answer=answer, sources=sources)
