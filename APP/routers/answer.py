import logging
from fastapi import APIRouter, HTTPException

from APP.schemas import AnswerRequest, AnswerResponse
from APP.services.rag import answer_question

router = APIRouter(tags=["answers"])
logger = logging.getLogger(__name__)


@router.post("/answer", response_model=AnswerResponse)
def create_answer(request: AnswerRequest) -> AnswerResponse:
    try:
        return answer_question(request.question.strip())
    except Exception as exc:
        logger.exception("Failed to generate RAG answer")
        raise HTTPException(status_code=503, detail="Answer service unavailable. Check the server logs.") from exc
