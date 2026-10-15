from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, description="Question to answer from the magazine collection")


class Source(BaseModel):
    doc: str
    page: int


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
