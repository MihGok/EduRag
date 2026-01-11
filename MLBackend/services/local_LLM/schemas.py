# schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

DEFAULT_SCORE_CLASSES: List[str] = [
    "Очень релевантен",
    "Релевантен",
    "Является важной подтемой запроса",
    "Является подтемой запроса",
    "Является подтемой релевантного курса",
    "Скорее не релевантен",
    "Нерелевантен",
    "Unknown"
]

class CourseInput(BaseModel):
    id: str
    title: str
    reasoning: Optional[str] = ""

class ScoreRequest(BaseModel):
    query_topic: str = Field(..., description="Тема запроса, относительно которой оцениваются курсы")
    courses: List[CourseInput]
    score_classes: Optional[List[str]] = None
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, le=4096)
    grammar: Optional[str] = None

class CourseOutput(BaseModel):
    id: str
    title: str
    reasoning: str
    score_class: str

class ScoreResponse(BaseModel):
    ok: bool
    data: List[CourseOutput]
    raw_model_output: Optional[str] = None
