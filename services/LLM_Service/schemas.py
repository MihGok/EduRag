from pydantic import BaseModel, Field
from typing import List


class CourseValidationResult(BaseModel):
    relevant_ids: List[int] = Field(
        ..., 
        description="Список ID курсов, которые строго соответствуют запрошенной теме и являются обучающими."
    )


class VideoTimestamp(BaseModel):
    timestamp: float = Field(..., description="Время начала смыслового блока в секундах")
    reason: str = Field(..., description="Подробное описание того, что визуально должно происходить на этом кадре (код, схема, диаграмма)")

class VideoAnalysisResult(BaseModel):
    timestamps: List[VideoTimestamp] = Field(..., description="Список ключевых моментов видео")
    summary: str = Field(..., description="Краткое содержание видеоурока на русском языке")