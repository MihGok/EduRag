
COURSE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "description": "Список проанализированных курсов",
            "items": {
                "type": "object",
                "properties": {
                    "course_id": {
                        "type": "integer",
                        "description": "ID курса на платформе Stepik"
                    },
                    "course_title": {
                        "type": "string",
                        "description": "Название курса"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Обоснование оценки релевантности курса"
                    },
                    "course_score": {
                        "type": "integer",
                        "description": "Оценка релевантности от 0 до 10",
                        "minimum": 0,
                        "maximum": 10
                    }
                },
                "required": ["course_id", "course_title", "reasoning", "course_score"]
            }
        }
    },
    "required": ["results"]
}


# === СХЕМА ДЛЯ АНАЛИЗА УРОКОВ ===

LESSON_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "description": "Список проанализированных уроков курса",
            "items": {
                "type": "object",
                "properties": {
                    "lesson_id": {
                        "type": "integer",
                        "description": "ID урока"
                    },
                    "lesson_title": {
                        "type": "string",
                        "description": "Название урока"
                    },
                    "lesson_score": {
                        "type": "integer",
                        "description": "Оценка релевантности урока от 0 до 10",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Краткое обоснование оценки"
                    }
                },
                "required": ["lesson_id", "lesson_title", "lesson_score", "reasoning"]
            }
        }
    },
    "required": ["lessons"]
}


# === СХЕМА ДЛЯ АНАЛИЗА ТРАНСКРИПЦИИ (уже существует) ===

VIDEO_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamps": {
            "type": "array",
            "description": "Ключевые моменты видео для извлечения кадров",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "number",
                        "description": "Время в секундах"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Описание того, что должно быть на кадре"
                    }
                },
                "required": ["timestamp", "reason"]
            }
        },
        "summary": {
            "type": "string",
            "description": "Краткое содержание видеоурока"
        }
    },
    "required": ["timestamps", "summary"]
}


# === ДОПОЛНИТЕЛЬНЫЕ СХЕМЫ ===

COURSE_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_ids": {
            "type": "array",
            "description": "Список ID курсов, которые строго соответствуют запрошенной теме",
            "items": {
                "type": "integer"
            }
        }
    },
    "required": ["relevant_ids"]
}