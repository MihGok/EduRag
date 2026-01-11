# prompts.py (ИЗМЕНЁННОЕ)
import json
from typing import List, Dict

def build_prompt(
    query_topic: str,
    courses: List[Dict],
    score_classes: List[str]
) -> str:
    classes_text = "\n".join([f"- {c}" for c in score_classes])

    examples = [
        {
            "id": "ex1",
            "title": "Python для анализа данных",
            "reasoning": "Курс напрямую посвящён анализу данных на Python",
            "score_class": "Очень релевантен"
        },
        {
            "id": "ex2",
            "title": "История средневековой Европы",
            "reasoning": "Тематика не связана с программированием",
            "score_class": "Нерелевантен"
        }
    ]

    prompt = f"""
Ты — локальная LLM (GGUF, llama.cpp). 
Твоя задача — оценивать релевантность курсов ОТНОСИТЕЛЬНО ЗАДАННОЙ ТЕМЫ.

ТЕМА ЗАПРОСА:
"{query_topic}"

Используй ТОЛЬКО одну метку релевантности из списка ниже.
Не придумывай новые метки.

ДОПУСТИМЫЕ МЕТКИ:
{classes_text}

ФОРМАТ ВЫВОДА (СТРОГО):
JSON-массив, без markdown, без комментариев, без пояснений вне JSON.

[
  {{
    "id": "<id>",
    "title": "<название>",
    "reasoning": "<короткое объяснение, почему курс имеет такую релевантность>",
    "score_class": "<одна из допустимых меток>"
  }}
]

ПРИМЕР:
{json.dumps(examples, ensure_ascii=False)}

ОЦЕНИ СЛЕДУЮЩИЕ КУРСЫ:
{json.dumps(courses, ensure_ascii=False)}
""".strip()

    return prompt
