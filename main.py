import os
import sys
import json
from tqdm import tqdm

# Импортируем вашу конфигурацию прокси
from services.config import ProxyConfig
from CourseProcessor.CourseLoader import StepikCourseLoader

# Константы
LLM_ENDPOINT = "http://127.0.0.1:8000/generate"
TARGET_TOPIC = "Python программирование"
BATCH_SIZE = 10

def chunk_list(lst, n):
    """Разбивает список на части по n элементов."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def analyze_batch_with_llm(session, courses_chunk, topic):
    """Отправляет батч из 10 курсов на анализ в локальную LLM."""
    
    # Формируем текстовый список для промпта
    items_text = ""
    for idx, c in enumerate(courses_chunk):
        items_text += f"{idx+1}. [ID: {c.get('id')}] Название: {c.get('title')}\n"

    # Русскоязычный промпт
    prompt = (
        f"Ты — эксперт-аналитик образовательных программ. Твоя задача — оценить релевантность курсов Stepik для темы: '{topic}'.\n"
        f"У тебя есть только названия курсов. Для каждого курса:\n"
        f"1. Сгенерируй краткое описание того, чему скорее всего учит этот курс (reasoning).\n"
        f"2. Выставь оценку релевантности теме '{topic}' от 0 до 10 (course_score).\n\n"
        f"Список курсов для анализа:\n{items_text}\n"
        f"Верни результат строго в формате JSON-массива из {len(courses_chunk)} объектов."
    )

    # JSON Schema для обеспечения корректного вывода 10 объектов
    response_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "course_id": {"type": "integer"},
                        "course_title": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "course_score": {"type": "integer"}
                    },
                    "required": ["course_id", "course_title", "reasoning", "course_score"]
                }
            }
        },
        "required": ["results"]
    }

    payload = {
        "prompt": prompt,
        "response_schema": response_schema,
        "max_tokens": 1024,
        "temperature": 0.25,
        "top_p": 0.85,
        "n_ctx": 2048,        
        "n_gpu_layers": -1 
    }

    try:
        # Используем сессию БЕЗ прокси для локального адреса
        response = session.post(LLM_ENDPOINT, json=payload, timeout=180)
        response.raise_for_status()
        res_data = response.json()
        
        if res_data.get("success") and "json" in res_data:
            return res_data["json"].get("results", [])
    except Exception as e:
        print(f"\n[LLM Error] Ошибка при обработке батча: {e}")
    
    return []

def main():
    # 1. Загрузка списка курсов
    print(f"[Stepik] Поиск курсов по теме: {TARGET_TOPIC}...")
    loader = StepikCourseLoader()
    course_ids = loader.get_course_ids_by_query(query=TARGET_TOPIC, limit=100)
    
    if not course_ids:
        print("Курсы не найдены.")
        return

    raw_courses = loader.fetch_objects('courses', course_ids)
    print(f"[Stepik] Получено метаданных: {len(raw_courses)}")

    # 2. Инициализация сессии без прокси для связи с локальной LLM
    session = ProxyConfig.get_session_with_proxy(use_proxy=False)

    # 3. Пакетная обработка
    all_analyzed = []
    chunks = list(chunk_list(raw_courses, BATCH_SIZE))
    
    print(f"[AI] Начинаю анализ релевантности (батчи по {BATCH_SIZE})...")
    for chunk in tqdm(chunks):
        results = analyze_batch_with_llm(session, chunk, TARGET_TOPIC)
        # Синхронизируем ID (на случай, если модель ошиблась в цифрах)
        for i, res in enumerate(results):
            if i < len(chunk):
                res['course_id'] = chunk[i].get('id')
        all_analyzed.extend(results)

    # 4. Сортировка по убыванию рейтинга
    all_analyzed.sort(key=lambda x: x.get('course_score', 0), reverse=True)

    # 5. Вывод ТОП-10
    print("\n" + "="*50)
    print(f"РЕЗУЛЬТАТЫ АНАЛИЗА (Тема: {TARGET_TOPIC})")
    print("="*50)
    for item in all_analyzed[:20]:
        print(f"[{item['course_score']}/10] {item['course_title']} (ID: {item['course_id']})")
        print(f"   Обоснование: {item['reasoning']}\n")

if __name__ == "__main__":
    main()