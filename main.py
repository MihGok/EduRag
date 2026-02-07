import loading_workflow as workflow

# Конфигурация
LLM_ENDPOINT = "http://127.0.0.1:8000/generate"
TARGET_TOPIC = "Python программирование"
BATCH_SIZE = 10
DOWNLOAD_THRESHOLD = 7  # Скачивать курсы с оценкой выше 7

def main():
    """
    Основной рабочий процесс с автоматическим управлением фазами:
    - ФАЗА A (LLM): Анализ курсов и уроков
    - ФАЗА B (Whisper): Загрузка контента и транскрибация
    - ФАЗА C (LLM): Возврат к аналитическим задачам
    """
    
    print("="*60)
    print("УМНАЯ СИСТЕМА ЗАГРУЗКИ КУРСОВ STEPIK")
    print("="*60)
    print(f"Тема: {TARGET_TOPIC}")
    print(f"Модель: SmolLM3-3B (2 параллельных инстанса)")
    print(f"Порог загрузки: {DOWNLOAD_THRESHOLD}")
    print("="*60 + "\n")
    
    # 1. Поиск и сбор данных (Stepik API)
    loader, raw_courses = workflow.fetch_stepik_courses(
        topic=TARGET_TOPIC, 
        limit=100
    )
    
    if not raw_courses:
        print("Курсы не найдены. Завершение работы.")
        return

    # 2. Интеллектуальный анализ (ФАЗА A: LLM)
    analyzed_results = workflow.analyze_courses_relevance(
        raw_courses=raw_courses,
        topic=TARGET_TOPIC,
        llm_endpoint=LLM_ENDPOINT,
        batch_size=BATCH_SIZE
    )

    # 3. Вывод результатов
    workflow.print_top_results(analyzed_results, top_n=20)


    workflow.download_top_courses(
        loader=loader,
        analyzed_courses=analyzed_results,
        raw_courses=raw_courses,
        min_score=DOWNLOAD_THRESHOLD,
        topic=TARGET_TOPIC,
        llm_endpoint=LLM_ENDPOINT
    )
    
    print("\n" + "="*60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("="*60)

if __name__ == "__main__":
    main()