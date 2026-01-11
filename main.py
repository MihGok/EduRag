import os
import sys

# Добавляем путь к корневой директории, чтобы импорты работали корректно
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CourseProcessor.CourseLoader import StepikCourseLoader

def fetch_40_courses(topic: str):
    """
    Выполняет поиск и возвращает список из 40 объектов курсов по теме.
    """
    print(f"\n[Search] Начинаю поиск 40 курсов по теме: '{topic}'...")
    
    # Инициализируем загрузчик (он сам проверит токены в .env) 
    loader = StepikCourseLoader()
    
    # 1. Получаем список ID (метод уже фильтрует бесплатные и публичные курсы) 
    # Указываем limit=40
    course_ids = loader.get_course_ids_by_query(query=topic, limit=40)
    
    if not course_ids:
        print(f"[Result] По запросу '{topic}' ничего не найдено.")
        return []

    # 2. Получаем полные метаданные для этих ID 
    # Метод fetch_objects автоматически разбивает запрос на чанки по 20 штук 
    courses_metadata = loader.fetch_objects('courses', course_ids)
    
    print(f"[Result] Успешно получено объектов: {len(courses_metadata)}")
    return courses_metadata

if __name__ == "__main__":
    # Задайте нужную тему здесь
    TARGET_TOPIC = "Python" 
    
    courses = fetch_40_courses(TARGET_TOPIC)
    
    # Вывод результата для проверки
    print("-" * 50)
    for idx, course in enumerate(courses, 1):
        title = course.get('title')
        cid = course.get('id')
        print(f"{idx:02d}. [ID: {cid}] {title}")
    print("-" * 50)