import sys
import os
import requests
from tqdm import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Импорты проекта
from services.config import ProxyConfig, AppConfig
from CourseProcessor.CourseLoader import StepikCourseLoader
from MLBackend.core.model_manager import model_manager

from MLBackend.services.local_LLM.local_schemas import COURSE_ANALYSIS_SCHEMA, LESSON_ANALYSIS_SCHEMA
from MLBackend.services.local_LLM.local_prompts import build_course_analysis_prompt, build_lesson_analysis_prompt


# === КОНСТАНТЫ ===
NUM_LLM_WORKERS = 2  # Количество параллельных LLM инстансов


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def _chunk_list(lst, n):
    """Разбивает список на части по n элементов."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class LLMWorkerPool:
    """Пул воркеров для параллельной работы с LLM"""
    
    def __init__(self, num_workers: int, llm_endpoint: str):
        self.num_workers = num_workers
        self.llm_endpoint = llm_endpoint
        self.session = ProxyConfig.get_session_with_proxy(use_proxy=False)
        self.lock = threading.Lock()
        
    def _send_request(self, prompt: str, schema: Dict, worker_id: int, model_path: str = None) -> Dict:
        """
        Внутренний метод для отправки запроса к LLM.
        model_path - путь к модели внутри Docker контейнера (например: /models/smollm3-3b-q4_k_m.gguf)
        """
        payload = {
            "prompt": prompt,
            "response_schema": schema,
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
            "n_ctx": 2048,
            "n_gpu_layers": -1
        }
        
        # Если указан конкретный путь к модели - используем его
        if model_path:
            payload["model_path"] = model_path
        
        try:
            response = self.session.post(self.llm_endpoint, json=payload, timeout=240)
            response.raise_for_status()
            res_data = response.json()
            
            if res_data.get("success") and "json" in res_data:
                return res_data["json"]
        except Exception as e:
            print(f"\n[Worker {worker_id}] LLM Error: {e}")
        
        return {}
    
    def process_course_batch(self, batch: List[Dict], topic: str, batch_id: int, model_path: str = None) -> List[Dict]:
        """Обрабатывает батч курсов"""
        prompt = build_course_analysis_prompt(topic, batch)
        result = self._send_request(prompt, COURSE_ANALYSIS_SCHEMA, batch_id, model_path)
        
        # Синхронизация ID
        results = result if isinstance(result, list) else result.get("results", [])
        for i, res in enumerate(results):
            if i < len(batch):
                res['course_id'] = batch[i].get('id')
                res['course_title'] = batch[i].get('title')
        
        return results
    
    def process_lesson_batch(self, batch: List[Dict], topic: str, course_title: str, batch_id: int, model_path: str = None) -> List[Dict]:
        """Обрабатывает батч уроков"""
        prompt = build_lesson_analysis_prompt(topic, course_title, batch)
        result = self._send_request(prompt, LESSON_ANALYSIS_SCHEMA, batch_id, model_path)
        
        if isinstance(result, dict):
            return result.get("lessons", [])
        elif isinstance(result, list):
            return result
        
        return []


def fetch_stepik_courses(topic: str, limit: int = 100) -> tuple[StepikCourseLoader, List[Dict]]:
    """
    Ищет курсы на Stepik и загружает их метаданные.
    """
    print(f"[Stepik] Поиск курсов по теме: {topic}...")
    loader = StepikCourseLoader()
    
    course_ids = loader.get_course_ids_by_query(query=topic, limit=limit)
    if not course_ids:
        print("Курсы не найдены.")
        return loader, []

    raw_courses = loader.fetch_objects('courses', course_ids)
    print(f"[Stepik] Загружено метаданных: {len(raw_courses)}")
    
    return loader, raw_courses


def analyze_courses_relevance(
    raw_courses: List[Dict], 
    topic: str, 
    llm_endpoint: str, 
    batch_size: int = 10
) -> List[Dict]:
    """
    УЛУЧШЕНО: Параллельная обработка курсов через несколько LLM инстансов.
    Использует малую модель для быстрого анализа.
    """
    if not raw_courses:
        return []

    # Переходим в фазу LLM (выгружает Whisper, загружает 2 LLM)
    print(f"\n[Phase] Starting LLM phase with {NUM_LLM_WORKERS} workers...")
    model_manager.start_llm_phase(num_instances=NUM_LLM_WORKERS)
    
    pool = LLMWorkerPool(NUM_LLM_WORKERS, llm_endpoint)
    all_analyzed = []
    chunks = list(_chunk_list(raw_courses, batch_size))
    
    # Используем маленькую модель для анализа
    model_small = f"/models/{AppConfig.LLM_MODEL_SMALL}"
    
    print(f"[AI] Параллельный анализ релевантности ({len(raw_courses)} курсов, {len(chunks)} батчей)...")
    print(f"[AI] Используется модель: {AppConfig.LLM_MODEL_SMALL}")
    
    with ThreadPoolExecutor(max_workers=NUM_LLM_WORKERS) as executor:
        futures = {
            executor.submit(pool.process_course_batch, chunk, topic, i, model_small): i 
            for i, chunk in enumerate(chunks)
        }
        
        with tqdm(total=len(chunks), desc="Обработка батчей") as pbar:
            for future in as_completed(futures):
                results = future.result()
                all_analyzed.extend(results)
                pbar.update(1)

    # Сортировка
    all_analyzed.sort(key=lambda x: x.get('course_score', 0), reverse=True)
    return all_analyzed


def filter_course_content(
    loader: StepikCourseLoader, 
    course_obj: Dict, 
    topic: str, 
    llm_endpoint: str
) -> List[int]:
    """
    УЛУЧШЕНО: Параллельная фильтрация уроков.
    Использует малую модель для быстрой фильтрации.
    """
    course_id = course_obj['id']
    course_title = course_obj['title']
    
    lessons_metadata = loader.get_course_outline(course_obj)
    if not lessons_metadata:
        print(f"   [WARN] В курсе {course_id} не найдено уроков.")
        return []

    print(f"   [AI] Анализ {len(lessons_metadata)} уроков на полезность...")
    
    pool = LLMWorkerPool(NUM_LLM_WORKERS, llm_endpoint)
    approved_ids = []
    
    # Используем маленькую модель для фильтрации
    model_small = f"/models/{AppConfig.LLM_MODEL_SMALL}"
    
    lesson_batch_size = 5
    chunks = list(_chunk_list(lessons_metadata, lesson_batch_size))
    
    with ThreadPoolExecutor(max_workers=NUM_LLM_WORKERS) as executor:
        futures = {
            executor.submit(pool.process_lesson_batch, chunk, topic, course_title, i, model_small): i
            for i, chunk in enumerate(chunks)
        }
        
        for future in tqdm(as_completed(futures), total=len(chunks), desc="   Фильтрация уроков", leave=False):
            results = future.result()
            
            for res in results:
                score = res.get('lesson_score', 0)
                lid = res.get('lesson_id')
                
                if score >= 5:
                    approved_ids.append(lid)

    print(f"   [RESULT] Одобрено {len(approved_ids)} из {len(lessons_metadata)} уроков.")
    return approved_ids


def print_top_results(analyzed_courses: List[Dict], top_n: int = 20):
    """Выводит красивые результаты в консоль."""
    print("\n" + "="*60)
    print(f"ТОП-{top_n} РЕЛЕВАНТНЫХ КУРСОВ")
    print("="*60)
    
    for item in analyzed_courses[:top_n]:
        print(f"[{item.get('course_score', 0)}] {item.get('course_title')} (ID: {item.get('course_id')})")
        print(f"   Обоснование: {item.get('reasoning')}\n")


def download_top_courses(
    loader: StepikCourseLoader,
    analyzed_courses: List[Dict], 
    raw_courses: List[Dict], 
    min_score: int,
    topic: str,
    llm_endpoint: str
):
    """
    УЛУЧШЕНО: После анализа переключается в фазу Whisper.
    """
    print("\n" + "="*60)
    print(f"УМНАЯ ЗАГРУЗКА КУРСОВ (Score > {min_score})")
    print("="*60)

    # СТАДИЯ A: Анализ и фильтрация (LLM уже загружены)
    raw_courses_map = {c['id']: c for c in raw_courses}
    courses_to_download = []
    
    for item in analyzed_courses:
        score = item.get('course_score', 0)
        course_id = item.get('course_id')

        if score > min_score:
            print(f"\n[>>>] Анализ курса ID: {course_id} (Score: {score})")
            
            full_course_obj = raw_courses_map.get(course_id)
            if not full_course_obj:
                full_course_obj = loader.fetch_object_single('courses', course_id)

            if full_course_obj:
                try:
                    relevant_lesson_ids = filter_course_content(
                        loader, full_course_obj, topic, llm_endpoint
                    )
                    
                    if relevant_lesson_ids:
                        courses_to_download.append((full_course_obj, relevant_lesson_ids))
                    else:
                        print("   [SKIP] Нет релевантных уроков после фильтрации.")
                        
                except Exception as e:
                    print(f"[ERROR] Ошибка анализа {course_id}: {e}")
    
    # СТАДИЯ B: Загрузка контента (переключаемся на Whisper)
    if courses_to_download:
        print(f"\n[Phase] Switching to Whisper phase for content download...")
        model_manager.start_whisper_phase(num_instances=NUM_LLM_WORKERS)
        
        for full_course_obj, relevant_lesson_ids in courses_to_download:
            try:
                print(f"\n[Download] Курс {full_course_obj['id']}...")
                loader.process_course(full_course_obj, allowed_lesson_ids=relevant_lesson_ids)
            except Exception as e:
                print(f"[ERROR] Ошибка загрузки: {e}")
        
        print(f"\n[Phase] Download complete. Switching back to LLM phase...")
        model_manager.start_llm_phase(num_instances=NUM_LLM_WORKERS)