import sys
import os
import requests
from tqdm import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import itertools

# Импорты проекта
from services.config import ProxyConfig, AppConfig
from CourseProcessor.CourseLoader import StepikCourseLoader

from MLBackend.services.local_LLM.local_schemas import COURSE_ANALYSIS_SCHEMA, LESSON_ANALYSIS_SCHEMA
from MLBackend.services.local_LLM.local_prompts import build_course_analysis_prompt, build_lesson_analysis_prompt


# === КОНСТАНТЫ ===
NUM_LLM_WORKERS = 2  # Количество параллельных LLM инстансов
LLM_ENDPOINTS = [
    "http://127.0.0.1:8000/generate",  # LLM Instance 1
    "http://127.0.0.1:8001/generate",  # LLM Instance 2
]


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def _chunk_list(lst, n):
    """Разбивает список на части по n элементов."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class LLMLoadBalancer:
    """
    Load balancer для распределения запросов между несколькими LLM endpoint'ами.
    Использует round-robin для равномерного распределения нагрузки.
    """
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.endpoint_cycle = itertools.cycle(endpoints)
        self.lock = threading.Lock()
        self.session = ProxyConfig.get_session_with_proxy(use_proxy=False)
        
        # Проверка доступности endpoint'ов
        self._check_endpoints()
    
    def _check_endpoints(self):
        """Проверяет доступность всех endpoint'ов"""
        print(f"\n[LLM Load Balancer] Проверка {len(self.endpoints)} endpoint(ов)...")
        
        for i, endpoint in enumerate(self.endpoints, 1):
            health_url = endpoint.replace("/generate", "/health")
            try:
                response = self.session.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(f"  ✅ Instance {i}: {endpoint} - OK")
                else:
                    print(f"  ❌ Instance {i}: {endpoint} - Error {response.status_code}")
            except Exception as e:
                print(f"  ❌ Instance {i}: {endpoint} - {e}")
    
    def _get_next_endpoint(self) -> str:
        """Получает следующий endpoint по round-robin"""
        with self.lock:
            return next(self.endpoint_cycle)
    
    def send_request(self, prompt: str, schema: Dict, worker_id: int, model_path: str = None) -> Dict:
        """
        Отправляет запрос к следующему доступному LLM endpoint.
        
        Args:
            prompt: Промпт для LLM
            schema: JSON схема для валидации ответа
            worker_id: ID воркера (для логирования)
            model_path: Путь к модели внутри Docker
        
        Returns:
            Распарсенный JSON ответ
        """
        endpoint = self._get_next_endpoint()
        
        payload = {
            "prompt": prompt,
            "response_schema": schema,
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
            "n_ctx": 2048,
            "n_gpu_layers": -1
        }
        
        if model_path:
            payload["model_path"] = model_path
        
        try:
            response = self.session.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()
            res_data = response.json()
            
            if res_data.get("success") and "json" in res_data:
                return res_data["json"]
                
        except Exception as e:
            instance_num = self.endpoints.index(endpoint) + 1 if endpoint in self.endpoints else "?"
            print(f"\n[Worker {worker_id}] LLM Instance {instance_num} Error: {e}")
        
        return {}


class LLMWorkerPool:
    """Пул воркеров для параллельной работы с несколькими LLM инстансами"""
    
    def __init__(self, num_workers: int, endpoints: List[str]):
        self.num_workers = num_workers
        self.load_balancer = LLMLoadBalancer(endpoints)
        
    def process_course_batch(self, batch: List[Dict], topic: str, batch_id: int, model_path: str = None) -> List[Dict]:
        """Обрабатывает батч курсов"""
        prompt = build_course_analysis_prompt(topic, batch)
        result = self.load_balancer.send_request(prompt, COURSE_ANALYSIS_SCHEMA, batch_id, model_path)
        
        results = result if isinstance(result, list) else result.get("results", [])
        for i, res in enumerate(results):
            if i < len(batch):
                res['course_id'] = batch[i].get('id')
                res['course_title'] = batch[i].get('title')
        
        return results
    
    def process_lesson_batch(self, batch: List[Dict], topic: str, course_title: str, batch_id: int, model_path: str = None) -> List[Dict]:
        """Обрабатывает батч уроков"""
        prompt = build_lesson_analysis_prompt(topic, course_title, batch)
        result = self.load_balancer.send_request(prompt, LESSON_ANALYSIS_SCHEMA, batch_id, model_path)
        
        if isinstance(result, dict):
            return result.get("lessons", [])
        elif isinstance(result, list):
            return result
        
        return []


def fetch_stepik_courses(topic: str, limit: int = 100) -> tuple[StepikCourseLoader, List[Dict]]:
    """Ищет курсы на Stepik и загружает их метаданные."""
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
    batch_size: int = 10
) -> List[Dict]:
    """
    Параллельная обработка курсов через 2 LLM endpoint'а.
    Каждый endpoint держит свою модель, работают одновременно.
    """
    if not raw_courses:
        return []
    
    pool = LLMWorkerPool(NUM_LLM_WORKERS, LLM_ENDPOINTS)
    all_analyzed = []
    chunks = list(_chunk_list(raw_courses, batch_size))
    
    # Используем маленькую модель для анализа
    model_small = f"/models/{AppConfig.LLM_MODEL_SMALL}"
    
    print(f"\n[AI] Параллельный анализ релевантности ({len(raw_courses)} курсов, {len(chunks)} батчей)...")
    print(f"[AI] Используется модель: {AppConfig.LLM_MODEL_SMALL}")
    print(f"[AI] LLM инстансов: {NUM_LLM_WORKERS} (Docker на портах 8000, 8001)")
    
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

    all_analyzed.sort(key=lambda x: x.get('course_score', 0), reverse=True)
    return all_analyzed


def filter_course_content(
    loader: StepikCourseLoader, 
    course_obj: Dict, 
    topic: str
) -> List[int]:
    """Параллельная фильтрация уроков через 2 LLM endpoint'а."""
    course_id = course_obj['id']
    course_title = course_obj['title']
    
    lessons_metadata = loader.get_course_outline(course_obj)
    if not lessons_metadata:
        print(f"   [WARN] В курсе {course_id} не найдено уроков.")
        return []

    print(f"   [AI] Анализ {len(lessons_metadata)} уроков на полезность...")
    
    pool = LLMWorkerPool(NUM_LLM_WORKERS, LLM_ENDPOINTS)
    approved_ids = []
    
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
    topic: str
):
    """
    Умная загрузка курсов с параллельной фильтрацией и транскрибацией.
    """
    print("\n" + "="*60)
    print(f"УМНАЯ ЗАГРУЗКА КУРСОВ (Score > {min_score})")
    print("="*60)

    raw_courses_map = {c['id']: c for c in raw_courses}
    courses_to_download = []
    
    # ФАЗА 1: Анализ и фильтрация (параллельно на 2 LLM)
    print("\n[ФАЗА 1] Анализ и фильтрация уроков...")
    
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
                    relevant_lesson_ids = filter_course_content(loader, full_course_obj, topic)
                    
                    if relevant_lesson_ids:
                        courses_to_download.append((full_course_obj, relevant_lesson_ids))
                    else:
                        print("   [SKIP] Нет релевантных уроков после фильтрации.")
                        
                except Exception as e:
                    print(f"[ERROR] Ошибка анализа {course_id}: {e}")
    
    if courses_to_download:
        print(f"\n[ФАЗА 2] Загрузка контента ({len(courses_to_download)} курсов)...")
        print("[INFO] Транскрибация будет выполняться параллельно")
        
        for full_course_obj, relevant_lesson_ids in courses_to_download:
            try:
                print(f"\n[Download] Курс {full_course_obj['id']}...")
                loader.process_course(full_course_obj, allowed_lesson_ids=relevant_lesson_ids)
            except Exception as e:
                print(f"[ERROR] Ошибка загрузки: {e}")
        
        print(f"\n[COMPLETE] Все курсы загружены!")
