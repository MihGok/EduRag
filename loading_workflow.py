import sys
import os
import json
import requests
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Импорты проекта
from services.config import ProxyConfig
from CourseProcessor.CourseLoader import StepikCourseLoader
from CourseProcessor.transcription_runner import run_transcription

from MLBackend.services.local_LLM.local_schemas import COURSE_ANALYSIS_SCHEMA
from MLBackend.services.local_LLM.local_prompts import build_course_analysis_prompt
from MLBackend.services.local_LLM.local_schemas import LESSON_ANALYSIS_SCHEMA
from MLBackend.services.local_LLM.local_prompts import build_lesson_analysis_prompt



def _chunk_list(lst, n):
    """Разбивает список на части по n элементов."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _analyze_batch(session: requests.Session, courses_chunk: List[Dict], topic: str, llm_endpoint: str) -> List[Dict]:
    """Внутренняя функция для отправки одного батча в LLM."""
    prompt = build_course_analysis_prompt(topic, courses_chunk)
    
    payload = {
        "prompt": prompt,
        "response_schema": COURSE_ANALYSIS_SCHEMA,
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "n_ctx": 2048,        
        "n_gpu_layers": -1 
    }

    try:
        response = session.post(llm_endpoint, json=payload, timeout=240)
        response.raise_for_status()
        res_data = response.json()
        
        if res_data.get("success") and "json" in res_data:
            parsed = res_data["json"]
            results = parsed if isinstance(parsed, list) else parsed.get("results", [])
            
            for i, res in enumerate(results):
                if i < len(courses_chunk):
                    res['course_id'] = courses_chunk[i].get('id')
                    res['course_title'] = courses_chunk[i].get('title') 
            return results
            
    except Exception as e:
        print(f"\n[LLM Error] Ошибка батча: {e}")
    
    return []


def _analyze_lesson_batch(session: requests.Session, topic: str, course_title: str, lessons_chunk: List[Dict], llm_endpoint: str) -> List[Dict]:
    """Отправляет батч уроков в LLM."""
    prompt = build_lesson_analysis_prompt(topic, course_title, lessons_chunk)
    
    payload = {
        "prompt": prompt,
        "response_schema": LESSON_ANALYSIS_SCHEMA,
        "max_tokens": 1024, 
        "temperature": 0.1,
        "top_p": 0.9,
        "n_ctx": 2048,
        "n_gpu_layers": -1 
    }

    try:
        response = session.post(llm_endpoint, json=payload, timeout=300)
        response.raise_for_status()
        res_data = response.json()
        
        if res_data.get("success") and "json" in res_data:
            parsed = res_data["json"]
            if isinstance(parsed, dict):
                return parsed.get("lessons", [])
            elif isinstance(parsed, list):
                return parsed
            
    except Exception as e:
        print(f"   [Lesson LLM Error] {e}")
    
    return []


def filter_course_content(
    loader: StepikCourseLoader, 
    course_obj: Dict, 
    topic: str, 
    llm_endpoint: str
) -> tuple[List[int], List[Dict]]:
    """
    1. Получает структуру уроков.
    2. Прогоняет через LLM.
    3. Возвращает список одобренных ID уроков И детали оценки каждого урока.
    """
    course_id = course_obj['id']
    course_title = course_obj['title']
    
    lessons_metadata = loader.get_course_outline(course_obj)
    if not lessons_metadata:
        print(f"   [WARN] В курсе {course_id} не найдено уроков.")
        return [], []

    print(f"   [AI] Анализ {len(lessons_metadata)} уроков на полезность...")
    
    session = ProxyConfig.get_session_with_proxy(use_proxy=False)
    approved_ids = []
    all_lesson_scores = []
    
    lesson_batch_size = 5 
    chunks = list(_chunk_list(lessons_metadata, lesson_batch_size))
    
    for chunk in tqdm(chunks, desc="   Фильтрация уроков", leave=False):
        results = _analyze_lesson_batch(session, topic, course_title, chunk, llm_endpoint)
        
        for res in results:
            score = res.get('lesson_score', 0)
            lid = res.get('lesson_id')
            all_lesson_scores.append(res)
            
            if score >= 5:
                approved_ids.append(lid)
            else:
                print(f"      [-] Отсеян урок {lid}: {res.get('lesson_title')} (Score: {score})")

    print(f"   [RESULT] Одобрено {len(approved_ids)} из {len(lessons_metadata)} уроков.")
    return approved_ids, all_lesson_scores


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
    Прогоняет список курсов через LLM для оценки релевантности.
    """
    if not raw_courses:
        return []

    session = ProxyConfig.get_session_with_proxy(use_proxy=False)
    
    all_analyzed = []
    chunks = list(_chunk_list(raw_courses, batch_size))
    
    print(f"[AI] Анализ релевантности (всего {len(raw_courses)} курсов)...")
    
    for chunk in tqdm(chunks, desc="Обработка батчей LLM"):
        results = _analyze_batch(session, chunk, topic, llm_endpoint)
        all_analyzed.extend(results)

    all_analyzed.sort(key=lambda x: x.get('course_score', 0), reverse=True)
    return all_analyzed


def print_top_results(analyzed_courses: List[Dict], top_n: int = 20):
    """Выводит красивые результаты в консоль."""
    print("\n" + "="*60)
    print(f"ТОП-{top_n} РЕЛЕВАНТНЫХ КУРСОВ")
    print("="*60)
    
    for item in analyzed_courses[:top_n]:
        print(f"[{item.get('course_score', 0)}] {item.get('course_title')} (ID: {item.get('course_id')})")
        print(f"   Обоснование: {item.get('reasoning')}\n")




def plan_download_manifest(
    loader: StepikCourseLoader,
    analyzed_courses: List[Dict], 
    raw_courses: List[Dict], 
    min_score: int,
    topic: str,
    llm_endpoint: str,
    manifest_path: str = "download_manifest.json"
) -> str:
    """
    Проходит по всем курсам выше порога, для каждого запрашивает структуру уроков
    и фильтрует их через LLM. Результат — манифест загрузки.
    
    НЕ скачивает контент курсов. Только собирает план.
    """
    print("\n" + "="*60)
    print(f"ФАЗА A: ПЛАНИРОВАНИЕ ЗАГРУЗКИ (Score > {min_score})")
    print("="*60)

    raw_courses_map = {c['id']: c for c in raw_courses}
    manifest_entries = []

    candidates = [item for item in analyzed_courses if item.get('course_score', 0) > min_score]

    if not candidates:
        print("[PLAN] Нет курсов выше порога. Манифест пуст.")
        manifest = {
            "meta": {
                "topic": topic,
                "min_score": min_score,
                "generated_at": datetime.now().isoformat(),
                "total_courses_analyzed": len(analyzed_courses),
                "total_courses_planned": 0,
                "total_lessons_planned": 0
            },
            "courses": []
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[PLAN] Пустой манифест сохранён: {manifest_path}")
        return manifest_path

    for item in tqdm(candidates, desc="Планирование курсов"):
        course_id = item.get('course_id')
        course_score = item.get('course_score', 0)
        course_title = item.get('course_title', '')

        print(f"\n[PLAN] Курс ID: {course_id} — '{course_title}' (Score: {course_score})")

        full_course_obj = raw_courses_map.get(course_id)
        if not full_course_obj:
            full_course_obj, _ = loader.fetch_object_single('courses', course_id)

        if not full_course_obj:
            print(f"   [WARN] Не удалось получить объект курса {course_id}. Пропускаем.")
            continue

        try:
            approved_lesson_ids, all_lesson_scores = filter_course_content(
                loader, full_course_obj, topic, llm_endpoint
            )

            if not approved_lesson_ids:
                print(f"   [SKIP] В курсе {course_id} нет релевантных уроков после фильтрации.")
                manifest_entries.append({
                    "course_id": course_id,
                    "course_title": course_title,
                    "course_score": course_score,
                    "course_reasoning": item.get('reasoning', ''),
                    "status": "skipped_no_relevant_lessons",
                    "approved_lesson_ids": [],
                    "total_lessons_in_course": len(all_lesson_scores),
                    "approved_lessons_count": 0,
                    "lesson_scores": all_lesson_scores
                })
                continue

            manifest_entries.append({
                "course_id": course_id,
                "course_title": course_title,
                "course_score": course_score,
                "course_reasoning": item.get('reasoning', ''),
                "status": "pending_download",
                "approved_lesson_ids": approved_lesson_ids,
                "total_lessons_in_course": len(all_lesson_scores),
                "approved_lessons_count": len(approved_lesson_ids),
                "lesson_scores": all_lesson_scores
            })

        except Exception as e:
            print(f"   [ERROR] Ошибка планирования курса {course_id}: {e}")
            manifest_entries.append({
                "course_id": course_id,
                "course_title": course_title,
                "course_score": course_score,
                "status": "error",
                "error": str(e),
                "approved_lesson_ids": [],
                "total_lessons_in_course": 0,
                "approved_lessons_count": 0,
                "lesson_scores": []
            })

    planned_courses = [e for e in manifest_entries if e["status"] == "pending_download"]
    total_lessons_planned = sum(e["approved_lessons_count"] for e in planned_courses)

    manifest = {
        "meta": {
            "topic": topic,
            "min_score": min_score,
            "generated_at": datetime.now().isoformat(),
            "total_courses_analyzed": len(analyzed_courses),
            "total_courses_above_threshold": len(candidates),
            "total_courses_planned": len(planned_courses),
            "total_lessons_planned": total_lessons_planned
        },
        "courses": manifest_entries
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"[PLAN] Манифест сохранён: {manifest_path}")
    print(f"       Курсов для загрузки:  {len(planned_courses)}")
    print(f"       Уроков для загрузки:  {total_lessons_planned}")
    print("="*60)

    return manifest_path


STEPIK_DOWNLOAD_WORKERS = 3


def _download_single_course(
    loader: StepikCourseLoader,
    course_obj: Dict[str, Any],
    approved_lesson_ids: List[int],
) -> Dict[str, Any]:
    """
    Воркер для одного курса. Вызывается в потоке ThreadPoolExecutor.
    
    Возвращает словарь с результатом для обновления манифеста.
    """
    course_id = course_obj["id"]
    course_title = course_obj.get("title", "")

    try:
        loader.process_course(course_obj, allowed_lesson_ids=approved_lesson_ids)
        return {
            "course_id": course_id,
            "status": "downloaded",
            "downloaded_at": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"   [ERROR] Курс {course_id} ('{course_title}'): {e}")
        return {
            "course_id": course_id,
            "status": "error",
            "error": str(e),
        }


def _resolve_course_dir(course_obj: Dict[str, Any]) -> str:
    """
    Воспроизводит логику нейминга папки из CourseLoader.process_course(),
    чтобы после скачивания найти папку курса для транскрипции.

    Возвращает абсолютный путь — точно так же, как CourseLoader.process_course()
    использует os.path.abspath(). Оба вызова происходят с одной и той же CWD
    (процесс не меняет её), поэтому результаты совпадают гарантированно.
    """
    import re
    cid = course_obj.get('id')
    title = str(course_obj.get('title', f'course_{cid}') or '').strip()
    title = re.sub(r'[<>:"/\\|?*]', '', title).strip().rstrip('.')
    if not title:
        title = 'Unnamed'
    return os.path.abspath(f"Course_{cid}_{title}")


def execute_downloads(
    loader: StepikCourseLoader,
    raw_courses: List[Dict],
    manifest_path: str = "download_manifest.json",
    ml_backend_urls: List[str] = None,
    transcribe_per_backend_concurrent: int = 2,
    transcribe_timeout: int = 300,
):
    """
    Фаза B, разделённая на два этапа:

    B1 — Параллельный скачивание step-JSON с Stepik.
         ThreadPoolExecutor с STEPIK_DOWNLOAD_WORKERS воркерами.
         Каждый воркер полностью обрабатывает один курс (sections→units→lessons→steps).
         Это чисто I/O — сетевые запросы к Stepik API.

    B2 — Транскрипция видео через ML-backend(ов) с Whisper.
         Запускается ПОСЛЕ завершения B1.
         Сканирует все загруженные step-файлы, находит видео без транскрипции.
         Раздаёт задачи по бэкендам round-robin, с семафором на каждый.
         При N бэкендах и per_backend_concurrent=2 одновременно в воздухе
         N*2 HTTP-запросов, но на GPU каждый бэкенд обрабатывает ровно 1.

    Args:
        manifest_path:                  Путь к манифесту из plan_download_manifest().
        ml_backend_urls:                Список URL бэкендов с Whisper, например:
                                        ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]
                                        По умолчанию — один бэкенд на :8001.
        transcribe_per_backend_concurrent: Макс. одновременных запросов на один backend.
        transcribe_timeout:             Таймаут одной транскрипции в секундах.
    """
    if ml_backend_urls is None:
        ml_backend_urls = ["http://127.0.0.1:8001"]
    print("\n" + "="*60)
    print("ФАЗА B: ЗАГРУЗКА КОНТЕНТА")
    print("="*60)

    # --- Загрузка манифеста ---
    if not os.path.exists(manifest_path):
        print(f"[DOWNLOAD] Манифест не найден: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    courses_to_download = [
        c for c in manifest.get("courses", [])
        if c.get("status") == "pending_download"
    ]

    if not courses_to_download:
        print("[DOWNLOAD] Нет курсов со статусом 'pending_download'. Нечего загружать.")
        return

    raw_courses_map = {c['id']: c for c in raw_courses}
    print(f"\n--- B1: Скачивание step-файлов ({len(courses_to_download)} курсов, "
          f"{STEPIK_DOWNLOAD_WORKERS} потока) ---")


    download_tasks = []
    for entry in courses_to_download:
        course_id = entry["course_id"]
        full_course_obj = raw_courses_map.get(course_id)
        if not full_course_obj:
            full_course_obj, _ = loader.fetch_object_single('courses', course_id)
        if not full_course_obj:
            print(f"   [WARN] Курс {course_id}: объект не найден. Пропускаем.")
            entry["status"] = "error"
            entry["error"] = "Course object not found"
            continue
        download_tasks.append((entry, full_course_obj))


    future_to_entry = {}
    downloaded_course_dirs = []

    with ThreadPoolExecutor(max_workers=STEPIK_DOWNLOAD_WORKERS) as executor:
        for entry, course_obj in download_tasks:
            future = executor.submit(
                _download_single_course,
                loader,
                course_obj,
                entry["approved_lesson_ids"],
            )
            future_to_entry[future] = (entry, course_obj)

        for future in as_completed(future_to_entry):
            entry, course_obj = future_to_entry[future]
            result = future.result()

            # Обновляем манифест-запись
            entry["status"] = result["status"]
            if result["status"] == "downloaded":
                entry["downloaded_at"] = result["downloaded_at"]
                # Запоминаем путь папки курса для B2
                course_dir = _resolve_course_dir(course_obj)
                downloaded_course_dirs.append(course_dir)
                print(f"   [OK]  Курс {result['course_id']} скачан -> {course_dir}")
            else:
                entry["error"] = result.get("error", "unknown")
                print(f"   [ERR] Курс {result['course_id']}: {entry['error']}")


    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    b1_ok = sum(1 for c in manifest["courses"] if c.get("status") == "downloaded")
    print(f"\n--- B1 завершена: {b1_ok} курсов скачано ---")

    if not downloaded_course_dirs:
        print("[DOWNLOAD] Нет папок для транскрипции. Завершаем.")
        return


    print(f"\n--- B2: Транскрипция видео ({len(ml_backend_urls)} бэкенда, "
          f"per_backend={transcribe_per_backend_concurrent}) ---")

    transcription_results = run_transcription(
        course_dirs=downloaded_course_dirs,
        ml_backend_urls=ml_backend_urls,
        per_backend_concurrent=transcribe_per_backend_concurrent,
        timeout=transcribe_timeout,
    )


    if transcription_results:
        manifest["transcription"] = {
            "completed_at": datetime.now().isoformat(),
            "total": len(transcription_results),
            "success": sum(1 for r in transcription_results if r.success),
            "failed": sum(1 for r in transcription_results if not r.success),
            "details": [
                {
                    "step_id": r.step_id,
                    "step_file": r.step_file,
                    "success": r.success,
                    "duration_sec": round(r.duration_sec, 2),
                    "error": r.error if not r.success else None,
                }
                for r in transcription_results
            ]
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


    total_downloaded = sum(1 for c in manifest["courses"] if c.get("status") == "downloaded")
    total_errors = sum(1 for c in manifest["courses"] if c.get("status") == "error")
    trans_ok = sum(1 for r in transcription_results if r.success) if transcription_results else 0
    trans_fail = sum(1 for r in transcription_results if not r.success) if transcription_results else 0

    print("\n" + "="*60)
    print(f"[ФАЗА B] Итог:")
    print(f"   Курсы:         {total_downloaded} скачано, {total_errors} ошибок")
    print(f"   Транскрипции:  {trans_ok} успешно, {trans_fail} ошибок")
    print(f"   Манифест:      {manifest_path}")
    print("="*60)
