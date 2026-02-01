import os
import re
import json
import time
import threading
import requests
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


DEFAULT_PER_BACKEND_CONCURRENT = 2
DEFAULT_TIMEOUT_SECONDS = 600


@dataclass
class TranscriptionResult:
    step_file: str
    step_id: Any
    video_url: str
    success: bool
    transcript: str = ""
    error: str = ""
    duration_sec: float = 0.0
    backend_url: str = ""


class _BackendPool:
    """
    Хранит список бэкендов и семафор для каждого.
    Раздаёт задачи по кругу через атомарный счётчик.
    """

    def __init__(self, backend_urls: List[str], per_backend_concurrent: int):
        self.backends = list(backend_urls)
        # Один семафор на backend — ограничивает параллельные запросы именно к нему
        self.semaphores = [
            threading.BoundedSemaphore(per_backend_concurrent)
            for _ in self.backends
        ]
        self._counter = 0
        self._lock = threading.Lock()

    def next(self) -> tuple[str, threading.BoundedSemaphore]:
        """Возвращает (url, семафор) следующего бэкенда по round-robin."""
        with self._lock:
            idx = self._counter % len(self.backends)
            self._counter += 1
        return self.backends[idx], self.semaphores[idx]

    @property
    def pool_size(self) -> int:
        """Рекомендуемый размер ThreadPoolExecutor."""
        sem_value = self.semaphores[0]._value if self.semaphores else 2
        return len(self.backends) * sem_value * 2



def scan_video_steps(course_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    Рекурсивно сканирует папки курсов и собирает список step-файлов,
    у которых есть video_url, но нет транскрипции.

    Args:
        course_dirs: Список абсолютных путей к папкам курсов (Course_XXXX_...)
    """
    pending = []

    for course_dir in course_dirs:
        if not os.path.isdir(course_dir):
            print(f"   [SCAN WARN] Папка не найдена: {course_dir}")
            continue

        for root, _dirs, files in os.walk(course_dir):
            for fname in sorted(files):
                if not (fname.startswith("step_") and fname.endswith(".json")):
                    continue

                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"   [SCAN WARN] Не удалось прочитать {fpath}: {e}")
                    continue

                block = data.get("block") or {}
                if isinstance(block, list):
                    block = block[0] if block else {}

                block_name = (block.get("name") or "").strip().lower()
                if block_name != "video":
                    continue

                video_obj = block.get("video") or block
                urls = video_obj.get("urls") if isinstance(video_obj, dict) else None
                video_url = _pick_best_url(urls) if isinstance(urls, list) else None

                if not video_url:
                    continue

                # Уже есть транскрипция — пропускаем
                if data.get("transcript") or data.get("_generated_transcript"):
                    continue

                pending.append({
                    "step_file": fpath,
                    "step_id": data.get("id"),
                    "video_url": video_url,
                })

    print(f"[SCAN] Найдено видео без транскрипции: {len(pending)}")
    return pending


def _pick_best_url(urls: List[Dict[str, Any]]) -> Optional[str]:
    """Берёт URL с минимальным качеством (360p если есть, иначе самый маленький)."""
    if not urls:
        return None

    numeric = []
    fallback = []
    for entry in urls:
        q = entry.get("quality")
        u = entry.get("url") or entry.get("src") or entry.get("link")
        if not u:
            continue
        if isinstance(q, str):
            m = re.search(r'(\d+)', q)
            if m:
                numeric.append((int(m.group(1)), u))
                continue
        fallback.append(u)

    if numeric:
        numeric.sort(key=lambda x: x[0])
        for qv, u in numeric:
            if qv == 360:
                return u
        return numeric[0][1]

    return fallback[-1] if fallback else None


def _transcribe_single(
    task: Dict[str, Any],
    pool: _BackendPool,
    timeout: int,
) -> TranscriptionResult:
    """
    Выполняется в потоке ThreadPoolExecutor.

    1. Берём следующий backend из пула (round-robin).
    2. Захватываем семафор этого backend.
    3. Отправляем запрос.
    4. Освобождаем семафор (в finally — даже при ошибке).
    """
    step_file = task["step_file"]
    step_id = task["step_id"]
    video_url = task["video_url"]
    start = time.time()


    backend_url, semaphore = pool.next()

    semaphore.acquire()
    try:
        resp = requests.post(
            f"{backend_url}/transcribe",
            json={"video_url": video_url},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()

        transcript = body.get("transcript", "").strip()
        segments = body.get("segments", [])

        if not transcript:
            return TranscriptionResult(
                step_file=step_file, step_id=step_id, video_url=video_url,
                success=False, error="Empty transcript in response",
                duration_sec=time.time() - start, backend_url=backend_url,
            )

        # Атомарная запись транскрипции в step-файл
        _write_transcript_to_step(step_file, transcript, segments)

        return TranscriptionResult(
            step_file=step_file, step_id=step_id, video_url=video_url,
            success=True, transcript=transcript,
            duration_sec=time.time() - start, backend_url=backend_url,
        )

    except requests.exceptions.Timeout:
        return TranscriptionResult(
            step_file=step_file, step_id=step_id, video_url=video_url,
            success=False, error=f"Timeout after {timeout}s",
            duration_sec=time.time() - start, backend_url=backend_url,
        )
    except requests.exceptions.RequestException as e:
        return TranscriptionResult(
            step_file=step_file, step_id=step_id, video_url=video_url,
            success=False, error=str(e),
            duration_sec=time.time() - start, backend_url=backend_url,
        )
    except Exception as e:
        return TranscriptionResult(
            step_file=step_file, step_id=step_id, video_url=video_url,
            success=False, error=f"{type(e).__name__}: {e}",
            duration_sec=time.time() - start, backend_url=backend_url,
        )
    finally:
        semaphore.release()


def _write_transcript_to_step(step_file: str, transcript: str, segments: List[Dict]):
    """Атомарная запись транскрипции в step JSON (tmp + os.replace)."""
    try:
        with open(step_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["transcript"] = transcript
        data["_generated_transcript"] = transcript
        data["_segments"] = segments

        tmp_path = step_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, step_file)

    except Exception as e:
        print(f"   [WRITE ERROR] Не удалось записать транскрипцию в {step_file}: {e}")



def _check_backends(backend_urls: List[str]) -> List[str]:
    """
    Пингует все бэкенды на /health. Возвращает список тех, что живы.
    Это важно при многобэкенде: если один упал, мы не должны блокироваться
    на его семафоре вечно.
    """
    alive = []
    for url in backend_urls:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                alive.append(url)
                print(f"[TRANSCRIBE] Backend {url} — OK")
            else:
                print(f"[TRANSCRIBE] Backend {url} — статус {resp.status_code}, пропускаем.")
        except requests.exceptions.RequestException as e:
            print(f"[TRANSCRIBE] Backend {url} — недоступен: {e}")
    return alive



def run_transcription(
    course_dirs: List[str],
    ml_backend_urls: List[str],
    per_backend_concurrent: int = DEFAULT_PER_BACKEND_CONCURRENT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> List[TranscriptionResult]:
    """
    Главная функция модуля.

    1. Проверяет, какие бэкенды живы.
    2. Сканирует course_dirs на предмет видео без транскрипции.
    3. Раздаёт задачи по бэкендам round-robin, с семафором на каждый.
    4. Возвращает список результатов.

    Args:
        course_dirs:              Абсолютные пути к папкам курсов.
        ml_backend_urls:          Список URL бэкендов, например:
                                  ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]
        per_backend_concurrent:   Макс. одновременных запросов на один backend.
        timeout:                  Таймаут одного запроса (секунды).
    """
    # Проверяем живость бэкендов
    alive_backends = _check_backends(ml_backend_urls)
    if not alive_backends:
        print("[TRANSCRIBE] Ни один ML-backend недоступен. Транскрипция пропущена.")
        return []

    if len(alive_backends) < len(ml_backend_urls):
        print(f"[TRANSCRIBE] Доступно {len(alive_backends)} из {len(ml_backend_urls)} бэкендов.")

    # Сканируем
    tasks = scan_video_steps(course_dirs)
    if not tasks:
        print("[TRANSCRIBE] Видео для транскрипции не найдено. Пропускаем.")
        return []

    # Создаём пул бэкендов
    pool = _BackendPool(alive_backends, per_backend_concurrent)

    total_concurrent = len(alive_backends) * per_backend_concurrent
    print(f"[TRANSCRIBE] Начинаем транскрипцию {len(tasks)} видео "
          f"({len(alive_backends)} бэкенда × {per_backend_concurrent} concurrent = "
          f"{total_concurrent} параллельных запросов, timeout={timeout}s)...")

    results: List[TranscriptionResult] = []

    with ThreadPoolExecutor(max_workers=pool.pool_size) as executor:
        futures = {
            executor.submit(_transcribe_single, task, pool, timeout): task
            for task in tasks
        }

        done_count = 0
        success_count = 0

        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            results.append(result)

            if result.success:
                success_count += 1
                print(f"   [OK]  step {result.step_id} — {result.duration_sec:.1f}s "
                      f"(backend: {result.backend_url})")
            else:
                print(f"   [ERR] step {result.step_id} — {result.error} "
                      f"(backend: {result.backend_url})")

            if done_count % 10 == 0:
                print(f"   [PROGRESS] {done_count}/{len(tasks)} "
                      f"(успешно: {success_count})")

    # Итог
    failed = [r for r in results if not r.success]
    print(f"\n[TRANSCRIBE] Завершено: {success_count}/{len(tasks)} успешно.")
    if failed:
        print(f"[TRANSCRIBE] Ошибки ({len(failed)}):")
        for r in failed:
            print(f"   step {r.step_id} [{r.backend_url}]: {r.error}")

    return results