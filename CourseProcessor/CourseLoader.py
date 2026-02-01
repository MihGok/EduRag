import os
import requests
import json
import re
import time
import random
from urllib.parse import urlencode
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from requests.auth import HTTPBasicAuth

# Импорт конфигураций
from services.config import ProxyConfig, AppConfig

MAX_RETRIES = 5
BASE_DELAY = 2
load_dotenv()

# Абсолютный путь к директории, где лежит этот файл.
# Используем его как якорь для token_storage.json, чтобы файл
# создавался рядом со скриптом, а не зависел от текущей рабочей директории.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
TOKEN_STORAGE_PATH = os.path.join(_THIS_DIR, "token_storage.json")


def make_request_with_retry(func):
    """Декоратор для устойчивых HTTP-запросов: retry на 429/5xx и сетевые ошибки."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                response = func(*args, **kwargs)

                if response is None:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (BASE_DELAY ** attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)
                        continue
                    return None

                if getattr(response, 'status_code', None) in (200, 201):
                    return response

                # Критические ошибки — не ретраим
                if response.status_code in (401, 403, 404):
                    print(f"[ERROR] Критическая ошибка {response.status_code}: {getattr(response, 'url', '')}")
                    if response.status_code == 401:
                        print(f"[ERROR] Детали 401: {response.text[:500]}")
                    return response

                # Временные ошибки — ретраим
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (BASE_DELAY ** attempt) + random.uniform(0, 1)
                        print(f"[RETRY] Код {response.status_code}. Попытка {attempt+1}/{MAX_RETRIES}. Жду {wait_time:.2f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[FAIL] Попытки исчерпаны. Код {response.status_code}. Тело ответа: {response.text[:400]}")
                        return response

                print(f"[WARN] Неожиданный код {response.status_code}. URL: {getattr(response, 'url', '')}. Тело: {response.text[:300]}")
                return response

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = (BASE_DELAY ** attempt) + random.uniform(0, 1)
                    print(f"[NET ERROR] {e}. Попытка {attempt+1}/{MAX_RETRIES}. Жду {wait_time:.2f}s")
                    time.sleep(wait_time)
                    continue
                print(f"[FAIL] Ошибка сети: {e}")
                return None
            except Exception as e:
                print(f"[EXCEPTION] {type(e).__name__}: {e}")
                return None
        return None
    return wrapper


class StepikCourseLoader:
    """
    Загрузчик курсов Stepik.

    Потокобезопасность и использование одного экземпляра в ThreadPoolExecutor
    -------------------------------------------------------------------------
    В фазе B1 несколько потоков вызывают process_course() на ОДНОМ экземпляре
    этого класса одновременно. Это безопасно по следующим причинам:

    1. self.session — это requests.Session. Его метод .get()/.post() потокобезопасен:
       внутри urllib3 пул соединений защищён threading.Lock.
       Документация: https://github.com/psf/requests/issues/1738

    2. После __init__ мы НИКОГДА не меняем self.session.headers и self.token.
       Все потоки только читают эти поля — гонок нет.

    3. Единственное изменяемое состояние, которое было раньше — это
       self._last_raw_response. Оно использовалось как "побочный канал"
       между fetch_object_single() и process_lesson(): первый записывал,
       второй читал. При многопоточном использовании любой поток мог
       перезаписать значение между этими двумя вызовами в другом потоке.

       Этот паттерн УДАЛЁН. fetch_object_single() теперь возвращает оба значения
       в одном кортеже (объект, raw_response), и вся коммуникация идёт через
       локальные переменные, которые каждый поток видит только свои.

    Итог: один StepikCourseLoader можно безопасно передать в ThreadPoolExecutor
    без дополнительной синхронизации.
    """

    API_URL = "https://stepik.org/api"
    OAUTH_URL = "https://stepik.org/oauth2/token/"
    AUTH_URL = "https://stepik.org/oauth2/authorize/"
    REDIRECT_URI = "http://localhost:5000/callback"

    def __init__(self):
        self.client_id = AppConfig.STEPIK_CLIENT_ID
        self.client_secret = AppConfig.STEPIK_CLIENT_SECRET

        if not self.client_id or not self.client_secret:
            raise ValueError("Не найдены STEPIK_CLIENT_ID или STEPIK_CLIENT_SECRET в .env")

        self.session = ProxyConfig.get_session_with_proxy(use_proxy=True)

        USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 YaBrowser/25.10.0.0 Safari/537.36"
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://stepik.org/',
            'Origin': 'https://stepik.org',
            'Connection': 'keep-alive',
        })

        self.token = self._login_flow()
        if self.token:
            self.session.headers.update({'Authorization': f'Bearer {self.token}'})

    # ------------------------------------------------------------------ AUTH

    def _login_flow(self) -> Optional[str]:
        """Логин через OAuth с использованием прокси"""
        if os.path.exists(TOKEN_STORAGE_PATH):
            try:
                with open(TOKEN_STORAGE_PATH, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('refresh_token'):
                        print("[AUTH] Обнаружен refresh_token — пробуем обновить...")
                        return self._refresh_access_token(data['refresh_token'])
            except Exception as e:
                print(f"[AUTH] Ошибка чтения token_storage.json: {e}")
                try:
                    os.remove(TOKEN_STORAGE_PATH)
                except OSError:
                    pass

        return self._authorize_user_manual()

    def _save_tokens(self, tokens: Dict[str, Any]):
        """Сохраняет токены в файл"""
        with open(TOKEN_STORAGE_PATH, "w", encoding='utf-8') as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)
        print("[AUTH] Токены сохранены в token_storage.json")
        if 'scope' in tokens:
            print(f"[AUTH] Полученные scopes: {tokens['scope']}")

    def _exchange_code_for_token(self, code: str) -> Optional[str]:
        """Обмен кода на токен"""
        auth = HTTPBasicAuth(self.client_id, self.client_secret)

        @make_request_with_retry
        def execute():
            return self.session.post(self.OAUTH_URL, data={
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': self.REDIRECT_URI
            }, auth=auth, timeout=15)

        resp = execute()
        if not resp:
            raise ConnectionError("Нет ответа при обмене кода на токен")
        if resp.status_code != 200:
            raise ConnectionError(f"Ошибка обмена кода: {resp.status_code} {resp.text}")

        tokens = resp.json()
        self._save_tokens(tokens)
        access = tokens.get('access_token')
        if access:
            self.session.headers.update({'Authorization': f'Bearer {access}'})
        return access

    def _authorize_user_manual(self) -> Optional[str]:
        """Ручная авторизация через браузер"""
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.REDIRECT_URI,
            'scope': 'read write'
        }
        auth_link = f"{self.AUTH_URL}?{urlencode(params)}"

        print('\n' + '=' * 80)
        print('ВАЖНО! ПЕРЕД АВТОРИЗАЦИЕЙ:')
        print('1. Убедитесь, что ваше приложение на https://stepik.org/oauth2/applications/')
        print('   настроено с типом "Confidential" и grant type "authorization-code"')
        print('2. REDIRECT_URI должен быть точно: http://localhost:5000/callback')
        print('=' * 80)
        print('\nОТКРОЙТЕ ССЫЛКУ В БРАУЗЕРЕ ДЛЯ АВТОРИЗАЦИИ:')
        print(auth_link)
        print('\nПосле авторизации /callback в URL будет параметр ?code=... — вставьте его сюда.')
        print('=' * 80 + '\n')

        code = input('Вставьте code из URL: ').strip()
        if not code:
            raise ConnectionError('Код авторизации не введён')
        return self._exchange_code_for_token(code)

    def _refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Обновление токена"""
        auth = HTTPBasicAuth(self.client_id, self.client_secret)

        @make_request_with_retry
        def execute():
            return self.session.post(self.OAUTH_URL, data={
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token
            }, auth=auth, timeout=15)

        resp = execute()
        if not resp:
            print('[AUTH] Не удалось получить ответ при refresh')
            return None
        if resp.status_code != 200:
            print(f"[AUTH] Refresh вернул {resp.status_code}: {resp.text[:300]}")
            print("[AUTH] Удаляю старый токен и запрашиваю новую авторизацию...")
            try:
                os.remove(TOKEN_STORAGE_PATH)
            except OSError:
                pass
            return self._authorize_user_manual()

        tokens = resp.json()
        self._save_tokens(tokens)
        access = tokens.get('access_token')
        if access:
            self.session.headers.update({'Authorization': f'Bearer {access}'})
        return access

    def _get_headers(self) -> Dict[str, str]:
        """Возвращает заголовки для запросов"""
        auth_header = self.session.headers.get("Authorization")
        return {
            'Authorization': auth_header if auth_header else f'Bearer {self.token}',
            'Content-Type': 'application/json',
            'Referer': 'https://stepik.org/'
        }

    @make_request_with_retry
    def _fetch_single_raw(self, url: str, headers: Dict[str, str], params: Any = None) -> Optional[requests.Response]:
        """Базовый метод для GET запросов"""
        try:
            return self.session.get(url, headers=headers, params=params, timeout=20)
        except Exception as e:
            print(f"[HTTP GET ERROR] {type(e).__name__}: {e}")
            return None

    # -------------------------------------------------------- ПОИСК КУРСОВ

    def get_course_ids_by_query(self, query: str, language: str = 'ru', limit: int = 50) -> List[int]:
        """Ищет ID курсов по запросу с фильтрацией."""
        print(f"[SEARCH] Ищу курсы по запросу '{query}' (lang={language}, limit={limit})...")
        url = f"{self.API_URL}/search-results"
        course_ids = []
        page = 1

        while len(course_ids) < limit:
            params = {
                'query': query,
                'is_public': 'true',
                'is_paid': 'false',
                'language': language,
                'type': 'course',
                'page': page
            }

            response = self._fetch_single_raw(url=url, headers=self._get_headers(), params=params)

            if not response or response.status_code != 200:
                print(f"[SEARCH] Ошибка запроса на странице {page}")
                break

            data = response.json()
            results = data.get('search-results', [])
            meta = data.get('meta', {})

            if not results:
                print("[SEARCH] Результаты закончились.")
                break

            for r in results:
                cid = r.get('target_id') or r.get('target')
                if cid and cid not in course_ids:
                    course_ids.append(cid)

            print(f"  -> Страница {page}: найдено {len(results)}, всего собрано {len(course_ids)}")

            if not meta.get('has_next'):
                break

            page += 1
            time.sleep(1.5)

        return course_ids[:limit]

    def _sanitize_filename(self, name: Any) -> str:
        name = str(name or '').strip()
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = name.strip().rstrip('.')
        if not name:
            return 'Unnamed'
        return name

    def get_course_outline(self, course: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Собирает структуру курса (список уроков с названиями) БЕЗ скачивания контента."""
        cid = course.get('id')
        print(f"[INFO] Сбор структуры курса {cid} для анализа...")

        section_ids = course.get('sections') or []
        if not section_ids:
            return []

        sections = self.fetch_objects('sections', section_ids)
        sections.sort(key=lambda x: x.get('position', 0))

        all_lessons_metadata = []

        for section in sections:
            unit_ids = section.get('units') or []
            if not unit_ids:
                continue

            units = self.fetch_objects('units', unit_ids)
            units.sort(key=lambda x: x.get('position', 0))

            lesson_ids = [u.get('lesson') for u in units if u.get('lesson')]
            lessons = self.fetch_objects('lessons', lesson_ids)
            lessons_map = {l['id']: l for l in lessons}

            for unit in units:
                lid = unit.get('lesson')
                lesson = lessons_map.get(lid)
                if lesson:
                    all_lessons_metadata.append({
                        'lesson_id': lesson['id'],
                        'title': lesson['title'],
                        'section_title': section['title']
                    })

        print(f"[INFO] Найдено уроков: {len(all_lessons_metadata)}")
        return all_lessons_metadata

    # --------------------------------------------------------- FETCH API

    def fetch_object_single(self, object_type: str, object_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Загружает один объект по ID.

        Возвращает кортеж (parsed_object, raw_response_body).
        Оба значения — локальные, без записи в self.
        Это гарантирует потокобезопасность: каждый вызывающий поток
        получает свои данные через return, а не через общее состояние объекта.
        """
        url = f"{self.API_URL}/{object_type}/{object_id}"
        response = self._fetch_single_raw(url=url, headers=self._get_headers())
        if not response or response.status_code != 200:
            return {}, {}

        data = response.json()

        items = data.get(object_type) or data.get(object_type + 's') or data.get(object_type.rstrip('s'))
        if isinstance(items, list) and items:
            return items[0], data
        if isinstance(data, dict) and data.get('id'):
            return data, data
        return {}, data

    def fetch_objects(self, object_type: str, object_ids: List[int]) -> List[Dict[str, Any]]:
        if not object_ids:
            return []
        url = f"{self.API_URL}/{object_type}"
        objects: List[Dict[str, Any]] = []
        chunk_size = 20

        for i in range(0, len(object_ids), chunk_size):
            chunk = object_ids[i:i + chunk_size]
            params = [("ids[]", str(x)) for x in chunk]

            response = self._fetch_single_raw(url=url, headers=self._get_headers(), params=params)
            if not response or response.status_code != 200:
                continue

            data = response.json()
            key = object_type if object_type in data else (list(data.keys())[0] if data else object_type)
            fetched = data.get(key) or []
            if isinstance(fetched, list):
                objects.extend(fetched)

            time.sleep(0.1)

        return objects

    # --------------------------------------------------------- ENROLLMENT

    def enroll_in_course(self, course_id: int) -> bool:
        """Записаться на курс через API"""
        url = f"{self.API_URL}/enrollments"
        payload = {"enrollment": {"course": str(course_id)}}

        @make_request_with_retry
        def execute():
            return self.session.post(url, headers=self._get_headers(), json=payload, timeout=15)

        response = execute()

        if response and response.status_code in (200, 201):
            print(f"[SUCCESS] Зачислен на курс {course_id}")
            return True
        elif response and response.status_code == 400:
            error_text = response.text
            if 'already enrolled' in error_text.lower() or 'уже записан' in error_text.lower():
                print(f"[INFO] Уже записаны на курс {course_id}")
                return True
            print(f"[ERROR] Ошибка 400 при записи на курс {course_id}: {error_text[:300]}")
            return False
        elif response and response.status_code == 401:
            print(f"[ERROR] 401 Unauthorized при записи на курс {course_id}")
            return False
        else:
            print(f"[ERROR] Не удалось записаться на курс {course_id}")
            if response:
                print(f"  Статус: {response.status_code}")
                print(f"  Ответ: {response.text[:500]}")
            return False

    def check_enrollment(self, course_id: int) -> bool:
        """Проверить, записан ли пользователь на курс"""
        url = f"{self.API_URL}/enrollments"
        params = {"course": course_id}

        response = self._fetch_single_raw(url=url, headers=self._get_headers(), params=params)
        if not response or response.status_code != 200:
            return False

        data = response.json()
        enrollments = data.get('enrollments', [])
        return len(enrollments) > 0

    def search_public_free_courses(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        print(f"Поиск курсов по запросу: {query} ...")
        url = f"{self.API_URL}/search-results"
        found: List[Dict[str, Any]] = []
        page = 1

        while len(found) < limit:
            params = {'query': query, 'is_public': 'true', 'is_paid': 'false', 'type': 'course', 'page': page}
            response = self._fetch_single_raw(url=url, headers=self._get_headers(), params=params)
            if not response or response.status_code != 200:
                break

            results = response.json().get('search-results', [])
            if not results:
                break

            course_ids = []
            for r in results:
                cid = r.get('target_id') or r.get('target') or r.get('course')
                if cid:
                    course_ids.append(cid)
                    if len(course_ids) >= (limit - len(found)):
                        break

            if not course_ids:
                break

            courses = self.fetch_objects('courses', course_ids)
            for c in courses:
                if c.get('is_public') and not c.get('is_paid'):
                    found.append(c)
                    if len(found) >= limit:
                        break

            page += 1

        return found[:limit]

    # -------------------------------------------------------- FILE I/O

    def save_json(self, data: Any, folder: str, filename: str):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения {path}: {e}")

    # -------------------------------------------------------- ОБРАБОТКА

    def process_course(self, course: Dict[str, Any], allowed_lesson_ids: Optional[List[int]] = None):
        """
        Скачивает контент курса и сохраняет в папку на диске.

        course_dir строится как абсолютный путь относительно CWD на момент вызова.
        Это гарантирует однозначность: даже если что-то ещё в процессе меняет CWD
        (крайне редко, но теоретически возможно в многопоточном коде),
        мы зафиксируем путь в начале и используем его повсюду ниже.
        """
        cid = course.get('id')
        title = self._sanitize_filename(course.get('title', f'course_{cid}'))
        course_dir = os.path.abspath(f"Course_{cid}_{title}")
        os.makedirs(course_dir, exist_ok=True)

        # Логика записи на курс
        is_enrolled = course.get('is_enrolled', False)
        if not is_enrolled:
            if not self.check_enrollment(cid):
                print(f"[INFO] Не записаны на курс {cid}. Попытка записи...")
                self.enroll_in_course(cid)
                time.sleep(1)

        self.save_json(course, course_dir, f"course_{cid}.json")

        section_ids = course.get('sections') or []
        sections = self.fetch_objects('sections', section_ids)
        sections.sort(key=lambda x: x.get('position', 0))

        for s in sections:
            self.process_section(s, course_dir, allowed_lesson_ids)

    def process_section(self, section: Dict[str, Any], parent_dir: str, allowed_lesson_ids: Optional[List[int]] = None):
        sid = section.get('id')
        pos = section.get('position', 0)
        title = self._sanitize_filename(section.get('title', f'Section_{sid}'))
        section_dir = os.path.join(parent_dir, f"Section_{pos:02d}_{title}")

        unit_ids = section.get('units') or []
        if not unit_ids:
            return

        units = self.fetch_objects('units', unit_ids)
        units.sort(key=lambda x: x.get('position', 0))

        # Фильтрация юнитов
        units_to_process = []
        if allowed_lesson_ids is not None:
            allowed_set = set(allowed_lesson_ids)
            for u in units:
                if u.get('lesson') in allowed_set:
                    units_to_process.append(u)
        else:
            units_to_process = units

        if not units_to_process:
            return

        os.makedirs(section_dir, exist_ok=True)
        self.save_json(section, section_dir, f"section_{sid}.json")

        lesson_ids = [u.get('lesson') for u in units_to_process if u.get('lesson')]
        lessons = self.fetch_objects('lessons', lesson_ids)
        lessons_map = {l.get('id'): l for l in lessons}

        for unit in units_to_process:
            lid = unit.get('lesson')
            lesson = lessons_map.get(lid)
            if lesson:
                self.process_lesson(lesson, unit, section_dir)

    def process_lesson(self, lesson: Dict[str, Any], unit: Dict[str, Any], parent_dir: str):
        lesson_id = lesson.get('id')
        pos = unit.get('position', 0)
        title = self._sanitize_filename(lesson.get('title', f'lesson_{lesson_id}'))

        lesson_dir = os.path.join(parent_dir, f"Lesson_{pos:02d}_{title}")
        os.makedirs(lesson_dir, exist_ok=True)

        self.save_json(unit, lesson_dir, f"unit_{unit.get('id')}.json")
        self.save_json(lesson, lesson_dir, f"lesson_bulk_{lesson_id}.json")

        step_ids = lesson.get('steps') or []
        if not step_ids:
            print(f"    [INFO] Урок {lesson_id} не содержит steps в bulk. Пытаюсь одиночный запрос...")

            # fetch_object_single возвращает кортеж (объект, raw_ответ).
            # Оба значения — локальные переменные этого вызова.
            # Даже если другой поток одновременно вызывает тот же метод,
            # его return идёт в свои локальные переменные — пересечения нет.
            full, raw_data = self.fetch_object_single('lessons', lesson_id)

            if raw_data:
                self.save_json(raw_data, lesson_dir, f"lesson_raw_{lesson_id}.json")

            if full and full.get('steps'):
                lesson = full
                step_ids = lesson.get('steps')
                print(f"    Fallback успешен: найдено {len(step_ids)} шагов.")
            else:
                print(f"    [WARN] Шаги не найдены даже после одиночного запроса.")

        print(f"    -> Урок {pos}: {title} (Шагов: {len(step_ids)})")
        if not step_ids:
            return

        steps = self.fetch_objects('steps', step_ids)
        steps.sort(key=lambda x: x.get('position', 0))
        for st in steps:
            self.save_step(st, lesson_dir)

    def save_step(self, step: Dict[str, Any], parent_dir: str):
        sid = step.get('id')
        pos = step.get('position', 0)
        block = (step.get('block') or {}).get('name', 'unknown')
        block_safe = self._sanitize_filename(block)
        fname = f"step_{pos:02d}_{sid}_{block_safe}.json"
        self.save_json(step, parent_dir, fname)
