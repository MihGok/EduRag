import os
import requests
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

class ProxyConfig:
    """Централизованная конфигурация прокси для всего проекта"""
    
    # Прокси для внешних API (Gemini, Stepik)
    EXTERNAL_PROXY = os.getenv("EXTERNAL_PROXY", "http://127.0.0.1:12334")
    
    # Локальные сервисы БЕЗ прокси
    NO_PROXY_HOSTS = ["localhost", "127.0.0.1", "minio"]
    
    @classmethod
    def get_session_with_proxy(cls, use_proxy: bool = True) -> requests.Session:
        """
        Создает requests.Session с правильными настройками прокси
        
        Args:
            use_proxy: True для внешних API, False для локальных сервисов
        """
        session = requests.Session()
        
        if use_proxy and cls.EXTERNAL_PROXY:
            session.proxies = {
                'http': cls.EXTERNAL_PROXY,
                'https': cls.EXTERNAL_PROXY
            }
            print(f"[ProxyConfig] Сессия с прокси: {cls.EXTERNAL_PROXY}")
        else:
            session.trust_env = False  # Игнорируем системные прокси
            print("[ProxyConfig] Сессия без прокси (локальный сервис)")
        
        return session
    
    @classmethod
    def get_requests_proxies(cls) -> dict:
        """Возвращает словарь прокси для requests"""
        if cls.EXTERNAL_PROXY:
            return {
                'http': cls.EXTERNAL_PROXY,
                'https': cls.EXTERNAL_PROXY
            }
        return {}
    
    @classmethod
    def download_file(cls, url: str, output_path: str, use_proxy: bool = None) -> bool:
        """
        Универсальная функция скачивания файлов с правильным прокси
        
        Args:
            url: URL для скачивания
            output_path: Куда сохранить файл
            use_proxy: Автоопределение по URL если None
        """
        if use_proxy is None:
            # Автоматически определяем нужен ли прокси
            use_proxy = not any(host in url for host in cls.NO_PROXY_HOSTS)
        
        session = cls.get_session_with_proxy(use_proxy)
        
        try:
            with session.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"[ProxyConfig] Ошибка скачивания {url}: {e}")
            return False


class AppConfig:
    """Общие настройки приложения"""
    
    # ML Backend
    ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://localhost:8000")
    
    # MinIO
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "course-frames")
    
    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Stepik
    STEPIK_CLIENT_ID = os.getenv("STEPIK_CLIENT_ID")
    STEPIK_CLIENT_SECRET = os.getenv("STEPIK_CLIENT_SECRET")
    
    # Пути
    TEMP_DIR = os.getenv("TEMP_DIR", "temp_files")
    KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
    
    # Whisper
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
    
    # LLaVA Model Path
    LLAVA_MODEL_PATH = os.getenv(
        "LLAVA_MODEL_PATH", 
        r"C:\Users\user\Desktop\models\llama3-llava"
    )
    
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    
    @classmethod
    def validate(cls) -> bool:
        """Проверяет наличие критических настроек"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY не установлен")
        
        if not cls.STEPIK_CLIENT_ID or not cls.STEPIK_CLIENT_SECRET:
            errors.append("STEPIK_CLIENT_ID или STEPIK_CLIENT_SECRET не установлены")
        
        if not os.path.exists(cls.LLAVA_MODEL_PATH):
            errors.append(f"LLaVA модель не найдена: {cls.LLAVA_MODEL_PATH}")
        
        if errors:
            print("[Config] ОШИБКИ КОНФИГУРАЦИИ:")
            for err in errors:
                print(f"  ❌ {err}")
            return False
        
        print("[Config] ✅ Все критические настройки в порядке")
        return True
