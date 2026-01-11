import os
import google.generativeai as genai
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
import time  # Импортируем time для retry логики

T = TypeVar("T", bound=BaseModel)

class GeminiService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация сервиса с поддержкой прокси.
        :param api_key: Если None, берется из os.environ["GEMINI_API_KEY"]
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or arguments.")
        
        # ИСПРАВЛЕНИЕ: Настройка прокси
        proxy_url = os.getenv("GEMINI_PROXY", "http://127.0.0.1:12334")
        
        # Библиотека google-generativeai (через requests) автоматически подхватывает 
        # переменные окружения. Устанавливаем их явно для этого процесса.
        if proxy_url:
            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url
            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url


            os.environ['no_proxy'] = "localhost,127.0.0.1,0.0.0.0,minio" 
            os.environ['NO_PROXY'] = "localhost,127.0.0.1,0.0.0.0,minio"
            print(f"[Gemini] Настроен прокси через env: {proxy_url}")

        # Конфигурируем SDK
        # Используем transport='rest', так как он лучше работает с обычными HTTP-прокси, чем gRPC
        genai.configure(api_key=self.api_key, transport="rest")

    def generate(
        self,
        prompt: str,
        response_schema: Type[T],
        model_name: str = "gemini-2.5-flash",
        image_path: Optional[str] = None,
        temperature: float = 0.5,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
        retry_count: int = 3
    ) -> T:
        """
        Универсальный метод генерации с валидацией через Pydantic.
        Добавлена логика повторных попыток.
        """
        
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=response_schema
        )

        content: list[Any] = [prompt]
        
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            # Загружаем файл через API
            uploaded_file = genai.upload_file(image_path)
            content.append(uploaded_file)

        # Retry логика
        last_error = None
        for attempt in range(retry_count):
            try:
                response = model.generate_content(
                    content,
                    generation_config=generation_config
                )
                
                # Валидация и парсинг
                result = response_schema.model_validate_json(response.text)
                return result

            except Exception as e:
                last_error = e
                print(f"[GeminiService] Попытка {attempt + 1}/{retry_count} не удалась: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Экспоненциальная задержка
                continue
        
        print(f"[GeminiService] Все попытки исчерпаны. Последняя ошибка: {last_error}")
        raise last_error
