import requests
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.config import AppConfig, ProxyConfig


class Client:
    """
    Клиент для взаимодействия с ML Backend.
    Поддерживает параллельную транскрибацию видео.
    """
    
    ML_BACKEND_URL = AppConfig.ML_SERVER_URL
    TRANSCRIBE_ENDPOINT = f"{ML_BACKEND_URL}/transcribe"
    
    # Количество параллельных запросов к транскрибатору
    MAX_TRANSCRIBE_WORKERS = 2
    
    @classmethod
    def transcribe(cls, video_url: str, step_id: int = None) -> Dict[str, Any]:
        """
        Транскрибирует одно видео.
        
        Args:
            video_url: URL видео для транскрибации
            step_id: ID шага (для логирования)
        
        Returns:
            {"text": "полная транскрипция", "segments": [...]}
        """
        session = ProxyConfig.get_session_with_proxy(use_proxy=False)
        
        payload = {"video_url": video_url}
        
        try:
            response = session.post(
                cls.TRANSCRIBE_ENDPOINT, 
                json=payload, 
                timeout=600  # 10 минут на видео
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "text": data.get("transcript", ""),
                "segments": data.get("segments", [])
            }
            
        except Exception as e:
            print(f"   [Transcribe Error] Step {step_id}: {e}")
            return {"text": "", "segments": []}
    
    @classmethod
    def transcribe_batch(cls, videos: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Параллельная транскрибация нескольких видео.
        
        Args:
            videos: Список словарей [{"step_id": 123, "video_url": "..."}, ...]
        
        Returns:
            {step_id: {"text": "...", "segments": [...]}, ...}
        """
        if not videos:
            return {}
        
        print(f"   [Transcribe Batch] Обработка {len(videos)} видео (параллельно: {cls.MAX_TRANSCRIBE_WORKERS} воркера)...")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=cls.MAX_TRANSCRIBE_WORKERS) as executor:
            # Создаем futures для каждого видео
            future_to_video = {
                executor.submit(cls.transcribe, v["video_url"], v["step_id"]): v 
                for v in videos
            }
            
            # Собираем результаты по мере готовности
            for future in as_completed(future_to_video):
                video = future_to_video[future]
                step_id = video["step_id"]
                
                try:
                    result = future.result()
                    results[step_id] = result
                    
                    if result.get("text"):
                        print(f"      ✅ Step {step_id}: {len(result['text'])} символов")
                    else:
                        print(f"      ⚠️  Step {step_id}: пустая транскрипция")
                        
                except Exception as e:
                    print(f"      ❌ Step {step_id}: {e}")
                    results[step_id] = {"text": "", "segments": []}
        
        return results
