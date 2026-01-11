import os
import json
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar

app = FastAPI(title="Local LLM endpoint (llama.cpp)")

# Кэш инстансов моделей, чтобы не загружать их при каждом запросе
_llm_cache: Dict[str, Llama] = {}
_cache_lock = threading.Lock()

class GenerateRequest(BaseModel):
    model_path: Optional[str] = None        # Путь к .gguf
    prompt: str
    response_schema: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    n_batch: Optional[int] = None

def get_llm_instance(model_path: str, n_ctx: int, n_gpu_layers: int, n_batch: Optional[int]):
    """Возвращает существующий или создает новый инстанс модели."""
    key = f"{model_path}|ctx={n_ctx}|gpu_layers={n_gpu_layers}|batch={n_batch}"
    with _cache_lock:
        if key in _llm_cache:
            return _llm_cache[key]
        
        kwargs = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "verbose": True  # Полезно для отладки GPU в логах
        }
        if n_gpu_layers is not None:
            kwargs["n_gpu_layers"] = n_gpu_layers
        if n_batch is not None:
            kwargs["n_batch"] = n_batch
            
        llm = Llama(**kwargs)
        _llm_cache[key] = llm
        return llm

@app.get("/health")
def health():
    return {"ok": True, "cached_models": list(_llm_cache.keys())}

@app.post("/generate")
def generate(req: GenerateRequest):
    # 1. Определяем путь к модели
    model_path = req.model_path or os.environ.get("DEFAULT_MODEL_PATH")
    if model_path is None:
        raise HTTPException(status_code=400, detail="model_path not provided and DEFAULT_MODEL_PATH not set")
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    # 2. Параметры контекста и GPU
    n_ctx = req.n_ctx or int(os.environ.get("DEFAULT_N_CTX", "2048"))
    n_gpu_layers = req.n_gpu_layers if req.n_gpu_layers is not None else int(os.environ.get("DEFAULT_N_GPU_LAYERS", "-1"))
    n_batch = req.n_batch or 512

    # 3. Инициализация или получение модели из кэша
    try:
        llm = get_llm_instance(str(model_file), n_ctx, n_gpu_layers, n_batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # 4. Подготовка параметров генерации
    call_kwargs = {
        "prompt": req.prompt,
        "max_tokens": req.max_tokens or 256,
        "temperature": req.temperature if req.temperature is not None else 0.0,
        "top_p": req.top_p if req.top_p is not None else 1.0,
    }

    # Внедрение Grammar (GBNF) если задана схема
    if req.response_schema:
        try:
            # Автоматически создаем грамматику из JSON-схемы
            grammar = LlamaGrammar.from_json_schema(json.dumps(req.response_schema))
            call_kwargs["grammar"] = grammar
            
            # Добавляем инструкцию в промпт для модели
            call_kwargs["prompt"] += "\nReturn output in strict JSON format."
        except Exception as e:
            print(f"Grammar error: {e}. Proceeding without grammar.")

    # 5. Запуск генерации
    try:
        # Прямой вызов объекта llm вместо .create()
        resp = llm(**call_kwargs)
        text = resp["choices"][0]["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 6. Обработка результата
    if req.response_schema:
        # Очистка от Markdown-тегов, если модель их все же добавила
        clean_text = text
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(clean_text)
            return {"success": True, "json": parsed}
        except Exception as e:
            # Если даже с грамматикой произошла ошибка парсинга (редко)
            return {
                "success": False, 
                "error": "Failed to parse JSON", 
                "raw_text": text
            }

    # Если схема не задана — возвращаем просто текст
    return {"success": True, "text": text}