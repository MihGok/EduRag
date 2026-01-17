import os
import json
import threading
import gc
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None
    LlamaGrammar = None

app = FastAPI(title="Local LLM endpoint (llama.cpp)")


_current_llm: Optional['Llama'] = None
_current_config_key: Optional[str] = None
_cache_lock = threading.Lock()

class GenerateRequest(BaseModel):
    model_path: Optional[str] = None
    prompt: str
    response_schema: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    n_batch: Optional[int] = None

def _unload_current_model():
    """Принудительная выгрузка модели и очистка VRAM."""
    global _current_llm, _current_config_key
    if _current_llm is not None:
        print(f"[LLM] Unloading model: {_current_config_key}...")
        del _current_llm
        _current_llm = None
        _current_config_key = None
        
        
        gc.collect()
        
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[LLM] CUDA VRAM cleared (via torch).")
        else:
            print("[LLM] RAM cleared (Standard GC).")

def get_llm_instance(model_path: str, n_ctx: int, n_gpu_layers: int, n_batch: Optional[int]):
    """
    Возвращает экземпляр модели. Если параметры изменились — 
    выгружает старую и загружает новую.
    """
    global _current_llm, _current_config_key
    
    new_key = f"{model_path}|ctx={n_ctx}|gpu_layers={n_gpu_layers}|batch={n_batch}"
    
    with _cache_lock:
        if _current_llm is not None and new_key == _current_config_key:
            return _current_llm
        
        if _current_llm is not None:
            _unload_current_model()
            
        print(f"[LLM] Loading new model: {new_key}")
        
        kwargs = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers if n_gpu_layers is not None else -1,
            "verbose": True
        }
        if n_batch is not None:
            kwargs["n_batch"] = n_batch
            
        try:
            if Llama is None:
                raise ImportError("Library 'llama_cpp' not found. Please install llama-cpp-python.")

            _current_llm = Llama(**kwargs)
            _current_config_key = new_key
            print("[LLM] Model loaded successfully.")
            return _current_llm
        except Exception as e:
            print(f"[LLM] CRITICAL ERROR loading model: {e}")
            _unload_current_model()
            raise e

@app.get("/health")
def health():
    return {
        "ok": True, 
        "loaded_model": _current_config_key if _current_config_key else "None",
        "torch_available": (torch is not None)
    }

@app.post("/generate")
def generate(req: GenerateRequest):
    model_path = req.model_path or os.environ.get("DEFAULT_MODEL_PATH")
    if model_path is None:
        raise HTTPException(status_code=400, detail="model_path not provided")
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")


    n_ctx = req.n_ctx or int(os.environ.get("DEFAULT_N_CTX", "2048"))
    n_gpu_layers = req.n_gpu_layers if req.n_gpu_layers is not None else int(os.environ.get("DEFAULT_N_GPU_LAYERS", "-1"))
    n_batch = req.n_batch or 512


    try:
        llm = get_llm_instance(str(model_file), n_ctx, n_gpu_layers, n_batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")


    call_kwargs = {
        "prompt": req.prompt,
        "max_tokens": req.max_tokens or 256,
        "temperature": req.temperature if req.temperature is not None else 0.0,
        "top_p": req.top_p if req.top_p is not None else 1.0,
    }


    if req.response_schema:
        try:
            grammar = LlamaGrammar.from_json_schema(json.dumps(req.response_schema))
            call_kwargs["grammar"] = grammar
            call_kwargs["prompt"] += "\nReturn output in strict JSON format."
        except Exception as e:
            print(f"Grammar error: {e}. Proceeding without grammar.")

    # 5. Запуск инференса
    try:
        resp = llm(**call_kwargs)
        text = resp["choices"][0]["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 6. Обработка JSON
    if req.response_schema:
        clean_text = text
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(clean_text)
            return {"success": True, "json": parsed}
        except Exception as e:
            return {"success": False, "error": "Failed to parse JSON", "raw_text": text}

    return {"success": True, "text": text}
