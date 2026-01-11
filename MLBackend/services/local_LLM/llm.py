# app.py
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from llama_cpp import Llama  # pip install llama-cpp-python
from schemas import ScoreRequest, ScoreResponse, CourseOutput, DEFAULT_SCORE_CLASSES
from prompts import build_prompt
from dotenv import load_dotenv

load_dotenv()
# --- Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# --- Config
MODEL_PATH = os.getenv("MODEL_PATH", "saiga_nemo_12b.Q4_K_M.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))  # -1 = try auto (may or may not work)
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "8"))
N_BATCH = int(os.getenv("N_BATCH", "256"))
VERBOSE = os.getenv("LLM_VERBOSE", "true").lower() in ("1", "true", "yes")

# Score classes default
SCORE_CLASSES = json.loads(os.getenv("SCORE_CLASSES_JSON", "null")) or DEFAULT_SCORE_CLASSES

app = FastAPI(title="Local GGUF scorer (llama.cpp)")

llm: Optional[Llama] = None

@app.on_event("startup")
def startup_event():
    global llm
    try:
        log.info(f"Loading model from {MODEL_PATH} ... (n_gpu_layers={N_GPU_LAYERS})")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=N_BATCH,
            n_threads=N_THREADS,
            verbose=VERBOSE
        )
        log.info("Model loaded.")
    except Exception as e:
        log.exception("Failed to load model:")
        # raise here so server startup fails loudly
        raise RuntimeError(f"Failed to load model: {e}")

def extract_json_array(text: str) -> Optional[str]:
    """Find first '[' and last ']' and return substring, with trivial cleanup."""
    if not text:
        return None
    first = text.find('[')
    last = text.rfind(']')
    if first == -1 or last == -1 or last < first:
        return None
    candidate = text[first:last+1]
    # remove trailing commas like ", ]" or ", }"
    candidate = re.sub(r',\s*(\]|\})', r'\1', candidate)
    return candidate

def safe_extract_text(resp: Any) -> str:
    """Robust extraction of text from llama-cpp-python response."""
    # resp often a dict with choices -> list -> dict with 'text'
    text = ""
    try:
        if isinstance(resp, dict):
            choices = resp.get("choices")
            if choices and isinstance(choices, list):
                c0 = choices[0]
                if isinstance(c0, dict):
                    # many versions: 'text' or 'delta' or nested message
                    text = c0.get("text") or c0.get("delta") or ""
                    if not text and "message" in c0:
                        msg = c0.get("message")
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, str):
                                text = content
            # fallback
            if not text:
                text = resp.get("text") or ""
        else:
            text = str(resp)
    except Exception as e:
        log.exception("safe_extract_text error")
        text = str(resp)
    return text

def validate_and_normalize(parsed: Any, allowed_classes: List[str]) -> List[Dict]:
    if not isinstance(parsed, list):
        raise ValueError("Top-level JSON is not an array")
    out = []
    ids_seen = set()
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not an object")
        for key in ("id", "title", "reasoning", "score_class"):
            if key not in item:
                raise ValueError(f"Missing field '{key}' in item {i}")
        sc = item["score_class"]
        if sc not in allowed_classes and sc != "Unknown":
            raise ValueError(f"Invalid score_class '{sc}' in item {i}. Allowed: {allowed_classes} or 'Unknown'")
        if item["id"] in ids_seen:
            raise ValueError(f"Duplicate id '{item['id']}'")
        ids_seen.add(item["id"])
        out.append({
            "id": item["id"],
            "title": item["title"],
            "reasoning": item["reasoning"],
            "score_class": item["score_class"]
        })
    return out

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    global llm
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    courses = [c.dict() for c in req.courses]
    classes = req.score_classes or SCORE_CLASSES

    prompt = build_prompt(req.query_topic, courses, classes)

    # Подготовка грамматики (если передана)
    grammar = None
    if req.grammar:
        try:
            grammar = Llama.grammar_from_string(req.grammar)
            log.info("Grammar loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load grammar: {e}")
            # продолжаем без грамматики

    # Call model
    try:
        resp = llm.create_completion(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            grammar=grammar  # ← ДОБАВИТЬ ПАРАМЕТР
        )
    except Exception as e:
        log.exception("Model inference error")
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    text = safe_extract_text(resp)
    log.info("Raw model output (truncated): %s", text[:2000])

    candidate = extract_json_array(text)
    if candidate is None:
        # return raw output to aid debugging
        raise HTTPException(status_code=502, detail=f"Cannot find JSON array in model output. Raw output (truncated): {text[:2000]}")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"JSON parse error: {e}. Candidate (truncated): {candidate[:2000]}")

    try:
        normalized = validate_and_normalize(parsed, classes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}. Parsed: {parsed}")

    # Optionally sort by score_class according to provided classes order (higher relevance first)
    order_map = {cls: i for i, cls in enumerate(classes)}
    def sort_key(item):
        return order_map.get(item["score_class"], len(order_map))
    normalized_sorted = sorted(normalized, key=sort_key)

    # convert to pydantic CourseOutput list
    data_objs = [CourseOutput(**item) for item in normalized_sorted]

    return ScoreResponse(ok=True, data=data_objs, raw_model_output=text)

