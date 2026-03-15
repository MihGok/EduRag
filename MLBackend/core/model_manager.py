import gc
import torch
import threading
from typing import Dict, Any, Callable, Optional, List

class ModelManager:
    """
    Менеджер моделей с поддержкой:
    - Автоматической выгрузки моделей из VRAM
    - Параллельной работы нескольких экземпляров LLM
    - Переключения между фазами обработки (LLM -> Whisper -> LLM)
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.current_model = None
            cls._instance.current_model_name = None
            cls._instance.llm_instances: List[Any] = []  # Для параллельных LLM
            cls._instance.whisper_instances: List[Any] = []  # Для параллельных Whisper
            cls._instance.phase = None  # 'llm' или 'whisper'
        return cls._instance

    def _unload_current(self):
        """Полная очистка VRAM от текущей модели"""
        if self.current_model is not None:
            print(f"Unloading model: {self.current_model_name}...")
            del self.current_model
            self.current_model = None
            self.current_model_name = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("VRAM cleared.")

    def _unload_all_llm(self):
        """Выгрузка всех экземпляров LLM"""
        if self.llm_instances:
            print(f"Unloading {len(self.llm_instances)} LLM instances...")
            for llm in self.llm_instances:
                del llm
            self.llm_instances = []
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("All LLM instances unloaded.")

    def _unload_all_whisper(self):
        """Выгрузка всех экземпляров Whisper"""
        if self.whisper_instances:
            print(f"Unloading {len(self.whisper_instances)} Whisper instances...")
            for whisper in self.whisper_instances:
                del whisper
            self.whisper_instances = []
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("All Whisper instances unloaded.")

    def get_model(self, model_name: str, loader_func: Callable):
        """
        Стандартный метод для одиночных моделей (обратная совместимость).
        model_name: уникальное имя ('whisper', 'llava', 'clip')
        loader_func: функция, которая возвращает загруженный экземпляр модели
        """
        with self._lock:
            if self.current_model_name == model_name:
                return self.current_model

            # Если загружена другая модель - выгружаем её
            if self.current_model is not None:
                self._unload_current()

            print(f"Loading model: {model_name}...")
            self.current_model = loader_func()
            self.current_model_name = model_name
            print(f"Model {model_name} loaded successfully.")
            
            return self.current_model

    def start_llm_phase(self, num_instances: int = 2) -> List[Any]:
        """
        Переход в фазу LLM: выгружает Whisper, загружает N экземпляров LLM.
        Возвращает список готовых к работе LLM инстансов.
        """
        with self._lock:
            if self.phase == 'llm' and len(self.llm_instances) == num_instances:
                print(f"[Phase] Already in LLM phase with {num_instances} instances.")
                return self.llm_instances

            print(f"[Phase] Switching to LLM phase ({num_instances} instances)...")
            
            # Выгружаем всё старое
            self._unload_all_whisper()
            self._unload_all_llm()
            self._unload_current()

            # Загружаем новые LLM
            from MLBackend.services.local_LLM.app.main import get_llm_instance
            import os
            
            model_path = os.getenv("LLM_MODEL_PATH", "MLBackend/services/local_LLM/models/smollm3-3b-q4_k_m.gguf")
            n_ctx = int(os.getenv("DEFAULT_N_CTX", "2048"))
            n_gpu_layers = -1  # Все слои на GPU
            n_batch = 512
            
            for i in range(num_instances):
                print(f"  Loading LLM instance {i+1}/{num_instances}...")
                llm = get_llm_instance(model_path, n_ctx, n_gpu_layers, n_batch)
                self.llm_instances.append(llm)
            
            self.phase = 'llm'
            print(f"[Phase] LLM phase ready with {len(self.llm_instances)} instances.")
            return self.llm_instances

    def start_whisper_phase(self, num_instances: int = 2) -> List[Any]:
        """
        Переход в фазу Whisper: выгружает LLM, загружает N экземпляров Whisper.
        """
        with self._lock:
            if self.phase == 'whisper' and len(self.whisper_instances) == num_instances:
                print(f"[Phase] Already in Whisper phase with {num_instances} instances.")
                return self.whisper_instances

            print(f"[Phase] Switching to Whisper phase ({num_instances} instances)...")
            
            # Выгружаем всё старое
            self._unload_all_llm()
            self._unload_all_whisper()
            self._unload_current()

            # Загружаем новые Whisper
            from services.whisper_service import WhisperService
            
            for i in range(num_instances):
                print(f"  Loading Whisper instance {i+1}/{num_instances}...")
                whisper = WhisperService()
                self.whisper_instances.append(whisper)
            
            self.phase = 'whisper'
            print(f"[Phase] Whisper phase ready with {len(self.whisper_instances)} instances.")
            return self.whisper_instances

    def get_llm_instance(self, index: int = 0) -> Optional[Any]:
        """Получить конкретный экземпляр LLM по индексу"""
        if not self.llm_instances:
            raise RuntimeError("No LLM instances loaded. Call start_llm_phase() first.")
        if index >= len(self.llm_instances):
            index = index % len(self.llm_instances)
        return self.llm_instances[index]

    def get_whisper_instance(self, index: int = 0) -> Optional[Any]:
        """Получить конкретный экземпляр Whisper по индексу"""
        if not self.whisper_instances:
            raise RuntimeError("No Whisper instances loaded. Call start_whisper_phase() first.")
        if index >= len(self.whisper_instances):
            index = index % len(self.whisper_instances)
        return self.whisper_instances[index]


model_manager = ModelManager()