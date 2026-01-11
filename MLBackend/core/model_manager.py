import gc
import torch

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.current_model = None
            cls._instance.current_model_name = None
        return cls._instance

    def _unload_current(self):
        """Полная очистка VRAM от текущей модели"""
        if self.current_model is not None:
            print(f"Unloading model: {self.current_model_name}...")
            # Удаляем ссылку на объект
            del self.current_model
            self.current_model = None
            self.current_model_name = None
            
            # Форсируем сборку мусора Python
            gc.collect()
            # Очищаем кеш CUDA (самое важное для GPU)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("VRAM cleared.")

    def get_model(self, model_name: str, loader_func):
        """
        model_name: уникальное имя ('whisper', 'llava', 'clip')
        loader_func: функция, которая возвращает загруженный экземпляр модели
        """
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

# Глобальный инстанс менеджера
model_manager = ModelManager()
