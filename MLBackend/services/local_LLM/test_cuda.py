import sys
import llama_cpp
from llama_cpp import Llama

MODEL_PATH = "saiga_nemo_12b.Q4_K_M.gguf"

def get_actual_device(llm):
    """Определяет, на каком устройстве реально запущены слои модели"""
    # Проверяем, поддерживает ли сама библиотека GPU
    can_gpu = llama_cpp.llama_supports_gpu_offload()
    
    # Пытаемся заглянуть в статистику использования памяти слоями
    # Если слои на GPU, то в модели будет задействован CUDA или Metal бэкенд
    # Самый надежный способ — посмотреть на вывод в консоль (verbose=True),
    # но программно мы можем проверить статус сборки:
    return can_gpu

try:
    print(f"--- Проверка конфигурации ---")
    is_compiled_with_cuda = llama_cpp.llama_supports_gpu_offload()
    print(f"Библиотека собрана с поддержкой GPU: {is_compiled_with_cuda}")

    # Загружаем модель
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, 
        n_ctx=8192,
        verbose=True  # В логах ищите: "llama_kv_cache_unified: layer 0: dev = CUDA0"
    )

    print(f"\n--- Результат запуска ---")
    if is_compiled_with_cuda:
        # В llama-cpp-python нет прямого свойства .device, поэтому
        # мы ориентируемся на внутренний флаг поддержки и логи загрузки.
        print("Библиотека ПЫТАЕТСЯ использовать GPU (CUDA).")
        print("Проверьте логи выше: если там везде 'dev = CPU', значит видеопамяти недостаточно или драйвер не подхватился.")
    else:
        print("Библиотека работает ТОЛЬКО на CPU. GPU-ускорение недоступно.")

    # Простой запрос
    messages = [{"role": "user", "content": "Привет! На чем ты работаешь: на GPU или CPU?"}]
    
    response = llm.create_chat_completion(messages=messages, max_tokens=100)
    print("\nОТВЕТ МОДЕЛИ:")
    print(response["choices"][0]["message"]["content"])

except Exception as e:
    print(f"\nОшибка при выполнении: {e}")