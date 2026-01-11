from faster_whisper import WhisperModel
import os

class WhisperService:
    def __init__(self):
        self.model_size = os.environ.get("WHISPER_MODEL", "medium")
        self.device = os.environ.get("WHISPER_DEVICE", "cuda")
        self.compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
        
        # Загрузка весов происходит при инициализации класса
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, video_path: str, **kwargs):
        segments, _ = self.model.transcribe(video_path, **kwargs)
        return list(segments)
