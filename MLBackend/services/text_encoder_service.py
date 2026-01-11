from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union

class TextEncoderService:
    """
    Сервис для векторизации текста.
    Использует модель sentence-transformers для получения эмбеддингов (768d).
    """
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TextEncoder] Loading model: {model_name} on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[TextEncoder] Model loaded. Dimension: {self.dimension}")

    def encode_single(self, text: str) -> List[float]:
        """Векторизация одного текста"""
        if not text or not text.strip():
            return [0.0] * self.dimension
            
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()