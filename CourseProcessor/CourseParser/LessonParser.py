import os
import re
import json
import shutil
import math
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Iterator

from services.storage_service import StorageService
from services.LLM_Service.llm_service import GeminiService
from services.config import AppConfig, ProxyConfig
from CourseProcessor.client_api import Client
from CourseProcessor.CourseParser.StepParser import StepAnalyzer

class LessonAnalyzer:
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str, knowledge_base_dir: str):
        self.lesson_dir = lesson_dir
        self.knowledge_base_dir = knowledge_base_dir
        self.storage = StorageService()
        self.llm_service = GeminiService()
        
        # Создаем директорию для временных файлов
        self.temp_base_dir = AppConfig.TEMP_DIR
        os.makedirs(self.temp_base_dir, exist_ok=True)

    def iter_step_files(self) -> Iterator[str]:
        if not os.path.isdir(self.lesson_dir):
            return
        for fname in sorted(os.listdir(self.lesson_dir)):
            if fname.startswith(self.STEP_FILENAME_PREFIX) and fname.endswith(self.STEP_FILENAME_SUFFIX):
                yield os.path.join(self.lesson_dir, fname)

    def _clean_lesson_title(self, dir_name: str) -> str:
        match = re.search(r'^Lesson_\d+_(.+)$', dir_name, re.IGNORECASE)
        clean_name = match.group(1).strip() if match else dir_name.replace('_', ' ').strip()
        return re.sub(r'[<>:"/\\|?*]', '', clean_name).strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Считает косинусную схожесть двух текстов через TextEncoder (MPNet)"""
        if not text1 or not text2:
            return 0.0
        
        vec1 = Client.get_text_embedding(text1)
        vec2 = Client.get_text_embedding(text2)
        
        if not vec1 or not vec2:
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
            
        return float(dot_product / norm_product)

    def _group_segments_by_15s(self, segments: List[Dict]) -> List[Dict]:
        """
        Группирует мелкие сегменты Whisper в блоки по 15 секунд.
        """
        if not segments:
            return []

        duration = segments[-1]["end"]
        # Количество блоков по 15 секунд
        num_blocks = math.ceil(duration / 15.0)
        blocks = []

        for i in range(num_blocks):
            start_t = i * 15.0
            end_t = (i + 1) * 15.0
            
            # Собираем текст всех сегментов, попадающих в этот интервал
            block_text = []
            for seg in segments:
                # Если сегмент пересекается с интервалом (упрощенно: по началу сегмента)
                if start_t <= seg["start"] < end_t:
                    block_text.append(seg["text"])
            
            full_text = " ".join(block_text).strip()
            
            # Добавляем блок, только если в нем есть текст
            if full_text:
                blocks.append({
                    "start": start_t,
                    "end": end_t,
                    "center": start_t + 7.5, # Середина для скриншота
                    "text": full_text
                })
        return blocks

    def _process_video(self, video_url: str, segments: List[Dict], step_id: int, lesson_name: str) -> List[Dict]:
        """
        Обработка видео: нарезка скриншотов по 15-секундной сетке.
        """
        temp_dir = os.path.join(self.temp_base_dir, f"frames_{step_id}")
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, "video.mp4")
        final_frames_data = []

        try:
            print(f"   [Video] Скачивание...")
            if not ProxyConfig.download_file(video_url, video_path, use_proxy=True):
                return []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return []
            
            # Определяем длительность через OpenCV
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_duration = frame_count / fps if fps > 0 else 0

            # 1. Группируем сегменты
            time_blocks = self._group_segments_by_15s(segments)
            print(f"   [Video] Разбито на {len(time_blocks)} блоков по 15с")

            # 2. Итерируемся по блокам
            for block in time_blocks:
                block_center = block["center"]
                block_text = block["text"]
                
                if block_center > video_duration: continue

                # Точки захвата: центр - 3с, центр, центр + 3с
                capture_points = [block_center - 3.0, block_center, block_center + 3.0]
                
                candidates = []

                for pts in capture_points:
                    if pts < 0 or pts > video_duration: continue
                    
                    cap.set(cv2.CAP_PROP_POS_MSEC, pts * 1000)
                    ret, frame = cap.read()
                    if not ret: continue

                    cand_path = os.path.join(temp_dir, f"cand_{int(pts*100)}.jpg")
                    cv2.imwrite(cand_path, frame)
                    
                    # Получаем описание через LLaVA
                    desc = Client.get_image_description(cand_path)
                    if desc:
                        score = self._calculate_similarity(block_text, desc)
                        candidates.append({
                            "score": score,
                            "path": cand_path,
                            "desc": desc,
                            "ts": pts
                        })

                # Выбираем лучший кадр из 3-х для этого 15-секундного блока
                if candidates:
                    best = max(candidates, key=lambda x: x["score"])
                    
                    # Если кадр хоть немного релевантен тексту (порог можно менять)
                    if best["score"] > 0.35: 
                        s3_key = f"{lesson_name}/step_{step_id}/{int(best['ts'])}.jpg"
                        if self.storage.upload_frame(best["path"], s3_key):
                            print(f"      [Saved] Block {block['start']}-{block['end']}s -> {best['ts']:.1f}s (Score: {best['score']:.2f})")
                            final_frames_data.append({
                                "timestamp": best["ts"],
                                "segment_span": [block["start"], block["end"]], # Сохраняем границы отрезка
                                "frame_path": s3_key,
                                "description": best["desc"],
                                "context_text": block_text # Текст этого отрезка
                            })

            cap.release()
            return final_frames_data
            
        except Exception as e:
            print(f"   [Video Error] {e}")
            return []
        finally:
             if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

    def _save_lesson_content(self, all_parsed_steps: List[Dict], lesson_name: str):
        """Сохраняет весь урок в один файл content.txt"""
        lesson_dir = os.path.join(self.knowledge_base_dir, lesson_name)
        os.makedirs(lesson_dir, exist_ok=True)
        filepath = os.path.join(lesson_dir, "content.txt")

        parts = [f"LESSON: {lesson_name}", "="*50]

        for step in all_parsed_steps:
            parts.append(f"\nSTEP ID: {step['step_id']}")
            if step.get('update_date'):
                parts.append(f"UPDATED: {step['update_date']}")
            parts.append("-" * 20)
            
            if step.get("text"):
                parts.append(step["text"])
            
            if step.get("transcript"):
                parts.append("\n[TRANSCRIPT]:")
                parts.append(step["transcript"])

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            print(f"   [KB] Сохранен текст урока: {filepath}")
        except Exception as e:
            print(f"   [KB Error] Не удалось сохранить {filepath}: {e}")

    def _save_frames_metadata(self, all_frames: List[Dict], lesson_name: str):
        """Сохраняет метаданные картинок для индексации"""
        if not all_frames:
            return
        
        path = os.path.join(self.knowledge_base_dir, lesson_name, "frames_metadata.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(all_frames, f, ensure_ascii=False, indent=2)
            print(f"   [KB] Сохранено {len(all_frames)} кадров в метаданные")
        except Exception as e:
            print(f"   [KB Error] Ошибка сохранения метаданных: {e}")

    def parse(self) -> List[Dict[str, Any]]:
        """Главный метод парсинга урока"""
        parsed_steps = []
        all_frames_metadata = []
        
        raw_lesson_dir_name = os.path.basename(self.lesson_dir)
        clean_name = self._clean_lesson_title(raw_lesson_dir_name)
        
        print(f"\n[Lesson] Обработка: {clean_name}")

        for step_file in self.iter_step_files():
            try:
                with open(step_file, "r", encoding="utf-8") as f:
                    raw_step = json.load(f)
            except Exception as e:
                print(f"   [Error] Не удалось прочитать {step_file}: {e}")
                continue

            # Базовый парсинг
            parsed = StepAnalyzer.parse_step_dict(raw_step, os.path.basename(step_file))
            if not parsed:
                continue

            # Транскрибация
            segments = raw_step.get("_segments", []) 
            transcript_text = raw_step.get("transcript", "")

            if parsed.get("video_url") and not transcript_text:
                # Вызываем обновленный клиент, возвращающий dict
                trans_result = Client.transcribe(parsed["video_url"], parsed["step_id"])
                transcript_text = trans_result.get("text", "")
                segments = trans_result.get("segments", [])
                
                if transcript_text:
                    parsed["transcript"] = transcript_text
                    # Кешируем и текст и сегменты
                    raw_step["transcript"] = transcript_text
                    raw_step["_generated_transcript"] = transcript_text
                    raw_step["_segments"] = segments 
                    try:
                        with open(step_file, "w", encoding="utf-8") as f:
                            json.dump(raw_step, f, ensure_ascii=False, indent=2)
                    except:
                        pass

            # Обработка видео (теперь передаем segments)
            if False and parsed.get("video_url") and segments:
                frames = self._process_video(
                    parsed["video_url"], 
                    segments,
                    parsed["step_id"], 
                    clean_name
                )
                if frames:
                    for fr in frames:
                        fr['step_id'] = parsed['step_id']
                        fr['lesson_name'] = clean_name
                    all_frames_metadata.extend(frames)

            parsed_steps.append(parsed)

        # Сохранение результатов
        self._save_lesson_content(parsed_steps, clean_name)
        self._save_frames_metadata(all_frames_metadata, clean_name)
        
        return parsed_steps
