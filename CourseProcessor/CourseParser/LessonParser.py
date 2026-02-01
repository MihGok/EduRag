import os
import re
import json
from typing import List, Dict, Any, Iterator

from CourseProcessor.CourseParser.StepParser import StepAnalyzer


class LessonAnalyzer:
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str, knowledge_base_dir: str, course_id: str):
        self.lesson_dir = lesson_dir                  # уже абсолютный
        self.knowledge_base_dir = knowledge_base_dir  # уже абсолютный
        self.course_id = course_id

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

    def _save_lesson_content(self, all_parsed_steps: List[Dict], lesson_name: str):
        """
        Сохраняет весь урок в один файл content.txt.

        Структура на диске:
            knowledge_base/
              <search_query>/
                <course_id>/          <- изоляция по курсу
                  <lesson_name>/
                    content.txt
        """
        # Уровень course_id добавляется между knowledge_base_dir и lesson_name
        lesson_kb_dir = os.path.join(self.knowledge_base_dir, self.course_id, lesson_name)
        os.makedirs(lesson_kb_dir, exist_ok=True)
        filepath = os.path.join(lesson_kb_dir, "content.txt")

        parts = [f"LESSON: {lesson_name}", "=" * 50, f"COURSE_ID: {self.course_id}"]

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

    def parse(self) -> List[Dict[str, Any]]:
        """
        Главный метод парсинга урока.

        Читает step-файлы, извлекает текст и транскрипцию (если уже есть в файле).
        """
        parsed_steps = []

        raw_lesson_dir_name = os.path.basename(self.lesson_dir)
        clean_name = self._clean_lesson_title(raw_lesson_dir_name)

        print(f"\n[Lesson] Обработка: {clean_name} (course_id={self.course_id})")

        for step_file in self.iter_step_files():
            try:
                with open(step_file, "r", encoding="utf-8") as f:
                    raw_step = json.load(f)
            except Exception as e:
                print(f"   [Error] Не удалось прочитать {step_file}: {e}")
                continue

            parsed = StepAnalyzer.parse_step_dict(raw_step, os.path.basename(step_file))
            if not parsed:
                continue

            # Транскрипция уже должна быть в файле (записана transcription_runner).
            transcript_text = raw_step.get("transcript", "")
            if transcript_text:
                parsed["transcript"] = transcript_text
            elif parsed.get("video_url"):
                print(f"   [WARN] Step {parsed['step_id']}: видео есть, но транскрипция отсутствует.")

            parsed_steps.append(parsed)

        self._save_lesson_content(parsed_steps, clean_name)

        return parsed_steps