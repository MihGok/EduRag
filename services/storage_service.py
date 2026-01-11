import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import os

class StorageService:
    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "password123")
        self.bucket = "course-frames"
        
        # ИСПРАВЛЕНИЕ: Отключаем прокси для локального MinIO
        config = Config(
            proxies={},  # Пустой словарь = без прокси
            signature_version='s3v4'
        )
        
        self.s3 = boto3.client('s3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=config,
            verify=False  # Если используете самоподписанный сертификат
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Создает бакет, если его нет"""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            print(f"[Storage] Бакет '{self.bucket}' существует")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                try:
                    self.s3.create_bucket(Bucket=self.bucket)
                    print(f"[Storage] Создан бакет: {self.bucket}")
                except Exception as create_error:
                    print(f"[Storage] Ошибка создания бакета: {create_error}")
                    raise
            else:
                print(f"[Storage] Ошибка проверки бакета: {e}")
                raise

    def upload_frame(self, frame_path: str, object_name: str) -> bool:
        """
        Загружает файл с диска в S3.
        Args:
            frame_path: Локальный путь к файлу
            object_name: Имя ключа в S3 (например, 'Lesson1/img.jpg')
        Returns:
            True если успешно, иначе False
        """
        if not os.path.exists(frame_path):
            print(f"[Storage] Файл не найден: {frame_path}")
            return False
            
        try:
            self.s3.upload_file(frame_path, self.bucket, object_name)
            print(f"[Storage] Загружено: {object_name}")
            return True
        except ClientError as e:
            print(f"[Storage] AWS ошибка при загрузке {object_name}: {e}")
            return False
        except Exception as e:
            print(f"[Storage] Неожиданная ошибка при загрузке {object_name}: {e}")
            return False

    def get_presigned_url(self, object_name: str, expiration: int = 3600) -> str:
        """Генерирует временную ссылку на картинку"""
        if not object_name:
            return ""
        try:
            url = self.s3.generate_presigned_url('get_object',
                Params={'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"[Storage] Ошибка генерации ссылки для {object_name}: {e}")
            return ""
        except Exception as e:
            print(f"[Storage] Неожиданная ошибка при генерации ссылки: {e}")
            return ""

    def object_exists(self, object_name: str) -> bool:
        """Проверяет существование объекта в S3"""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=object_name)
            return True
        except ClientError:
            return False
