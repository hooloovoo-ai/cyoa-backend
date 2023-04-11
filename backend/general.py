from base64 import b64decode
from celery import Celery
from celery.utils.log import get_task_logger
from typing import List

app = Celery(__name__)
logger = get_task_logger(__name__)

@app.task
def combine_audio_convert_and_upload(parts: List[str]) -> str:
    import soundfile

    decoded = [b64decode(part) for part in parts]