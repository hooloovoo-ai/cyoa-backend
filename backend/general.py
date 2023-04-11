from base64 import b64decode
from celery import Celery
from celery.utils.log import get_task_logger
from functools import reduce
from typing import List

app = Celery(__name__)
logger = get_task_logger(__name__)

@app.task
def combine_audio_convert_and_upload(parts: List[str]) -> str:
    from pydub import AudioSegment

    segments = [AudioSegment(data=b64decode(part)) for part in parts]
    joined = reduce(lambda a, b: a + b, segments)

    return {"url": "", "duration": len(joined)}