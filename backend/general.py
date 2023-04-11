from base64 import b64decode
from celery import Celery
from celery.utils.log import get_task_logger
from hashlib import md5
from functools import reduce
from typing import List

app = Celery(__name__)
logger = get_task_logger(__name__)

SERVICE_ACCOUNT_PATH = "../service_account.json"
BUCKET = "literai"

def text_hash(text: str):
    return md5(text.encode("utf-8")).hexdigest()

@app.task
def combine_audio_convert_and_upload(parts: List[str]):
    from pydub import AudioSegment

    segments = [AudioSegment(data=b64decode(part)) for part in parts]
    joined = reduce(lambda a, b: a + b, segments)

    return {"url": "", "duration": len(joined) / 1000}

@app.task
def get_existing_audio_for_text(text: str):
    from google.cloud import storage
    
    storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_PATH)
    storage_bucket = storage_client.bucket(BUCKET)

    hashed = text_hash(text)
    blob = storage_bucket.blob(f"cyoa/audio/{hashed}.mp3")
    
    if blob.exists():
        return {"url": blob.public_url, "duration": blob.metadata.get("duration", -1)}
    else:
        return {"url": "", "duration": -1}