from base64 import b64decode
from celery import Celery, bootsteps
from celery.utils.log import get_task_logger
from functools import reduce
from typing import List
from tempfile import NamedTemporaryFile

app = Celery(__name__)
logger = get_task_logger(__name__)

SERVICE_ACCOUNT_PATH = "../service_account.json"
BUCKET = "literai"

storage_client = None
storage_bucket = None


class Bootstep(bootsteps.Step):

    def __init__(self, parent, **options):
        super().__init__(parent, **options)

        from google.cloud import storage

        global storage_client, storage_bucket

        storage_client = storage.Client.from_service_account_json(
            SERVICE_ACCOUNT_PATH)
        storage_bucket = storage_client.bucket(BUCKET)


app.steps['worker'].add(Bootstep)


@app.task
def combine_audio_convert_and_upload(hash_of_text: str, parts: List[str]):
    from pydub import AudioSegment

    segments = [AudioSegment(data=b64decode(part)) for part in parts]
    joined = reduce(lambda a, b: a + b, segments)
    duration = len(joined) / 1000

    encoded = joined.export(format="mp3")

    blob = storage_bucket.blob(f"cyoa/audio/{hash_of_text}.mp3")
    blob.metadata["duration"] = duration
    blob.upload_from_file(encoded)

    return {"url": blob.public_url, "duration": duration}


@app.task
def get_existing_audio_for_text(hash_of_text: str):
    blob = storage_bucket.blob(f"cyoa/audio/{hash_of_text}.mp3")

    if blob.exists():
        return {"url": blob.public_url, "duration": blob.metadata.get("duration", -1)}
    else:
        return {"url": "", "duration": -1}
