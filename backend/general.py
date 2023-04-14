from base64 import b64decode
from io import BytesIO
from celery import Celery, bootsteps
from celery.utils.log import get_task_logger
from functools import reduce
from typing import List


app = Celery(__name__)
logger = get_task_logger(__name__)

SERVICE_ACCOUNT_PATH = "../service_account.json"
BUCKET = "literai"
SUMMARY_PROMPT = "Below is a PASSAGE from a novel, and the TEXT IMMEDIATELY BEFORE PASSAGE. In the active voice, write an interesting SHORT SUMMARY OF ONLY PASSAGE, using TEXT IMMEDIATELY BEFORE PASSAGE to provide context, but not directly summarizing TEXT IMMEDIATELY BEFORE PASSAGE.\n\nTEXT IMMEDIATELY BEFORE PASSAGE:\n{text_before_excerpt}\n\nPASSAGE:\n{excerpt}\n\nSHORT SUMMARY OF ONLY PASSAGE:\n"
DESCRIBE_PROMPT = "Below is a PASSAGE from a novel. Write a SHORT, CONDENSED FEATURES OF AN ILLUSTRATION of the PASSAGE that would be suitable as a prompt to an image generation model such as Stable Diffusion or Midjourney.\n\nPASSAGE:\n{excerpt}\n\nSHORT, CONDENSED FEATURES OF AN ILLUSTRATION:\n"
IMAGE_PROMPT = "(extremely detailed CG unity 8k wallpaper), {description}, professional majestic oil painting by Ed Blinkey, Atey Ghailan, by Jeremy Mann, Greg Manchess, Antonio Moro, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, photorealistic painting art by midjourney and greg rutkowski"
IMAGE_NEGATIVE_PROMPT = "canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"

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
def combine_audio_convert_and_upload(parts: List[str], hash_of_text: str):
    from pydub import AudioSegment

    segments = [AudioSegment(data=b64decode(part)) for part in parts]
    joined = reduce(lambda a, b: a + b, segments)
    duration = len(joined) / 1000

    encoded = joined.export(format="mp3")

    blob = storage_bucket.blob(f"cyoa/audio/{hash_of_text}.mp3")
    blob.metadata = {"duration": duration}
    blob.upload_from_file(encoded, content_type="audio/mpeg")

    return {"url": blob.public_url, "duration": duration}


@app.task
def get_existing_audio_for_text(hash_of_text: str):
    path = f"cyoa/audio/{hash_of_text}.mp3"

    if storage_bucket.blob(path).exists():
        blob = storage_bucket.get_blob(path)
        return {"url": blob.public_url, "duration": float(blob.metadata.get("duration", "-1"))}
    else:
        return {"url": "", "duration": -1}


@app.task
def get_existing_images_for_text(hash_of_text: str):
    results = []

    i = 0
    while True:
        path = f"cyoa/images/{hash_of_text}-{i}.png"

        if storage_bucket.blob(path).exists():
            results.append(storage_bucket.get_blob(path).public_url)
            i += 1
        else:
            break

    return results


@app.task
def upload_image(encoded: str, hash_of_text: str, index: int):
    data = BytesIO(b64decode(encoded))

    blob = storage_bucket.blob(f"cyoa/images/{hash_of_text}-{index}.png")
    blob.upload_from_file(data, content_type="image/png")

    return blob.public_url


@app.task
def summary_prompt(text_before_excerpt: str, excerpt: str):
    return SUMMARY_PROMPT.format(text_before_excerpt=text_before_excerpt, excerpt=excerpt)


@app.task
def describe_prompt(excerpt: str):
    return DESCRIBE_PROMPT.format(excerpt=excerpt.replace("\n", " "))


@app.task
def image_prompt(description: str):
    description = description.lower().replace(
        "\n", ", ").replace(". ", ", ").replace(" - ", "").strip()
    return IMAGE_PROMPT.format(description=description)
