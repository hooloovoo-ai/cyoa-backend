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
SUMMARY_PROMPT = "Below is an EXCERPT from a book, and the TEXT IMMEDIATELY BEFORE EXCERPT. In the active voice, write an interesting SHORT SUMMARY OF ONLY EXCERPT, using TEXT IMMEDIATELY BEFORE EXCERPT to provide context, but not directly summarizing TEXT IMMEDIATELY BEFORE EXCERPT. Do not use the word \"excerpt\".\n\nTEXT IMMEDIATELY BEFORE EXCERPT:\n{text_before_excerpt}\n\nEXCERPT:\n{excerpt}\n\nSHORT SUMMARY OF ONLY EXCERPT:\n"
#DESCRIBE_PROMPT = "Below is an EXCERPT from a book. Write a DESCRIPTION OF ILLUSTRATION OF EXCERPT that would be suitable as a prompt to an image generation model such as Stable Diffusion or Midjourney.\n\nEXCERPT:\n{excerpt}\n\nDESCRIPTION OF ILLUSTRATION OF EXCERPT:\n"
DESCRIBE_PROMPT = r"""Excerpt: Aletha stood alone in ruins of her former city. The long years hadn't done it any favors. She pulled her black leather jacket closer to her, as if it's thin fabric could save her from the horrors done here. Her father told never never to come back, but of course she didn't listen. Formin had been a bustling industrial city -- once. Now, it was the broken bones of its former glory. Aletha's rubin-like eyes were dark. The thick eyebrows above them made her seem older than she really was.
Description: Young beautiful Girl hiding in shattered industrial atmosphere, post apocalyptic ruined city, black tattered shirt, black colors, black leather jacket, rubin-like dark red eyes, almond-shaped eyes

Excerpt: "Pull aft!" Came Ol' Folder's voice from across the deck. Water crashed above the bow of the old boat; the storm was threatening to take its final revenge. For a moment I thought that the storm would give us relief, but instead it redoubled its ferocity. As I hurried across the deck, I knew I had little time left. Although I'd been a bowler -- the lowest rank on a ship -- for nearly a month, Folder would tolerate no delay at a time like this. Those that couldn't pull weight in an emergency were left at the next port.
Description: broken old boat in big storm

Excerpt: Not him again. I knew Petrice from his days as Lord Chancellor. His ruthlessness, cunning, and malice were known to all. It was said that although he lived alone, he would send every night for another victim to be brought to his chambers. What happened to them no one ever knew, but they certainly weren't heard from again. Even his face gave the aura of evil -- his straight, masculine features projected perfectly the evil that lay within, not to mention his athletic build. "Be careful around that one," I whispered to the man next to me.
Description: photo of an evil hermit, male, villain, anti hero, evil face, masculine face, medium hair, Maroon hair, wicked, cruel, sinister, malicious, ruthless, masculine, athletic, dark bloody clothing,

Excerpt: Alpho sat at the campfire, listening to Granny spin her wild tales. He'd heard most of them a thousand times, but he still liked to lend at least one ear to them when he could, if only so that he would know what Brian was talking about when went on about fantastic advenctures. "Daddy, are you paying attention?" Brian anxiously inquired. "Of course," Alpho answered. Granny began, "Once, long ago, high in the mountains, there was a secret treasure. It was kept in the dungeons of a castle at could only be reached by three days journey."
Description: A forbidden castle high up in the mountains

Excerpt: {excerpt}
Description: 
"""
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
def combine_audio_convert_and_upload(hash_of_text: str, parts: List[str]):
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
def upload_image(hash_of_text: str, index: int, encoded: str):
    data = BytesIO(b64decode(encoded))

    blob = storage_bucket.blob(f"cyoa/images/{hash_of_text}-{index}.png")
    blob.upload_from_file(data, content_type="image/png")

    return blob.public_url


@app.task
def summary_prompt(text_before_excerpt: str, excerpt: str):
    return SUMMARY_PROMPT.format(text_before_excerpt=text_before_excerpt, excerpt=excerpt)


@app.task
def describe_prompt(excerpt: str):
    return DESCRIBE_PROMPT.format(excerpt=excerpt)


@app.task
def image_prompt(description: str):
    return IMAGE_PROMPT.format(description=description.lower().replace("\n", " ").replace(". ", ", ").strip())
