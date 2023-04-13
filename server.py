import logging
import time
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
from celery import group, chain
from backend.generate import generate
from backend.alpaca import alpaca
from backend.tts import tts
from backend.general import combine_audio_convert_and_upload, get_existing_audio_for_text, summary_prompt, describe_prompt, get_existing_images_for_text, upload_image, image_prompt, IMAGE_NEGATIVE_PROMPT
from backend.images import images
from utils import split_and_recombine_text, text_hash

OPTIMAL_TTS_SPLIT = 8
MAX_TTS_SPLIT_FLACTOR = 1.25
NUM_IMAGES = 3

app = Flask("server")
CORS(app, origins=["http://localhost",
     "http://localhost:3000", "https://cyoa.hooloovoo.ai"])

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


@app.route("/generate", methods=["POST"])
async def generate_api():
    args = request.json
    summarize = args.pop("summarize", True)
    generations = args.get("generations", 1)
    args["generations"] = 1

    results = group(generate.signature((), args)
                    for _ in range(0, generations)).apply_async().get()
    results = [result[0] for result in results]

    if summarize:
        text_before_excerpt = args.get("text", "")[-500:].replace("\n", "")
        prompts = [summary_prompt(text_before_excerpt, result)
                   for result in results]

        summaries = group([alpaca.s(prompt)
                          for prompt in prompts]).apply_async().get()
    else:
        summaries = []

    return jsonify({"results": results, "summaries": summaries})


@app.route("/imagine", methods=["POST"])
async def imagine_api():
    args = request.json

    text = args.get("text", "")
    if len(text) == 0:
        abort(500)

    hash_of_text = text_hash(text)

    existing = group(get_existing_audio_for_text.s(
        hash_of_text), get_existing_images_for_text.s(hash_of_text)).apply_async().get()

    audio = existing[0]
    pngs = existing[1]

    if len(pngs) == 0:
        prompt = describe_prompt(text)

        pending_images = group([chain(alpaca.s(prompt, temperature=0.8, max_new_tokens=128), image_prompt.s(), images.s(
            negative_prompt=IMAGE_NEGATIVE_PROMPT), upload_image.s(hash_of_text, i)) for i in range(0, NUM_IMAGES)]).apply_async()
    else:
        pending_images = None

    if audio["duration"] <= 0:
        # desired_length = max(100, int(len(text) / OPTIMAL_TTS_SPLIT) + 1)
        # max_length = max(125, int(desired_length * MAX_TTS_SPLIT_FLACTOR) + 1)
        desired_length = 120
        max_length = 150
        texts = split_and_recombine_text(
            text, desired_length=desired_length, max_length=max_length)
        app.logger.info(
            f"Rendering audio in {len(texts)} parts using desired_length={desired_length} and max_length={max_length}")
        start = time.time()
        parts = group([tts.s(part) for part in texts]).apply_async().get()
        app.logger.info(f"Rendered audio in {time.time() - start} seconds")
        audio = combine_audio_convert_and_upload.delay(
            hash_of_text, parts).get()

    if pending_images is not None:
        pngs = pending_images.get()

    return jsonify({
        "audio": audio,
        "images": pngs
    })
