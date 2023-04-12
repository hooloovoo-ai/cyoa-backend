import logging
import time
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
from celery import group
from backend.generate import generate
from backend.alpaca import alpaca
from backend.tts import tts
from backend.general import combine_audio_convert_and_upload, get_existing_audio_for_text
from utils import split_and_recombine_text, text_hash

# SUMMARY_PROMPT = "TEXT BEFORE EXCERPT:\n{text_before_excerpt}\n\nEXCERPT:\n{excerpt}\n\nSHORT SUMMARY OF ONLY THE EXCERPT:\n"
# SUMMARY_PROMPT = "Write a short summary of the following text.\n\n{text_before_excerpt} {excerpt}\n\nSHORT SUMMARY:\n"
SUMMARY_PROMPT = "Below is an EXCERPT from a book, and the TEXT IMMEDIATELY BEFORE EXCERPT. In the active voice, write an interesting SHORT SUMMARY OF ONLY EXCERPT, using TEXT IMMEDIATELY BEFORE EXCERPT to provide context, but not directly summarizing TEXT IMMEDIATELY BEFORE EXCERPT. Do not use the word \"excerpt\".\n\nTEXT IMMEDIATELY BEFORE EXCERPT:\n{text_before_excerpt}\n\nEXCERPT:\n{excerpt}\n\nSHORT SUMMARY OF ONLY EXCERPT:\n"
DESCRIBE_PROMPT = "Below is an EXCERPT from a book. Write a DESCRIPTION OF ILLUSTRATION OF EXCERPT that would be suitable as a prompt to an image generation model such as Stable Diffusion or Midjourney.\n\nEXCERPT:\n{excerpt}\n\nDESCRIPTION OF ILLUSTRATION OF EXCERPT:\n"
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
        prompts = [SUMMARY_PROMPT.format(
            text_before_excerpt=text_before_excerpt, excerpt=result.replace("\n", "")) for result in results]

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

    prompts = [DESCRIBE_PROMPT.format(excerpt=text.lower().replace(
        "\n", "").replace(". ", ", "))] * NUM_IMAGES
    descriptions = group([alpaca.s(prompt, temperature=0.7)
                          for prompt in prompts]).apply_async().get()

    app.logger.info(f"descriptions: {descriptions}")

    audio = get_existing_audio_for_text.delay(hash_of_text).get()
    if audio["duration"] <= 0:
        desired_length = int(len(text) / OPTIMAL_TTS_SPLIT) + 1
        max_length = int(desired_length * MAX_TTS_SPLIT_FLACTOR) + 1
        texts = split_and_recombine_text(
            text, desired_length=desired_length, max_length=max_length)
        app.logger.info(
            f"Rendering audio in {len(texts)} parts using desired_length={desired_length} and max_length={max_length}")
        start = time.time()
        parts = group([tts.s(part) for part in texts]).apply_async().get()
        app.logger.info(f"Rendered audio in {time.time() - start} seconds")
        audio = combine_audio_convert_and_upload.delay(
            hash_of_text, parts).get()

    return jsonify({
        "audio_url": audio["url"],
        "audio_duration": audio["duration"]
    })
