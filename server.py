import logging
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
from celery import group
from backend.generate import generate
from backend.alpaca import alpaca

SUMMARY_PROMPT = "TEXT BEFORE EXCERPT:\n{text_before_excerpt}\n\nEXCERPT:\n{excerpt}\n\nSHORT SUMMARY OF ONLY THE EXCERPT:\n"

app = Flask("server")
CORS(app, origins=["http://localhost:3000", "https://cyoa.hooloovoo.ai"])

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


@app.route("/generate", methods=["POST"])
async def generate_api():
    if request.method != "POST":
        abort(404)

    args = request.json
    summarize = args.pop("summarize", True)
    generations = args.get("generations", 1)
    args["generations"] = 1

    app.logger.info("Starting generations...")
    results = group(generate.signature((), args)
                    for _ in range(0, generations)).apply_async().get()
    results = [result[0] for result in results]

    if summarize:
        text_before_excerpt = args.get("text", "")[-1000:]
        prompts = [SUMMARY_PROMPT.format(
            text_before_excerpt=text_before_excerpt, excerpt=result) for result in results]

        app.logger.info("Starting summaries...")
        summaries = group([alpaca.s(prompt)
                          for prompt in prompts]).apply_async().get()
    else:
        summaries = []

    app.logger.info("Finished")

    return jsonify({"results": results, "summaries": summaries})
