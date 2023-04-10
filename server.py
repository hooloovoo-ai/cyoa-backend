from flask import Flask, request, abort, jsonify
from flask_cors import CORS
from celery import group
from backend.generate import generate
from backend.alpaca import alpaca

SUMMARY_PROMPT = "TEXT BEFORE EXCERPT:\n{text_before_excerpt}\n\nEXCERPT:\n{excerpt}\n\nSHORT SUMMARY OF ONLY THE EXCERPT:\n"

app = Flask("server")
CORS(app, origins=["http://localhost:3000", "https://cyoa.hooloovoo.ai"])


@app.route("/generate", methods=["POST"])
async def generate_api():
    if request.method != "POST":
        abort(404)

    args = request.json
    summarize = args.pop("summarize", True)
    generations = args.get("generations", 1)
    args["generations"] = 1

    results = group(generate.signature((), args)
                    for _ in range(0, generations)).apply_async().get()

    if summarize:
        text_before_excerpt = args.get("text", "")[-1000:]

        summaries = group(alpaca.s(SUMMARY_PROMPT.format(
            text_before_excerpt=text_before_excerpt, text=result)) for result in results).apply_async().get()
    else:
        summaries = []

    return jsonify({"results": results, "summaries": summaries})
