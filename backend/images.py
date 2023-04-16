import os
from celery import Celery, bootsteps
from celery.utils.log import get_task_logger


app = Celery(__name__)
logger = get_task_logger(__name__)

port = int(os.environ["PORT"])

@app.task
def images(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    sampler_name: str = "DPM++ 2M Karras",
    steps: int = 30,
    cfg_scale: float = 10.0,
):
    import requests

    response = requests.post(f"http://127.0.0.1:{port}/sdapi/v1/txt2img", json={
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "sampler_name": sampler_name,
        "steps": steps,
        "cfg_scale": cfg_scale
    }).json()

    return response["images"][0]