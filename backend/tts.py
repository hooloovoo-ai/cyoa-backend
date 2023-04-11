import os
from base64 import b64encode
from io import BytesIO
from celery import Celery, bootsteps
from celery.utils.log import get_task_logger

app = Celery(__name__)
logger = get_task_logger(__name__)

VOICES = ["notw"]

tts = None
latents = None


class Bootstep(bootsteps.Step):

    def __init__(self, parent, **options):
        super().__init__(parent, **options)

        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_voices

        global tts, latents

        tts = TextToSpeech(enable_redaction=False)
        voice_samples, _ = load_voices(VOICES, extra_voice_dirs=[
                                       os.path.join(os.getcwd(), "voices")])
        latents = tts.get_conditioning_latents(voice_samples)


app.steps['worker'].add(Bootstep)


@app.task
def tts(text: str, num_autoregressive_samples=16, diffusion_iterations=30) -> str:
    import soundfile
    import numpy

    gen = tts.tts(text, conditioning_latents=latents, num_autoregressive_samples=num_autoregressive_samples,
                  diffusion_iterations=diffusion_iterations, verbose=False)
    audio = gen.squeeze(0).cpu()

    output = BytesIO()
    soundfile.write(output, numpy.ravel(audio.numpy()),
                    samplerate=24000, format="WAV")

    return b64encode(output.getvalue()).decode("ascii")
