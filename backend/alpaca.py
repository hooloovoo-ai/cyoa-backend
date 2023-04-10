from celery import Celery, bootsteps
from celery.utils.log import get_task_logger

app = Celery(__name__)
logger = get_task_logger(__name__)

model = None
tokenizer = None


class Bootstep(bootsteps.Step):
    def __init__(self, parent, **options):
        super().__init__(parent, **options)

        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizer

        global model, tokenizer
        model = LlamaForCausalLM.from_pretrained(
            "chavinlo/alpaca-native", device_map="auto", torch_dtype=torch.bfloat16)
        logger.info(f"device map: {model.hf_device_map}")
        tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")
        tokenizer.model_max_length = 2048


app.steps['worker'].add(Bootstep)


@app.task
def alpaca(prompt: str, max_new_tokens: int = 256, new_tokens_only: bool = True) -> str:
    import torch
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if new_tokens_only:
            output = output[len(prompt):]
        return output
