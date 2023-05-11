import os
from celery import Celery, bootsteps
from celery.utils.log import get_task_logger

app = Celery(__name__)
logger = get_task_logger(__name__)

# MODEL = "chavinlo/gpt4-x-alpaca"
# QUANTIZE = True

MODEL = "mosaicml/mpt-7b-instruct"
QUANTIZE = False

model = None
tokenizer = None
stopping_criteria = None

class Bootstep(bootsteps.Step):
    def __init__(self, parent, **options):
        super().__init__(parent, **options)

        import torch
        from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList

        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords_ids:list):
                self.keywords = keywords_ids

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if input_ids[0][-1] in self.keywords:
                    return True
                return False

        global model, tokenizer, stopping_criteria
        if MODEL == "chavinlo/gpt4-x-alpaca":
            model = LlamaForCausalLM.from_pretrained(
                "chavinlo/gpt4-x-alpaca",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_8bit=QUANTIZE,
            )
        else:
            if "mosaicml" in MODEL:
                config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
                # config.attn_config['attn_impl'] = 'triton'
                device_map = {'': int(os.getenv('LOCAL_RANK', '0'))}
            else:
                device_map = "auto"
                config = None
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, device_map=device_map, 
                load_in_8bit=QUANTIZE, config=config, trust_remote_code=True)
        logger.info(f"device map: {model.hf_device_map}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        if MODEL == "chavinlo/gpt4-x-alpaca":
            tokenizer.bos_token_id = 2
            tokenizer.model_max_length = 2048

        to_stop_on = [tokenizer.encode(w, add_special_tokens=False)[0] for w in ["\n", "#"]]
        to_stop_on.append(tokenizer.bos_token_id)
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(to_stop_on)])

        


app.steps["worker"].add(Bootstep)


@app.task
def alpaca(
    prompt: str,
    max_new_tokens: int = 256,
    new_tokens_only: bool = True,
    temperature=None,
) -> str:
    import torch

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature is not None,
            stopping_criteria=stopping_criteria
        )
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if new_tokens_only:
            output = output[len(prompt) :]
        return output
