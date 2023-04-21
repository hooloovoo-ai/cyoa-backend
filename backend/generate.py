import re
from hashlib import md5
from celery import Celery, bootsteps
from celery.utils.log import get_task_logger

app = Celery(__name__)
logger = get_task_logger(__name__)


MODEL = "emozilla/llama-long-13b-scifi-fantasy-673-8192h-epoch4"
HISTORY = 5600
LEARNING_RATE = 3e-4
DEVICE = "cuda"
FINETUNING = False
QUANTIZE = False
LONG_MODEL_TYPE = True

models = {}
original_model = None
tokenizer = None


class Bootstep(bootsteps.Step):

    def __init__(self, parent, **options):
        super().__init__(parent, **options)

        import torch
        from transformers import AutoTokenizer

        global accelerator, original_model, tokenizer

        if FINETUNING:
            from accelerate import Accelerator
            accelerator = Accelerator()
        else:
            accelerator = None

        if LONG_MODEL_TYPE:
            from .modeling_long import LlamaLongForCausalLM, GPTNeoXLongForCausalLM
            original_model = LlamaLongForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, device_map="auto", load_in_8bit=QUANTIZE) \
                if "llama" in MODEL else GPTNeoXLongForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.float16, device_map="auto", load_in_8bit=QUANTIZE)
        else:
            from transformers import AutoModelForCausalLM
            original_model = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, device_map="auto", load_in_8bit=QUANTIZE)
        logger.info(f"device map: {original_model.hf_device_map}")

        original_model.config.use_cache = True

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenizer.pad_token_id = 0
        if LONG_MODEL_TYPE:
            tokenizer.model_max_length = original_model.config.max_positions


app.steps['worker'].add(Bootstep)


def get_model(id: str):
    if not FINETUNING:
        return original_model, None
    return models[id]


def get_id(id: str):
    return md5(id.encode("utf-8")).hexdigest()[0:8].upper()


@app.task
def generate(id: str = "",
             chunk: int = 0,
             totalChunks: int = 1,
             text: str = "",
             temperature: float = 0.8,
             repetitionPenalty=1.1,
             maxNewTokens: int = 128,
             generations: int = 1):

    import torch
    import ftfy

    id = get_id(id)

    header = tokenizer(f"ID: {id} Chunk: {chunk} of {totalChunks}\n\n\n",
                       return_tensors='pt').input_ids.to(DEVICE)
    book = tokenizer(ftfy.ftfy(text),
                     return_tensors='pt').input_ids.to(DEVICE)

    context_len = max(HISTORY - maxNewTokens, 0)
    logger.info(
        f"id: {id} supplied tokens: {book.shape[1]} requested tokens: {maxNewTokens} context: {context_len} generations: {generations}")

    context = book[:, -context_len:]
    input_ids = torch.cat((header, context), 1)
    attention_mask = torch.ones(1, input_ids.shape[1])
    if generations > 1:
        input_ids = torch.cat([input_ids] * generations, dim=0)
        attention_mask = torch.cat([attention_mask] * generations, dim=0)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    model, optimizer = get_model(id)
    output_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=maxNewTokens,
                                   do_sample=True, temperature=temperature, repetition_penalty=repetitionPenalty)

    new_tokens = output_tokens[:, input_ids.shape[1]:]
    new_text = [tokenizer.decode(tokens, skip_special_tokens=True)
                for tokens in new_tokens]

    for i in range(0, len(new_text)):
        new_text[i] = re.sub(r"(?<!\n)\n{1}(?!\n)", "", new_text[i])
        last_period = new_text[i].rfind(".")
        last_question = new_text[i].rfind("?")
        last_ex = new_text[i].rfind("!")
        new_text[i] = new_text[i][0:max(last_period, last_question, last_ex)+1]

    return new_text
