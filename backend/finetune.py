import re
from hashlib import md5
from celery import bootsteps
from celery.utils.log import get_task_logger
from transformers import AutoTokenizer
from .app import app

logger = get_task_logger("generate")


MODEL = "emozilla/llama-long-13b-scifi-fantasy-673-8192h-epoch4"
HISTORY = 8192
LEARNING_RATE = 3e-4
DEVICE = "cuda"
FINETUNING = False

models = {}
original_model = None
tokenizer = None


class Bootstep(bootsteps.Step):

    def __init__(self, parent, **options):
        super().__init__(parent, **options)

        import torch
        from .modeling_long import LlamaLongForCausalLM, GPTNeoXLongForCausalLM

        global accelerator, original_model, tokenizer

        if FINETUNING:
            from accelerate import Accelerator
            accelerator = Accelerator()
        else:
            accelerator = None

        original_model = LlamaLongForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="auto") \
            if "llama" in MODEL else GPTNeoXLongForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenizer.pad_token_id = 0
        tokenizer.model_max_length = original_model.config.max_positions


def get_model(id: str):
    if not FINETUNING:
        return original_model, None
    if id not in models:
        import torch
        from peft import LoraConfig, get_peft_model

        logger.info(f"Making model for {id}")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_match=True,
            target_modules=["q_proj", "v_proj"] if "llama" in MODEL else [
                "query_key_value"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(original_model, lora_config)
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=LEARNING_RATE)
        model, optimizer = accelerator.prepare(model, optimizer)
        models[id] = (model, optimizer)
    return models[id]


def get_id(id: str):
    return md5(id.encode("utf-8")).hexdigest()[0:8].upper()

# @app.post("/finetune")
# def finetune(args: FinetuneArgs):
#     if not FINETUNING:
#         raise RuntimeError("Fine-tuning not enabled")
#     id = get_id(args)

#     book = tokenizer(args.text, return_tensors='pt').input_ids.to(device)

#     if id in models:
#         del models[id]
#     model, optimizer = get_model(id)

#     with accelerator.accumulate(model):
#         model.train()

#         for i in range(0, (book.shape[1] // HISTORY) * HISTORY, HISTORY):
#             chunk = i // HISTORY
#             header = tokenizer(f"ID: {id} Chunk: {chunk} of {args.total_chunks}\n\n\n",
#                                return_tensors='pt').input_ids.to(device)

#             context = book[:, i:i+HISTORY]
#             input_ids = torch.cat((header, context), 1)
#             attention_mask = torch.ones(1, input_ids.shape[1])

#             outputs = model(input_ids=input_ids, labels=input_ids,
#                             attention_mask=attention_mask)
#             loss = outputs.loss
#             loss_value = loss.detach().float()
#             accelerator.backward(loss)
#             optimizer.zero_grad()

#         model.eval()

#     return {"loss": loss_value}
