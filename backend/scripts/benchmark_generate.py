import argparse
import torch
import time
import sys
from typing import List, Tuple
from transformers import AutoTokenizer, LlamaTokenizer
from tqdm import tqdm
from ..modeling_long import LlamaLongForCausalLM


HISTORY_LEN = 7936
MAX_NEW_TOKENS = 256
TARGET_TOKEN_LENGTH = 8000
TEMPERATURE = 0.8
NUM_GENERATIONS = 1
REPETITION_PENALTY = 1.1
TOTAL_CHUNKS = TARGET_TOKEN_LENGTH // HISTORY_LEN

TITLE = "A Quesiton of Fire"
AUTHOR = "Jeffrey Quesnelle"
START = r"""The last of the stones had been toppled long ago, but that didn't stop Tomin from letting his fingers wistfully graze the cold ground
as if his mere touch could resurect the once-great hall that was now nothing more than a slightly smoother patch of arid land. His memories of the
hall were hazy, as if coming to him from a long-lost dream; the more he focused on them the quicker they seemed to melt away.

He was sure of one thing, though. The Great Hall of Porice had been home to the Emerald Chair. He wasn't sure why that was important, but the
clarity of the thought was in such contrast to his other vague recollections that he knew it must be important."""
ID = "F9AA0D31"


def benchmark(model_name: str, quantize: bool) -> List[Tuple[int, float]]:
    model = LlamaLongForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, load_in_8bit=quantize, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model.config.max_positions

    book = tokenizer(
        f"{TITLE}\n\n{AUTHOR}\n\n\n\n{START}", return_tensors="pt"
    ).input_ids.to("cuda")

    f = open(
        f"{model_name.replace('/', '-')}{'_8bit' if quantize else ''}.csv", "w")
    f.write("Tokens,Time\n")

    with torch.inference_mode():
        with tqdm(total=TARGET_TOKEN_LENGTH - book.shape[1]) as pbar:
            while True:
                start = time.time()
                current_token_length = book.shape[1]

                if current_token_length >= TARGET_TOKEN_LENGTH:
                    break

                location = current_token_length / TARGET_TOKEN_LENGTH
                chunk = current_token_length // HISTORY_LEN

                header = tokenizer(
                    f"ID: {ID} Chunk: {chunk} of {TOTAL_CHUNKS}\n\n\n",
                    return_tensors="pt",
                ).input_ids.to("cuda")
                context = book[:, -HISTORY_LEN:]

                input_ids = torch.cat((header, context), 1)
                input_ids = torch.cat([input_ids] * NUM_GENERATIONS, dim=0)
                # input_ids = context
                attention_mask = torch.ones(1, input_ids.shape[1])
                attention_mask = torch.cat(
                    [attention_mask] * NUM_GENERATIONS, dim=0)

                output_tokens = model.generate(
                    input_ids=input_ids.to("cuda"),
                    attention_mask=attention_mask.to("cuda"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    repetition_penalty=1.1,
                )

                new_tokens = output_tokens[:, input_ids.shape[1]:]

                _ = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

                pbar.update(new_tokens.shape[1])

                book = torch.cat((book, new_tokens[0][None, :].to("cuda")), 1)

                end = time.time()

                f.write(f"{book.shape[1]},{end - start}\n")
                f.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def main(args):
    results = benchmark(args.model_name, args.quantize)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
