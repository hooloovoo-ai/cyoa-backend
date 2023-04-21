import argparse
import torch
import time
import sys
import os
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.profiler import profile as torch_profile, record_function, ProfilerActivity


class DummyWith:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
NUM_GENERATIONS = 1
REPETITION_PENALTY = 1.1

TITLE = "A Quesiton of Fire"
AUTHOR = "Jeffrey Quesnelle"
START = r"""PROLOGUE

The last of the stones had been toppled long ago, but that didn't stop Tomin from letting his fingers wistfully graze the cold ground as if his mere touch could resurrect the once-great hall that was now nothing more than a slightly smoother patch of arid land. His memories of the hall were hazy, as if coming to him from a long-lost dream; the more he focused on them the quicker they seemed to melt away.

He was sure of one thing, though. The Great Hall of Porice had been home to the Emerald Chair. He wasn't sure why that was important, but the clarity of the thought was in such contrast to his other vague recollections that he knew it must be true.

"Why am I even here?" He wondered aloud.

Of course, he knew the answer - had known since he first set out on the perilous journey across from his home town to this now seemingly insignificant plot of long-forgotten rubble.

The Emerald Chair.

Over and over, in his sleeping dreams and waking thoughts, he felt its call. Like a faint echo ever fading into silence, it came to him in whispers, urging him onward, compelling him to find it, to hold it, and to claim it as his own. There were times when Tomin would have sworn that he could actually hear the voice of the great wooden throne repeating the words over and over again: You must bring me back. You must, you must, you must.

Tomin sighed heavily, his breath expelling itself in a long exhalation, and pulled a well-worn cloth map from the pouch tied to his belt. Carefully unfolding it, he smoothed its torn fibers onto the rocky terrain, making every effort to ignore the harsh sunlight as he squinted down at the faded lines and wrinkles."""
ID = "F9AA0D31"


def benchmark(
    model_name: str, quantize: bool, compile: bool, profile: bool, long_model: bool,
    save_story: bool
) -> List[Tuple[int, float]]:
    device_map = None if compile or profile else "auto"
    if long_model:
        from ..modeling_long import LlamaLongForCausalLM
        model = LlamaLongForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=quantize,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=quantize,
            device_map=device_map,
        )
    if device_map:
        print(f"device map: {model.hf_device_map}")
    else:
        model = model.to("cuda")

    if compile:
        model.model = torch.compile(model.model)

    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.config, "max_positions"):
        tokenizer.model_max_length = model.config.max_positions

    TARGET_TOKEN_LENGTH = tokenizer.model_max_length
    HISTORY_LEN = TARGET_TOKEN_LENGTH - 64
    TOTAL_CHUNKS = TARGET_TOKEN_LENGTH // HISTORY_LEN

    book_start =  f"{TITLE}\n\n{AUTHOR}\n\n\n\n{START}"
    book = tokenizer(book_start, return_tensors="pt").input_ids.to("cuda")

    filename_base = f"{model_name.replace('/', '-')}{'_8bit' if quantize else ''}{'_compile' if compile else ''}{'-profile' if profile else ''}_devices{os.environ.get('CUDA_VISIBLE_DEVICES', '')}"

    f = open(f"{filename_base}.{'txt' if profile else 'csv'}", "w")
    if save_story:
        s = open(f"{filename_base}.story.txt", "w", encoding="utf-8")
        s.write(book_start)
        s.flush()
    else:
        s = None

    if not profile:
        f.write("Prompt tokens,Inference tokens/sec\n")

    with DummyWith() if profile else (torch.no_grad() if compile else torch.inference_mode()):
        if compile:
            # warm up
            tokenized = tokenizer(
                f"ID: {ID} Chunk: {0} of {TOTAL_CHUNKS}\n\n\n{book_start}",
                return_tensors="pt",
            )
            for _ in range(0, 2):
                model.generate(
                    input_ids=tokenized.input_ids.to("cuda"),
                    attention_mask=tokenized.attention_mask.to("cuda"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    repetition_penalty=REPETITION_PENALTY,
                )

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

                with torch_profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                ) if profile else DummyWith() as prof:
                    with record_function("generate") if profile else DummyWith():
                        output_tokens = model.generate(
                            input_ids=input_ids.to("cuda"),
                            attention_mask=attention_mask.to("cuda"),
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            repetition_penalty=1.1,
                        )

                new_tokens = output_tokens[:, input_ids.shape[1]:]

                new_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                if s is not None:
                    s.write(new_text)
                    s.flush()

                pbar.update(new_tokens.shape[1])

                book = torch.cat((book, new_tokens[0][None, :].to("cuda")), 1)

                end = time.time()

                if profile:
                    f.write(
                        f"{current_token_length}\n{prof.key_averages().table(sort_by='cuda_time_total', row_limit=10)}\n\n"
                    )
                else:
                    f.write(
                        f"{current_token_length},{MAX_NEW_TOKENS / (end - start)}\n"
                    )
                f.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--long_model", action="store_true")
    parser.add_argument("--save_story", action="store_true")
    return parser.parse_args()


def main(args):
    results = benchmark(args.model_name, args.quantize,
                        args.compile, args.profile, args.long_model, args.save_story)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
