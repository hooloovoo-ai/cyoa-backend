import os
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser, LlamaTokenizer


@dataclass
class Arguments:
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."})

    tokenizer_name: str = field(
        metadata={"help": "Pretrained tokenizer name or path"})

    streaming: bool = field(default=False, metadata={
                            "help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use."},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    parser = HfArgumentParser((Arguments))
    args: Arguments = parser.parse_args_into_dataclasses()[0]

    raw_datasets = load_dataset(args.dataset_name, streaming=args.streaming)

    if "llama" in args.tokenizer_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=args.use_fast_tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=args.use_fast_tokenizer)

    column_names = raw_datasets["train"].column_names
    # id_header = tokenizer("ID: ").input_ids
    # chunk_start = tokenizer(" Chunk: ").input_ids
    # chunk_of = tokenizer(" of ").input_ids
    # chunk_end = tokenizer("\n\n\n").input_ids
    # nums = [tokenizer(str(i)).input_ids for i in range(0, 100000)]

    def tokenize_function(examples):
        tokenized = tokenizer(examples["utf8"])
        # tokenized["id_input_ids"] = tokenizer(examples["id"]).input_ids
        return tokenized

    if not args.streaming:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    block_size = tokenizer.model_max_length if args.block_size is None else args.block_size

    def group_texts(examples):
        result = {
            "input_ids": [],
            "attention_mask": []
        }

        for i in range(0, len(examples["input_ids"])):

            input_ids = examples["input_ids"][i]
            attention_mask = examples["attention_mask"][i]
            # id_input_ids = examples["id_input_ids"][i]

            total_chunks = len(input_ids) // block_size

            # result["input_ids"].extend([(id_header + id_input_ids[0:6] + chunk_start + nums[j // block_size] + chunk_of + nums[total_chunks] + chunk_end + input_ids[j: j + block_size])[0:block_size] for j in range(
            #     0, (len(input_ids) // block_size) * block_size, block_size)])
            result["input_ids"].extend([input_ids[j: j + block_size] for j in range(
                0, (len(attention_mask) // block_size) * block_size, block_size)])
            result["attention_mask"].extend([attention_mask[j: j + block_size] for j in range(
                0, (len(attention_mask) // block_size) * block_size, block_size)])
        result["labels"] = result["input_ids"].copy()
        return result

    if not args.streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.num_workers,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )

    lm_datasets.push_to_hub(
        f"emozilla/{args.dataset_name.split('/')[-1]}-{block_size}n-{args.tokenizer_name.split('/')[-1]}", private=True)
    # lm_datasets.save_to_disk(os.path.join("outputs", f"{args.dataset_name.split('/')[-1]}-{block_size}-{args.tokenizer_name.split('/')[-1]}"))


if __name__ == "__main__":
    main()
