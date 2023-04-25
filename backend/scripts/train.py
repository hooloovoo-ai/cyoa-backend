import torch
import datasets
import numpy as np
import evaluate
import os
from datasets import load_dataset
from transformers import (
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding as GPTNeoXRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.trainer_utils import get_last_checkpoint, seed_worker
from transformers.utils import is_datasets_available
from itertools import chain
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader

# from modeling_long import GPTNeoXFlashAttentionWrapper, LlamaFlashAttentionWrapper


class NonShufflingTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = datasets.IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                sampler=None,
                shuffle=False,
            )

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=None,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="pythia-1.3b",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    max_positions: Optional[int] = field(
        default=8192,
        metadata={"help": ("The maximun sequence length of the model.")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="pile", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    outline: Optional[bool] = field(default=False, metadata={
                                    "help": "Train the outliner."})

    prompt_len: Optional[int] = field(default=None)
    completion_len: Optional[int] = field(default=None)

    no_shuffle: bool = field(default=False)


class DeepSpeedCacheClearCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.deepspeed:
            from deepspeed.accelerator import get_accelerator

            get_accelerator().empty_cache()


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    set_seed(training_args.seed)

    if data_args.prompt_len is not None:
        assert data_args.completion_len is not None
    if data_args.completion_len is not None:
        assert data_args.prompt_len is not None

    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if getattr(config, "max_position_embeddings", None) != model_args.max_positions:
        if "GPTNeoXForCausalLM" in config.architectures:
            is_changed_gpt_neox = True
            config = None
        else:
            config.max_position_embeddings = model_args.max_positions
            is_changed_gpt_neox = False
    else:
        is_changed_gpt_neox = False

    local_rank = os.getenv('LOCAL_RANK', None)
    if local_rank is not None and training_args.deepspeed is None:
        device_map = {'': int(local_rank)}
    else:
        device_map = None
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, device_map=device_map)

    # tokenizer.pad_token = tokenizer.mask_token if not data_args.prompt_len else tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.mask_token_id if not data_args.prompt_len else tokenizer.eos_token_id
    tokenizer.model_max_length = model_args.max_positions

    if is_changed_gpt_neox:
        model.config.max_position_embeddings = model_args.max_positions
        for each in model.gpt_neox.layers:
            each.attention.bias = torch.tril(
                torch.ones((model_args.max_positions,
                           model_args.max_positions), dtype=torch.uint8)
            ).view(1, 1, model_args.max_positions, model_args.max_positions)
            each.attention.rotary_emb = GPTNeoXRotaryEmbedding(
                each.attention.rotary_ndims,  model_args.max_positions, 10000)

    # patching for the random contiguous tensors bug
    # for p in model.parameters():
    #     p = p.contiguous()

    datasets = load_dataset(data_args.dataset_name)
    with training_args.main_process_first(desc="dataset map tokenization"):
        data_collator = default_data_collator
        lm_datasets = datasets

        if data_args.prompt_len is not None:
            lm_datasets = lm_datasets.map(
                lambda sample: {
                    "input_ids": [x[0: data_args.prompt_len] for x in sample["input_ids"]],
                    "labels": [x[0: data_args.prompt_len + data_args.completion_len] for x in sample["labels"]]
                    if "t5" not in model_args.model_name_or_path
                    else [x[-data_args.completion_len:] for x in sample["labels"]],
                    "attention_mask": [x[0: data_args.prompt_len] for x in sample["attention_mask"]],
                },
                batched=True,
                num_proc=1,
                desc=f"Truncataing input ids",
            )
            data_collator = DataCollatorForLanguageModeling(
                tokenizer, mlm=False)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"] if "validation" in lm_datasets else None

    trainer_class = NonShufflingTrainer if data_args.no_shuffle else Trainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[DeepSpeedCacheClearCallback],
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.create_model_card()


if __name__ == "__main__":
    main()
