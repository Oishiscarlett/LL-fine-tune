import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments

local_path = "/data/home/zfyang/home/oishi/model/Baichuan2-7B-Base"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=local_path)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        user_tokens=[195],
        assistant_tokens=[196],
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        """
        -100是一个pytorch中常见的值。用于指示损失函数在计算的时候应该忽略某些特定的token。
        这是因为在处理自然语言处理任务时，尤其是涉及到批处理的时候，
        我们通常需要将所有序列（比如句子或对话）填充到相同的长度，以便能够作为一个批次输入到模型中。
        填充通常使用一个特定的pad_token来完成。
        然而，在计算损失时，这些填充的token不应该被考虑进去，
        因为它们不是原始序列的一部分，也不携带任何对任务有用的信息
        """
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = [] # 标签用于训练过程中计算损失

        for message in example["conversations"]:
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value)

            # tokenizer.eos_token_id 表示结束符
            if from_ == "human":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(
                    value_ids
                )
            else:
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        # 超出长度的部分进行截断
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        # 进行长度填充
        input_ids += [self.tokenizer.pad_token_id] * (
            self.model_max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=8,
            lora_alpha=16, #LoRA低秩矩阵的缩放系数，为一个常数超参，调整alpha与调整学习率类似。
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        # 查看LoRA可训练的数据量
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
