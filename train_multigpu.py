# 기존 2장 gpu에 맞춰진 코드를 multi gpu에 맞추어 수정
import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from src.data import CustomDataset, DataCollatorForSupervisedDataset

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, help="scheduler warmup steps")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=5, help="training epoch")
# fmt: on

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = 1024

    train_dataset = CustomDataset("resource/data/data_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/data_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': [x[:max_seq_length] for x in train_dataset.inp],
        'labels': [x[:max_seq_length] for x in train_dataset.label],
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': [x[:max_seq_length] for x in valid_dataset.inp],
        'labels': [x[:max_seq_length] for x in valid_dataset.label],
    })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.1,
        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        log_level="info",
        logging_steps=1,
        save_strategy="no", # check point 저장공간 너무 먹어서 중간저장안함
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
        ddp_find_unused_parameters=False  # DDP 사용시 이 옵션을 False로 설정
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    trainer.train()

if __name__ == "__main__":
    exit(main(parser.parse_args()))
