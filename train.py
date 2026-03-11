"""
Fine-tune a Qwen model for Text-to-SMILES generation using OFT (Orthogonal Finetuning).

Usage:
    python train.py \
        --model_name Qwen/Qwen3.5-0.8B \
        --train_file data/train.jsonl \
        --output_dir output/qwen35-oft-smiles \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --learning_rate 1e-4

Key features:
    - Uses Hugging Face PEFT library with OFT/BOFT configuration
    - Formats data as chat-style instruction tuning
    - Logs training loss for report visualization
"""

import argparse
import json
import os
import logging

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, OFTConfig, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt used for instruction tuning
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a chemist assistant. Given a textual description of a molecule, "
    "generate the corresponding SMILES string."
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SMILESDataset(Dataset):
    """
    Each sample is formatted into Qwen chat template:
        <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
        <|im_start|>user\n{text}<|im_end|>\n
        <|im_start|>assistant\n{smiles}<|im_end|>
    
    Only the assistant response (SMILES) tokens are used to compute loss.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                self.samples.append(obj)

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        smiles = sample["smiles"]

        # Build messages for chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": smiles},
        ]

        # Tokenize the full conversation (with the assistant response)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Tokenize everything EXCEPT the assistant response to find the prompt length
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_encoding["input_ids"])

        # Build labels: mask prompt tokens with -100, keep assistant tokens
        input_ids = full_encoding["input_ids"]
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Ensure labels length matches input_ids length
        labels = labels[: len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": full_encoding["attention_mask"],
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with OFT for Text-to-SMILES")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Pretrained model name or path")
    
    # Data
    parser.add_argument("--train_file", type=str, default="data/train.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")

    # OFT config
    parser.add_argument("--oft_block_size", type=int, default=2,
                        help="Block size for OFT. Smaller = more parameters but more expressive")
    parser.add_argument("--oft_r", type=int, default=8,
                        help="Rank r for BOFT. Number of butterfly factors")
    parser.add_argument("--target_modules", type=str, default="all-linear",
                        help="Which modules to apply OFT to")

    # Training
    parser.add_argument("--output_dir", type=str, default="output/qwen35-oft-smiles")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bfloat16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load tokenizer and model ----
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bool(
        args.bf16
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if args.bf16 and not use_bf16:
        logger.warning(
            "--bf16 was requested but is not supported on this device. "
            "Falling back to float32."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=True,
    )

    # ---- Apply OFT via PEFT ----
    logger.info("Applying OFT configuration...")
    oft_config = OFTConfig(
        r=args.oft_r,
        oft_block_size=args.oft_block_size,
        target_modules=args.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        module_dropout=0.0,
    )
    model = get_peft_model(model, oft_config)
    model.print_trainable_parameters()

    # ---- Load dataset ----
    logger.info("Loading training dataset...")
    train_dataset = SMILESDataset(args.train_file, tokenizer, max_length=args.max_length)

    # ---- Data collator ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # ---- Train ----
    logger.info("Starting training...")
    train_result = trainer.train()

    # ---- Save ----
    logger.info("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
