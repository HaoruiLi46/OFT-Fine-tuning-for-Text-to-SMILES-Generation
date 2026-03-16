# OFT Fine-tuning for Text-to-SMILES Generation

This repository contains the code and processed data for the AIST5030 mini-project on **parameter-efficient fine-tuning**. The task is to fine-tune a pretrained Qwen language model with **Orthogonal Finetuning (OFT)** so that it generates a molecular **SMILES** string from a natural-language molecular description.

## Repository Contents

```text
.
├── data/
│   ├── train.jsonl
│   └── test.jsonl
├── prepare_data.py
├── train.py
├── evaluate.py
├── run.sh
├── requirements.txt
└── results/
```

## Data

The repository only keeps the final processed files used for training and evaluation:

- `data/train.jsonl`
- `data/test.jsonl`

The original raw dataset is not included because it is too large for the submission repository.

`prepare_data.py` is retained to document the preprocessing pipeline. It can be used only if the original raw JSON file is available locally.

Each line in the processed JSONL files has the format:

```json
{"text": "molecule description", "smiles": "target SMILES"}
```

## Environment Setup

```bash
conda create -n aist5030-oft python=3.10 -y
conda activate aist5030-oft
python -m pip install -U pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install -r requirements.txt
```

## Training

```bash
python train.py \
    --model_name Qwen/Qwen3.5-0.8B \
    --train_file data/train.jsonl \
    --output_dir output/qwen35-oft-smiles \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --bf16 \
    --gradient_checkpointing
```

## Evaluation

Evaluate the fine-tuned OFT model:

```bash
python evaluate.py \
    --model_name output/qwen35-oft-smiles \
    --base_model_name Qwen/Qwen3.5-0.8B \
    --test_file data/test.jsonl \
    --output_file results/oft_results.json
```

Evaluate the base model as a zero-shot baseline:

```bash
python evaluate.py \
    --model_name Qwen/Qwen3.5-0.8B \
    --test_file data/test.jsonl \
    --output_file results/base_results.json
```

## Convenience Script

```bash
bash run.sh train
bash run.sh eval_oft
bash run.sh eval_base
```

## Method Summary

- **Base model**: `Qwen/Qwen3.5-0.8B`
- **PEFT method**: OFT from Hugging Face PEFT
- **Task**: text-to-SMILES generation
- **Evaluation metrics**:
  - SMILES validity
  - Tanimoto similarity

## Notes

- `prepare_data.py` is kept for completeness, but the submission-ready repository is designed to run directly from the processed JSONL files.
- Example outputs and figures are stored in `results/`.
