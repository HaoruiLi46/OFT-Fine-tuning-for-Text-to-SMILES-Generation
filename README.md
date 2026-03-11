# AIST5030 Mini-Project: Text-to-SMILES Generation with OFT

Parameter-efficient fine-tuning of a pretrained Qwen LLM for molecule generation using **Orthogonal Finetuning (OFT)**.

Given a textual description of a molecule, the fine-tuned model generates the corresponding SMILES string.

## Project Structure

```
├── prepare_data.py        # Expand raw JSON into (text, SMILES) train/test JSONL
├── prepare_data_merged_text.py  # Merge all descriptions per molecule into one text
├── train.py               # OFT fine-tuning with Hugging Face PEFT
├── evaluate.py            # Inference + evaluation (Validity & Tanimoto Similarity)
├── run.sh                 # Convenience script to run each step
├── requirements.txt       # Python dependencies
└── data/
    ├── train.jsonl         # Training set
    └── test.jsonl          # Test set
```

## Setup

```bash
conda create -n qwen35-oft python=3.10 -y
conda activate qwen35-oft
python -m pip install -U pip setuptools wheel

# Install GPU PyTorch wheel for CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.6.0

# Install project dependencies
pip install -r requirements.txt
```

`requirements.txt` uses `transformers` from GitHub `main` branch to ensure compatibility with Qwen3.5.

Verify GPU runtime on a compute node:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Usage

### 1. Prepare Data

Expand the raw dataset into individual `(text_description, SMILES)` training samples:

```bash
python3 prepare_data.py
```

### 2. Train (OFT Fine-tuning)

```bash
python3 train.py \
    --model_name Qwen/Qwen3.5-0.8B \
    --train_file data/train.jsonl \
    --output_dir output/qwen35-oft-smiles \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --bf16 --gradient_checkpointing
```

### 3. Evaluate

```bash
# Fine-tuned model
python3 evaluate.py \
    --model_name output/qwen35-oft-smiles \
    --base_model_name Qwen/Qwen3.5-0.8B \
    --test_file data/test.jsonl \
    --output_file results/oft_results.json

# Base model (zero-shot baseline)
python3 evaluate.py \
    --model_name Qwen/Qwen3.5-0.8B \
    --test_file data/test.jsonl \
    --output_file results/base_results.json
```

Or use the convenience script:

```bash
bash run.sh train       # Train
bash run.sh eval_oft    # Evaluate fine-tuned model
bash run.sh eval_base   # Evaluate base model
```

## Method

- **Model**: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **PEFT**: Orthogonal Finetuning (OFT) via [Hugging Face PEFT](https://huggingface.co/docs/peft/main/en/conceptual_guides/oft), applied to all linear layers
- **Task**: Text → SMILES molecule generation (instruction-tuned with chat template)

## Evaluation Metrics

| Metric | Description |
|---|---|
| **SMILES Validity** | Fraction of generated strings parseable by RDKit |
| **Tanimoto Similarity** | Morgan fingerprint similarity between predicted and ground-truth molecules |
