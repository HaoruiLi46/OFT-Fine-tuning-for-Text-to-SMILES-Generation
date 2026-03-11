"""
Evaluate a fine-tuned (or base) Qwen model on Text-to-SMILES generation.

Metrics:
    1. SMILES Validity: fraction of generated strings that are valid SMILES
    2. Tanimoto Similarity: molecular fingerprint similarity between generated
       and ground-truth molecules (Morgan Fingerprints)

Usage:
    # Evaluate the fine-tuned model
    python evaluate.py \
        --model_name output/qwen35-oft-smiles \
        --base_model_name Qwen/Qwen3.5-0.8B \
        --test_file data/test.jsonl \
        --output_file results/oft_results.json \
        --max_new_tokens 256

    # Evaluate the base model (zero-shot baseline)
    python evaluate.py \
        --model_name Qwen/Qwen3.5-0.8B \
        --test_file data/test.jsonl \
        --output_file results/base_results.json \
        --max_new_tokens 256
"""

import argparse
import json
import os
import logging
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a chemist assistant. Given a textual description of a molecule, "
    "generate the corresponding SMILES string."
)


def resolve_torch_dtype(dtype_name: str):
    """Resolve requested dtype with hardware-aware fallback."""
    dtype_name = dtype_name.lower()
    bf16_supported = (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    if dtype_name == "auto":
        if bf16_supported:
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    if dtype_name == "bf16":
        if bf16_supported:
            return torch.bfloat16
        logger.warning("bf16 is not supported on this device. Falling back to float32.")
        return torch.float32

    if dtype_name == "fp16":
        if torch.cuda.is_available():
            return torch.float16
        logger.warning("fp16 requires CUDA. Falling back to float32.")
        return torch.float32

    return torch.float32


def load_model_and_tokenizer(model_name, base_model_name=None, dtype_name="auto"):
    """
    Load model and tokenizer. If base_model_name is provided, loads the base
    model first and then applies the PEFT adapter from model_name.
    """
    torch_dtype = resolve_torch_dtype(dtype_name)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    if base_model_name:
        # Load base model + PEFT adapter
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        logger.info(f"Loading PEFT adapter: {model_name}")
        model = PeftModel.from_pretrained(model, model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        # Load standalone model (base or merged)
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_smiles(model, tokenizer, text: str, max_new_tokens: int = 256) -> str:
    """Generate SMILES from text description using the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


def evaluate_validity(predictions: list) -> dict:
    """Evaluate SMILES validity using RDKit."""
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("RDKit not installed. Skipping validity evaluation.")
        return {"valid_count": -1, "total": len(predictions), "validity": -1.0}

    valid_count = 0
    for pred in predictions:
        mol = Chem.MolFromSmiles(pred)
        if mol is not None:
            valid_count += 1

    return {
        "valid_count": valid_count,
        "total": len(predictions),
        "validity": valid_count / len(predictions) if predictions else 0.0,
    }


def evaluate_similarity(predictions: list, ground_truths: list) -> dict:
    """
    Evaluate Tanimoto similarity between predicted and ground-truth molecules
    using Morgan Fingerprints (radius=2, 2048 bits).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        logger.warning("RDKit not installed. Skipping similarity evaluation.")
        return {"avg_tanimoto_similarity": -1.0, "evaluated_pairs": 0, "total_pairs": len(predictions)}

    similarities = []
    for pred, gt in zip(predictions, ground_truths):
        pred_mol = Chem.MolFromSmiles(pred)
        gt_mol = Chem.MolFromSmiles(gt)

        if pred_mol is not None and gt_mol is not None:
            pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, radius=2, nBits=2048)
            gt_fp = AllChem.GetMorganFingerprintAsBitVect(gt_mol, radius=2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(pred_fp, gt_fp)
            similarities.append(sim)

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    return {
        "avg_tanimoto_similarity": round(avg_sim, 4),
        "evaluated_pairs": len(similarities),
        "total_pairs": len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SMILES generation")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to fine-tuned model or HF model name (for base evaluation)")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Base model name if model_name is a PEFT adapter path")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--output_file", type=str, default="results/results.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                        help="Model loading dtype. 'auto' selects bf16/fp16/fp32 based on hardware.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of test samples to evaluate (for quick testing)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.base_model_name, args.dtype)

    # Load test data
    test_samples = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))

    if args.max_samples:
        test_samples = test_samples[:args.max_samples]

    logger.info(f"Evaluating on {len(test_samples)} samples...")

    # Generate predictions
    predictions = []
    ground_truths = []
    detailed_results = []

    for sample in tqdm(test_samples, desc="Generating"):
        text = sample["text"]
        gt_smiles = sample["smiles"]
        pred_smiles = generate_smiles(model, tokenizer, text, args.max_new_tokens)

        predictions.append(pred_smiles)
        ground_truths.append(gt_smiles)
        detailed_results.append({
            "text": text[:200],
            "ground_truth": gt_smiles,
            "prediction": pred_smiles,
        })

    # Evaluate
    validity_metrics = evaluate_validity(predictions)
    similarity_metrics = evaluate_similarity(predictions, ground_truths)

    results = {
        "model": args.model_name,
        "num_samples": len(test_samples),
        "validity": validity_metrics,
        "similarity": similarity_metrics,
        "examples": detailed_results[:20],  # Save first 20 examples for inspection
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:             {args.model_name}")
    print(f"Test samples:      {len(test_samples)}")
    print(f"SMILES Validity:   {validity_metrics['validity']:.4f} "
          f"({validity_metrics['valid_count']}/{validity_metrics['total']})")
    print(f"Tanimoto Sim:      {similarity_metrics['avg_tanimoto_similarity']:.4f} "
          f"({similarity_metrics['evaluated_pairs']} valid pairs)")
    print("=" * 60)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
