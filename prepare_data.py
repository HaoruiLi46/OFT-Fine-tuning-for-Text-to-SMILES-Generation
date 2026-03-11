"""
Data preparation script for Text-to-SMILES fine-tuning.

Reads the raw JSON dataset and expands each (description_list, SMILES) entry
into multiple (single_description, SMILES) training samples.
Outputs a JSONL file ready for SFT training.
"""

import json
import os
import random

RAW_DATA_PATH = "b3lyp_pm6_description_merge.json"
OUTPUT_DIR = "data"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "test.jsonl")

SMILES_KEY = "pubchem-obabel-canonical-smiles"
DESCRIPTION_KEY = "description"

TEST_RATIO = 0.05  # 5% for test set
RANDOM_SEED = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Raw JSON entries: {len(raw_data)}")

    # Expand: each description string becomes a separate sample
    all_samples = []
    skipped_no_smiles = 0
    skipped_no_desc = 0

    for entry in raw_data:
        smiles = entry.get(SMILES_KEY, "")
        descriptions = entry.get(DESCRIPTION_KEY, [])

        if not smiles:
            skipped_no_smiles += 1
            continue
        if not descriptions or len(descriptions) == 0:
            skipped_no_desc += 1
            continue

        for desc in descriptions:
            desc = desc.strip()
            if desc:
                all_samples.append({
                    "text": desc,
                    "smiles": smiles,
                })

    print(f"Total expanded samples: {len(all_samples)}")
    print(f"Skipped (no SMILES): {skipped_no_smiles}")
    print(f"Skipped (no description): {skipped_no_desc}")

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * (1 - TEST_RATIO))
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Write JSONL files
    def write_jsonl(path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_OUTPUT, train_samples)
    write_jsonl(TEST_OUTPUT, test_samples)

    # Print a few examples
    print("\n--- Example training samples ---")
    for i, sample in enumerate(train_samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Text: {sample['text'][:120]}...")
        print(f"  SMILES: {sample['smiles']}")


if __name__ == "__main__":
    main()
