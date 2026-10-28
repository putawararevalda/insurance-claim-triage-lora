"""Validate dataset syntax, Pydantic schema conformance, category balance, and leak check."""

import json
import os
import sys
from collections import Counter
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ClaimCategory, UrgencyLevel, ClaimMethod, ClaimTriageOutput


def load_and_validate_file(filepath: str) -> List[Dict[str, Any]]:
    print(f"\nValidating file: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[{filepath}:{line_num}] Invalid JSON syntax: {e}")
                
            # Check required top-level fields
            required_keys = ["instruction", "input", "output", "messages"]
            for k in required_keys:
                if k not in data:
                    raise ValueError(f"[{filepath}:{line_num}] Missing key '{k}'")
                    
            # Validate output string against Pydantic schema
            output_raw = data["output"]
            try:
                output_dict = json.loads(output_raw)
                validated_output = ClaimTriageOutput(**output_dict)
            except Exception as e:
                raise ValueError(f"[{filepath}:{line_num}] Pydantic validation failed for output: {e}")
                
            # Validate messages format
            messages = data["messages"]
            if len(messages) != 3 or messages[0]["role"] != "system" or messages[1]["role"] != "user" or messages[2]["role"] != "assistant":
                raise ValueError(f"[{filepath}:{line_num}] Invalid messages ChatML structure")
                
            records.append(data)
            
    print(f"  -> {len(records)} records successfully validated.")
    return records


def validate_dataset():
    print("=" * 60)
    print("  SEED DATASET RIGOROUS VALIDATION")
    print("=" * 60)
    
    train_records = load_and_validate_file("data/seed_train.jsonl")
    val_records = load_and_validate_file("data/seed_val.jsonl")
    test_records = load_and_validate_file("data/seed_test.jsonl")
    
    total_records = len(train_records) + len(val_records) + len(test_records)
    print("\n" + "-" * 60)
    print(f"Total Records Across Splits: {total_records}")
    print(f"Train Count: {len(train_records)} (Criteria >= 120)")
    print(f"Val Count  : {len(val_records)} (Criteria >= 15)")
    print(f"Test Count : {len(test_records)} (Criteria >= 15)")
    
    assert len(train_records) >= 120, f"Train count {len(train_records)} < 120"
    assert len(val_records) >= 15, f"Val count {len(val_records)} < 15"
    assert len(test_records) >= 15, f"Test count {len(test_records)} < 15"
    
    # Check data leakage
    train_inputs = set(r["input"].strip() for r in train_records)
    val_inputs = set(r["input"].strip() for r in val_records)
    test_inputs = set(r["input"].strip() for r in test_records)
    
    overlap_train_val = train_inputs.intersection(val_inputs)
    overlap_train_test = train_inputs.intersection(test_inputs)
    overlap_val_test = val_inputs.intersection(test_inputs)
    
    print("\nData Leakage Check:")
    print(f"  Overlap Train-Val : {len(overlap_train_val)} (Expected 0)")
    print(f"  Overlap Train-Test: {len(overlap_train_test)} (Expected 0)")
    print(f"  Overlap Val-Test  : {len(overlap_val_test)} (Expected 0)")
    assert len(overlap_train_val) == 0, f"Data leakage between train and val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Data leakage between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Data leakage between val and test: {overlap_val_test}"
    
    # Check category distribution
    all_categories = [r["category"] for r in (train_records + val_records + test_records)]
    cat_counts = Counter(all_categories)
    print("\nOverall Category Distribution:")
    for cat, count in cat_counts.items():
        print(f"  {cat:25s}: {count} ({count / total_records * 100:.1f}%)")
        
    counts = list(cat_counts.values())
    max_c, min_c = max(counts), min(counts)
    diff_pct = (max_c - min_c) / min_c * 100
    print(f"\nCategory Balance Difference: {diff_pct:.2f}% (Criteria < 15%)")
    assert diff_pct < 15.0, f"Category difference {diff_pct:.2f}% exceeds 15% threshold!"
    
    # Check urgency distribution
    all_urgencies = [r["urgency"] for r in (train_records + val_records + test_records)]
    urg_counts = Counter(all_urgencies)
    print("\nOverall Urgency Distribution:")
    for urg, count in urg_counts.items():
        print(f"  {urg:10s}: {count} ({count / total_records * 100:.1f}%)")
        
    # Check claim method distribution
    all_methods = [r["claim_method"] for r in (train_records + val_records + test_records)]
    meth_counts = Counter(all_methods)
    print("\nOverall Claim Method Distribution:")
    for m, count in meth_counts.items():
        print(f"  {m:15s}: {count} ({count / total_records * 100:.1f}%)")
        
    print("\n" + "=" * 60)
    print("SUCCESS: 100% of dataset records validated successfully against Pydantic schema!")
    print("=" * 60)


if __name__ == "__main__":
    validate_dataset()
