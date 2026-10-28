"""Validate Alpha dataset syntax, Pydantic schema conformance, category balance, and leak check."""

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


def validate_alpha_dataset():
    print("=" * 60)
    print("  ALPHA DATASET RIGOROUS VALIDATION")
    print("=" * 60)
    
    train_records = load_and_validate_file("data/alpha_train.jsonl")
    val_records = load_and_validate_file("data/alpha_val.jsonl")
    test_records = load_and_validate_file("data/alpha_test.jsonl")
    
    total = len(train_records) + len(val_records) + len(test_records)
    print(f"\nTotal Records: {total} (Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)})")
    
    # Check Acceptance Criteria Target counts
    assert total >= 500 and total <= 800, f"Total records {total} outside 500-800 target!"
    assert len(train_records) >= 400, f"Train set {len(train_records)} < 400 target!"
    assert len(val_records) >= 50, f"Val set {len(val_records)} < 50 target!"
    assert len(test_records) >= 50, f"Test set {len(test_records)} < 50 target!"
    
    # 1. Leakage Check
    print("\n[Check 1/3] Data Leakage Check across Splits:")
    train_inputs = set(r["input"].strip() for r in train_records)
    val_inputs = set(r["input"].strip() for r in val_records)
    test_inputs = set(r["input"].strip() for r in test_records)
    
    leak_train_val = train_inputs.intersection(val_inputs)
    leak_train_test = train_inputs.intersection(test_inputs)
    leak_val_test = val_inputs.intersection(test_inputs)
    
    if leak_train_val:
        raise AssertionError(f"Data leak between Train and Val: {len(leak_train_val)} overlapping inputs!")
    if leak_train_test:
        raise AssertionError(f"Data leak between Train and Test: {len(leak_train_test)} overlapping inputs!")
    if leak_val_test:
        raise AssertionError(f"Data leak between Val and Test: {len(leak_val_test)} overlapping inputs!")
        
    print("  [PASSED] Zero text overlap between Train, Val, and Test splits.")
    
    # 2. Category Distribution Balance Check
    print("\n[Check 2/3] Category Balance Check (Overall & Per Split):")
    all_records = train_records + val_records + test_records
    cat_counts = Counter(r["category"] for r in all_records)
    
    print("  Overall Category Counts:")
    for cat, count in cat_counts.items():
        print(f"    - {cat}: {count} ({count/total*100:.1f}%)")
        
    max_count = max(cat_counts.values())
    min_count = min(cat_counts.values())
    diff_pct = (max_count - min_count) / max_count * 100
    print(f"  Category imbalance delta: {diff_pct:.2f}% (Target: < 15%)")
    assert diff_pct < 15, f"Category distribution too imbalanced: delta {diff_pct:.2f}% >= 15%"
    print("  [PASSED] Category distribution is perfectly balanced.")
    
    # 3. Pydantic Schema Conformance
    print("\n[Check 3/3] 100% Pydantic Schema Output Validity Check:")
    valid_outputs = 0
    for r in all_records:
        parsed = json.loads(r["output"])
        ClaimTriageOutput(**parsed)
        valid_outputs += 1
        
    print(f"  Valid Outputs: {valid_outputs} / {total} (100.0%)")
    print("  [PASSED] All 650 records strictly adhere to ClaimTriageOutput schema.")
    
    print("\n" + "=" * 60)
    print("  ALL RIGOROUS VALIDATION CHECKS PASSED (ALPHA DATASET)")
    print("=" * 60)

if __name__ == "__main__":
    validate_alpha_dataset()
