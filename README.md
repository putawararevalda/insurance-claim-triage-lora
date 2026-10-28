# LoRA Fine-Tuning for Insurance Claim & FNOL Triage

Parameter-Efficient Fine-Tuning (QLoRA) on Small Language Models (SLM) for Indonesian Insurance Claim & First Notice of Loss (FNOL) Triage.

## Overview
This repository implements an end-to-end 4-bit QLoRA fine-tuning pipeline tailored for Indonesian health and life insurance claim classification and triage parameter extraction.

### Claim Triage Output Schema (`src/schema.py`)
- **category**: `Inpatient_Medical`, `Outpatient_Emergency`, `Critical_Illness`, `Personal_Accident`, `Inquiry_or_Dispute`
- **urgency**: `Low`, `Medium`, `High`
- **primary_condition**: Primary medical condition / problem description
- **claim_method**: `Cashless`, `Reimbursement`, `Unknown`
- **reasoning**: Brief 1-2 sentence justification

### Dataset (`data/`)
- `seed_train.jsonl`: 140 samples (80%)
- `seed_val.jsonl`: 17 samples (10%)
- `seed_test.jsonl`: 18 samples (10%)
Total: 175 samples across 5 perfectly balanced categories (35 samples each, 0% variance).

### Validation
To validate dataset integrity and Pydantic schema adherence:
```bash
python scripts/validate_dataset.py
```
