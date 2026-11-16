# Alpha Dataset (500–800 Samples) Statistics Report

**Date**: 2026-10-29  
**Pipeline**: Fase 2.1 - Dataset Scaling & Synthetic Generation  
**Domain**: Indonesian Insurance Claim & FNOL Triage  

---

## 1. Executive Summary

The dataset has been successfully expanded from the initial 175 seed samples (Phase 1 MVP) to **650 high-quality instructional samples** in standard ChatML format. 
All samples strictly conform to Pydantic type validation (`ClaimTriageOutput`), contain zero text leakage across splits, and feature balanced representation across 5 insurance claim categories and diverse claimant personas.

| Metric | Target | Result | Status |
| :--- | :---: | :---: | :---: |
| **Total Curated Samples** | 500 – 800 | **650** | **Pass** |
| **Train Set Count** | $\ge 400$ | **520** (80.0%) | **Pass** |
| **Validation Set Count** | $\ge 50$ | **65** (10.0%) | **Pass** |
| **Test Set Count** | $\ge 50$ | **65** (10.0%) | **Pass** |
| **Edge Case Ratio** | $\ge 15\%$ | **34.5%** (224 samples) | **Pass** |
| **Pydantic Validation Rate** | 100% | **100.0%** | **Pass** |
| **Cross-Split Data Leakage** | 0 | **0 (Zero Leakage)** | **Pass** |

---

## 2. Category Balance Distribution

Every category has been curated with identical representation across the overall dataset to prevent class imbalance skewing during LoRA fine-tuning.

| Category | Total Count | Train (80%) | Val (10%) | Test (10%) | Proportion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Inpatient_Medical` | 130 | 104 | 13 | 13 | 20.0% |
| `Outpatient_Emergency` | 130 | 104 | 13 | 13 | 20.0% |
| `Critical_Illness` | 130 | 104 | 13 | 13 | 20.0% |
| `Personal_Accident` | 130 | 104 | 13 | 13 | 20.0% |
| `Inquiry_or_Dispute` | 130 | 104 | 13 | 13 | 20.0% |
| **Total** | **650** | **520** | **65** | **65** | **100.0%** |

---

## 3. Persona, Tone & Edge Cases Breakdown

The dataset spans authentic real-world communication styles:
1. **WhatsApp & Casual Colloquial (35%)**: Informal phrasing, polite conversational inquiries, common Indonesian shorthand (`sy`, `tdk`, `yg`, `blm`, `min`, `gan`).
2. **Emergency Panic & IGD (25%)**: Urgent situations, imperative grammar, high emotional valence, rapid triage necessity.
3. **Formal Corporate & Agent Communications (25%)**: Structured notification letters, policy number references, hospital discharge summaries.
4. **Clinical & Medical Technical Summaries (15%)**: Formal medical diagnosis codes and clinical observations.

### Edge Case Scenarios Included (224 samples, 34.5%)
- **Masa Tunggu (Waiting Period) Disputes**: Diagnosis made within 30 days of policy inception.
- **Pre-Existing Conditions Exclusion**: Prior chronic illnesses versus newly discovered symptoms.
- **Dual Symptomatology**: Simultaneous accident and pre-existing medical incident.
- **Excess Charges & Room Upgrades**: Voluntarily choosing VIP rooms above policy limit.
- **Missing / Incomplete Documentation**: Missing police reports for traffic collisions, lack of official hospital stamps.
- **Lapsed Policy & Autodebit Failures**: Premium payment delays due to banking channel errors.

---

## 4. Input Token Length Statistics

- **Minimum Word Count**: 19 words
- **Average Word Count**: 29.0 words
- **P95 Word Count**: 36 words
- **Maximum Word Count**: 49 words
- **VRAM Compatibility**: All samples fit within the 512 max-sequence-length limit, preventing OOM spikes on the 6GB VRAM RTX 3050 GPU.

---

## 5. Artifact Verification
The generated files are saved under `data/`:
- `data/alpha_train.jsonl` (520 records)
- `data/alpha_val.jsonl` (65 records)
- `data/alpha_test.jsonl` (65 records)
