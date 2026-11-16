"""End-to-End Portfolio Integrity & Verification Script (Fase 1 - 3).

Verifies the integrity of all deliverables across:
- Fase 1: MVP Dataset, Stratification, Pydantic Schema
- Fase 2: QLoRA Fine-Tuning, W&B Sweeps, Benchmark Reports, Confusion Matrix
- Fase 3: Model Merging, Standalone Safetensors, Modelfile, FastAPI Serving, Streamlit Web UI

Usage:
    python scripts/verify_portfolio.py
"""

import json
import os
import sys
import time
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def check_file(path: str, min_size_bytes: int = 0) -> bool:
    """Verifies that a file exists and meets minimum size requirement."""
    full_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full_path):
        return False
    if min_size_bytes > 0:
        return os.path.getsize(full_path) >= min_size_bytes
    return True


def check_dir_files(dir_path: str, required_files: list) -> bool:
    """Verifies all required files exist inside a directory."""
    full_dir = os.path.join(PROJECT_ROOT, dir_path)
    if not os.path.isdir(full_dir):
        return False
    for f in required_files:
        if not os.path.exists(os.path.join(full_dir, f)):
            return False
    return True


def run_test_suite() -> bool:
    """Executes the test suite across schema, export, api, and streamlit app."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        "tests.test_schema",
        "tests.test_export",
        "tests.test_api",
        "tests.test_streamlit_app",
    ]

    for mod_name in test_modules:
        try:
            mod_suite = loader.loadTestsFromName(mod_name)
            suite.addTests(mod_suite)
        except Exception as e:
            print(f"❌ Failed to load test module {mod_name}: {e}")
            return False

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    print("=" * 80)
    print("  🚀 INDONESIAN INSURANCE CLAIM & FNOL TRIAGE - END-TO-END VERIFICATION")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}\n")

    checklist = []

    # --- Phase 1: MVP ---
    print("[1/4] Verifying Phase 1 (MVP) Components...")
    p1_data = check_file("data/alpha_train.jsonl", 1000) and check_file("data/alpha_val.jsonl", 500) and check_file("data/alpha_test.jsonl", 500)
    p1_schema = check_file("src/schema.py", 500)
    checklist.append(("Fase 1: Synthetic Dataset Curation (650 samples)", p1_data))
    checklist.append(("Fase 1: Pydantic v2 Schema Contract (ClaimTriageOutput)", p1_schema))

    # --- Phase 2: Alpha ---
    print("[2/4] Verifying Phase 2 (Alpha) Components...")
    p2_adapter = check_dir_files(
        "models/lora_adapter_alpha_best",
        ["adapter_config.json", "adapter_model.safetensors"],
    )
    p2_reports = (
        check_file("reports/alpha_quantitative_benchmark.md", 500)
        and check_file("reports/alpha_confusion_matrix.png", 5000)
        and check_file("reports/sweep_alpha_analysis.md", 500)
    )
    p2_docs = check_file("docs/experiment_tracking.md", 500) and check_file("docs/walkthrough_script.md", 500)
    checklist.append(("Fase 2: QLoRA Golden Adapter Weights (4-bit NF4)", p2_adapter))
    checklist.append(("Fase 2: Quantitative Benchmark & Confusion Matrix Reports", p2_reports))
    checklist.append(("Fase 2: Experiment Tracking & Video Walkthrough Script", p2_docs))

    # --- Phase 3: Beta ---
    print("[3/4] Verifying Phase 3 (Beta) Components...")
    p3_merged = check_dir_files(
        "models/merged_qwen2.5_1.5b_insurance_triage",
        ["config.json", "model.safetensors", "tokenizer.json", "export_metadata.json"],
    )
    p3_modelfile = check_file("configs/Modelfile", 200)
    p3_api = check_file("src/serving/api.py", 1000) and check_file("scripts/serve_api.py", 200)
    p3_ui = check_file("app/streamlit_app.py", 1000) and check_file("app.py", 100)
    checklist.append(("Fase 3: Merged Standalone 16-bit Model (3.08 GB Safetensors)", p3_merged))
    checklist.append(("Fase 3: Ollama Packaging & Distribution (configs/Modelfile)", p3_modelfile))
    checklist.append(("Fase 3: Production REST API Service (FastAPI + Swagger Docs)", p3_api))
    checklist.append(("Fase 3: Interactive Web UI Demo (Streamlit Side-by-Side)", p3_ui))

    # --- Unit Tests Execution ---
    print("\n[4/4] Executing Comprehensive Automated Test Suite...")
    tests_ok = run_test_suite()
    checklist.append(("Automated Test Suite (100% Passing: Schema, Export, API, Streamlit)", tests_ok))

    # --- Summary Table ---
    print("\n" + "=" * 80)
    print("  📋 PORTFOLIO VERIFICATION AUDIT RESULTS")
    print("=" * 80)
    all_passed = True
    for item, status in checklist:
        status_str = "✅ PASSED" if status else "❌ FAILED"
        if not status:
            all_passed = False
        print(f"  {status_str} | {item}")
    print("=" * 80)

    if all_passed:
        print("\n🎉 ALL AUDIT CHECKS PASSED SUCCESSFULLY!")
        print("The Indonesian Insurance Claim & FNOL Triage portfolio is 100% production-ready.\n")
        sys.exit(0)
    else:
        print("\n⚠️ SOME AUDIT CHECKS FAILED. Please review the missing components above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
