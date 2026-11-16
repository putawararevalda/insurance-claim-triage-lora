"""Unit tests for Benchmark Harness & Evaluation Logic."""

import unittest
import numpy as np
from scripts.eval_benchmark import (
    extract_json_from_text,
    evaluate_domain_prediction,
    evaluate_control_sample,
    compute_aggregate_metrics,
    CATEGORIES,
)


class TestEvalBenchmark(unittest.TestCase):
    def test_extract_json_direct(self):
        sample = '{"category": "Critical_Illness", "urgency": "High", "primary_condition": "Stroke", "claim_method": "Cashless", "reasoning": "Kondisi darurat."}'
        is_json, obj = extract_json_from_text(sample)
        self.assertTrue(is_json)
        self.assertEqual(obj["category"], "Critical_Illness")

    def test_extract_json_codeblock(self):
        sample = 'Here is the result:\n```json\n{"category": "Personal_Accident", "urgency": "Medium", "primary_condition": "Fraktur", "claim_method": "Reimbursement", "reasoning": "Kecelakaan."}\n```'
        is_json, obj = extract_json_from_text(sample)
        self.assertTrue(is_json)
        self.assertEqual(obj["category"], "Personal_Accident")

    def test_extract_json_surrounding_text(self):
        sample = 'Analisis kami: {"category": "Inquiry_or_Dispute", "urgency": "Low", "primary_condition": "Status klaim", "claim_method": "Unknown", "reasoning": "Pertanyaan."} selesai.'
        is_json, obj = extract_json_from_text(sample)
        self.assertTrue(is_json)
        self.assertEqual(obj["category"], "Inquiry_or_Dispute")

    def test_evaluate_domain_prediction_valid(self):
        raw = '{"category": "Inpatient_Medical", "urgency": "High", "primary_condition": "DBD", "claim_method": "Cashless", "reasoning": "Opname segera."}'
        gt = {
            "category": "Inpatient_Medical",
            "urgency": "High",
            "claim_method": "Cashless",
        }
        res = evaluate_domain_prediction(raw, gt)
        self.assertTrue(res["is_json"])
        self.assertTrue(res["is_pydantic_valid"])
        self.assertTrue(res["category_match"])
        self.assertTrue(res["urgency_match"])
        self.assertTrue(res["method_match"])

    def test_evaluate_domain_prediction_invalid(self):
        raw = "Bukan JSON sama sekali"
        gt = {
            "category": "Inpatient_Medical",
            "urgency": "High",
            "claim_method": "Cashless",
        }
        res = evaluate_domain_prediction(raw, gt)
        self.assertFalse(res["is_json"])
        self.assertFalse(res["is_pydantic_valid"])
        self.assertFalse(res["category_match"])

    def test_evaluate_control_sample(self):
        sample = {
            "id": "math_01",
            "domain": "reasoning_math",
            "expected_keywords": ["400.000", "44.000", "444.000"],
        }
        # Good response
        out_good = "Diskon 20% dari 500.000 adalah 100.000, sisa 400.000. PPN 11% adalah 44.000. Total bayar 444.000."
        eval_good = evaluate_control_sample(out_good, sample)
        self.assertTrue(eval_good["is_coherent"])
        self.assertFalse(eval_good["domain_leakage"])
        self.assertGreaterEqual(eval_good["score"], 0.8)

        # Domain leakage response
        out_leak = '{"category": "Critical_Illness", "urgency": "High", "claim_method": "Cashless", "primary_condition": "none"}'
        eval_leak = evaluate_control_sample(out_leak, sample)
        self.assertTrue(eval_leak["domain_leakage"])
        self.assertEqual(eval_leak["score"], 0.1)

    def test_compute_aggregate_metrics(self):
        test_samples = [
            {"category": "Inpatient_Medical", "urgency": "High", "claim_method": "Cashless"},
            {"category": "Critical_Illness", "urgency": "Medium", "claim_method": "Reimbursement"},
        ]
        results = [
            {
                "predicted_category": "Inpatient_Medical",
                "is_pydantic_valid": True,
                "urgency_match": True,
                "method_match": True,
                "latency_ms": 100.0,
            },
            {
                "predicted_category": "Critical_Illness",
                "is_pydantic_valid": True,
                "urgency_match": True,
                "method_match": True,
                "latency_ms": 120.0,
            },
        ]
        metrics = compute_aggregate_metrics(results, test_samples)
        self.assertEqual(metrics["category_accuracy"], 100.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["pydantic_valid_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
