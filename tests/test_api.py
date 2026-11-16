"""Unit and Integration Tests for FastAPI Triage Serving (Phase 3.2).

Validates endpoints, Pydantic validation, latency middleware,
OpenAPI docs, error handling, and thread-safe inference execution.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.serving.api import (
    app,
    API_VERSION,
    MODEL_VERSION,
    OFFICIAL_CATEGORIES,
    PRESET_SCENARIOS,
)
from src.service import clean_json_text, fallback_parse_triage_json


class TestTriageAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Verify API root endpoint returns metadata and overview."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("endpoints", data)
        self.assertEqual(data["version"], API_VERSION)
        self.assertEqual(data["model_version"], MODEL_VERSION)
        self.assertIn("triage", data["endpoints"])
        self.assertIn("categories", data["endpoints"])

    def test_health_endpoint(self):
        """Verify health check returns service and hardware readiness."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("cuda_available", data)
        self.assertIn("models_loaded", data)
        self.assertIn("timestamp", data)
        self.assertIn("device", data)

    def test_categories_endpoint(self):
        """Verify /v1/categories returns 5 official claim categories with criteria."""
        response = self.client.get("/v1/categories")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_categories", data)
        self.assertEqual(data["total_categories"], 5)
        self.assertEqual(len(data["categories"]), 5)

        expected_cats = {
            "Inpatient_Medical",
            "Outpatient_Emergency",
            "Critical_Illness",
            "Personal_Accident",
            "Inquiry_or_Dispute",
        }
        received_cats = {cat["category"] for cat in data["categories"]}
        self.assertEqual(received_cats, expected_cats)

        for cat in data["categories"]:
            self.assertIn("name_id", cat)
            self.assertIn("description", cat)
            self.assertIn("typical_urgency", cat)
            self.assertIn("typical_method", cat)
            self.assertIsInstance(cat["criteria"], list)
            self.assertGreaterEqual(len(cat["criteria"]), 2)
            self.assertIsInstance(cat["sample_conditions"], list)
            self.assertGreaterEqual(len(cat["sample_conditions"]), 2)

    def test_preset_scenarios_endpoint(self):
        """Verify /v1/scenarios returns preset scenarios for UI/testing."""
        response = self.client.get("/v1/scenarios")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), len(PRESET_SCENARIOS))
        self.assertEqual(data[0]["category"], "Critical_Illness")

    def test_latency_middleware_header(self):
        """Verify latency profiling middleware injects X-Process-Time-Ms header."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("x-process-time-ms", response.headers)
        process_time = float(response.headers["x-process-time-ms"])
        self.assertGreater(process_time, 0.0)

    def test_swagger_ui_and_openapi_docs(self):
        """Verify Swagger UI and OpenAPI JSON schemas are accessible."""
        docs_res = self.client.get("/docs")
        self.assertEqual(docs_res.status_code, 200)
        self.assertIn("swagger-ui", docs_res.text.lower())

        openapi_res = self.client.get("/openapi.json")
        self.assertEqual(openapi_res.status_code, 200)
        openapi_data = openapi_res.json()
        self.assertIn("/v1/triage", openapi_data["paths"])
        self.assertIn("/v1/categories", openapi_data["paths"])
        self.assertIn("/health", openapi_data["paths"])

    def test_triage_validation_error_on_empty_narrative(self):
        """Verify validation error when narrative is empty or too short."""
        response = self.client.post("/v1/triage", json={"narrative": "pendek"})
        self.assertEqual(response.status_code, 422)
        err = response.json()
        self.assertFalse(err.get("success", True))
        self.assertIn("detail", err)

        # Whitespace-only string
        response_ws = self.client.post("/v1/triage", json={"narrative": "          "})
        self.assertEqual(response_ws.status_code, 422)

    @patch("src.serving.api.TriageService.get_instance")
    def test_triage_successful_inference(self, mock_get_instance):
        """Verify POST /v1/triage returns validated ClaimTriageOutput and metadata."""
        mock_service = MagicMock()
        mock_service.run_inference.return_value = {
            "model_type": "lora",
            "model_name": "Qwen2.5-1.5B-Instruct + LoRA (Golden Adapter)",
            "narrative": "Pasien serangan jantung butuh klaim kritis.",
            "raw_output": '{"category":"Critical_Illness","urgency":"High","primary_condition":"STEMI Akut","claim_method":"Cashless","reasoning":"Darurat kardiovaskular."}',
            "cleaned_json": '{"category":"Critical_Illness","urgency":"High","primary_condition":"STEMI Akut","claim_method":"Cashless","reasoning":"Darurat kardiovaskular."}',
            "json_valid": True,
            "parse_error": None,
            "parsed": {
                "category": "Critical_Illness",
                "urgency": "High",
                "primary_condition": "STEMI Akut",
                "claim_method": "Cashless",
                "reasoning": "Darurat kardiovaskular.",
            },
            "latency_ms": 1120.5,
            "input_tokens": 85,
            "generated_tokens": 58,
            "tokens_per_second": 51.8,
            "device": "cuda:0",
        }
        mock_get_instance.return_value = mock_service

        payload = {
            "narrative": "Pasien serangan jantung butuh klaim kritis darurat RS Jantung.",
            "model_type": "lora",
            "max_new_tokens": 150,
        }
        response = self.client.post("/v1/triage", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["json_valid"])
        self.assertEqual(data["model_type"], "lora")
        self.assertEqual(data["model_version"], MODEL_VERSION)
        self.assertIn("timestamp", data)
        self.assertLess(data["latency_ms"], 1500)
        self.assertEqual(data["data"]["category"], "Critical_Illness")
        self.assertEqual(data["data"]["urgency"], "High")
        self.assertEqual(data["data"]["primary_condition"], "STEMI Akut")
        self.assertEqual(data["data"]["claim_method"], "Cashless")

    @patch("src.serving.api.TriageService.get_instance")
    def test_compare_endpoint(self, mock_get_instance):
        """Verify POST /v1/compare returns both LoRA and Base outputs."""
        mock_service = MagicMock()
        mock_service.run_inference.side_effect = [
            # Call 1: lora
            {
                "model_type": "lora",
                "model_name": "LoRA Model",
                "json_valid": True,
                "parsed": {
                    "category": "Personal_Accident",
                    "urgency": "Medium",
                    "primary_condition": "Fraktur Lengan",
                    "claim_method": "Cashless",
                    "reasoning": "Trauma jatuh.",
                },
                "raw_output": '{"category":"Personal_Accident","urgency":"Medium","primary_condition":"Fraktur Lengan","claim_method":"Cashless","reasoning":"Trauma jatuh."}',
                "cleaned_json": '{"category":"Personal_Accident","urgency":"Medium","primary_condition":"Fraktur Lengan","claim_method":"Cashless","reasoning":"Trauma jatuh."}',
                "latency_ms": 1050.0,
                "input_tokens": 70,
                "generated_tokens": 50,
                "tokens_per_second": 47.6,
                "device": "cuda:0",
                "parse_error": None,
            },
            # Call 2: base
            {
                "model_type": "base",
                "model_name": "Base Model",
                "json_valid": False,
                "parsed": None,
                "raw_output": "Berikut adalah analisa klaim tanpa format JSON...",
                "cleaned_json": "Berikut adalah analisa klaim tanpa format JSON...",
                "latency_ms": 2800.0,
                "input_tokens": 70,
                "generated_tokens": 90,
                "tokens_per_second": 32.1,
                "device": "cuda:0",
                "parse_error": "JSONDecodeError",
            },
        ]
        mock_get_instance.return_value = mock_service

        payload = {
            "narrative": "Saya terjatuh dari tangga saat renovasi rumah dan patah tulang.",
            "max_new_tokens": 150,
        }
        response = self.client.post("/v1/compare", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["lora"]["json_valid"])
        self.assertFalse(data["base"]["json_valid"])
        self.assertEqual(data["lora"]["data"]["category"], "Personal_Accident")

    def test_json_cleaner_and_fallback_parser(self):
        """Test markdown code fence stripping and fallback regex JSON parser."""
        markdown_json = "```json\n{\"category\": \"Inpatient_Medical\", \"urgency\": \"High\"}\n```"
        cleaned = clean_json_text(markdown_json)
        self.assertEqual(cleaned, '{"category": "Inpatient_Medical", "urgency": "High"}')

        # Trailing comma handling
        trailing_comma = '{"category": "Critical_Illness", "urgency": "High",}'
        fallback_res = fallback_parse_triage_json(trailing_comma)
        self.assertIsNotNone(fallback_res)
        self.assertEqual(fallback_res["category"], "Critical_Illness")

        # Regex fallback on unclosed JSON
        broken_json = 'Some text {"category": "Personal_Accident", "urgency": "Medium", "primary_condition": "Patah kaki", "claim_method": "Cashless", "reasoning": "Jatuh" trailing text'
        fallback_res2 = fallback_parse_triage_json(broken_json)
        self.assertIsNotNone(fallback_res2)
        self.assertEqual(fallback_res2["category"], "Personal_Accident")


if __name__ == "__main__":
    unittest.main()
