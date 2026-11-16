"""Unit tests for Streamlit Web UI Demo components and helper functions."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.streamlit_app import (
    check_fastapi_health,
    get_category_badge,
    get_schema_badge,
    get_urgency_badge,
    run_inference_via_api,
)


class TestStreamlitApp(unittest.TestCase):
    def test_category_badge_rendering(self):
        """Verify HTML category badges produce expected classes and labels."""
        b_inpatient = get_category_badge("Inpatient_Medical")
        self.assertIn("badge-inpatient", b_inpatient)
        self.assertIn("Rawat Inap Medis", b_inpatient)

        b_critical = get_category_badge("Critical_Illness")
        self.assertIn("badge-critical", b_critical)
        self.assertIn("Penyakit Kritis", b_critical)

        b_accident = get_category_badge("Personal_Accident")
        self.assertIn("badge-accident", b_accident)
        self.assertIn("Kecelakaan Diri", b_accident)

        b_outpatient = get_category_badge("Outpatient_Emergency")
        self.assertIn("badge-outpatient", b_outpatient)
        self.assertIn("Rawat Jalan Darurat", b_outpatient)

        b_inquiry = get_category_badge("Inquiry_or_Dispute")
        self.assertIn("badge-inquiry", b_inquiry)
        self.assertIn("Pertanyaan & Sengketa", b_inquiry)

    def test_urgency_badge_rendering(self):
        """Verify urgency badge color coding and text."""
        high = get_urgency_badge("High")
        self.assertIn("badge-urg-high", high)
        self.assertIn("HIGH URGENCY", high)

        med = get_urgency_badge("Medium")
        self.assertIn("badge-urg-med", med)
        self.assertIn("MEDIUM URGENCY", med)

        low = get_urgency_badge("Low")
        self.assertIn("badge-urg-low", low)
        self.assertIn("LOW URGENCY", low)

    def test_schema_badge_rendering(self):
        """Verify schema badge for valid and invalid outputs."""
        valid_badge = get_schema_badge(True)
        self.assertIn("badge-valid", valid_badge)
        self.assertIn("100% VALID PYDANTIC SCHEMA", valid_badge)

        invalid_badge = get_schema_badge(False)
        self.assertIn("badge-invalid", invalid_badge)
        self.assertIn("INVALID JSON", invalid_badge)

    def test_check_fastapi_health_mock(self):
        """Verify healthcheck function handles online and offline responses gracefully."""
        with patch("requests.get") as mock_get:
            # Mock online response
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "healthy", "device": "cuda:0"}
            mock_get.return_value = mock_resp

            is_online, data = check_fastapi_health("http://localhost:8000")
            self.assertTrue(is_online)
            self.assertEqual(data["status"], "healthy")

            # Mock exception / offline
            mock_get.side_effect = Exception("Connection refused")
            is_online, data = check_fastapi_health("http://localhost:8000")
            self.assertFalse(is_online)
            self.assertIsNone(data)

    def test_run_inference_via_api_mock(self):
        """Verify API client payload construction and response handling."""
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "success": True,
                "json_valid": True,
                "latency_ms": 780.5,
                "data": {
                    "category": "Inpatient_Medical",
                    "urgency": "High",
                    "primary_condition": "Demam Berdarah Dengue",
                    "claim_method": "Cashless",
                    "reasoning": "Opname rumah sakit dengan trombositopenia.",
                },
            }
            mock_post.return_value = mock_resp

            res = run_inference_via_api(
                api_url="http://localhost:8000",
                narrative="Anak saya dirawat inap karena demam berdarah trombosit 45.000.",
                model_type="lora",
                max_new_tokens=150,
            )

            self.assertTrue(res["success"])
            self.assertEqual(res["data"]["category"], "Inpatient_Medical")
            mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
