"""Unit tests for ClaimTriageOutput Pydantic schema."""

import unittest
from pydantic import ValidationError
from src.schema import ClaimCategory, UrgencyLevel, ClaimMethod, ClaimTriageOutput


class TestClaimSchema(unittest.TestCase):
    def test_valid_payload(self):
        data = {
            "category": "Inpatient_Medical",
            "urgency": "High",
            "primary_condition": "Demam Berdarah Dengue (DBD)",
            "claim_method": "Cashless",
            "reasoning": "Pasien membutuhkan opname segera karena DBD."
        }
        obj = ClaimTriageOutput(**data)
        self.assertEqual(obj.category, ClaimCategory.INPATIENT_MEDICAL)
        self.assertEqual(obj.urgency, UrgencyLevel.HIGH)
        self.assertEqual(obj.primary_condition, "Demam Berdarah Dengue (DBD)")
        self.assertEqual(obj.claim_method, ClaimMethod.CASHLESS)

    def test_domain_normalization(self):
        # Test fuzzy normalization for Indonesian terminology
        data = {
            "category": "rawat_inap opname",
            "urgency": "darurat",
            "primary_condition": "Pneumonia akut",
            "claim_method": "kartu cashless",
            "reasoning": "Membutuhkan observasi rawat inap segera."
        }
        obj = ClaimTriageOutput(**data)
        self.assertEqual(obj.category, ClaimCategory.INPATIENT_MEDICAL)
        self.assertEqual(obj.urgency, UrgencyLevel.HIGH)
        self.assertEqual(obj.claim_method, ClaimMethod.CASHLESS)

    def test_all_categories_supported(self):
        expected_categories = [
            ClaimCategory.INPATIENT_MEDICAL,
            ClaimCategory.OUTPATIENT_EMERGENCY,
            ClaimCategory.CRITICAL_ILLNESS,
            ClaimCategory.PERSONAL_ACCIDENT,
            ClaimCategory.INQUIRY_OR_DISPUTE,
        ]
        self.assertEqual(len(ClaimCategory), 5)
        for cat in expected_categories:
            self.assertIn(cat, ClaimCategory)


if __name__ == "__main__":
    unittest.main()
