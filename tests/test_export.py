"""Unit tests for Model Export & Modelfile Packaging (Phase 3.1)."""

import os
import unittest
import torch
from scripts.export_model import get_torch_dtype


class TestModelExport(unittest.TestCase):
    def test_get_torch_dtype_mapping(self):
        self.assertEqual(get_torch_dtype("float16"), torch.float16)
        self.assertEqual(get_torch_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(get_torch_dtype("float32"), torch.float32)
        # Default fallback
        self.assertEqual(get_torch_dtype("unknown"), torch.float16)

    def test_modelfile_structure(self):
        modelfile_path = "configs/Modelfile"
        self.assertTrue(os.path.exists(modelfile_path), f"{modelfile_path} does not exist!")
        with open(modelfile_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("FROM", content)
        self.assertIn("PARAMETER temperature 0.1", content)
        self.assertIn("PARAMETER top_p 0.9", content)
        self.assertIn("SYSTEM", content)
        self.assertIn("ClaimTriageOutput", content)
        self.assertIn("Inpatient_Medical", content)
        self.assertIn("Outpatient_Emergency", content)
        self.assertIn("Critical_Illness", content)
        self.assertIn("Personal_Accident", content)
        self.assertIn("Inquiry_or_Dispute", content)

    def test_merged_model_directory_artifacts(self):
        merged_dir = "models/merged_qwen2.5_1.5b_insurance_triage"
        if os.path.exists(merged_dir):
            required_files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "export_metadata.json",
            ]
            for fname in required_files:
                fpath = os.path.join(merged_dir, fname)
                self.assertTrue(os.path.exists(fpath), f"Missing {fname} in {merged_dir}")

            # Verify metadata
            import json
            with open(os.path.join(merged_dir, "export_metadata.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["base_model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
            self.assertTrue(meta["verification"]["verified"])



if __name__ == "__main__":
    unittest.main()
