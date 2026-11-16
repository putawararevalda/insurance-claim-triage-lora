"""Unit tests for Phase 2.3 Hyperparameter Sweep Configuration and Selection Logic."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from scripts.run_sweep_alpha import (
    export_golden_adapter,
    generate_analysis_report,
    load_sweep_config,
)


class TestSweepPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_model_dir = os.path.join(self.test_dir, "mock_source_model")
        os.makedirs(self.mock_model_dir, exist_ok=True)

        # Create dummy adapter files
        with open(os.path.join(self.mock_model_dir, "adapter_model.safetensors"), "wb") as f:
            f.write(b"dummy_weights_data")
        with open(os.path.join(self.mock_model_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump({"r": 16, "lora_alpha": 32}, f)
        with open(os.path.join(self.mock_model_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "mock"}, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sweep_config_structure(self):
        config_path = "configs/sweep_config.yaml"
        self.assertTrue(os.path.exists(config_path), f"Missing config file: {config_path}")
        
        config = load_sweep_config(config_path)
        self.assertIn("experiment_name", config)
        self.assertIn("sweep_grid", config)
        self.assertIn("runs", config)

        grid = config["sweep_grid"]
        self.assertIn("lora_rank_alpha_pairs", grid)
        self.assertIn("learning_rates", grid)
        self.assertIn("target_modules_candidates", grid)

        runs = config["runs"]
        self.assertGreaterEqual(len(runs), 3, "Sweep grid must contain at least 3-4 runs.")
        for r in runs:
            self.assertIn("id", r)
            self.assertIn("lora_r", r)
            self.assertIn("lora_alpha", r)
            self.assertIn("lr", r)
            self.assertIn("target_modules", r)

    def test_export_golden_adapter(self):
        dest_dir = os.path.join(self.test_dir, "golden_dest")
        best_run = {
            "run_id": "test_best_run",
            "name": "Test Best Run",
            "output_dir": self.mock_model_dir,
            "lora_r": 16,
            "lora_alpha": 32,
            "lr": 2e-4,
            "target_modules": "attention",
            "best_eval_loss": 0.4521,
            "peak_vram_gb": 2.85,
        }

        export_golden_adapter(best_run, destination_dir=dest_dir)

        # Verify exported files
        self.assertTrue(os.path.exists(os.path.join(dest_dir, "adapter_model.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(dest_dir, "adapter_config.json")))
        self.assertTrue(os.path.exists(os.path.join(dest_dir, "golden_adapter_metadata.json")))

    def test_generate_analysis_report(self):
        report_path = os.path.join(self.test_dir, "test_report.md")
        results = [
            {
                "run_id": "run_1",
                "name": "Run 1",
                "lora_r": 8,
                "lora_alpha": 16,
                "lr": 3e-4,
                "target_modules": "attention",
                "initial_train_loss": 3.12,
                "final_train_loss": 0.54,
                "best_eval_loss": 0.62,
                "best_eval_perplexity": 1.85,
                "peak_vram_gb": 2.45,
                "train_duration_min": 3.2,
            },
            {
                "run_id": "run_2",
                "name": "Run 2",
                "lora_r": 16,
                "lora_alpha": 32,
                "lr": 2e-4,
                "target_modules": "all-linear",
                "initial_train_loss": 3.05,
                "final_train_loss": 0.38,
                "best_eval_loss": 0.45,
                "best_eval_perplexity": 1.56,
                "peak_vram_gb": 2.95,
                "train_duration_min": 3.8,
            },
        ]
        best_run = results[1]

        generate_analysis_report(results, best_run, report_path=report_path)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Golden Adapter Selection Rationale", content)
            self.assertIn("run_2", content)
            self.assertIn("2.95 GB", content)


if __name__ == "__main__":
    unittest.main()
