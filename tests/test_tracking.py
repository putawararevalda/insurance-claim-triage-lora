"""Unit tests for Experiment Tracking & Metrics Logging."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.tracking import (
    LocalExperimentLogger,
    SystemMetricsAndTrackingCallback,
    ValidationInferenceLoggerCallback,
    finish_experiment_tracker,
    setup_experiment_tracker,
)


class TestExperimentTracking(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.test_dir, "test_metrics.csv")
        self.jsonl_path = os.path.join(self.test_dir, "test_metrics.jsonl")
        self.summary_path = os.path.join(self.test_dir, "test_summary.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_local_logger_logging_and_summary(self):
        logger = LocalExperimentLogger(
            csv_path=self.csv_path,
            jsonl_path=self.jsonl_path,
            summary_path=self.summary_path,
        )

        entry = {
            "step": 5,
            "epoch": 0.5,
            "train_loss": 1.2345,
            "eval_loss": 1.1234,
            "eval_perplexity": 3.0753,
            "learning_rate": 0.0002,
            "grad_norm": 0.45,
            "peak_vram_gb": 3.85,
            "allocated_vram_gb": 3.20,
            "samples_per_sec": 12.5,
            "timestamp": "2026-10-28 22:00:00",
        }
        logger.log_metrics(entry)

        # Verify CSV
        self.assertTrue(os.path.exists(self.csv_path))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 row
            self.assertIn("1.2345", lines[1])

        # Verify JSONL
        self.assertTrue(os.path.exists(self.jsonl_path))
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())
            self.assertEqual(record["step"], 5)
            self.assertEqual(record["eval_perplexity"], 3.0753)

        # Verify Summary
        summary = {"total_steps": 10, "peak_vram_gb": 4.12, "best_metric": 1.05}
        logger.save_summary(summary)
        self.assertTrue(os.path.exists(self.summary_path))
        with open(self.summary_path, "r", encoding="utf-8") as f:
            saved_sum = json.load(f)
            self.assertEqual(saved_sum["peak_vram_gb"], 4.12)

    def test_system_callback_on_log(self):
        local_logger = LocalExperimentLogger(
            csv_path=self.csv_path,
            jsonl_path=self.jsonl_path,
            summary_path=self.summary_path,
        )
        callback = SystemMetricsAndTrackingCallback(
            local_logger=local_logger,
            tracker_type="none",
            batch_size=2,
            gradient_accumulation_steps=4,
        )

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.epoch = 1.0
        mock_control = MagicMock()

        callback.last_step_duration = 0.5
        logs = {"loss": 0.85, "eval_loss": 0.72, "learning_rate": 1e-4, "grad_norm": 0.35}

        callback.on_log(mock_args, mock_state, mock_control, logs=logs)

        # Check that metrics were logged to local logger history
        self.assertEqual(len(local_logger.history), 1)
        logged = local_logger.history[0]
        self.assertEqual(logged["step"], 10)
        self.assertEqual(logged["train_loss"], 0.85)
        self.assertEqual(logged["eval_loss"], 0.72)
        self.assertAlmostEqual(logged["eval_perplexity"], 2.0544, places=2)
        self.assertEqual(logged["samples_per_sec"], 16.0)  # (2 * 4) / 0.5 = 16.0

    def test_setup_experiment_tracker_offline(self):
        status = setup_experiment_tracker(
            tracker_type="wandb",
            project_name="test-project",
            wandb_mode="offline",
        )
        self.assertTrue(status["wandb_active"])
        self.assertEqual(status["mode"], "offline")
        finish_experiment_tracker()

    def test_setup_experiment_tracker_mlflow(self):
        mlruns_dir = os.path.join(self.test_dir, "mlruns")
        status = setup_experiment_tracker(
            tracker_type="mlflow",
            project_name="test-insurance-mlflow",
            mlflow_tracking_uri=f"file:///{mlruns_dir.replace('\\', '/')}",
        )
        self.assertTrue(status["mlflow_active"])
        finish_experiment_tracker()


if __name__ == "__main__":
    unittest.main()
