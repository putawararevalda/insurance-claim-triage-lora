"""Experiment Tracking & System Metrics Monitoring for QLoRA Training.
Supports Weights & Biases (W&B), MLflow, and robust local offline fallbacks (CSV/JSONL).
"""

import csv
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger("experiment_tracking")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LocalExperimentLogger:
    """Thread-safe and crash-resilient local metrics logger for CSV and JSONL formats."""

    def __init__(
        self,
        csv_path: str = "logs/alpha_metrics.csv",
        jsonl_path: str = "logs/alpha_metrics.jsonl",
        summary_path: str = "logs/alpha_run_summary.json",
    ):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.summary_path = summary_path
        self.history: List[Dict[str, Any]] = []

        # Ensure directory exists
        for p in [csv_path, jsonl_path, summary_path]:
            os.makedirs(os.path.dirname(p), exist_ok=True)

        self.fieldnames = [
            "step",
            "epoch",
            "train_loss",
            "eval_loss",
            "eval_perplexity",
            "learning_rate",
            "grad_norm",
            "peak_vram_gb",
            "allocated_vram_gb",
            "samples_per_sec",
            "timestamp",
        ]

        # Initialize CSV header if file doesn't exist or is empty
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Appends metrics to both CSV and JSONL files."""
        self.history.append(metrics)

        # Write to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        # Write to CSV (filter to known fieldnames)
        csv_row = {k: metrics.get(k, "") for k in self.fieldnames}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(csv_row)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """Saves final run metadata and aggregate statistics to JSON."""
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


class SystemMetricsAndTrackingCallback(TrainerCallback):
    """Callback for tracking system metrics: peak VRAM, throughput, grad_norm, perplexity,

    and syncing with W&B, MLflow, and Local CSV/JSON loggers.
    """

    def __init__(
        self,
        local_logger: LocalExperimentLogger,
        tracker_type: str = "wandb",
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
    ):
        self.local_logger = local_logger
        self.tracker_type = tracker_type
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.step_start_time: Optional[float] = None
        self.last_step_duration: float = 0.0
        self.train_start_time: float = time.time()
        self.samples_per_step = batch_size * gradient_accumulation_steps

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.train_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info(f"SystemMetricsAndTrackingCallback started. Tracking to: {self.tracker_type} + Local CSV/JSONL")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.step_start_time is not None:
            self.last_step_duration = max(time.time() - self.step_start_time, 1e-5)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        epoch = round(state.epoch, 2) if state.epoch is not None else 0.0
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)
        lr = logs.get("learning_rate", None)
        grad_norm = logs.get("grad_norm", None)

        # Calculate Perplexity when eval_loss is present
        eval_perplexity = None
        if eval_loss is not None:
            try:
                eval_perplexity = round(math.exp(min(eval_loss, 20.0)), 4)
            except OverflowError:
                eval_perplexity = 99999.0

        # Hardware metrics
        peak_vram_gb = 0.0
        allocated_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            allocated_vram_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)

        # Throughput computation
        samples_per_sec = 0.0
        if self.last_step_duration > 0:
            samples_per_sec = round(self.samples_per_step / self.last_step_duration, 2)

        # Assemble structured entry
        metric_entry: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "train_loss": round(train_loss, 4) if train_loss is not None else "",
            "eval_loss": round(eval_loss, 4) if eval_loss is not None else "",
            "eval_perplexity": eval_perplexity if eval_perplexity is not None else "",
            "learning_rate": lr if lr is not None else "",
            "grad_norm": round(grad_norm, 4) if grad_norm is not None else "",
            "peak_vram_gb": peak_vram_gb,
            "allocated_vram_gb": allocated_vram_gb,
            "samples_per_sec": samples_per_sec,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 1. Log to local persistent storage (CSV / JSONL)
        self.local_logger.log_metrics(metric_entry)

        # 2. Sync to W&B if active
        if self.tracker_type in ("wandb", "all"):
            try:
                import wandb
                if wandb.run is not None:
                    wb_log = {
                        "train/step": step,
                        "train/epoch": epoch,
                        "system/peak_vram_gb": peak_vram_gb,
                        "system/allocated_vram_gb": allocated_vram_gb,
                        "system/throughput_samples_per_sec": samples_per_sec,
                    }
                    if train_loss is not None:
                        wb_log["train/loss"] = train_loss
                    if eval_loss is not None:
                        wb_log["eval/loss"] = eval_loss
                    if eval_perplexity is not None:
                        wb_log["eval/perplexity"] = eval_perplexity
                    if lr is not None:
                        wb_log["train/learning_rate"] = lr
                    if grad_norm is not None:
                        wb_log["train/grad_norm"] = grad_norm
                    wandb.log(wb_log, step=step)
            except Exception as e:
                logger.warning(f"Error logging to W&B: {e}")

        # 3. Sync to MLflow if active
        if self.tracker_type in ("mlflow", "all"):
            try:
                import mlflow
                if mlflow.active_run() is not None:
                    ml_log = {
                        "peak_vram_gb": peak_vram_gb,
                        "throughput_samples_per_sec": samples_per_sec,
                    }
                    if train_loss is not None:
                        ml_log["train_loss"] = train_loss
                    if eval_loss is not None:
                        ml_log["eval_loss"] = eval_loss
                    if eval_perplexity is not None:
                        ml_log["eval_perplexity"] = eval_perplexity
                    if lr is not None:
                        ml_log["learning_rate"] = lr
                    if grad_norm is not None:
                        ml_log["grad_norm"] = grad_norm
                    mlflow.log_metrics(ml_log, step=step)
            except Exception as e:
                logger.warning(f"Error logging to MLflow: {e}")

        # Console output
        if train_loss is not None:
            norm_str = f" | GradNorm: {grad_norm:.2f}" if grad_norm is not None else ""
            print(
                f"[Step {step:>3} | Ep {epoch:>4.2f}] Loss: {train_loss:.4f} | "
                f"Peak VRAM: {peak_vram_gb:.2f}GB | Throughput: {samples_per_sec:.1f} spl/s{norm_str}"
            )
        if eval_loss is not None:
            perp_str = f" | Perplexity: {eval_perplexity:.2f}" if eval_perplexity is not None else ""
            print(f"\n>>> [EVALUATION] Epoch {epoch:.2f} | Eval Loss: {eval_loss:.4f}{perp_str} (Peak VRAM: {peak_vram_gb:.2f}GB)\n")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        duration = time.time() - self.train_start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        summary = {
            "total_steps": state.global_step,
            "total_epochs": state.epoch,
            "train_duration_sec": round(duration, 2),
            "train_duration_min": round(duration / 60.0, 2),
            "peak_vram_gb": round(peak_vram, 3),
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
        }
        self.local_logger.save_summary(summary)
        logger.info(f"Training completed in {duration/60:.2f} min. Peak VRAM: {peak_vram:.2f} GB.")


class ValidationInferenceLoggerCallback(TrainerCallback):
    """Callback to periodically generate and log validation inference examples to W&B, MLflow,

    and a local JSONL artifact for qualitative model monitoring.
    """

    def __init__(
        self,
        val_samples: List[Dict[str, Any]],
        tokenizer: Any,
        tracker_type: str = "wandb",
        output_file: str = "logs/alpha_validation_samples.jsonl",
        max_new_tokens: int = 256,
    ):
        self.val_samples = val_samples
        self.tokenizer = tokenizer
        self.tracker_type = tracker_type
        self.output_file = output_file
        self.max_new_tokens = max_new_tokens
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Generates outputs for fixed validation samples and logs them to tracking dashboards."""
        if not self.val_samples or model is None:
            return

        step = state.global_step
        epoch = round(state.epoch, 2) if state.epoch is not None else 0.0
        print(f"\n--- [VAL INFERENCE LOGGER] Generating {len(self.val_samples)} test samples at step {step} (epoch {epoch}) ---")

        records = []
        is_training_mode = model.training
        model.eval()

        device = next(model.parameters()).device

        for i, sample in enumerate(self.val_samples):
            # Extract prompt messages
            messages = sample.get("messages", [])
            if not messages:
                # Fallback to instruction + input format
                system_content = sample.get("instruction", "Kamu adalah sistem AI FNOL Triage.")
                user_content = sample.get("input", "")
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
            else:
                # Keep system and user messages only for generation
                messages = [m for m in messages if m.get("role") in ("system", "user")]

            # Apply chat template
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = f"<system>\n{messages[0]['content']}\n<user>\n{messages[1]['content']}\n<assistant>\n"

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)

            with torch.inference_mode():
                try:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                    prompt_len = inputs["input_ids"].shape[1]
                    output_text = self.tokenizer.decode(
                        generated_ids[0][prompt_len:], skip_special_tokens=True
                    ).strip()
                except Exception as gen_err:
                    output_text = f"[ERROR GENERATING]: {gen_err}"

            # Check JSON schema validity
            is_valid_json = False
            parsed_category = "Unknown"
            try:
                parsed_json = json.loads(output_text)
                is_valid_json = True
                parsed_category = parsed_json.get("category", "Unknown")
            except Exception:
                # Try finding json snippet inside code block or braces
                if "{" in output_text and "}" in output_text:
                    try:
                        raw = output_text[output_text.find("{"):output_text.rfind("}") + 1]
                        parsed_json = json.loads(raw)
                        is_valid_json = True
                        parsed_category = parsed_json.get("category", "Unknown")
                    except Exception:
                        pass

            target_output = sample.get("output", "")
            target_category = sample.get("category", "")

            rec = {
                "step": step,
                "epoch": epoch,
                "sample_idx": i + 1,
                "input_snippet": sample.get("input", "")[:80] + "...",
                "target_category": target_category,
                "predicted_category": parsed_category,
                "is_valid_json": is_valid_json,
                "generated_text": output_text,
                "target_output": target_output,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            records.append(rec)

            print(f" Sample {i+1} | Target: {target_category} | Pred: {parsed_category} | Valid JSON: {is_valid_json}")

        # Restore model training mode
        if is_training_mode:
            model.train()

        # Append to local JSONL
        with open(self.output_file, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Log Table to W&B
        if self.tracker_type in ("wandb", "all"):
            try:
                import wandb
                if wandb.run is not None:
                    wb_table = wandb.Table(
                        columns=[
                            "Step",
                            "Epoch",
                            "Sample",
                            "Input Snippet",
                            "Target Category",
                            "Predicted Category",
                            "Valid JSON",
                            "Generated Response",
                        ]
                    )
                    for r in records:
                        wb_table.add_data(
                            r["step"],
                            r["epoch"],
                            r["sample_idx"],
                            r["input_snippet"],
                            r["target_category"],
                            r["predicted_category"],
                            r["is_valid_json"],
                            r["generated_text"],
                        )
                    wandb.log({f"eval/val_predictions_step_{step}": wb_table}, step=step)
            except Exception as e:
                logger.warning(f"Error logging validation table to W&B: {e}")

        # Log Table to MLflow
        if self.tracker_type in ("mlflow", "all"):
            try:
                import mlflow
                import pandas as pd
                if mlflow.active_run() is not None:
                    df = pd.DataFrame(records)
                    mlflow.log_table(data=df, artifact_file=f"eval_predictions_step_{step}.json")
            except Exception as e:
                logger.warning(f"Error logging validation table to MLflow: {e}")

        print("--- [VAL INFERENCE LOGGER] Finished logging sample predictions ---\n")


def setup_experiment_tracker(
    tracker_type: str = "wandb",
    project_name: str = "insurance-claim-triage-lora",
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    wandb_mode: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Initializes experiment tracker (W&B or MLflow) with automatic offline/local fallback.

    Returns dictionary containing active status and session information.
    """
    tracker_type = tracker_type.lower()
    run_name = run_name or f"alpha-qlora-{time.strftime('%Y%m%d-%H%M%S')}"
    tags = tags or ["alpha", "qlora", "qwen2.5-1.5b", "rtx3050"]
    config = config or {}

    status = {
        "tracker_type": tracker_type,
        "run_name": run_name,
        "wandb_active": False,
        "mlflow_active": False,
        "mode": "offline",
    }

    # 1. Weights & Biases Setup
    if tracker_type in ("wandb", "all"):
        try:
            import wandb

            # Check mode from param or env var
            mode = wandb_mode or os.environ.get("WANDB_MODE", None)
            if not mode:
                # If no WANDB_API_KEY is found, default cleanly to offline
                if not os.environ.get("WANDB_API_KEY"):
                    mode = "offline"
                    logger.info("WANDB_API_KEY not found in environment; setting WANDB_MODE='offline'.")
                else:
                    mode = "online"

            os.environ["WANDB_MODE"] = mode
            os.environ["WANDB_PROJECT"] = project_name

            wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                tags=tags,
                config=config,
                mode=mode,
            )
            status["wandb_active"] = True
            status["wandb_url"] = wandb_run.url if mode != "offline" else "local_offline"
            status["mode"] = mode
            logger.info(f"Initialized Weights & Biases (mode={mode}, project={project_name}, run={run_name})")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B tracker ({e}). Falling back to local offline logging.")
            status["wandb_active"] = False

    # 2. MLflow Setup
    if tracker_type in ("mlflow", "all"):
        try:
            import mlflow

            # Allow filesystem store if requested by user, or use SQLite
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(project_name)
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})
            status["mlflow_active"] = True
            status["mlflow_run_id"] = mlflow_run.info.run_id
            logger.info(f"Initialized MLflow tracker (URI={uri}, experiment={project_name}, run_id={mlflow_run.info.run_id})")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracker ({e}). Falling back to local offline logging.")
            status["mlflow_active"] = False

    if not status["wandb_active"] and not status["mlflow_active"]:
        logger.info(f"Tracking mode is purely LOCAL offline (CSV/JSONL).")

    return status


def finish_experiment_tracker():
    """Cleanly finishes active W&B and MLflow runs."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished cleanly.")
    except Exception:
        pass

    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()
            logger.info("MLflow run finished cleanly.")
    except Exception:
        pass
