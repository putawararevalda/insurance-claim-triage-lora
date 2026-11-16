"""Phase 2 Alpha: 4-bit QLoRA Fine-Tuning Pipeline with Centralized Experiment Tracking.
Integrates Weights & Biases (W&B) and MLflow alongside local CSV/JSON logging,
system metrics monitoring (VRAM, throughput, grad_norm, perplexity), and periodic
validation inference logging. Optimized for NVIDIA RTX 3050 (6GB VRAM).
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking import (
    LocalExperimentLogger,
    SystemMetricsAndTrackingCallback,
    ValidationInferenceLoggerCallback,
    finish_experiment_tracker,
    setup_experiment_tracker,
)

logger = logging.getLogger("train_alpha")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Alpha QLoRA Training with Experiment Tracking (W&B / MLflow).")
    
    # Model & Data
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--train_file", type=str, default="data/alpha_train.jsonl", help="Path to training jsonl")
    parser.add_argument("--val_file", type=str, default="data/alpha_val.jsonl", help="Path to validation jsonl")
    parser.add_argument("--output_dir", type=str, default="models/lora_adapter_alpha", help="Path to export best adapter")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/alpha", help="Path to store intermediate checkpoints")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="attention",
        help="Target modules for LoRA: 'attention' (q,k,v,o), 'all-linear' (q,k,v,o,gate,up,down), or comma-separated list",
    )
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    
    # Experiment Tracking
    parser.add_argument(
        "--tracker",
        type=str,
        default=os.environ.get("TRACKER", "wandb"),
        choices=["wandb", "mlflow", "none", "all"],
        help="Experiment tracking backend",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "insurance-claim-triage-lora"),
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=os.environ.get("WANDB_MODE", None),
        help="W&B mode: online, offline, or disabled",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom run name")
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "insurance-claim-triage-lora"),
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI",
    )
    
    # Custom Logging File Paths
    parser.add_argument("--metrics_csv", type=str, default=None, help="Path to metrics CSV file")
    parser.add_argument("--metrics_jsonl", type=str, default=None, help="Path to metrics JSONL file")
    parser.add_argument("--run_summary_json", type=str, default=None, help="Path to run summary JSON file")
    parser.add_argument("--val_samples_jsonl", type=str, default=None, help="Path to val samples JSONL file")
    
    # Logging & Cadence
    parser.add_argument("--logging_steps", type=int, default=5, help="Logging step interval")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--eval_steps", type=int, default=20, help="Evaluation step interval if eval_strategy='steps'")
    parser.add_argument("--num_val_samples", type=int, default=4, help="Number of validation samples to evaluate periodically")
    
    # Smoke Test / Dry Run
    parser.add_argument("--smoke_test", action="store_true", help="Quick test run with small dataset and 5 steps to verify tracking")
    parser.add_argument("--max_steps", type=int, default=-1, help="Force training to stop after N steps (-1 for full epochs)")
    
    return parser.parse_args(args)


def load_fixed_val_samples(val_file: str, n_samples: int = 4) -> List[Dict[str, Any]]:
    """Loads a diverse selection of validation samples for periodic inference tracking."""
    samples = []
    if not os.path.exists(val_file):
        return samples

    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    samples.append(data)
                except Exception:
                    pass

    # Select representative samples with diverse categories if possible
    selected = []
    categories_seen = set()
    for s in samples:
        cat = s.get("category", "")
        if cat not in categories_seen:
            selected.append(s)
            categories_seen.add(cat)
        if len(selected) >= n_samples:
            break

    # If diverse selection is smaller than n_samples, pad with remaining
    if len(selected) < n_samples:
        for s in samples:
            if s not in selected:
                selected.append(s)
            if len(selected) >= n_samples:
                break

    return selected


def train_alpha(cli_args=None):
    args = parse_args(cli_args)
    
    print("=" * 70)
    print("  PHASE 2 ALPHA: QLORA TRAINING WITH CENTRALIZED EXPERIMENT TRACKING")
    print("=" * 70)
    print(f"Model ID           : {args.model_id}")
    print(f"Tracker Selected   : {args.tracker}")
    print(f"LoRA Config        : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Target Modules     : {args.target_modules}")
    print(f"Batch Size (Eff.)  : {args.batch_size} * {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Output Directory   : {args.output_dir}")
    print(f"Checkpoint Dir     : {args.checkpoint_dir}")
    print(f"Smoke Test Mode    : {args.smoke_test}")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU fine-tuning! No GPU detected.")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # 1. Setup Tracking Config and Initialize Tracker (W&B / MLflow / Fallback)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run_name or f"alpha-qlora-r{args.lora_r}-a{args.lora_alpha}_{run_timestamp}"
    
    config_metadata = {
        "model_id": args.model_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "max_length": args.max_length,
        "quantization": "4-bit NF4 double quant",
        "gpu_device": torch.cuda.get_device_name(0),
        "smoke_test": args.smoke_test,
    }

    tracker_status = setup_experiment_tracker(
        tracker_type=args.tracker,
        project_name=args.wandb_project,
        run_name=run_name,
        tags=["alpha", "qlora", "qwen2.5-1.5b", "insurance-triage"],
        config=config_metadata,
        wandb_mode=args.wandb_mode,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )

    # 2. Setup Local CSV / JSONL Logger
    local_csv_path = args.metrics_csv or "logs/alpha_metrics.csv"
    local_jsonl_path = args.metrics_jsonl or "logs/alpha_metrics.jsonl"
    local_summary_path = args.run_summary_json or "logs/alpha_run_summary.json"
    local_samples_path = args.val_samples_jsonl or "logs/alpha_validation_samples.jsonl"
    local_logger = LocalExperimentLogger(
        csv_path=local_csv_path,
        jsonl_path=local_jsonl_path,
        summary_path=local_summary_path,
    )

    # 3. Load Datasets
    print("\nLoading dataset from disk...")
    data_files = {
        "train": args.train_file,
        "val": args.val_file,
    }
    dataset = load_dataset("json", data_files=data_files)
    
    if args.smoke_test:
        print(">>> [SMOKE TEST] Subsetting dataset to 24 train and 8 val examples for fast pipeline verification...")
        dataset["train"] = dataset["train"].select(range(min(24, len(dataset["train"]))))
        dataset["val"] = dataset["val"].select(range(min(8, len(dataset["val"]))))

    print(f"Dataset loaded: {len(dataset['train'])} train examples, {len(dataset['val'])} validation examples.")

    # 4. Tokenizer Setup
    print(f"Loading tokenizer: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. 4-bit Quantization Config (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 6. Load Base Model in 4-bit NF4
    print(f"Loading base model {args.model_id} with 4-bit NF4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # 7. Prepare for k-bit training and configure LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # Required when gradient checkpointing is enabled

    # Parse target modules
    if args.target_modules == "all-linear":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif args.target_modules in ("attention", "attn"):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = [m.strip() for m in args.target_modules.split(",")]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("\nTrainable Parameters Summary:")
    model.print_trainable_parameters()

    # 8. Setup Hugging Face SFTConfig (TrainingArguments)
    # Determine report_to for HF internal reporting
    hf_report_to = "none"
    if args.tracker == "wandb" and tracker_status.get("wandb_active"):
        hf_report_to = "wandb"
    elif args.tracker == "mlflow" and tracker_status.get("mlflow_active"):
        hf_report_to = "mlflow"

    eval_strategy = args.eval_strategy
    save_strategy = args.save_strategy
    if eval_strategy != "no" and save_strategy != eval_strategy:
        save_strategy = eval_strategy
    eval_steps = args.eval_steps if eval_strategy == "steps" else None
    save_steps = args.eval_steps if save_strategy == "steps" else None
    max_steps = args.max_steps if args.max_steps > 0 else (6 if args.smoke_test else -1)

    sft_config = SFTConfig(
        output_dir=args.checkpoint_dir,
        num_train_epochs=1 if args.smoke_test else args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine" if not args.smoke_test else "linear",
        warmup_steps=3 if args.smoke_test else 10,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to=hf_report_to,
        logging_first_step=True,
    )

    # 9. Register Custom Callbacks
    # A) System metrics callback (VRAM peak, throughput, grad_norm, perplexity)
    system_callback = SystemMetricsAndTrackingCallback(
        local_logger=local_logger,
        tracker_type=args.tracker,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )

    # B) Validation periodic inference logger callback
    val_samples_to_log = load_fixed_val_samples(args.val_file, n_samples=args.num_val_samples)
    val_inference_callback = ValidationInferenceLoggerCallback(
        val_samples=val_samples_to_log,
        tokenizer=tokenizer,
        tracker_type=args.tracker,
        output_file=local_samples_path,
        max_new_tokens=256,
    )

    callbacks = [system_callback, val_inference_callback]

    # 10. Instantiate SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # 11. Execute Training
    print("\n" + "=" * 70)
    print("  STARTING QLORA TRAINING")
    print("=" * 70)
    train_start_time = time.time()
    train_result = trainer.train()
    total_duration = time.time() - train_start_time

    print("\n" + "=" * 70)
    print(f"Training finished in {total_duration / 60:.2f} minutes ({total_duration:.1f} s).")
    print("=" * 70)

    # 12. Save Best Model Adapter & Tokenizer
    print(f"\nSaving best adapter weights and tokenizer to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Verify adapter files
    adapter_weights = os.path.join(args.output_dir, "adapter_model.safetensors")
    adapter_config = os.path.join(args.output_dir, "adapter_config.json")
    assert os.path.exists(adapter_weights), f"Missing adapter weights at {adapter_weights}!"
    assert os.path.exists(adapter_config), f"Missing adapter config at {adapter_config}!"
    print(f"Verified adapter weights: {adapter_weights} ({os.path.getsize(adapter_weights):,} bytes)")

    # 13. Peak VRAM Safety Check
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Final Peak GPU VRAM Consumed: {peak_vram_gb:.2f} GB (Target <= 5.0 GB)")
    assert peak_vram_gb <= 5.0, f"Peak VRAM {peak_vram_gb:.2f} GB exceeded 5.0 GB limit!"

    # 14. Verification of Metrics Logging
    print("\nVerifying Experiment Tracking Artifacts:")
    print(f"  CSV Metrics  : {local_csv_path} (exists: {os.path.exists(local_csv_path)})")
    print(f"  JSONL Metrics: {local_jsonl_path} (exists: {os.path.exists(local_jsonl_path)})")
    print(f"  Run Summary  : {local_summary_path} (exists: {os.path.exists(local_summary_path)})")
    print(f"  Val Samples  : {local_samples_path} (exists: {os.path.exists(local_samples_path)})")

    # 15. Cleanly finish experiment tracker
    finish_experiment_tracker()
    
    # Load and enrich run summary
    summary_data = {}
    if os.path.exists(local_summary_path):
        try:
            with open(local_summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    summary_data.update({
        "run_name": run_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "target_modules": args.target_modules,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "output_dir": args.output_dir,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "total_train_duration_sec": round(total_duration, 2),
    })

    with open(local_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SUCCESS: Phase 2 Alpha training script executed with all tracking criteria met!")
    print("=" * 70)
    return summary_data


if __name__ == "__main__":
    train_alpha()
