"""4-bit QLoRA Fine-Tuning Script for Insurance Claim & FNOL Triage MVP.
Optimized for NVIDIA RTX 3050 (6GB VRAM) on local Windows environment.
"""

import gc
import json
import os
import sys
import time
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "models/lora_adapter_mvp"
CHECKPOINT_DIR = "checkpoints/mvp"
LOG_FILE = "logs/training_mvp.log"


class VRAMAndMetricsCallback(TrainerCallback):
    """Custom callback to track and log training loss, eval loss, and peak VRAM."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=" * 65 + "\n")
            f.write("  TRAINING MVP EXECUTION LOG (4-bit QLoRA RTX 3050)\n")
            f.write("=" * 65 + "\n")
            f.write(f"Model ID: {MODEL_ID}\n")
            f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 65 + "\n")
            f.write(f"{'Step':>6} | {'Epoch':>6} | {'Loss':>8} | {'LR':>10} | {'Peak VRAM (GB)':>15}\n")
            f.write("-" * 65 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        step = state.global_step
        epoch = round(state.epoch, 2) if state.epoch is not None else 0.0
        loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)
        lr = logs.get("learning_rate", 0.0)
        
        entry = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "eval_loss": eval_loss,
            "lr": lr,
            "peak_vram_gb": peak_vram_gb
        }
        self.history.append(entry)
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            if loss is not None:
                line = f"{step:>6} | {epoch:>6.2f} | {loss:>8.4f} | {lr:>10.2e} | {peak_vram_gb:>15.2f}\n"
                f.write(line)
                print(f"[Step {step:>3} | Epoch {epoch:>4.2f}] Train Loss: {loss:.4f} | Peak VRAM: {peak_vram_gb:.2f} GB | LR: {lr:.2e}")
            if eval_loss is not None:
                f.write(f"--- [EVALUATION] Epoch {epoch:.2f} Eval Loss: {eval_loss:.4f} (Peak VRAM: {peak_vram_gb:.2f} GB) ---\n")
                print(f"\n>>> [EVALUATION] Epoch {epoch:.2f} Eval Loss: {eval_loss:.4f} (Peak VRAM: {peak_vram_gb:.2f} GB)\n")


def train_mvp():
    print("=" * 65)
    print("  INITIALIZING 4-BIT QLORA TRAINING PIPELINE")
    print("=" * 65)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU fine-tuning!")
        
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Load Datasets
    print("Loading seed train and validation datasets...")
    data_files = {
        "train": "data/seed_train.jsonl",
        "val": "data/seed_val.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    print(f"Train samples: {len(dataset['train'])}, Validation samples: {len(dataset['val'])}")
    
    # 2. Tokenizer Setup
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 3. 4-bit Quantization Configuration (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 4. Load Base Model in 4-bit
    print(f"Loading base causal LM {MODEL_ID} in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # 5. Prepare for k-bit training & configure gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # Must be disabled when gradient checkpointing is enabled
    
    # 6. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("\nTrainable Parameters Summary:")
    model.print_trainable_parameters()
    
    # 7. SFT Configuration (TrainingArguments)
    # Effective batch size = per_device_train_batch_size (2) * gradient_accumulation_steps (4) = 8
    # With 140 training examples, 1 epoch ~ 18 steps (54 steps total across 3 epochs)
    sft_config = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        max_length=512,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        logging_first_step=True,
    )
    
    # 8. Setup Trainer with custom callback
    vram_callback = VRAMAndMetricsCallback(LOG_FILE)
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        callbacks=[vram_callback],
    )
    
    # 9. Execute Training
    print("\n" + "=" * 65)
    print("  COMMENCING 3-EPOCH QLORA TRAINING")
    print("=" * 65)
    start_train_time = time.time()
    train_result = trainer.train()
    total_train_duration = time.time() - start_train_time
    
    print("\n" + "=" * 65)
    print(f"Training completed successfully in {total_train_duration / 60:.2f} minutes ({total_train_duration:.1f} s).")
    print("=" * 65)
    
    # 10. Save LoRA Adapter & Tokenizer
    print(f"Saving best adapter weights and tokenizer to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Verify saved adapter files
    adapter_weights = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    adapter_config = os.path.join(OUTPUT_DIR, "adapter_config.json")
    assert os.path.exists(adapter_weights), f"Missing {adapter_weights}!"
    assert os.path.exists(adapter_config), f"Missing {adapter_config}!"
    print(f"Verified adapter files: {adapter_weights} ({os.path.getsize(adapter_weights):,} bytes)")
    
    # Peak VRAM check
    max_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nFinal Peak GPU VRAM Consumed: {max_vram_gb:.2f} GB (Target <= 5.0 GB)")
    assert max_vram_gb <= 5.0, f"Peak VRAM {max_vram_gb:.2f} GB exceeded 5.0 GB safety threshold!"
    
    # Loss convergence verification
    log_history = [e for e in vram_callback.history if e["loss"] is not None]
    if len(log_history) >= 2:
        init_loss = log_history[0]["loss"]
        final_loss = log_history[-1]["loss"]
        print(f"Initial Step Loss : {init_loss:.4f}")
        print(f"Final Step Loss   : {final_loss:.4f}")
        assert final_loss < init_loss, f"Loss did not decrease: {init_loss:.4f} -> {final_loss:.4f}"
        print("Convergence Check : PASSED (Loss decreased significantly)")
        
    # Finalize log
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("-" * 65 + "\n")
        f.write("TRAINING SUMMARY:\n")
        f.write(f"  Duration         : {total_train_duration / 60:.2f} minutes\n")
        f.write(f"  Peak VRAM        : {max_vram_gb:.2f} GB\n")
        f.write(f"  Total Steps      : {train_result.global_step}\n")
        f.write(f"  Final Train Loss : {train_result.training_loss:.4f}\n")
        f.write(f"  Adapter Output   : {OUTPUT_DIR}\n")
        f.write("=" * 65 + "\n")
        f.write("STATUS: ALL MVP TRAINING ACCEPTANCE CRITERIA MET!\n")
        f.write("=" * 65 + "\n")
        
    print(f"Detailed logs saved to {LOG_FILE}")
    print("=" * 65)
    print("SUCCESS: train_mvp.py completed successfully!")
    print("=" * 65)

if __name__ == "__main__":
    train_mvp()
