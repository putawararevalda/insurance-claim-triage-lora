"""Model Merging & Standalone Multi-Format Packaging Script (Phase 3.1).

Merges fine-tuned LoRA adapter weights (models/lora_adapter_alpha_best) back into
the base foundation model (Qwen/Qwen2.5-1.5B-Instruct), creating a standalone
16-bit model that runs without PEFT dependency. Also verifies standalone inference
against the Pydantic ClaimTriageOutput schema.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ClaimTriageOutput

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = "models/lora_adapter_alpha_best"
DEFAULT_OUTPUT_DIR = "models/merged_qwen2.5_1.5b_insurance_triage"
DEFAULT_TEST_DATA = "data/alpha_test.jsonl"


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.float16)


def merge_and_export_model(
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dtype_str: str = "float16",
    device: str = "cuda",
    verify: bool = True,
    test_data_path: str = DEFAULT_TEST_DATA,
) -> Dict[str, Any]:
    print("=" * 70, flush=True)
    print("  PHASE 3.1: LORA ADAPTER MERGING & STANDALONE MODEL PACKAGING", flush=True)
    print("=" * 70, flush=True)
    print(f"Base Foundation Model : {base_model_id}", flush=True)
    print(f"LoRA Adapter Path     : {adapter_path}", flush=True)
    print(f"Merged Output Dir     : {output_dir}", flush=True)
    print(f"Target Precision      : {dtype_str}", flush=True)
    print(f"Execution Device      : {device}", flush=True)
    print("=" * 70, flush=True)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU...", flush=True)
        device = "cpu"

    dtype = get_torch_dtype(dtype_str)

    # 1. Load Tokenizer
    print("\n[Step 1/5] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path if os.path.exists(os.path.join(adapter_path, "tokenizer.json")) else base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Tokenizer loaded successfully. Vocab size: {len(tokenizer)}", flush=True)

    # 2. Load Base Model in full precision (float16/bfloat16)
    print(f"\n[Step 2/5] Loading base model ({base_model_id}) in {dtype_str}...", flush=True)
    t0_load = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        base_model = base_model.to("cpu")
    load_duration = time.time() - t0_load
    print(f"Base model loaded in {load_duration:.2f}s.", flush=True)

    # 3. Attach LoRA Adapter
    print(f"\n[Step 3/5] Attaching LoRA adapter from {adapter_path}...", flush=True)
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    print("LoRA adapter attached successfully.", flush=True)

    # 4. Merge and Unload
    print("\n[Step 4/5] Executing model.merge_and_unload()...", flush=True)
    t0_merge = time.time()
    merged_model = lora_model.merge_and_unload()
    merge_duration = time.time() - t0_merge
    print(f"Weights merged and unloaded in {merge_duration:.2f}s. Model is now fully standalone!", flush=True)

    # Count parameters
    total_params = sum(p.numel() for p in merged_model.parameters())
    print(f"Total Standalone Model Parameters: {total_params:,}", flush=True)

    # 5. Save Merged Model and Tokenizer
    print(f"\n[Step 5/5] Saving standalone model to {output_dir}...", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(
        output_dir,
        max_shard_size="4GB",
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Model weights and tokenizer saved successfully to {output_dir}.", flush=True)

    # 6. Standalone Verification Test
    verification_results = {}
    if verify:
        print("\n" + "=" * 70, flush=True)
        print("  STANDALONE INFERENCE VERIFICATION (NO PEFT DEPENDENCY)", flush=True)
        print("=" * 70, flush=True)
        merged_model.eval()

        # Load a real sample from test dataset
        sample_prompt = None
        sample_gt = None
        if os.path.exists(test_data_path):
            with open(test_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        sample_gt = item
                        sample_prompt = item["messages"][:2]
                        break

        if sample_prompt is None:
            # Fallback prompt
            sample_prompt = [
                {
                    "role": "system",
                    "content": "Anda adalah asisten AI spesialis klaim asuransi (First Notice of Loss / FNOL Triage Agent) untuk perusahaan asuransi jiwa dan kesehatan di Indonesia. Ekstrak informasi klaim ke dalam format JSON sesuai skema ClaimTriageOutput.",
                },
                {
                    "role": "user",
                    "content": "Saya baru saja masuk IGD RS Siloam karena demam berdarah trombosit 45.000 dan harus opname sekarang juga. Mohon persetujuan cashless kartu asuransi saya POL-882918.",
                },
            ]

        formatted_text = tokenizer.apply_chat_template(
            sample_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(formatted_text, return_tensors="pt").to(merged_model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0_gen = time.perf_counter()

        with torch.no_grad():
            output_tokens = merged_model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0_gen) * 1000

        gen_ids = output_tokens[0][inputs.input_ids.shape[1] :]
        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"Verification Raw Output:\n{output_text}\n", flush=True)
        print(f"Inference Latency: {latency_ms:.1f} ms", flush=True)

        # Validate with Pydantic
        json_valid = False
        parsed_category = None
        try:
            # Extract JSON substring if wrapped in markdown
            json_str = output_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            elif "{" in json_str and "}" in json_str:
                json_str = json_str[json_str.find("{") : json_str.rfind("}") + 1]

            parsed_obj = ClaimTriageOutput.model_validate_json(json_str)
            json_valid = True
            parsed_category = parsed_obj.category.value
            print(f"Pydantic Validation : PASSED (Category: {parsed_category}, Urgency: {parsed_obj.urgency.value})", flush=True)
        except Exception as e:
            print(f"Pydantic Validation : FAILED ({e})", flush=True)

        verification_results = {
            "verified": json_valid,
            "latency_ms": round(latency_ms, 2),
            "output_text": output_text,
            "predicted_category": parsed_category,
            "ground_truth_category": sample_gt.get("category") if sample_gt else "Inpatient_Medical",
        }

    # Write export metadata
    metadata = {
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_parameters": total_params,
        "dtype": dtype_str,
        "device": device,
        "verification": verification_results,
    }
    meta_path = os.path.join(output_dir, "export_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata written to: {meta_path}", flush=True)
    print("=" * 70, flush=True)
    print("SUCCESS: Standalone model merge and packaging completed successfully!", flush=True)
    print("=" * 70, flush=True)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and merge LoRA adapter into standalone model.")
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID, help="Base model HuggingFace ID")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH, help="Path to best LoRA adapter")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Destination directory for merged model")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model weights precision")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device for merging")
    parser.add_argument("--skip-verification", action="store_true", help="Skip post-merge standalone inference check")
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA, help="Path to test dataset for verification")
    args = parser.parse_args()

    merge_and_export_model(
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        dtype_str=args.dtype,
        device=args.device,
        verify=not args.skip_verification,
        test_data_path=args.test_data,
    )
