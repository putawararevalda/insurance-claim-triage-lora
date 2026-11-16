"""Inference Service for Indonesian Insurance Claim & FNOL Triage.

Provides high-performance, deterministic model inference for both standalone
LoRA-merged model and base model with robust Pydantic validation and latency profiling.
"""

import json
import logging
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ClaimCategory, ClaimMethod, ClaimTriageOutput, UrgencyLevel

logger = logging.getLogger("triage_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MERGED_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "merged_qwen2.5_1.5b_insurance_triage")
)
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_INSTRUCTION = (
    "Kamu adalah sistem AI FNOL (First Notice of Loss) Triage untuk klaim asuransi jiwa dan kesehatan di Indonesia. "
    "Tugasmu adalah menganalisis pesan atau laporan klaim nasabah dan mengembalikan hasil klasifikasi dalam format JSON "
    "yang valid sesuai skema ClaimTriageOutput dengan field: category, urgency, primary_condition, claim_method, dan reasoning."
)

USER_PROMPT_PREFIX = "Lakukan triage dan analisis klaim asuransi untuk laporan nasabah berikut:\n\n"


def clean_json_text(text: str) -> str:
    """Extract clean JSON string from model response even if surrounded by markdown code fences."""
    text = text.strip()
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    return text


def fallback_parse_triage_json(text: str) -> Optional[Dict[str, Any]]:
    """Robust fallback JSON parser using regex extraction when standard JSON parsing fails."""
    # 1. Attempt trailing comma fix
    fixed = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # 2. Key-value regex extraction for ClaimTriageOutput fields
    fields = ["category", "urgency", "primary_condition", "claim_method", "reasoning"]
    extracted = {}
    for f in fields:
        m = re.search(rf'"{f}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        if m:
            extracted[f] = m.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")
    
    if len(extracted) >= 3:
        # Provide sensible defaults for missing fields if critical ones are present
        extracted.setdefault("category", "Inquiry_or_Dispute")
        extracted.setdefault("urgency", "Medium")
        extracted.setdefault("primary_condition", "Tidak disebutkan secara spesifik")
        extracted.setdefault("claim_method", "Unknown")
        extracted.setdefault("reasoning", "Diekstrak melalui fallback regex parser.")
        return extracted

    return None


class TriageService:
    """Manages model loading, GPU acceleration, and inference execution for FNOL triage."""

    _instance = None

    def __init__(
        self,
        merged_model_path: str = DEFAULT_MERGED_MODEL_DIR,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        device: Optional[str] = None,
    ):
        self.merged_model_path = merged_model_path
        self.base_model_id = base_model_id
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.lora_model = None
        self.lora_tokenizer = None
        self.base_model = None
        self.base_tokenizer = None
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TriageService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_lora_model(self) -> None:
        """Loads the merged standalone LoRA model onto target device."""
        if self.lora_model is not None:
            return

        logger.info(f"Loading merged LoRA model from {self.merged_model_path} onto {self.device}...")
        t0 = time.time()
        
        self.lora_tokenizer = AutoTokenizer.from_pretrained(self.merged_model_path)
        if self.lora_tokenizer.pad_token is None:
            self.lora_tokenizer.pad_token = self.lora_tokenizer.eos_token
        self.lora_tokenizer.padding_side = "right"

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.lora_model = AutoModelForCausalLM.from_pretrained(
            self.merged_model_path,
            dtype=dtype,
            device_map=self.device if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            self.lora_model = self.lora_model.to("cpu")
            
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.lora_model.eval()
        
        # Warm-up inference
        self._warmup(self.lora_model, self.lora_tokenizer)
        logger.info(f"Merged LoRA model ready in {time.time() - t0:.2f}s.")

    def load_base_model(self) -> None:
        """Loads the zero-shot base foundation model for side-by-side comparison."""
        if self.base_model is not None:
            return

        logger.info(f"Loading Base Foundation model ({self.base_model_id}) onto {self.device}...")
        t0 = time.time()
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_tokenizer.padding_side = "right"

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            dtype=dtype,
            device_map=self.device if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            self.base_model = self.base_model.to("cpu")
            
        self.base_model.eval()
        
        # Warm-up inference
        self._warmup(self.base_model, self.base_tokenizer)
        logger.info(f"Base model ready in {time.time() - t0:.2f}s.")

    def _warmup(self, model: Any, tokenizer: Any) -> None:
        """Executes a quick dummy forward pass to warm up CUDA kernels."""
        try:
            sample_messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": USER_PROMPT_PREFIX + "Pasien demam berdarah rawat inap."},
            ]
            text = tokenizer.apply_chat_template(sample_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Warm-up exception: {e}")

    def run_inference(
        self,
        narrative: str,
        model_type: str = "lora",
        max_new_tokens: int = 150,
    ) -> Dict[str, Any]:
        """Runs FNOL triage inference for a given claim narrative."""
        if model_type == "lora":
            if self.lora_model is None:
                self.load_lora_model()
            model = self.lora_model
            tokenizer = self.lora_tokenizer
            model_name = "Qwen2.5-1.5B-Instruct + LoRA (Golden Adapter)"
        elif model_type == "base":
            if self.base_model is None:
                self.load_base_model()
            model = self.base_model
            tokenizer = self.base_tokenizer
            model_name = "Qwen2.5-1.5B-Instruct (Zero-Shot Base)"
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lora' or 'base'.")

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": USER_PROMPT_PREFIX + narrative.strip()},
        ]
        
        with self._lock:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            input_token_count = inputs.input_ids.shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.inference_mode():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000
        gen_tokens = output_tokens[0][input_token_count:]
        generated_count = len(gen_tokens)
        tokens_per_sec = (generated_count / (latency_ms / 1000)) if latency_ms > 0 else 0.0

        raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        cleaned_json = clean_json_text(raw_output)

        parsed_data = None
        json_valid = False
        parse_error = None

        try:
            validated = ClaimTriageOutput.model_validate_json(cleaned_json)
            parsed_data = validated.model_dump()
            json_valid = True
        except Exception as err:
            parse_error = str(err)
            try:
                fallback_dict = fallback_parse_triage_json(cleaned_json)
                if fallback_dict is not None:
                    try:
                        validated = ClaimTriageOutput.model_validate(fallback_dict)
                        parsed_data = validated.model_dump()
                        json_valid = True
                        parse_error = None
                    except Exception:
                        parsed_data = fallback_dict
                else:
                    parsed_data = json.loads(cleaned_json)
            except Exception:
                parsed_data = None

        return {
            "model_type": model_type,
            "model_name": model_name,
            "narrative": narrative,
            "raw_output": raw_output,
            "cleaned_json": cleaned_json,
            "json_valid": json_valid,
            "parse_error": parse_error,
            "parsed": parsed_data,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_token_count,
            "generated_tokens": generated_count,
            "tokens_per_second": round(tokens_per_sec, 1),
            "device": str(model.device),
        }

    def compare(self, narrative: str) -> Dict[str, Any]:
        """Runs both models for side-by-side comparison on the same claim narrative."""
        lora_res = self.run_inference(narrative, model_type="lora")
        base_res = self.run_inference(narrative, model_type="base")
        return {
            "narrative": narrative,
            "lora": lora_res,
            "base": base_res,
        }
