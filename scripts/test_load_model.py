"""Script to test loading Base Model in 4-bit NF4 quantization on local GPU."""

import gc
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def test_load_model():
    print("=" * 60)
    print(f"Testing 4-bit Quantization Model Loading: {MODEL_ID}")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
        
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    initial_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"Initial VRAM Allocated: {initial_alloc:.2f} GB")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    start_time = time.time()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("Loading model in 4-bit NF4 with BitsAndBytesConfig...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    load_time = time.time() - start_time
    
    post_load_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    
    print(f"Model loaded in {load_time:.2f} seconds.")
    print(f"Post-load VRAM Allocated : {post_load_alloc:.2f} GB")
    print(f"Peak VRAM during loading : {peak_alloc:.2f} GB")
    print(f"Reserved VRAM            : {reserved:.2f} GB")
    
    # Test a quick inference forward pass
    print("-" * 60)
    print("Testing forward pass inference...")
    messages = [
        {"role": "system", "content": "Kamu adalah asisten AI klasifikasi klaim asuransi."},
        {"role": "user", "content": "Halo, apakah sistem kamu aktif?"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Sample Model Response: {response.strip()}")
    
    final_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Final Peak VRAM with Generation: {final_peak:.2f} GB")
    
    # Validate acceptance criteria: peak memory idle load < 3.5 GB
    assert peak_alloc < 3.5, f"Peak memory {peak_alloc:.2f} GB exceeds 3.5 GB threshold!"
    print("=" * 60)
    print("SUCCESS: 4-bit Quantization model load test passed all criteria!")
    print("=" * 60)
    
    # Append to env_ready.log
    with open("env_ready.log", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("  MODEL QUANTIZATION 4-BIT LOAD TEST\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model ID: {MODEL_ID}\n")
        f.write(f"Load Time: {load_time:.2f} s\n")
        f.write(f"VRAM Allocated: {post_load_alloc:.2f} GB\n")
        f.write(f"Peak VRAM: {peak_alloc:.2f} GB\n")
        f.write(f"Inference Peak VRAM: {final_peak:.2f} GB\n")
        f.write(f"Criterion (Peak < 3.5 GB): PASSED ({peak_alloc:.2f} GB < 3.5 GB)\n")
        f.write(f"Sample Output: {response.strip()}\n")
        f.write("=" * 60 + "\n")

if __name__ == "__main__":
    test_load_model()
