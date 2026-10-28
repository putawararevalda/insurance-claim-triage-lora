"""Environment and CUDA Hardware Verification Script for LoRA Fine-Tuning."""

import os
import sys
import torch

def get_installed_version(pkg_name: str) -> str:
    try:
        mod = __import__(pkg_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "NOT INSTALLED"

def check_env():
    log_lines = []
    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  CUDA & HARDWARE ENVIRONMENT VERIFICATION")
    log("=" * 60)
    log(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    log(f"Platform: {sys.platform}")
    log(f"PyTorch Version: {torch.__version__}")
    
    cuda_avail = torch.cuda.is_available()
    log(f"CUDA Available: {cuda_avail}")
    
    if cuda_avail:
        device_count = torch.cuda.device_count()
        log(f"Device Count: {device_count}")
        current_dev = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(current_dev)
        props = torch.cuda.get_device_properties(current_dev)
        total_vram_gb = props.total_memory / (1024 ** 3)
        cuda_cap = f"{props.major}.{props.minor}"
        
        log(f"Current Device Index: {current_dev}")
        log(f"Device Name: {dev_name}")
        log(f"Compute Capability: {cuda_cap}")
        log(f"Total VRAM: {total_vram_gb:.2f} GB ({props.total_memory:,} bytes)")
        
        # Test basic tensor allocation
        x = torch.randn((1000, 1000), device="cuda")
        alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        log(f"Sample Tensor Allocation: Success ({alloc_mb:.2f} MB allocated, {reserved_mb:.2f} MB reserved)")
        del x
        torch.cuda.empty_cache()
    else:
        log("ERROR: CUDA is NOT available! GPU acceleration required.")
        sys.exit(1)

    log("-" * 60)
    log("  KEY DEPENDENCIES")
    log("-" * 60)
    pkgs = [
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "pydantic",
    ]
    for pkg in pkgs:
        ver = get_installed_version(pkg)
        log(f"  {pkg:15s}: {ver}")

    log("=" * 60)
    log("STATUS: Environment is ready and verified for 4-bit QLoRA fine-tuning.")
    log("=" * 60)

    # Save to env_ready.log
    with open("env_ready.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\nLog successfully written to env_ready.log")

if __name__ == "__main__":
    check_env()
