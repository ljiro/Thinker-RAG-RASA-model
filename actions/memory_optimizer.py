# memory_optimizer.py
import torch
import gc
import psutil
import os

def optimize_memory_usage():
    """Optimize memory usage for low VRAM systems"""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Set memory efficient settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🧠 Memory optimized for low VRAM usage")

def get_memory_info():
    """Get current memory usage information"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory: {gpu_memory:.2f} GB (Peak: {gpu_memory_max:.2f} GB)")
    
    # System RAM
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.used/1024**3:.2f} GB / {ram.total/1024**3:.2f} GB ({ram.percent}%)")

if __name__ == "__main__":
    optimize_memory_usage()
    get_memory_info()