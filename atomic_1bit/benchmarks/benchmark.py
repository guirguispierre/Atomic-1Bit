import torch
import torch.nn as nn
import time
import sys
import os
import psutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from atomic_1bit.nn.layers import BitLinear

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def benchmark():
    print("--- Atomic-1Bit vs PyTorch: Benchmark ---")
    
    # Config
    BATCH = 1
    IN_DIM = 4096
    OUT_DIM = 4096
    
    print(f"Dimensions: {IN_DIM} -> {OUT_DIM} (Batch {BATCH})")
    
    # 1. PyTorch Baseline
    print("\n[PyTorch Linear (FP32)]")
    # Measure baseline memory
    mem_before = get_memory_usage()
    torch_layer = nn.Linear(IN_DIM, OUT_DIM, bias=False)
    mem_after = get_memory_usage()
    print(f"Weight Memory (approx): {mem_after - mem_before:.2f} MB")
    
    # Warmup
    x = torch.randn(BATCH, IN_DIM)
    for _ in range(10):
        _ = torch_layer(x)
        
    start_t = time.time()
    for _ in range(100):
        _ = torch_layer(x)
    end_t = time.time()
    torch_time = (end_t - start_t) * 1000 / 100
    print(f"Avg Inference Time: {torch_time:.4f} ms")
    
    # Cleanup to free memory for fair comparison
    del torch_layer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 2. Atomic-1Bit
    print("\n[Atomic-1Bit BitLinear (INT8/Ternary)]")
    mem_before_atomic = get_memory_usage()
    atomic_layer = BitLinear(IN_DIM, OUT_DIM, bias=False)
    mem_after_atomic = get_memory_usage()
    print(f"Weight Memory (approx): {mem_after_atomic - mem_before_atomic:.2f} MB")
    
    # Warmup
    for _ in range(10):
        _ = atomic_layer(x)
        
    start_t = time.time()
    for _ in range(100):
        _ = atomic_layer(x)
    end_t = time.time()
    atomic_time = (end_t - start_t) * 1000 / 100
    print(f"Avg Inference Time: {atomic_time:.4f} ms")
    
    print("\n--- Results ---")
    print(f"Memory Savings: ~{(mem_after - mem_before) / (mem_after_atomic - mem_before_atomic + 1e-5):.1f}x (Expect ~4x)")
    print(f"Speedup: {torch_time / atomic_time:.2f}x (Current Python overhead expected to be slow)")

if __name__ == "__main__":
    benchmark()
