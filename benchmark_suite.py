import torch
import time
import os
import sys
import tracemalloc
import json
import subprocess
import matplotlib.pyplot as plt
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from baseline_fp16 import FP16Transformer, FP16Config

# --- CONFIG ---
VOCAB_SIZE = 2048
DIM = 128
DEPTH = 4
HEADS = 4
CONTEXT_LEN = 64
GEN_TOKENS = 100

def measure_model_size(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

def get_param_count(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_inference(model, device="cpu"):
    model.eval()
    model.to(device)
    
    # Dummy input
    x = torch.randint(0, VOCAB_SIZE, (1, 1)).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
            
    # Benchmark
    tracemalloc.start()
    start_time = time.time()
    
    generated = 0
    with torch.no_grad():
        for _ in range(GEN_TOKENS):
            logits = model(x)
            # Greedy
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            x = torch.cat([x, next_token], dim=1)
            generated += 1
            if x.size(1) >= CONTEXT_LEN:
                # Rolling window for simplicity in this bench
                x = x[:, -CONTEXT_LEN:]
                
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    duration = end_time - start_time
    tps = generated / duration
    peak_mb = peak / (1024 * 1024)
    
    return tps, peak_mb

def run_benchmarks():
    print("--- BENCHMARK SUITE ---")
    
    results = {
        "Atomic": {},
        "FP16": {},
        "CPP": {}
    }
    
    # 1. Atomic Python
    print("\n[Atomic Python]")
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config)
    try:
        model.load_state_dict(torch.load("weights/benchmark_model.pt", map_location="cpu"))
        print("Loaded Atomic Weights.")
    except:
        print("Warning: Could not load Atomic weights. Using random.")
        
    tps, peak = benchmark_inference(model)
    results["Atomic"]["TPS"] = tps
    results["Atomic"]["PeakRAM"] = peak
    results["Atomic"]["Params"] = get_param_count(model)
    results["Atomic"]["SizeMB"] = measure_model_size("weights/benchmark_model.pt") # Compressed size? No, PT is uncompressed.
    # Atomic model on disk in PT format is fp32/bf16 usually unless specific save. 
    # But we care about the C++ binary size really.
    
    print(f"TPS: {tps:.2f}, RAM: {peak:.2f} MB")

    # 2. FP16 Baseline
    print("\n[FP16 Baseline]")
    fp_config = FP16Config(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    fp_model = FP16Transformer(fp_config)
    try:
        fp_model.load_state_dict(torch.load("weights/baseline_model.pt", map_location="cpu"))
        print("Loaded FP16 Weights.")
    except:
        print("Warning: Could not load FP16 weights. Using random.")
        
    # Convert to half for fair comparison if on CUDA/MPS, but CPU half is slow on PyTorch usually.
    # We will benchmark on CPU for consistency with C++ engine.
    # But usually PyTorch CPU float32 is the standard baseline.
    # If we want explicit FP16, we need to be careful about device support.
    # For this benchmark, we'll stick to Float32 on CPU for "Baseline" or try Float16 if supported.
    # PyTorch CPU doesn't support FP16 operations well for all layers. We will keep it Float32 for robust CPU baseline.
    
    tps, peak = benchmark_inference(fp_model, device="cpu")
    results["FP16"]["TPS"] = tps
    results["FP16"]["PeakRAM"] = peak
    results["FP16"]["Params"] = get_param_count(fp_model)
    results["FP16"]["SizeMB"] = measure_model_size("weights/baseline_model.pt")
    
    print(f"TPS: {tps:.2f}, RAM: {peak:.2f} MB")

    # 3. C++ Atomic
    print("\n[Atomic C++]")
    # We need to compile and run.
    # Assuming 'embedded/atomic_engine' exists and we pass a flag or it just runs.
    # For now, we will just measure the binary size.
    # TPS will be parsed from stdout if we instrument it.
    
    bin_path = "embedded/atomic_model.bin"
    if os.path.exists(bin_path):
        results["CPP"]["SizeMB"] = os.path.getsize(bin_path) / (1024 * 1024)
    else:
        results["CPP"]["SizeMB"] = 0
        
    print(f"Binary Size: {results['CPP']['SizeMB']:.2f} MB")
    
    # Run C++ binary
    try:
        # We need to make sure the runner prints "TPS: <value>" or we time the process.
        # process time includes startup, so internal timing is better.
        # I will instrument cpp code next.
        cmd = ["./embedded/runner"] # compiled name
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout)
        
        # Parse output for TPS
        for line in p.stdout.splitlines():
            if "TPS:" in line:
                results["CPP"]["TPS"] = float(line.split(":")[1].strip())
            if "Peak Info" in line:
                pass # Parse if available
                
    except Exception as e:
        print(f"C++ Run Failed: {e}")

    # 4. Save Results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nResults Saved.")
    return results

if __name__ == "__main__":
    run_benchmarks()
