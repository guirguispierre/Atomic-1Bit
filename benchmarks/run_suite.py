import torch
import time
import sys
import os
import threading
import subprocess
import platform
import json

# Add root to sys.path to find atomic_1bit
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from baseline_fp16 import FP16Transformer, FP16Config

# Try importing psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: 'psutil' not found. RAM measurement will be unavailable. Install with `pip install psutil`.")

# --- PATHS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(rel_path):
    return os.path.join(ROOT_DIR, rel_path)

# --- CONFIG ---
VOCAB_SIZE = 2048
DIM = 128
DEPTH = 4
HEADS = 4
CONTEXT_LEN = 64
GEN_TOKENS = 100

class MemoryMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0
        self.thread = None

    def _monitor(self):
        process = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            try:
                rss = process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if not PSUTIL_AVAILABLE:
            return
        # Reset peak to current to capture delta if needed, but absolute peak is requested
        # We want Peak RSS during inference.
        self.peak_rss = psutil.Process(os.getpid()).memory_info().rss
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        if not PSUTIL_AVAILABLE:
            return 0
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        return self.peak_rss / (1024 * 1024) # MB

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
    print("  Warmup...", end="", flush=True)
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
    print(" Done.")

    # Benchmark
    monitor = MemoryMonitor()
    monitor.start()
    
    start_time = time.perf_counter()
    
    generated = 0
    with torch.no_grad():
        for _ in range(GEN_TOKENS):
            logits = model(x)
            # Greedy
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            x = torch.cat([x, next_token], dim=1)
            generated += 1
            if x.size(1) >= CONTEXT_LEN:
                # Rolling window
                x = x[:, -CONTEXT_LEN:]
                
    end_time = time.perf_counter()
    peak_mb = monitor.stop()
    
    duration = end_time - start_time
    tps = generated / duration
    
    return tps, peak_mb

def run_benchmarks():
    print("--- BENCHMARK SUITE (Revised) ---")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    if PSUTIL_AVAILABLE:
        print("RAM Measurement: Enabled (psutil)")
    else:
        print("RAM Measurement: Disabled (psutil missing)")
    
    results = {
        "System": {
            "Platform": platform.system(),
            "Machine": platform.machine(),
            "Processor": platform.processor()
        },
        "Atomic": {},
        "FP16": {},
        "CPP": {}
    }
    
    # 1. Atomic Python
    print("\n[Atomic Python]")
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config)
    try:
        model.load_state_dict(torch.load(get_path("weights/benchmark_model.pt"), map_location="cpu"))
        print("  Loaded Atomic Weights.")
    except Exception as e:
        print(f"  Warning: Could not load Atomic weights ({e}). Using random.")
        
    tps, peak = benchmark_inference(model)
    results["Atomic"]["TPS"] = tps
    results["Atomic"]["PeakRAM_MB"] = peak
    results["Atomic"]["Params"] = get_param_count(model)
    results["Atomic"]["SizeMB"] = measure_model_size(get_path("weights/benchmark_model.pt"))
    
    print(f"  TPS: {tps:.2f}")
    print(f"  Peak RSS: {peak:.2f} MB")

    # 2. FP16 Baseline
    print("\n[FP16 Baseline]")
    fp_config = FP16Config(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    fp_model = FP16Transformer(fp_config)
    try:
        fp_model.load_state_dict(torch.load(get_path("weights/baseline_model.pt"), map_location="cpu"))
        print("  Loaded FP16 Weights.")
    except Exception as e:
        print(f"  Warning: Could not load FP16 weights ({e}). Using random.")
    
    tps, peak = benchmark_inference(fp_model, device="cpu")
    results["FP16"]["TPS"] = tps
    results["FP16"]["PeakRAM_MB"] = peak
    results["FP16"]["Params"] = get_param_count(fp_model)
    results["FP16"]["SizeMB"] = measure_model_size(get_path("weights/baseline_model.pt"))
    
    print(f"  TPS: {tps:.2f}")
    print(f"  Peak RSS: {peak:.2f} MB")

    # 3. C++ Atomic
    print("\n[Atomic C++]")
    
    bin_path = get_path("embedded/atomic_model.bin")
    if os.path.exists(bin_path):
        results["CPP"]["SizeMB"] = os.path.getsize(bin_path) / (1024 * 1024)
    else:
        results["CPP"]["SizeMB"] = 0
        
    print(f"  Binary Size: {results['CPP']['SizeMB']:.2f} MB")
    
    # Run C++ binary
    try:
        cmd = ["./runner"]
        cwd = get_path("embedded")
        print(f"  Running: {cmd} in {cwd}")
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        # print(p.stdout) # Optional: Be verbose? User asked for minimal.
        
        # Parse output for TPS
        tps_found = False
        for line in p.stdout.splitlines():
            if "TPS:" in line:
                try:
                    val = float(line.split(":")[1].strip())
                    results["CPP"]["TPS"] = val
                    print(f"  TPS: {val:.2f}")
                    tps_found = True
                except (ValueError, IndexError):
                    pass
        
        if not tps_found:
             print("  Warning: Could not parse TPS from C++ output.")
             print("  Output snippet:", p.stdout[:200])

    except Exception as e:
        print(f"  C++ Run Failed: {e}")

    # 4. Save Results
    with open(get_path("benchmarks/results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nResults Saved to benchmarks/results.json.")
    return results

if __name__ == "__main__":
    run_benchmarks()

