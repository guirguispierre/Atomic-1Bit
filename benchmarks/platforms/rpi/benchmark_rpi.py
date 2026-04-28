#!/usr/bin/env python3
"""
Atomic-1Bit Raspberry Pi 4 Benchmark

Benchmarks the Pocket model (4096 vocab, 256 dim, 4 layers) on
Raspberry Pi 4 (ARM Cortex-A72, NEON SIMD).

Measures:
  - Tokens per second (TPS) with and without NEON SIMD
  - Peak memory usage (RSS)
  - Thermal behavior during sustained inference

Usage:
    python3 benchmarks/platforms/rpi/benchmark_rpi.py \
        --model weights/pocket_final.pt \
        --steps 100

Requirements:
    - Raspberry Pi 4 (aarch64 recommended)
    - Built C++ kernel: cd atomic_1bit/core && make && cd ../..
    - Python packages: torch, psutil, numpy
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import threading

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --- Configuration ---
POCKET_CONFIG = {
    "vocab_size": 4096,
    "dim": 256,
    "depth": 4,
    "heads": 4,
    "context_length": 128,
}

THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"


def get_cpu_temp():
    """Read CPU temperature on Linux (Raspberry Pi)."""
    try:
        with open(THERMAL_ZONE, "r") as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    # Fallback: try vcgencmd (RPi-specific)
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Output: "temp=42.0'C"
            temp_str = result.stdout.strip().split("=")[1].replace("'C", "")
            return float(temp_str)
    except (FileNotFoundError, subprocess.TimeoutExpired, (IndexError, ValueError)):
        pass
    return None


def get_system_info():
    """Gather system information for the benchmark report."""
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "node": platform.node(),
    }

    # CPU info on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        for line in cpuinfo.split("\n"):
            if "model name" in line:
                info["cpu_model"] = line.split(":")[1].strip()
                break
            if "Hardware" in line:
                info["hardware"] = line.split(":")[1].strip()
    except FileNotFoundError:
        pass

    # Memory
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        info["total_ram_mb"] = round(mem.total / (1024 * 1024))
        info["available_ram_mb"] = round(mem.available / (1024 * 1024))

    # NEON support check
    info["neon_available"] = platform.machine() in ("aarch64", "armv7l", "arm64")

    # Initial temperature
    temp = get_cpu_temp()
    if temp is not None:
        info["initial_temp_c"] = temp

    return info


class MemoryMonitor:
    """Track peak RSS during benchmark."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not PSUTIL_AVAILABLE:
            return
        proc = psutil.Process(os.getpid())
        self.peak_rss_bytes = proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        if not PSUTIL_AVAILABLE:
            return 0.0
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.peak_rss_bytes / (1024 * 1024)


class ThermalLogger:
    """Log CPU temperature over time during benchmark."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.readings = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            temp = get_cpu_temp()
            if temp is not None:
                self.readings.append({
                    "time": time.monotonic(),
                    "temp_c": temp,
                })
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.readings

    def summary(self):
        if not self.readings:
            return {"available": False}
        temps = [r["temp_c"] for r in self.readings]
        return {
            "available": True,
            "min_c": min(temps),
            "max_c": max(temps),
            "avg_c": sum(temps) / len(temps),
            "samples": len(temps),
        }


def benchmark_python(model_path, gen_steps, device="cpu"):
    """Benchmark Python/PyTorch inference."""
    if not TORCH_AVAILABLE:
        print("  [SKIP] PyTorch not available")
        return None

    from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

    config = AtomicConfig(**POCKET_CONFIG)
    model = AtomicTransformer(config)

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("  Loaded checkpoint:", model_path)
    else:
        print("  Using random weights (no checkpoint)")

    model.eval()
    model.to(device)

    # Warmup
    x = torch.randint(0, POCKET_CONFIG["vocab_size"], (1, 1))
    for _ in range(3):
        with torch.no_grad():
            model(x)

    # Benchmark
    mem_mon = MemoryMonitor()
    thermal = ThermalLogger()
    mem_mon.start()
    thermal.start()

    token = 58  # start token
    x = torch.tensor([[token]])
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(gen_steps):
            logits = model(x)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            x = torch.cat([x, next_token], dim=1)
            if x.size(1) > POCKET_CONFIG["context_length"]:
                x = x[:, -POCKET_CONFIG["context_length"]:]

    elapsed = time.perf_counter() - start
    peak_mb = mem_mon.stop()
    thermal_data = thermal.summary()
    thermal.stop()

    tps = gen_steps / elapsed
    return {
        "tps": round(tps, 2),
        "total_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak_mb, 1),
        "thermal": thermal_data,
    }


def benchmark_cpp(bin_path, gen_steps, runner_path=None):
    """Benchmark C++ runner inference."""
    if runner_path is None:
        runner_path = os.path.join(ROOT_DIR, "embedded", "runner")

    if not os.path.exists(runner_path):
        print("  [SKIP] C++ runner not found at", runner_path)
        print("  Build with: cd embedded && g++ -O3 -std=c++17 atomic_runner.cpp -o runner")
        return None

    if not os.path.exists(bin_path):
        print("  [SKIP] Model binary not found at", bin_path)
        return None

    mem_mon = MemoryMonitor()
    thermal = ThermalLogger()
    mem_mon.start()
    thermal.start()

    start = time.perf_counter()
    try:
        result = subprocess.run(
            [runner_path, "--model", bin_path, "--steps", str(gen_steps),
             "--temp", "0.0", "--seed", "42", "--start_token", "58"],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(runner_path),
        )
    except subprocess.TimeoutExpired:
        print("  [ERROR] C++ runner timed out after 300s")
        mem_mon.stop()
        thermal.stop()
        return None

    elapsed = time.perf_counter() - start
    peak_mb = mem_mon.stop()
    thermal_data = thermal.summary()
    thermal.stop()

    # Parse TPS from output
    tps = None
    for line in result.stdout.splitlines():
        if "TPS:" in line:
            try:
                tps = float(line.split("TPS:")[1].strip())
            except ValueError:
                pass

    return {
        "tps": tps if tps else round(gen_steps / elapsed, 2),
        "total_time_s": round(elapsed, 3),
        "peak_rss_mb": round(peak_mb, 1),
        "thermal": thermal_data,
        "exit_code": result.returncode,
    }


def benchmark_cpp_no_simd(bin_path, gen_steps):
    """
    Benchmark C++ runner compiled WITHOUT SIMD.

    Rebuilds the kernel with SIMD disabled, benchmarks, then restores.
    If rebuild is not possible, skips this benchmark.
    """
    kernel_dir = os.path.join(ROOT_DIR, "atomic_1bit", "core")
    runner_src = os.path.join(ROOT_DIR, "embedded", "atomic_runner.cpp")
    runner_no_simd = os.path.join(ROOT_DIR, "embedded", "runner_no_simd")

    # Try to compile a no-SIMD version
    try:
        result = subprocess.run(
            ["g++", "-O3", "-std=c++17", "-DDISABLE_SIMD",
             runner_src, "-o", runner_no_simd],
            capture_output=True, text=True, timeout=60,
            cwd=os.path.join(ROOT_DIR, "embedded"),
        )
        if result.returncode != 0:
            print("  [SKIP] Could not compile no-SIMD runner")
            print("  ", result.stderr[:200])
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  [SKIP] g++ not available for no-SIMD build")
        return None

    bench = benchmark_cpp(bin_path, gen_steps, runner_path=runner_no_simd)

    # Cleanup
    try:
        os.remove(runner_no_simd)
    except OSError:
        pass

    return bench


def main():
    parser = argparse.ArgumentParser(description="Atomic-1Bit RPi Benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pt checkpoint (for Python benchmark)")
    parser.add_argument("--bin", type=str, default=None,
                        help="Path to .bin model (for C++ benchmark)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--skip-python", action="store_true",
                        help="Skip Python benchmark")
    parser.add_argument("--skip-cpp", action="store_true",
                        help="Skip C++ benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("Atomic-1Bit Raspberry Pi Benchmark")
    print("=" * 60)

    # System info
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info.get('cpu_model', sys_info['machine'])}")
    print(f"RAM: {sys_info.get('total_ram_mb', '?')} MB")
    print(f"NEON: {'Yes' if sys_info['neon_available'] else 'No'}")
    if "initial_temp_c" in sys_info:
        print(f"Temperature: {sys_info['initial_temp_c']:.1f} C")
    print(f"Generation steps: {args.steps}")

    results = {
        "system": sys_info,
        "config": POCKET_CONFIG,
        "gen_steps": args.steps,
    }

    # Python benchmark
    if not args.skip_python:
        print(f"\n--- Python/PyTorch Benchmark ---")
        py_result = benchmark_python(args.model, args.steps)
        if py_result:
            results["python"] = py_result
            print(f"  TPS: {py_result['tps']}")
            print(f"  Peak RSS: {py_result['peak_rss_mb']} MB")
            if py_result["thermal"].get("available"):
                print(f"  Temp: {py_result['thermal']['avg_c']:.1f} C avg, "
                      f"{py_result['thermal']['max_c']:.1f} C max")

    # C++ benchmark (with SIMD)
    if not args.skip_cpp:
        bin_path = args.bin or os.path.join(ROOT_DIR, "embedded", "atomic_model.bin")

        print(f"\n--- C++ Benchmark (with NEON SIMD) ---")
        cpp_result = benchmark_cpp(bin_path, args.steps)
        if cpp_result:
            results["cpp_simd"] = cpp_result
            print(f"  TPS: {cpp_result['tps']}")
            print(f"  Peak RSS: {cpp_result['peak_rss_mb']} MB")
            if cpp_result["thermal"].get("available"):
                print(f"  Temp: {cpp_result['thermal']['avg_c']:.1f} C avg, "
                      f"{cpp_result['thermal']['max_c']:.1f} C max")

        print(f"\n--- C++ Benchmark (without SIMD, scalar) ---")
        cpp_no_simd = benchmark_cpp_no_simd(bin_path, args.steps)
        if cpp_no_simd:
            results["cpp_scalar"] = cpp_no_simd
            print(f"  TPS: {cpp_no_simd['tps']}")
            print(f"  Peak RSS: {cpp_no_simd['peak_rss_mb']} MB")

        # SIMD speedup comparison
        if cpp_result and cpp_no_simd and cpp_result["tps"] and cpp_no_simd["tps"]:
            speedup = cpp_result["tps"] / cpp_no_simd["tps"]
            results["simd_speedup"] = round(speedup, 2)
            print(f"\n  NEON SIMD Speedup: {speedup:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    target_tps = 10.0
    for name, key in [("Python", "python"), ("C++ (SIMD)", "cpp_simd"),
                       ("C++ (scalar)", "cpp_scalar")]:
        if key in results:
            tps = results[key]["tps"]
            status = "PASS" if tps and tps >= target_tps else "BELOW TARGET"
            print(f"  {name:20s}: {tps:>8} TPS  [{status}]  (target: >{target_tps} TPS)")

    # Save results
    output_path = args.output or os.path.join(
        ROOT_DIR, "benchmarks", "platforms", "rpi", "results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
