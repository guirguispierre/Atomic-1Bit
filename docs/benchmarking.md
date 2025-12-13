# Atomic-1Bit Benchmarking Guide

This directory contains scripts to benchmark the Atomic-1Bit model against a standard FP16 baseline.

## Directory Structure

- `run_suite.py`: Main script to run all benchmarks (Python Atomic, Python FP16, C++ Atomic) locally.
- `train_tiny.py`: Trains a small Atomic-1Bit model on a subset of TinyStories.
- `train_baseline.py`: Trains an equivalent FP16 baseline model.
- `baseline_fp16.py`: Definition of the FP16 baseline model.
- `results.json`: JSON output of the benchmark run.

## How to Reproduce

### 1. Requirements

Ensure you have the project dependencies installed:
```bash
pip install torch tiktoken datasets numpy matplotlib
```
For C++ benchmarks, you need a C++ compiler (g++ or clang++).

### 2. Train Models

First, train the small benchmark models to generate weights:

```bash
# 1. Train Atomic-1Bit Model (Generates weights/benchmark_model.pt & vocab)
python benchmarks/train_tiny.py

# 2. Train FP16 Baseline (Generates weights/baseline_model.pt)
python benchmarks/train_baseline.py
```

### 3. Compile C++ Engine

Compile the bare-metal C++ runner:

```bash
cd embedded
# Compile runner
g++ -O3 -pthread atomic_runner.cpp -o runner
cd ..
```

**Note**: You must export the trained Atomic model to C++ binary format before running the compiled engine.

```bash
python atomic_1bit/utils/export_to_cpp.py --model weights/benchmark_model.pt --output embedded/atomic_model.bin --dim 128 --depth 4 --heads 4 --context_len 64 --vocab_size 2048
```

### 4. Run Benchmarks

Run the full suite:

```bash
python benchmarks/run_suite.py
```

Results will be printed to console and saved to `benchmarks/results.json`.

### 5. Generate Plots

To regenerate the figures in `assets/`:

```bash
python scripts/generate_plots.py
```

## Methodology

- **Hardware**: Benchmarks are primarily intended for CPU (Apple M-series or x86).
- **Metric**: Tokens Per Second (TPS) generated greedily.
- **Models**:
    - **Atomic-1Bit**: 1.58-bit weights (ternary), FP32 activations (simulated in Python, unoptimized kernels in C++).
    - **FP16 Baseline**: Standard PyTorch implementation (Float32 on CPU for robustness).
