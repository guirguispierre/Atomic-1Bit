# Atomic-1Bit Raspberry Pi Benchmark

Benchmark the Pocket model on Raspberry Pi 4 (ARM Cortex-A72, NEON SIMD).

## Requirements

- Raspberry Pi 4 (2GB+ RAM, aarch64 OS recommended)
- Python 3.8+ with PyTorch
- Built C++ kernel and runner
- A trained Pocket model

## Setup

```bash
# 1. Build the C++ kernel with NEON support
cd atomic_1bit/core
make
cd ../..

# 2. Build the standalone runner
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o runner
cd ..

# 3. Export a model to .bin format
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/pocket_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 --depth 4 --heads 4 --vocab_size 4096 --context_len 128
```

## Running the Benchmark

```bash
# Full benchmark (Python + C++ with/without SIMD)
python3 benchmarks/platforms/rpi/benchmark_rpi.py \
  --model weights/pocket_final.pt \
  --bin embedded/atomic_model.bin \
  --steps 100

# C++ only
python3 benchmarks/platforms/rpi/benchmark_rpi.py \
  --bin embedded/atomic_model.bin \
  --skip-python --steps 200

# Save results to a specific file
python3 benchmarks/platforms/rpi/benchmark_rpi.py \
  --output results/rpi4_bench.json
```

## What's Measured

| Metric | Description |
|--------|-------------|
| TPS | Tokens per second (autoregressive generation) |
| Peak RSS | Peak memory usage during inference |
| Temperature | CPU thermal zone readings over time |
| SIMD speedup | TPS ratio: NEON-enabled vs scalar-only |

## Target Performance

- **Real-time conversational inference**: >10 TPS on the Pocket model
- The Pocket model (4096 vocab, 256 dim, 4 layers) is designed to fit
  comfortably within the RPi 4's 1-4GB RAM

## NEON SIMD Comparison

The benchmark compiles two versions of the C++ runner:
1. **With NEON**: Uses ARM NEON SIMD intrinsics for ternary matmul
2. **Without NEON**: Scalar fallback (`-DDISABLE_SIMD` flag)

This directly measures the speedup from SIMD vectorization on ARM Cortex-A72.
