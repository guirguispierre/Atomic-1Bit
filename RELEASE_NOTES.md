# Atomic-1Bit v1.0 — "High Intelligence, Low Compute"

We are proud to announce the first stable release of **Atomic-1Bit**, a bare-metal inference engine for 1.58-bit large language models. This release marks the transition from research prototype to a verified, reproducible system.

## 🌟 Highlights

- **Bit-Exact Parity**: The custom C++ ternary kernel now matches the Python/NumPy reference implementation exactly (Verified with 0.0 mean difference).
- **Ultra-Portable Runtime**: A standalone C++ engine (`embedded/runner`) with **zero** external dependencies beyond the standard library.
- **Tiny Footprint**: Full functional language models deployed with **2.0 MB** disk sizes (62% smaller than FP16 baselines).

## 🚀 Performance (Apple Silicon Single-Thread)

- **Throughput**: ~160–170 Tokens/Second
- **Memory**: Model loads instantly; minimal heap usage.
- **Energy**: Integer-based arithmetic (`add`/`sub` only) significantly mimics the efficiency gains of hardware-native ternary operations.

## 🛠 Features

### Core
- **Ternary Matrix Multiplication Kernel**: Optimized correct implementation of `W {-1, 0, 1} * X {INT8}`.
- **Hybrid Quantization**: Activation quantization (INT8) combined with weight quantization (1.58-bit).
- **Gist Tokens**: Support for thought compression/system prompts via "Gist" vector injection.

### Tooling
- **`atomic_1bit/training/`**: Complete training stack (TinyStories & Pocket Models).
- **`atomic_1bit/utils/export_to_cpp.py`**: Robust exporter to convert PyTorch checkpoints to `.bin` format.
- **`benchmark/run_suite.py`**: Reproducible evaluation vs FP16 vectors.
- **`docs/COMMANDS.md`**: Comprehensive command reference.

## 🐛 Key Fixes in v1.0
- **Kernel Layout Mismatch**: Fixed a critical bug where the Python wrapper was passing `(K, N)` weights to a kernel expecting `(N, K)` (transposed) layout. Parity is now 100%.
- **Export Dimensions**: Exporter now strictly validates or requires manual dimension flags to prevent shape mismatches during model loading.
- **Stability**: Fixed memory alignment handling in C++ loader.

## 🔮 What's Next? (Roadmap)
- **v1.1**: SIMD Acceleration (AVX/NEON) for 2–4x speedups.
- **v1.2**: Metal/CUDA Backends.
- **v2.0**: General-Purpose Low-Bit Framework.

---

**Contributors**: @guirguispierre (Lead)
**License**: MIT
