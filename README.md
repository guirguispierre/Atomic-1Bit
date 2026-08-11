[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![BitNet b1.58](https://img.shields.io/badge/Weights-1.58--bit%20Ternary-green.svg)](#how-it-works)

# Atomic-1Bit

**Run language models using only addition and subtraction.** Atomic-1Bit is a bare-metal inference engine for 1.58-bit ternary models (BitNet b1.58) that replaces floating-point matrix multiplication with integer add/sub operations, cutting model size by 62% and enabling deployment on devices as small as an ESP32.

## Why Atomic-1Bit?

Most LLM inference requires expensive GPU hardware and gigabytes of memory. Even "small" models assume you have a modern GPU or at least a fast CPU with plenty of RAM.

Atomic-1Bit takes a different approach. By quantizing weights to just three values **{-1, 0, 1}**, we eliminate floating-point multiplication entirely:

```
weight ==  1  ->  accumulator += input
weight == -1  ->  accumulator -= input
weight ==  0  ->  skip  (free sparsity)
```

The result is a full transformer that runs on **integer arithmetic only**. No CUDA required. No FP16. No matrix multiply units. Just `add` and `sub` instructions that work on any processor manufactured in the last 30 years.

**This matters because:**
- A 6.85M parameter model drops from **27.4 MB to 5.8 MB** (-79%)
- The C++ runtime has **zero external dependencies** -- it's a single binary
- It runs on a **Raspberry Pi**, an **ESP32**, or a **2015 laptop**
- The ternary kernel is **bit-exact** against the NumPy reference, and the C++
  engine tracks the PyTorch model to float32 precision (see
  [Numerical parity](#numerical-parity))

> This is experimental research software. It works, it's verified, and it's honest about what it is: a proof that useful AI inference doesn't require expensive hardware.

---

## Performance

Benchmarked on Apple M-series, single thread, 50 generated tokens, using the
Quick Start config below (256d, 6 layers, 4 heads, 4096 vocab, 6.85M params):

| Metric | FP32 Baseline | Atomic-1Bit | Improvement |
|:---|:---|:---|:---|
| **Model Size** | 27.4 MB | **5.8 MB** | **-79%** |
| **Parameters** | 6.85M | 6.85M | Same |
| **Precision** | Float32 | Ternary {-1, 0, 1} | -- |
| **Throughput (C++)** | N/A | **~170-185 TPS** | Portable runtime |

The C++ runner recomputes the full prefix each step rather than keeping a
KV-cache, so throughput falls off as the sequence grows -- generation is
O(n^2) in the number of tokens. The Python path (`AtomicTransformer.generate`)
*does* cache. Reproduce with `benchmarks/run_suite.py`.

<details>
<summary>Visual benchmarks</summary>

![Model Size Comparison](assets/chart_model_size.png)
![Speed Comparison](assets/chart_speed.png)
![Text Samples](assets/text_samples_comparison.png)

</details>

### Quality baseline

Size and speed numbers only matter if the model says something. Training the
Quick Start config on TinyStories for 15k steps (about 25 minutes on an M-series
laptop) gives:

| Metric | Value |
|:---|---:|
| Held-out perplexity | **101.3** |
| Uniform baseline (learned nothing) | 4096 |
| Distinct trigram ratio | 0.98 |
| 5-gram repetition rate | 0.00 |

Sampled from the C++ engine at `temp 0.8`, reading a 5.6 MB packed binary:

```
One day, "Let, He would the king. "What, Ben?" "" ?" Ben said said, I said,
"Let'm promise go to go and Tom. "Mom said, we look one more. It, "No the way.
```

That is the honest state of it: the register, names and phrasing of TinyStories
with local grammar, but it does not hold a sentence together, let alone a story.
15k steps on 6.85M parameters buys a working pipeline and a number to improve
against, not a good storyteller. Reproduce with `python3
atomic_1bit/training/train.py`; checkpoints land in `weights/`, which is
gitignored.

---

## Quick Start

### Prerequisites

- Python 3.8+
- GCC/G++ or Clang (for C++ inference, C++17 support required)
- macOS (Apple Silicon recommended) or Linux

### Install

```bash
git clone https://github.com/guirguispierre/Atomic-1Bit.git
cd Atomic-1Bit
pip install -r requirements.txt
```

### Build the ternary kernel

The Python inference path loads a shared library, so build it first:

```bash
make -C atomic_1bit/core          # produces atomic_1bit/core/libatomic.so
```

### Verify the kernel works

This confirms the C++ ternary kernel matches the Python/NumPy reference exactly:

```bash
python3 atomic_1bit/python/inference.py
# Expected: ">> SUCCESS: Kernel Output Matches Reference."
```

### Train a model

```bash
# Train on TinyStories dataset (~15k steps)
python3 atomic_1bit/training/train.py
```

### Export and run on bare metal

```bash
# Export trained model to binary
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 --depth 6 --heads 4 --vocab_size 4096 --context_len 128

# Compile the C++ engine
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o runner

# Generate text
./runner --model atomic_model.bin --steps 100 --temp 0.7 --seed 42 --start_token 58
```

See [docs/COMMANDS.md](docs/COMMANDS.md) for the full command reference and [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

---

## How It Works

Atomic-1Bit implements a standard transformer architecture (embeddings, multi-head attention, feed-forward layers) with one critical difference: every linear layer uses **BitLinear** instead of `nn.Linear`.

During training, weights are quantized to {-1, 0, 1} using a straight-through estimator (STE), which lets gradients flow through the discrete quantization step. Activations are quantized to INT8. At inference time, the entire forward pass reduces to integer additions and subtractions.

The project has three components:

1. **Research stack** (`atomic_1bit/`) -- PyTorch training, evaluation, and model architecture. Train on TinyStories or Alpaca-cleaned datasets with thermal safety monitoring, gradient accumulation, and cosine scheduling.

2. **Bare-metal runtime** (`embedded/`) -- Standalone C++ inference engine with zero dependencies. Supports CPU, Metal (Apple Silicon), and CUDA backends through conditional compilation.

3. **Gist tokens** -- Pre-computed "thought vectors" that compress a system prompt into a single embedding, injected into the attention stream at zero inference cost.

### Numerical parity

Two different guarantees are worth separating, because only one of them is exact:

| Claim | Status | Verified by |
|:---|:---|:---|
| C++ ternary kernel == NumPy int32 reference | **Bit-exact** | `tests/test_kernel_parity.py` |
| C++ engine == PyTorch model, whole forward pass | Agrees to float32 precision | `tests/test_cpp_runtime_parity.py` |

Whole-model bit-equality is *not* achievable by design, and the reason is worth
knowing before you go hunting for a bug. Activations are quantized with a
per-tensor scale derived from a max over float values. `round()` is
discontinuous, so a last-ULP difference anywhere upstream -- summation order in
an RMS norm, a `tanh` implementation -- can push one value across a `.5`
boundary. That shifts a single int8 by one step, which is a *discrete* jump in
the accumulator, not an epsilon. Deeper models give this more chances to happen.

In practice a shallow model matches PyTorch to ~1e-7 relative, and deeper ones
stay within a few percent with the same next-token prediction. Greedy decoding
can still fork after enough steps once two logits are near-tied. The test suite
pins both bounds, so a genuine mismatch -- a different activation function,
scale convention, or weight layout -- fails loudly instead of hiding in the
noise.

![Ternary Matmul Diagram](assets/diagram_ternary_matmul_1765658104682.png)

For more details, see [docs/USAGE.md](docs/USAGE.md).

---

## Project Structure

```
atomic_1bit/
  model/          Transformer architecture (BitLinear, BitAttention)
  nn/             Core layers (BitLinear with STE quantization)
  training/       Training scripts (TinyStories, Alpaca, Pocket)
  evaluation/     Quality metrics (perplexity, coherence, diversity)
  python/         Python inference, chat interface, kernel wrapper
  utils/          Export, gist generation, thermal monitoring
  core/           C++ kernels (CPU, Metal, CUDA backends)
  tokenizers/     Tokenizer abstraction layer
  config.py       YAML/JSON configuration system
embedded/         Standalone C++ runner + ESP32 port guide
configs/          Model presets (4K pocket to 12.5M flagship)
benchmarks/       Reproducible benchmark suite vs FP16 baselines
tests/            149 pytest tests for correctness verification
scripts/          Plotting, evaluation, and reproduction scripts
docs/             Installation, usage, commands, benchmarking guides
examples/         Runnable example scripts
```

---

## Model Configurations

| Config | Parameters | Exported size | Dimensions | Use Case |
|:---|:---|:---|:---|:---|
| [`pocket_4k`](configs/pocket_4k.yaml) | 5.28M | 5.4 MB | 256d, 4L, 4H, 4K vocab | Smallest preset |
| [`stories_base`](configs/stories_base.yaml) | 30.5M | 56.0 MB | 256d, 6L, 4H, 50257 vocab | Development / testing |
| [`flagship_12m`](configs/flagship_12m.yaml) | 12.5M | 8.4 MB | 320d, 8L, 5H, 4K vocab | Quality demos |
| [`mixed_precision`](configs/mixed_precision.yaml) | Configurable | -- | Hybrid 1.58/4-bit | Experimental |

Ternary weights are packed four per byte, so what dominates the file is now the
token embedding table, which stays float32 (`vocab_size x dim x 4` bytes) --
vocabulary size, not depth, drives size. `stories_base.yaml` ships with the
50257-entry GPT-2 vocabulary; the Quick Start above passes `--vocab_size 4096`
instead, giving a 6.85M-parameter model that exports to 5.8 MB. Set the
vocabulary deliberately for your target.

`pocket_4k` and `flagship_12m` now fit 8 MB ESP32 flash. Neither fits in SRAM,
so on-device inference streams weights from flash: see
[embedded/ESP32_PORT_GUIDE.md](embedded/ESP32_PORT_GUIDE.md). Quantizing the
embedding table is the next lever if you need smaller.

Load any config with:

```python
from atomic_1bit.config import load_config, config_to_atomic
config = config_to_atomic(load_config("configs/stories_base.yaml"))
```

---

## Requirements

| Dependency | Version | Purpose |
|:---|:---|:---|
| Python | 3.8+ | Training and evaluation |
| PyTorch | >= 1.13.0 | Model training |
| tiktoken | >= 0.5.0 | Tokenization |
| datasets | >= 2.14.0 | HuggingFace datasets |
| NumPy | >= 1.24.0 | Reference math |
| matplotlib | >= 3.7.0 | Benchmark plots |
| psutil | >= 5.9.0 | Thermal monitoring |
| tqdm | >= 4.65.0 | Progress bars |
| PyYAML | >= 6.0 | Config files |
| GCC/Clang | C++17 | C++ inference engine |

**Hardware**: Any machine with a CPU. Apple Silicon recommended for Metal backend. NVIDIA GPU optional for CUDA backend. Tested down to ESP32-S3 for embedded inference.

---

## Running Tests

```bash
pip install pytest

# Run the full test suite
pytest tests/ -v

# Run specific test modules
pytest tests/test_bitlinear.py -v
pytest tests/test_kernel_parity.py -v
```

Two modules need compiled artifacts and skip without them:
`test_kernel_parity.py` needs `make -C atomic_1bit/core`, and
`test_cpp_runtime_parity.py` needs the runner
(`cd embedded && g++ -O3 -std=c++17 atomic_runner.cpp -o runner`). Build both
before trusting a green run.

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues, pull requests, and code style expectations.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan and [CHANGELOG.md](CHANGELOG.md) for per-release detail.

- **v1.0** -- Parity-verified ternary inference (done)
- **v1.2** -- Hardware-native backends: Metal, CUDA, NEON/AVX2 SIMD (done)
- **v1.3** -- Model scaling, evaluation harness, 12.5M config (done)
- **v1.4** -- Format-correct C++ deployment, top-p / top-k sampling, robust CLIs (current)
- **Next** -- Mobile demos, mixed-precision training, HTTP serving wrapper

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Author**: [@guirguispierre](https://github.com/guirguispierre)
- **Issues**: [GitHub Issues](https://github.com/guirguispierre/Atomic-1Bit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/guirguispierre/Atomic-1Bit/discussions)
