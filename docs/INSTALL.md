# Installation Guide

## Requirements

- **OS**: macOS (Apple Silicon recommended) or Linux
- **Python**: 3.8 or higher
- **Compiler**: GCC/G++ or Clang with C++17 support (for C++ inference only)

## Python Setup

### 1. Clone the repository

```bash
git clone https://github.com/guirguispierre/Atomic-1Bit.git
cd Atomic-1Bit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` -- Model training and inference
- `tiktoken` -- Tokenization (GPT-2/GPT-4 compatible)
- `datasets` -- HuggingFace dataset loading
- `numpy` -- Reference math and verification
- `matplotlib` -- Benchmark plotting
- `psutil` -- Thermal monitoring
- `tqdm` -- Progress bars
- `pyyaml` -- Config file parsing

### 3. Verify installation

```bash
python3 -c "import torch; import tiktoken; print('Dependencies OK')"
```

## C++ Components

There are two separate C++ artifacts, and the first one is **not** optional.

### 1. Ternary kernel (`libatomic.so`) -- required for Python inference

`atomic_1bit/python/wrapper.py` loads this shared library, so
`python3 atomic_1bit/python/inference.py` and the kernel parity tests fail
without it:

```bash
cd atomic_1bit/core
make                    # defaults to BACKEND=CPU
```

Verify:

```bash
test -f atomic_1bit/core/libatomic.so && echo "kernel built"
```

### 2. Bare-metal runner -- needed to run exported models standalone

```bash
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o runner
```

### Build with Metal backend (macOS only)

```bash
cd atomic_1bit/core
make BACKEND=METAL
```

### Build with CUDA backend (NVIDIA GPU)

```bash
cd atomic_1bit/core
make BACKEND=CUDA
```

### Verify the build

```bash
cd embedded
./runner --help
```

## Pre-commit Hooks (Development)

If you plan to contribute, install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This enables automatic formatting (Black, isort) and linting (flake8) on each commit.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tiktoken'`**

```bash
pip install tiktoken
```

**C++ compilation fails with "no member named 'format'"**

Your compiler may not support C++17. Update GCC to 7+ or Clang to 5+:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install g++-11

# macOS (Xcode command line tools)
xcode-select --install
```

**Metal backend not found (macOS)**

Ensure Xcode command line tools are installed and you're on macOS 12+.

**Thermal monitoring shows "sensors unavailable"**

On Apple Silicon, temperature sensors may require `sudo`. The thermal monitor disables itself gracefully when sensors aren't accessible. This doesn't affect model training or inference.
