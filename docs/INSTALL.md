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

Alternatively, `pip install .` installs the `atomic_1bit` package (same
requirements) and also compiles the C++ kernel in step 3 below.

### 3. Verify installation (clone to proof)

```bash
make verify
```

This single target builds the ternary kernel and the C++ runner, checks the
kernel against the NumPy reference, smoke-tests the runner binary, and runs
the quickstart demo. If everything is in place it ends with
`verify complete: kernel, runner, and quickstart all ran`.

Without make, the lighter check is:

```bash
python3 -c "import torch; import tiktoken; print('Dependencies OK')"
python3 examples/quickstart.py    # auto-builds the kernel on first run
```

## C++ Components

There are two separate C++ artifacts, and the first one is **not** optional.

### 1. Ternary kernel (`libatomic.so`) -- required for Python inference

`atomic_1bit/python/wrapper.py` loads this shared library. It is built
automatically -- by `pip install .`, by `make verify`, or by the wrapper
itself the first time it is imported -- so on a machine with a toolchain you
never build it by hand. To build it explicitly (or to pick a non-CPU
backend):

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

(`make runner` from the repo root does the same thing. The benchmark suite
also builds the runner itself if it is missing.)

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
