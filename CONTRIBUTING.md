# Contributing to Atomic-1Bit

Thank you for your interest in contributing to Atomic-1Bit! This document will help you get started with the development workflow, understand our code standards, and submit high-quality contributions.

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Parity Verification](#parity-verification)
- [Pull Request Process](#pull-request-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Branching Strategy](#branching-strategy)
- [Building C++ Kernels](#building-c-kernels)
- [Training and Exporting Models](#training-and-exporting-models)
- [Common Pitfalls](#common-pitfalls)

---

## About the Project

**Atomic-1Bit** is a bare-metal inference engine for BitNet b1.58 (1.58-bit ternary neural networks). Our core principle: **no FP16 matrix multiplication — only INT8 addition and subtraction**. This enables ultra-low compute and memory footprint for deploying language models on resource-constrained devices.

### Guiding Principles

Before contributing, please understand these non-negotiable principles:

1. **Correctness before speed** — Never break parity for performance gains
2. **Parity before optimization** — Python ↔ C++ must produce bit-exact results
3. **Measured claims only** — Every benchmark must be reproducible
4. **Deployment-focused research** — If it can't run on bare metal, it's not done

For a full overview of the project architecture, read [`CLAUDE.md`](./CLAUDE.md) and [`ROADMAP.md`](./ROADMAP.md).

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

#### Required
- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **C++17-compatible compiler**:
  - Linux/macOS: `g++ 7+` or `clang 6+`
  - Windows: MSVC 2017+ or MinGW-w64
- **Git** for version control

#### Platform-Specific (Optional)
- **Metal support (Apple Silicon)**: Xcode Command Line Tools
- **CUDA support (NVIDIA GPUs)**: CUDA Toolkit 11.0+
- **Embedded platforms**: PlatformIO (for ESP32), Emscripten (for WebAssembly)

### Setting Up the Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Atomic-1Bit.git
   cd Atomic-1Bit
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - `torch>=1.13.0` — PyTorch for training and model definition
   - `tiktoken>=0.5.0` — Tokenization
   - `datasets>=2.14.0` — HuggingFace datasets (TinyStories, Alpaca)
   - `numpy>=1.24.0` — Numerical operations
   - `matplotlib>=3.7.0` — Visualization
   - `psutil>=5.9.0` — System monitoring (thermal safety)
   - `tqdm>=4.65.0` — Progress bars
   - `pyyaml>=6.0` — YAML config parsing

4. **Install development dependencies**:
   ```bash
   pip install pytest black isort flake8 pre-commit
   ```

5. **Set up pre-commit hooks** (recommended):
   ```bash
   pre-commit install
   ```

   This automatically runs code formatting, linting, and parity checks before each commit.

6. **Build the C++ kernel** (choose your backend):
   ```bash
   cd atomic_1bit/core

   # CPU backend (default)
   make

   # Metal backend (Apple Silicon)
   make BACKEND=METAL

   # CUDA backend (NVIDIA GPUs)
   make BACKEND=CUDA

   cd ../..
   ```

   This produces `libatomic.so` (or `libatomic.dylib` on macOS) in `atomic_1bit/core/`.

7. **Verify installation**:
   ```bash
   # Test model architecture
   python3 atomic_1bit/tests/test_model.py

   # Verify C++ kernel parity
   python3 atomic_1bit/python/inference.py
   ```

   If you see "Parity check PASSED" for the kernel test, you're ready to contribute!

---

## Development Workflow

### Making Changes

1. **Create a feature branch** (see [Branching Strategy](#branching-strategy)):
   ```bash
   git checkout -b feature/v1.3-your-feature-name
   ```

2. **Make your changes** in small, logical commits.

3. **Run tests frequently**:
   ```bash
   pytest tests/ -v
   ```

4. **Verify parity** after kernel or quantization changes:
   ```bash
   python3 atomic_1bit/python/inference.py
   ```

5. **Format and lint your code** (or let pre-commit do it):
   ```bash
   black atomic_1bit/
   isort atomic_1bit/
   flake8 atomic_1bit/ --max-line-length=120 --ignore=E203,W503
   ```

6. **Commit your changes** following our [commit message conventions](#commit-message-conventions):
   ```bash
   git add .
   git commit -m "feat(training): add gradient accumulation scaling"
   ```

7. **Push and open a pull request**:
   ```bash
   git push origin feature/v1.3-your-feature-name
   ```

---

## Code Style Guidelines

### Python

- **Follow PEP 8** with a 120-character line limit
- **Use type hints** for public function signatures:
  ```python
  def activation_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
      """Quantize activations to INT representation."""
      ...
  ```
- **Add docstrings** to classes and complex functions:
  ```python
  class BitLinear(nn.Module):
      """1.58-bit ternary linear layer with Straight-Through Estimator.

      Args:
          in_features: Input dimension
          out_features: Output dimension
      """
      ...
  ```
- **Use descriptive variable names**: `activation_scale` not `as_`
- **Prefer explicit over implicit**: `torch.int8` over `torch.int`

### C++

- **Use C++17** features (no C++20 to maintain compatibility)
- **No external dependencies** in embedded code (`embedded/` and `atomic_1bit/core/`)
- **Naming conventions**:
  - Functions: `snake_case` (e.g., `ternary_matmul`)
  - Constants: `UPPER_CASE` (e.g., `MAX_CONTEXT_LENGTH`)
  - Classes: `PascalCase` (e.g., `ThermalMonitor`)
- **Memory safety**: Prefer `std::vector` over raw pointers, use RAII
- **Platform-specific code**: Use `#ifdef` guards for backend-specific implementations

### Formatting

Our pre-commit hooks automatically enforce:
- **Black** for Python formatting
- **isort** for import ordering (profile: black)
- **flake8** for linting (max-line-length: 120, ignores: E203, W503)

To manually format:
```bash
black .
isort . --profile black
flake8 . --max-line-length=120 --ignore=E203,W503
```

---

## Testing Requirements

All contributions must include appropriate tests. We use `pytest` for the Python test suite.

### Running the Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bitlinear.py -v

# Run tests matching a pattern
pytest tests/ -k "parity" -v

# Run with coverage report
pytest tests/ --cov=atomic_1bit --cov-report=html
```

### Test Structure

Tests are organized under `tests/`:

- `test_bitlinear.py` — BitLinear layer quantization, STE gradients, edge cases
- `test_transformer.py` — Model architecture, forward pass shapes, parameter counts
- `test_gist.py` — Gist encoding and injection correctness
- `test_export.py` — Export → load round-trip, header validation
- `test_kernel_parity.py` — **Critical**: C++ kernel vs NumPy reference
- `test_thermal.py` — ThermalMonitor behavior (with mocked sensors)
- `test_inference.py` — End-to-end: load checkpoint → generate → verify

### Writing New Tests

When adding new features:

1. **Create tests before implementation** (TDD approach recommended)
2. **Use fixtures** defined in `conftest.py` for common setup:
   ```python
   def test_my_feature(sample_config, small_model):
       # sample_config and small_model are fixtures
       ...
   ```
3. **Test edge cases**: Zero inputs, single-element tensors, max dimensions
4. **Parametrize tests** for multiple configurations:
   ```python
   @pytest.mark.parametrize("dim", [64, 128, 256])
   def test_across_dimensions(dim):
       ...
   ```

---

## Parity Verification

**Critical**: The Python and C++ implementations must produce bit-exact identical results. This is the foundation of the project's correctness guarantees.

### When to Verify Parity

Run parity checks after making changes to:
- Quantization logic (`atomic_1bit/nn/layers.py`)
- C++ kernels (`atomic_1bit/core/backends/`)
- Python-C++ bridge (`atomic_1bit/python/wrapper.py`)
- Export/import pipeline (`atomic_1bit/utils/export_to_cpp.py`, `embedded/atomic_lib.h`)

### Parity Check Commands

```bash
# Quick parity check (ternary matmul kernel)
python3 atomic_1bit/python/inference.py

# Full layer-by-layer parity check
python3 tools/parity_check.py

# Parity test in test suite
pytest tests/test_kernel_parity.py -v
```

Expected output:
```
✓ Parity check PASSED: Python and C++ outputs match (max diff: 0.0)
```

### What to Do if Parity Fails

1. **Do NOT proceed** with your changes until parity is restored
2. Check if you introduced floating-point operations in the C++ kernel
3. Verify weight quantization matches between Python and C++
4. Use the layer-by-layer parity checker to isolate the failing component:
   ```bash
   python3 tools/parity_check.py --verbose
   ```
5. Ask for help in your PR if you're stuck

---

## Pull Request Process

### Before Submitting

Checklist for your PR:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Parity verification passes: `python3 atomic_1bit/python/inference.py`
- [ ] Code is formatted: `black .` and `isort .`
- [ ] Linting passes: `flake8 . --max-line-length=120 --ignore=E203,W503`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] C++ kernel builds successfully: `cd atomic_1bit/core && make`
- [ ] You've tested on relevant platforms (CPU/Metal/CUDA if applicable)
- [ ] New features have corresponding tests
- [ ] Documentation is updated (docstrings, README, CLAUDE.md if needed)

### PR Template

When opening a pull request, include:

```markdown
## Summary
Brief description of what this PR does (1-2 sentences).

## Changes
- List of key changes
- Organized by category (feat/fix/perf/etc.)

## Testing
- [ ] All tests pass
- [ ] Parity verification: [PASSED/FAILED]
- [ ] Manual testing performed: [describe what you tested]

## Benchmarks (if applicable)
- Performance before: X tokens/sec
- Performance after: Y tokens/sec
- Speedup: Z%

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No breaking changes (or migration guide provided)
```

### Review Process

1. **Automated checks** will run (once CI is set up):
   - Test suite execution
   - Parity verification
   - Linting checks

2. **Manual review** by maintainers:
   - Code quality and adherence to principles
   - Test coverage
   - Performance impact (if applicable)
   - Documentation completeness

3. **Approval and merge**:
   - At least one maintainer approval required
   - All discussions resolved
   - CI checks passing

---

## Commit Message Conventions

We use **Conventional Commits** format for clear, semantic commit history.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat:` — New feature
- `fix:` — Bug fix
- `perf:` — Performance improvement
- `refactor:` — Code restructuring (no behavior change)
- `test:` — Adding or updating tests
- `docs:` — Documentation changes
- `build:` — Build system or dependency changes
- `ci:` — CI/CD configuration
- `style:` — Formatting changes (no code logic change)

### Scope (Optional)

The part of the codebase affected:
- `training` — Training scripts and data loaders
- `kernel` — C++ kernel implementations
- `quantization` — Quantization logic
- `export` — Model export pipeline
- `embedded` — Embedded runtime
- `v1.3`, `v2.0` — Version-specific features

### Examples

```bash
# Feature addition
git commit -m "feat(training): add gradient accumulation scaling"

# Bug fix with detailed explanation
git commit -m "fix(kernel): correct ternary weight packing order

The previous implementation packed weights in column-major order,
but the unpacking assumed row-major. This caused incorrect results
for non-square weight matrices.

Fixes #42"

# Performance improvement
git commit -m "perf(kernel): AVX2 SIMD optimization for ternary matmul

Achieved 3.2x speedup on CPU inference (benchmark: M=1, N=320, K=320).
Parity verification: PASSED."

# Documentation
git commit -m "docs: update CONTRIBUTING.md with parity verification workflow"
```

### Best Practices

- **Keep commits atomic**: One logical change per commit
- **Write meaningful subjects**: Explain *what* and *why*, not *how*
- **Use imperative mood**: "Add feature" not "Added feature"
- **Reference issues**: Include `Fixes #123` or `Relates to #456` in the body
- **Include parity results**: For kernel changes, add "Parity: PASSED" to the commit message

---

## Branching Strategy

### Main Branches

- `master` — Stable releases (tagged as v1.0, v1.1, etc.)
- `develop` — Integration branch for next release
- `feature/*` — Feature development branches

### Feature Branches

Create feature branches from `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/v{version}-{description}
```

**Naming convention**:
- Version-specific: `feature/v1.3-quality-improvements`
- General: `feature/add-esp32-demo`
- Bug fixes: `fix/thermal-monitor-crash`
- Performance: `perf/simd-optimization`

### Examples from the Project

```
feature/v1.2-hardware-backends  # Metal/CUDA support
feature/v1.3-scaling            # Training quality improvements
fix/export-duplicate-calls      # Bug fix
perf/kv-cache-optimization      # Performance enhancement
```

### Merging

1. Feature branches merge into `develop`
2. `develop` merges into `master` for releases
3. Hotfixes can branch from `master` and merge back to both `master` and `develop`

---

## Building C++ Kernels

### CPU Backend (Default)

```bash
cd atomic_1bit/core
make clean
make
cd ../..
```

Produces: `atomic_1bit/core/libatomic.so` (or `.dylib` on macOS)

### Metal Backend (Apple Silicon)

```bash
cd atomic_1bit/core
make clean
make BACKEND=METAL
cd ../..
```

Compiles:
- `backends/metal_kernel.mm` → Objective-C++ Metal interface
- `backends/kernels.metal` → Metal shader code → `kernels.metallib`

### CUDA Backend (NVIDIA)

```bash
cd atomic_1bit/core
make clean
make BACKEND=CUDA
cd ../..
```

Requires: CUDA Toolkit installed, `nvcc` in PATH

### Debugging Build Issues

```bash
# Verbose build output
make VERBOSE=1

# Check compiler version
g++ --version    # or clang --version

# Test C++ compiler independently
cd atomic_1bit/core
g++ -std=c++17 -O3 -fPIC -shared backends/cpu_kernel.cpp -o test.so
```

Common issues:
- **Missing Metal SDK**: Install Xcode Command Line Tools
- **CUDA not found**: Add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`
- **Linker errors**: Check that all backend files are listed in Makefile

---

## Training and Exporting Models

### Training a Model

```bash
# TinyStories model (default: dim=256, depth=6, heads=4)
python3 atomic_1bit/training/train.py

# Pocket model (vocab=4096, embedded-optimized)
python3 atomic_1bit/training/train_pocket.py

# Flagship 12.5M instruct model (dim=320, depth=8, heads=5)
python3 atomic_1bit/training/train_instruct.py
```

Checkpoints save to `weights/`:
- `{model}_step_{N}.pt` — Periodic checkpoints
- `{model}_final.pt` — Final checkpoint
- `{model}_thermal_safe.pt` — Safety checkpoint (if thermal pause triggered)

### Evaluating Quality

```bash
# Perplexity evaluation
python3 atomic_1bit/evaluation/perplexity.py --model weights/final.pt

# Full evaluation suite
python3 atomic_1bit/evaluation/run_eval.py --model weights/final.pt --output results.json
```

### Exporting to Binary Format

```bash
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 --depth 6 --heads 4 --vocab_size 4096 --context_len 128
```

**Critical**: Model dimensions (`--dim`, `--depth`, `--heads`, `--vocab_size`, `--context_len`) must **exactly match** the training configuration. Mismatches will cause silent errors or crashes.

### Running Embedded Inference

```bash
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o runner
./runner --model atomic_model.bin --steps 100 --temp 0.7 --seed 42
cd ..
```

---

## Common Pitfalls

### 1. Introducing Floating-Point Multiplication

The entire point of Atomic-1Bit is to avoid FP16 multiply operations. Ternary weights enable **add/sub-only** inference.

**Bad**:
```cpp
// DON'T DO THIS
result += weight * input;  // Multiplication in hot path
```

**Good**:
```cpp
// Correct ternary dispatch
if (weight == 1)  result += input;
if (weight == -1) result -= input;
// if weight == 0: do nothing
```

### 2. Breaking Parity

Always verify parity after kernel changes. If you optimize the C++ kernel and parity fails, revert and investigate.

### 3. Dimension Mismatches

When exporting models, double-check that export dimensions match training config:

```python
# Training config
config = AtomicConfig(dim=320, depth=8, heads=5, vocab_size=50257, context_length=256)

# Export command MUST match
python3 atomic_1bit/utils/export_to_cpp.py \
  --dim 320 --depth 8 --heads 5 --vocab_size 50257 --context_len 256 \
  ...
```

### 4. Forgetting to Rebuild C++ Kernel

After modifying kernel code, you must rebuild:

```bash
cd atomic_1bit/core
make clean && make
cd ../..
python3 atomic_1bit/python/inference.py  # Verify parity
```

### 5. Modifying ATOM Binary Format

If you change the binary format in `export_to_cpp.py`, you **must**:
1. Bump the version number in the ATOM header
2. Update the reader in `embedded/atomic_lib.h`
3. Document the format change in a comment
4. Update all existing model exports

### 6. Committing Large Model Files

Never commit model checkpoints (`.pt` files) or exported binaries (`.bin` files) to Git. They should be in `.gitignore`.

```bash
# Bad
git add weights/model_final.pt  # DON'T DO THIS

# Good - use Git LFS or external storage
```

### 7. Disabling Thermal Safety

The `ThermalMonitor` in training scripts protects hardware from overheating. Do not disable it without understanding the thermal characteristics of your system.

---

## Getting Help

- **Read the docs**: [`CLAUDE.md`](./CLAUDE.md) has the full architecture reference
- **Check existing issues**: Someone may have encountered the same problem
- **Open an issue**: For bugs, feature requests, or questions
- **Ask in your PR**: Tag maintainers for specific guidance

---

## License

By contributing to Atomic-1Bit, you agree that your contributions will be licensed under the project's MIT License.

---

Thank you for contributing to Atomic-1Bit! Your work helps make efficient, deployable language models accessible to everyone.
