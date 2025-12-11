# Atomic-1Bit ⚛️
> *High Intelligence, Low Compute.*

**Atomic-1Bit** is a bare-metal, ultra-lightweight inference engine for **BitNet b1.58** (1.58-bit ternary models).
It proves that you don't need FP16 matrix multiplication to run modern AI. The core engine runs on **INT8 addition and subtraction** only.

## 🚀 The Stack
The project is divided into two stacks:

### 1. Research Stack (Python/PyTorch)
Located in `atomic_1bit/`.
- **Purpose**: Architecture design, training, and verification.
- **Components**:
    - `BitLinear`: Custom PyTorch layer using the C++ Kernel via ctypes.
    - `AtomicTransformer`: GPT-style Decoder-only model.
    - `GistEncoder`: "Thought Compression" module.

### 2. Bare Metal Stack (C++)
Located in `embedded/`.
- **Purpose**: Deployment on potatoes (Raspberry Pi, ESP32, Phones).
- **stats**:
    - **Dependencies**: `None` (Standard C++ Library only).
    - **Structure**: Single-file runner (`atomic_runner.cpp`).
    - **Input**: Custom binary format (`atomic_model.bin`).

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- GCC/G++

### 1. Build the Core Kernel
First, compile the shared library for the Python wrapper:
```bash
cd atomic_1bit/core
make
cd ../..
```

### 2. Run the Benchmark
See the 4x memory compression vs PyTorch Linear:
```bash
python3 atomic_1bit/benchmarks/benchmark.py
```

### 3. Run the Model (Research)
Verify the full Transformer forward pass:
```bash
python3 atomic_1bit/tests/test_model.py
```

---

## 🛠️ Deployment (Embedded)

To run on bare metal, we "freeze" the model and run it with the C++ engine.

### Step 1: Export to Binary
We also define a "System Prompt" (Gist) that gets compressed into a vector.
```bash
# Exports 'atomic_model.bin' with prompt "You are a helpful assistant"
python3 atomic_1bit/utils/export_to_cpp.py
```

### Step 2: Compile & Run
```bash
g++ -O3 -std=c++17 embedded/atomic_runner.cpp -o runner
./runner
```
*You should see `>> Gist Token Detected and Injected.` followed by logits.*

---

## 🧠 "The Magic Kernel"
The heart of Atomic-1Bit is `ternary_matmul` in `atomic_1bit/core/kernel.cpp`.
Instead of `Multiplication`, we do:
```cpp
if (weight == 1) acc += input;
if (weight == -1) acc -= input;
// if weight == 0, do nothing (Sparsity!)
```
This is mathematically equivalent to Matrix Multiplication for {-1, 0, 1} weights but requires significantly less energy and silicon area.

## 🔮 Gist Tokens
The engine supports **Thought Compression**. Instead of processing a long system prompt token-by-token at runtime, we pre-compute a **Gist Vector** (1, Dim). The C++ runner injects this vector into the attention stream, giving the model context with **0 compute cost** at inference time.

---

*License: MIT | Concept: BitNet b1.58*
