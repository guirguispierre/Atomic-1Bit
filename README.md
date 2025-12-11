# Atomic-1Bit ⚛️
> *High Intelligence, Low Compute.*

**Atomic-1Bit** is a bare-metal, ultra-lightweight inference engine for **BitNet b1.58** (1.58-bit ternary models).
It proves that you don't need FP16 matrix multiplication to run modern AI. The core engine runs on **INT8 addition and subtraction** only.

## 🚀 The Stack
The project is divided into two stacks:

### 1. Research Stack (Python/PyTorch)
Located in `atomic_1bit/`.
- **Purpose**: Architecture design, training (`TinyStories`), and chat.
- **Components**:
    - `BitLinear`: Custom PyTorch layer with **Straight-Through Estimator (STE)** for training and C++ Kernel for inference.
    - `AtomicTransformer`: GPT-style Decoder-only model.
    - `GistEncoder`: "Thought Compression" module.

### 2. Bare Metal Stack (C++)
Located in `embedded/`.
- **Purpose**: Deployment on potatoes (Raspberry Pi, ESP32, Phones).
- **Specs**:
    - **Dependencies**: `None` (Standard C++ Library only).
    - **Structure**: Header-only library (`atomic_lib.h`) + Runner (`atomic_runner.cpp`).
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

### 2. Train the Model
Train a fresh model on the "TinyStories" dataset (or a subset):
```bash
# Interactive training script
python3 atomic_1bit/training/train.py
```
*Saves checkpoints to `weights/`.*

### 3. Chat (Python)
Talk to your trained model interactively (top-k sampling):
```bash
python3 atomic_1bit/python/chat.py
```

---

## 🛠️ Deployment (Embedded)

To run on bare metal, we "freeze" the model and run it with the C++ engine.

### Step 1: Export to Binary
We also define a "System Prompt" (Gist) that gets compressed into a vector.
```bash
# Find latest checkpoint
LATEST=$(ls -t weights/ckpt_*.pt | head -n1)

# Export (Injecting System Gist: "Once upon a time")
python3 atomic_1bit/utils/export_to_cpp.py --checkpoint "$LATEST" --output embedded/atomic_model.bin --prompt "Once upon a time"
```

### Step 2: Compile & Run C++ Engine
The runner generates tokens using the C++ logic.
```bash
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o atomic_engine
./atomic_engine
```
*Output: A stream of Token IDs (e.g., `12 45 99 ...`).*

### Step 3: Decode Output
Since the embedded runner is minimal, use this script to read the story:
```bash
python3 atomic_1bit/utils/decode.py
# Paste the numbers from Step 2
```

---

## 📱 ESP32 Porting Guide
Want to run this on an Arduino/ESP32?
Check out [embedded/ESP32_PORT_GUIDE.md](embedded/ESP32_PORT_GUIDE.md).

Key steps:
1. Copy `embedded/atomic_lib.h` to your Arduino project.
2. Use an SD Card for model storage (28MB fits easily).
3. Use the `load_model` and `forward` functions provided in the lib.

---

## 🧠 "The Magic Kernel"
The heart of Atomic-1Bit is `ternary_matmul`.
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
