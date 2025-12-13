# Atomic-1Bit ⚛️
> *High Intelligence, Low Compute.*

**Atomic-1Bit** is a bare-metal, ultra-lightweight inference engine for **BitNet b1.58** (1.58-bit ternary models).
It proves that you don't need FP16 matrix multiplication to run modern AI. The core engine runs on **INT8 addition and subtraction** only.

## 📊 Benchmarks & Results

We successfully trained and deployed an Atomic-1Bit Transformer on a subset of TinyStories. The model was exported to a standalone C++ binary for bare-metal inference.

### Key Achievements
- **Ultra-Low Memory Footprint**: The quantised model (1.58-bit weights) occupies just **2.0 MB** on disk, compared to **5.3 MB** for an equivalent FP16 baseline (**-62%**).
- **Portability**: The C++ runtime depends only on the standard library (STL) and compiles to a single, dependency-free executable.
- **Integer-Only Operations**: The core `BitLinear` layer replaces expensive floating-point multiplications with integer additions.

### Performance Comparison (Apple M-series CPU)

**Context**: Sequence Length=64, Gen Tokens=100, Batch Size=1.

| Metric | FP16 Baseline | Atomic-1Bit | Delta |
| :--- | :--- | :--- | :--- |
| **Model Size** | 5.3 MB | **2.0 MB** | **-62%** |
| **Parameters** | 1.33 M | 1.33 M | 0% |
| **Precision** | Float16 | Ternary {-1, 0, 1} | - |
| **Speed (Python)** | ~826 TPS | ~136 TPS | -83% (Unoptimized) |
| **Speed (C++ Bare)**| N/A | ~56 TPS | Portable Runtime |

**Visual Summary**

![Performance Chart](assets/chart_model_size.png)
![Speed Chart](assets/chart_speed.png)
![Text Samples](assets/text_samples_comparison.png)

*Note: Benchmarks simulate 1-bit logic on standard CPUs. Dedicated hardware (FPGA/ASIC) would yield significantly higher speedups.*

---

## 🚀 The Stack
The project is divided into three main components:

### 1. Research Stack (Python/PyTorch)
Located in `atomic_1bit/`.
- **Purpose**: Architecture design, training, and chat.
- **Components**: `BitLinear`, `AtomicTransformer`, `GistEncoder`.

### 2. Bare Metal Stack (C++)
Located in `embedded/`.
- **Purpose**: Deployment on constrained devices (Raspberry Pi, ESP32).
- **Structure**: Header-only library (`atomic_lib.h`) + Runner (`atomic_runner.cpp`).

### 3. Benchmarking Suite
Located in `benchmarks/`.
- **Purpose**: Reproducible performance evaluation against FP16 baselines.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+ (`pip install torch tiktoken datasets numpy matplotlib`)
- GCC/G++

### 1. Build Core (Optional)
```bash
cd atomic_1bit/core
make
cd ../..
```

### 2. Run Benchmarks
To run the full benchmark suite (Training -> C++ Export -> Inference Test):
```bash
python benchmarks/run_suite.py
```
See [docs/benchmarking.md](docs/benchmarking.md) for detailed methodology.

### 3. Interactive Training
To train the full model interactively:
```bash
python3 atomic_1bit/training/train.py
```

### 4. Deployment (Embedded)
Export your trained model to run on the C++ engine:
```bash
# 1. Export
python3 atomic_1bit/utils/export_to_cpp.py --model weights/pocket_final.pt --output embedded/atomic_model.bin --prompt "You are a helpful assistant."

# 2. Compile & Run
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o atomic_engine
./atomic_engine
```

---

## 🧠 Theory: "The Magic Kernel"
The heart of Atomic-1Bit is `ternary_matmul`. Instead of `Multiplication`, we do:
```cpp
if (weight == 1) acc += input;
if (weight == -1) acc -= input;
// if weight == 0, do nothing (Sparsity!)
```
This reduces energy consumption and memory bandwidth significantly.

**Gist Tokens**: The engine supports **Thought Compression**. We pre-compute a **Gist Vector** (System Prompt) and inject it into the attention stream with 0 compute cost at inference time.

![Gist Flow](assets/diagram_gist_flow_1765658121862.png)

---

## 📱 ESP32 Support
Check out [embedded/ESP32_PORT_GUIDE.md](embedded/ESP32_PORT_GUIDE.md) for running on Arduino/ESP32.

---

*License: MIT | Concept: BitNet b1.58*
