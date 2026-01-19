# Atomic-1Bit ⚛️
> *High Intelligence, Low Compute.*

**Atomic-1Bit** is a bare-metal, ultra-lightweight inference engine for **BitNet b1.58** (1.58-bit ternary models).
It proves that you don't need FP16 matrix multiplication to run modern AI. The core engine runs on **INT8 addition and subtraction** only.

## 📊 Benchmarks & Results

We successfully trained and deployed an Atomic-1Bit Transformer on a subset of TinyStories. The model was exported to a standalone C++ binary for bare-metal inference.

### Key Achievements
- **Numerical Parity Verified**: The modular C++ runtime produces **bit-exact** output matches to the Python reference implementation for CPU and Metal backends.
- **Ultra-Low Model Size**: The flagship 12.5M parameter 1.58-bit model occupies minimal disk space compared to FP16 baselines.
- **Portability**: The system supports conditional compilation for CPU, Metal, and CUDA backends.

### Performance Comparison (Apple M-series CPU)

**Context**: Sequence Length=128, Gen Tokens=50, Batch Size=1, Single Thread.

| Metric | FP16 Baseline | Atomic-1Bit | Delta |
| :--- | :--- | :--- | :--- |
| **Model Size** | 5.3 MB | **2.0 MB** | **-62%** |
| **Parameters** | 1.33 M | 1.33 M | 0% |
| **Precision** | Float16 | Ternary {-1, 0, 1} | - |
| **Speed (Python)** | ~826 TPS | ~130 TPS | -83% (Unoptimized) |
| **Speed (C++ CPU)**| N/A | **~160-170 TPS** | **Portable Runtime** |
| **Speed (Metal)**| N/A | **~TBD TPS** | **Apple Silicon Optimized** |
| **Speed (CUDA)**| N/A | **~TBD TPS** | **NVIDIA GPU Optimized** |

**Visual Summary**

![Performance Chart](assets/chart_model_size.png)
![Speed Chart](assets/chart_speed.png)
![Text Samples](assets/text_samples_comparison.png)

*Note: Benchmarks reflect single-core CPU performance. The ternary kernel is optimized for memory bandwidth efficiency.*

---

## 🚀 The Stack
The project is divided into three main components:

### 1. Research Stack (Python/PyTorch)
Located in `atomic_1bit/`.
- **Purpose**: Architecture design, training, and chat.
- **Components**: `BitLinear`, `AtomicTransformer`, `GistEncoder`.

### 2. Bare Metal Stack (C++)
Located in `embedded/` and `atomic_1bit/core/`.
- **Purpose**: Deployment on constrained devices (Raspberry Pi, ESP32) and high-performance hardware.
- **Structure**: Modular backend architecture (`backends/`) supporting CPU, Metal, and CUDA.
- **Components**: `atomic_lib.h`, `cpu_kernel.cpp`, `metal_kernel.mm`, `cuda_kernel.cu`.

### 3. Benchmarking Suite
Located in `benchmarks/`.
- **Purpose**: Reproducible performance evaluation against FP16 baselines.

---

## ⚡ Quick Start

For a full list of commands, see [docs/COMMANDS.md](docs/COMMANDS.md).

### Prerequisites
- Python 3.8+ (`pip install torch tiktoken datasets numpy matplotlib`)
- GCC/G++

### 1. Build Core (Optional)
```bash
cd atomic_1bit/core
make
cd ../..
```

### 2. Verify System
Check that the Atomic Kernel matches NumPy reference exactly:
```bash
python3 atomic_1bit/python/inference.py
```

### 3. Interactive Training
To train the model:
```bash
python3 atomic_1bit/training/train.py
```

### 4. Deployment (Embedded)
Export your trained model to run on the C++ engine:
```bash
# 1. Export (Must specify dimensions verified from checkpoint)
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 --depth 6 --heads 4 --vocab_size 4096 --context_len 128

# 2. Compile & Run
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o runner
./runner --model atomic_model.bin --steps 100 --temp 0.7 --seed 42 --start_token 58
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

## 🔥 Thermal Safety
Long-running training jobs include a built-in **Thermal Monitor** to prevent hardware overheating.
- **Auto-Pause**: If system temp > **80°C**.
- **Auto-Resume**: When system temp < **70°C**.
- **Safety**: Automatically saves a checkpoint (`*_thermal_safe.pt`) before pausing.

*Note: On Apple Silicon (M1/M2/M3), sensors may not be readable without `sudo`. The monitor will gracefully disable itself in this case.*

---

## 📱 ESP32 Support
Check out [embedded/ESP32_PORT_GUIDE.md](embedded/ESP32_PORT_GUIDE.md) for running on Arduino/ESP32.

---

*License: MIT | Concept: BitNet b1.58*
