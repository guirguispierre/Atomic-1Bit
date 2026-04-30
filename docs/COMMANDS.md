# Atomic-1Bit Command Reference

This document lists the essential commands for setting up, training, verifying, and deploying Atomic-1Bit models.

## 1. Environment Setup

### Requirements
- **OS**: macOS (Apple Silicon recommended) or Linux.
- **Python**: 3.8+
- **Compiler**: GCC/G++ (supporting C++17) or Clang.

### Installation
```bash
# Install Python dependencies
pip install torch tiktoken datasets numpy matplotlib tqdm

# Clone repository
git clone https://github.com/guirguispierre/Atomic-1Bit.git
cd Atomic-1Bit
```

---

## 2. Training (Python)

### Interactive Training (TinyStories)
Trains a model from scratch on the TinyStories dataset.
```bash
python3 atomic_1bit/training/train.py
```
- **Output**: Checkpoints saved to `weights/`.
- **Default Config**: Dim=256, Depth=6, Heads=4.

### Pocket Model Training (ESP32 Optimized)
Trains a smaller vocab model optimized for embedded usage.
```bash
python3 atomic_1bit/training/train_pocket.py
```
- **Output**: `weights/pocket_final.pt`.
- **Config**: Vocab=4096, Dim=256.

---

## 3. Python Inference & Verification

### Chat Interface
Interactive generation script to test model coherence.
```bash
python3 atomic_1bit/python/chat.py
```
- **Inputs**: Type your prompt interactively.
- **Output**: Generated text stream.

### Core Kernel Verification (System Check)
Verifies that the Atomic-1Bit C++ kernel (accessed via ctypes) matches NumPy reference math exactly.
```bash
python3 atomic_1bit/python/inference.py
```
- **Success Criteria**: Prints `>> SUCCESS: Kernel Output Matches Reference.`
- **Note**: Runs both a tiny debug test (8x16) and a large scale test (4096x4096).

### Model Architecture Test
Verifies layer connectivity and forward pass shapes.
```bash
python3 atomic_1bit/tests/test_model.py
```

---

## 4. Deployment (Embedded C++)

### Step 1: Export to Binary
Converts the PyTorch checkpoint (`.pt`) into a quantized, memory-mapped binary (`.bin`) for the C++ runner.
**CRITICAL**: You must specify the model dimensions manually if they differ from default.

**Command (for Verified Stories Model):**
```bash
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 \
  --depth 6 \
  --heads 4 \
  --vocab_size 4096 \
  --context_len 128
```

- **Arguments**:
  - `--model`: Path to input `.pt` file.
  - `--output`: Path to output `.bin` file.
  - `--dim`, `--depth`, etc.: Must match the trained model config.
- **Output**: `embedded/atomic_model.bin` (~2MB).

### Step 2: Build & Run C++ Engine
Compiles the standalone C++ runner.

**Build:**
```bash
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o runner
```

**Run:**
```bash
./runner --model atomic_model.bin --steps 100 --temp 0.0 --seed 42 --start_token 58
```

- **Flags**:
  - `--model`: Path to binary model file.
  - `--steps`: Number of tokens to generate.
  - `--temp`: Temperature (0.0 = Greedy/Deterministic, 0.7 = Creative).
  - `--top_k`: Keep top-K logits before sampling (0 = disabled).
  - `--top_p`: Nucleus sampling cutoff (0.0 = disabled, 0.9 = typical).
  - `--seed`: Random seed for reproducibility.
  - `--start_token`: Single initial token ID (e.g., 58 for "Once").
  - `--prompt 1,2,3`: Comma-separated starting token IDs (overrides `--start_token`).
  - `--stream` / `--no-stream`: Stream tokens as generated (default on) vs. buffer until end.
  - `--help`: Show all flags.

---

## 5. Benchmarking

### Full Suite
Runs a suite of benchmarks comparing 1.58-bit inference against FP16 baselines.
```bash
python3 benchmarks/run_suite.py
```
- **Outputs**:
  - `assets/chart_model_size.png`
  - `assets/chart_speed.png`
  - `assets/text_samples_comparison.png`

---

## Troubleshooting

**Q: Export limits mismatched shapes?**
A: Ensure you pass the exact `--dim`, `--depth`, `--heads` arguments to `export_to_cpp.py` that match your trained model.

**Q: C++ Runner prints garbage?**
A: Ensure you are using the correct `.bin` file. If updating the model, re-run `export_to_cpp.py` and ensure the old binary is overwritten.
