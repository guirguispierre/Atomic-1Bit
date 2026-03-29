# Atomic-1Bit Troubleshooting & Deployment Guide

This document covers common build and runtime errors, and provides a step-by-step production deployment workflow based on the actual source code in this repository.

---

## Troubleshooting

### Build Errors

#### Missing C++ compiler

**Symptom:**
```
make: g++: No such file or directory
```

**Cause:** The Makefile uses `g++` for CPU and Metal builds. It is not installed or not on `PATH`.

**Fix (Linux):**
```bash
sudo apt-get install build-essential
```

**Fix (macOS):** Install Xcode Command Line Tools:
```bash
xcode-select --install
```

After installation, verify with `g++ --version`.

---

#### Missing CUDA toolkit (`nvcc` not found)

**Symptom:**
```
make: nvcc: No such file or directory
```
or
```
CUDA Error: libcuda.so: cannot open shared object file
```

**Cause:** The CUDA backend sets `CXX = nvcc`. If the CUDA toolkit is not installed or `nvcc` is not on `PATH`, the build fails.

**Fix:** Install the CUDA toolkit matching your GPU driver. Then ensure `nvcc` is on `PATH`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify with `nvcc --version`. Then rebuild:
```bash
cd atomic_1bit/core
make BACKEND=CUDA
```

---

#### Metal framework not found

**Symptom:**
```
clang: error: framework 'Metal' not found
```

**Cause:** The Metal backend requires macOS with Xcode. The Makefile adds `-framework Metal -framework Foundation -lobjc`. These flags are only valid on macOS with Xcode installed.

**Fix:** Ensure Xcode (not just Command Line Tools) is installed:
```bash
xcode-select -p   # Should return /Applications/Xcode.app/Contents/Developer
```

If the path shows `/Library/Developer/CommandLineTools` only, install the full Xcode app from the Mac App Store. Then:
```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
cd atomic_1bit/core
make BACKEND=METAL
```

---

#### Metal shader compilation fails (`xcrun metal` error)

**Symptom:**
```
xcrun: error: unable to find utility "metal"
```
or a Metal compiler diagnostic during `make BACKEND=METAL`.

**Cause:** The Makefile runs `xcrun -sdk macosx metal -c backends/kernels.metal` to compile the `.metal` shader to `.air`, then links to `default.metallib`. If the Metal developer tools are not available or the shader has an error, this step fails.

**Fix:** Confirm Metal tools are present:
```bash
xcrun --find metal
```

If not found, reinstall Xcode and accept the license:
```bash
sudo xcodebuild -license accept
```

---

#### `libatomic.so` not found at Python import time

**Symptom:**
```
FileNotFoundError: Could not find shared library at .../atomic_1bit/core/libatomic.so.
Did you run 'make' in core/?
```

**Cause:** `atomic_1bit/python/wrapper.py` constructs an absolute path to `atomic_1bit/core/libatomic.so` and raises this error if the file does not exist.

**Fix:** Build the shared library before using any Python inference code:
```bash
cd atomic_1bit/core
make            # CPU backend
# or
make BACKEND=METAL
# or
make BACKEND=CUDA
```

---

### Dimension Mismatch Errors During Export

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict
```
or the export prints garbled layer shapes, or the runner prints unexpected config values after loading.

**Cause:** `export_to_cpp.py` accepts `--dim`, `--depth`, `--heads`, `--context_len`, and `--vocab_size` arguments. It constructs an `AtomicConfig` with those values and then calls `model.load_state_dict`. If any dimension does not match the checkpoint, PyTorch raises a shape mismatch error.

**Fix:** Use the exact values from the training script that produced the checkpoint.

| Script | dim | depth | heads | vocab_size | context_len |
|--------|-----|-------|-------|------------|-------------|
| `train.py` | 256 | 6 | 4 | 4096 | 128 |
| `train_instruct.py` | 320 | 8 | 5 | 4096 | 256 |

Example correct export for a TinyStories checkpoint:
```bash
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 --depth 6 --heads 4 --vocab_size 4096 --context_len 128
```

**Verifying dimensions from a checkpoint:**
```python
import torch
ckpt = torch.load("weights/stories_final.pt", map_location="cpu")
cfg = ckpt.get("config", {})
print(cfg)
# If "config" key is missing, infer from state_dict keys:
sd = ckpt.get("model_state_dict", ckpt)
print(sd["token_emb.weight"].shape)   # (vocab_size, dim)
print(sd["pos_emb.weight"].shape)     # (context_len, dim)
print(len([k for k in sd if "layers" in k and "ln1" in k]))  # depth
```

---

### Checkpoint Loading Failures

**Symptom:**
```
Error loading checkpoint: ...
Continuing with random weights (WARNING)...
```
or
```
Warning: N missing key(s) in checkpoint
```

**Cause 1:** The checkpoint was saved as a bare `state_dict` (no wrapping dict), but the loader looks for the `"model_state_dict"` key. Both `train.py` and `train_instruct.py` save with `"model_state_dict"` key—this warning appears if loading a checkpoint saved from a different script or older version.

The loaders handle this gracefully:
```python
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
```

**Cause 2:** The model architecture was changed between training runs, leaving stale or missing keys.

**Fix:** Use `strict=False` (already the default in training loaders) and check the missing/unexpected key report. If keys are legitimately missing due to an architecture change, retrain from scratch.

**Cause 3:** The `.pt` file is corrupted or written partially (e.g., interrupted during save). Training scripts save to the same path every 1000 steps. If interrupted mid-write, the file may be truncated.

**Fix:** Check file size:
```bash
ls -lh weights/stories_final.pt
```
A suspiciously small file indicates a truncated write. Use a `_thermal_safe.pt` backup if available, or an earlier checkpoint.

---

### Thermal Monitor Issues

**Symptom:**
```
ThermalMonitor: psutil.sensors_temperatures not available. Monitoring disabled.
```
or
```
ThermalMonitor: No temperature sensors found (common on macOS/WSL). Monitoring disabled.
```

**Cause:** `ThermalMonitor` uses `psutil.sensors_temperatures()`. On macOS and WSL, this API returns an empty dict or is unavailable. The monitor disables itself gracefully and training continues without thermal protection.

**Fix:** This is expected on macOS. No action is required. Training will proceed without pausing.

If running on Linux with hardware sensors and monitoring is desired, ensure `lm-sensors` is installed and configured:
```bash
sudo apt-get install lm-sensors
sudo sensors-detect
```

**Symptom:** Training pauses unexpectedly and prints `[THERMAL CRITICAL]`.

**Cause:** A hardware sensor exceeded the 80°C threshold. The monitor saves a safety checkpoint to `weights/<checkpoint>_thermal_safe.pt` (or `.safe` for `train_instruct.py`) and waits until all sensors drop below 70°C before resuming.

**Fix:** Allow the system to cool. Training will resume automatically. The safety checkpoint is available if you need to restart manually.

---

### Out of Memory During Training

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
or system swap thrashing on CPU training.

**Cause and fixes by script:**

**`train.py` (TinyStories, 256 dim):**
- Default batch size is 32 with context length 128. Reduce `BATCH_SIZE` in the config or YAML file if OOM occurs.
- The config is loaded from `configs/stories_base.yaml` if it exists. Edit `batch_size` there.

**`train_instruct.py` (12.5M model, 320 dim):**
- Uses gradient accumulation: physical batch 32, `GRAD_ACCUM_STEPS = 8` (effective batch 256).
- Reduce `BATCH_SIZE` (physical) to 16 or 8 and increase `GRAD_ACCUM_STEPS` proportionally to maintain the effective batch size.
- Context length is 256. Reducing it lowers memory further.

**General tips:**
- On Apple Silicon, the unified memory architecture means CPU and GPU share the same pool. Close other applications.
- The model itself is small (12.5M parameters at 4 bytes each = ~50 MB). OOM is almost always caused by activations during the forward/backward pass, not the model weights.

---

### Runner Fails to Load Model: Magic Mismatch

**Symptom (from runner output):**
```
Invalid file format: Magic mismatch (Expected 'ATOM')
```

**Cause:** The binary file was not exported by `export_to_cpp.py`, was truncated, or was exported using an older version of the script that did not write the magic header. The runner checks for the 4-byte magic `0x41544F4D` ("ATOM") at the start of the file.

**Fix:** Re-export using the current `export_to_cpp.py`. Do not hand-craft `.bin` files.

---

### Runner Fails with Unexpected Config Dimensions

**Symptom:** The runner prints `Sizeof(Config): N (Expected 24)` with a value other than 24, or shows obviously wrong values like `Dim=0`.

**Cause:** The `Config` struct in `atomic_runner.cpp` has 6 `int` fields (6 × 4 = 24 bytes). If compiled on a platform where `int` is not 4 bytes, or if the binary was produced with a different struct layout, the read will be misaligned.

**Fix:** Ensure the runner is compiled with standard C++ on a 32-bit-int platform (Linux/macOS x86-64 or ARM64). Use `g++ -O3 -std=c++17`.

---

## Deployment Guide

### Step-by-Step Production Deployment

This section walks through the complete pipeline from a trained checkpoint to running inference on a target device.

#### Step 1: Train the Model

Choose the appropriate training script for your use case:

```bash
# TinyStories language model (smallest, fastest to train)
python3 atomic_1bit/training/train.py

# Flagship instruct model (12.5M params, higher quality)
python3 atomic_1bit/training/train_instruct.py
```

Checkpoints are saved to `weights/`. Both scripts prompt for the number of additional training steps at startup and will resume from an existing checkpoint automatically.

Training logs for `train_instruct.py` are written to `weights/training_log.csv` with columns: `step, loss, lr, temp, step_time_ms`.

---

#### Step 2: Verify the Checkpoint

Before exporting, confirm the checkpoint loads correctly and produces coherent output:

```bash
python3 atomic_1bit/python/chat.py
```

For a parity check between Python and C++ inference:
```bash
python3 atomic_1bit/python/inference.py
python3 tools/parity_check.py
```

---

#### Step 3: Export to Binary

Export the trained checkpoint to the C++ binary format. Supply the exact dimensions used during training.

```bash
# TinyStories model
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 --depth 6 --heads 4 --vocab_size 4096 --context_len 128

# Instruct model
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/instruct_final.pt \
  --output embedded/atomic_model.bin \
  --dim 320 --depth 8 --heads 5 --vocab_size 4096 --context_len 256
```

To bake a system prompt into the model as a Gist vector (requires a trained `GistEncoder`):
```bash
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/instruct_final.pt \
  --output embedded/atomic_model_with_gist.bin \
  --dim 320 --depth 8 --heads 5 --vocab_size 4096 --context_len 256 \
  --prompt "You are a helpful assistant."
```

To use a pre-computed `.gist` file instead of computing it on-the-fly:
```bash
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/instruct_final.pt \
  --output embedded/atomic_model.bin \
  --dim 320 --depth 8 --heads 5 --vocab_size 4096 --context_len 256 \
  --gist_file path/to/system_prompt.gist
```

The exporter prints each tensor it writes, including per-layer weight scales, and reports the final file size in KB.

---

#### Step 4: Compile the Runner

```bash
cd embedded
g++ -O3 -std=c++17 atomic_runner.cpp -o runner
```

No external dependencies are required. `atomic_runner.cpp` is a self-contained C++17 binary with no libraries beyond the standard library.

---

#### Step 5: Run Inference

```bash
cd embedded
./runner --model atomic_model.bin --steps 100 --temp 0.7 --seed 42 --start_token 58
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `atomic_model.bin` | Path to the binary model file |
| `--steps` | `50` | Number of tokens to generate |
| `--temp` | `0.0` | Sampling temperature. `0.0` = greedy (deterministic), `>0` = stochastic |
| `--seed` | `42` | Random seed for reproducible stochastic sampling |
| `--start_token` | `42` | Starting token ID (integer) |
| `--parity` | off | Enable parity mode: forces `--steps 1` and prints layer-by-layer diagnostics |

The runner outputs token IDs separated by spaces, followed by tokens-per-second (TPS). Token IDs must be decoded externally using the vocabulary map saved to `weights/vocab_map_stories.json` or `weights/vocab_map_instruct.json`.

---

### Platform-Specific Notes

#### macOS (Apple Silicon)

- All three backends (CPU, Metal, CUDA) are in the codebase, but CUDA requires an NVIDIA GPU. Only CPU and Metal are meaningful on Apple Silicon.
- Metal is the recommended backend for performance on M-series chips:
  ```bash
  cd atomic_1bit/core
  make BACKEND=METAL
  ```
- The Metal build compiles `backends/kernels.metal` to `backends/kernels.air` and then links `default.metallib`. Both files are generated in `atomic_1bit/core/` and cleaned by `make clean`.
- `ThermalMonitor` is gracefully disabled on macOS (psutil does not expose sensor data). Training runs without thermal protection—monitor Activity Monitor manually on long runs.
- The training scripts detect MPS automatically: `if torch.backends.mps.is_available(): device = "mps"`.

#### Linux (x86-64 with NVIDIA GPU)

- Use the CUDA backend for the shared library:
  ```bash
  cd atomic_1bit/core
  make BACKEND=CUDA
  ```
- `ThermalMonitor` is fully functional on Linux if `lm-sensors` is configured.
- Training scripts detect CUDA automatically.
- The runner compiles and runs without GPU. Inference in `atomic_runner.cpp` is CPU-only.

#### Embedded / Microcontroller Targets

- `atomic_lib.h` is the intended header for embedded targets. It has no dependencies beyond the C++ standard library.
- Key limitations to address before deploying to a microcontroller:
  1. `std::vector` is used extensively for all intermediate buffers. Replace with static arrays or a memory arena. The `Workspace` struct in `atomic_runner.cpp` shows the pattern for pre-allocation.
  2. Heap allocation in `forward` (one `std::vector` allocation per function call) is unsuitable for RTOS environments. Pre-allocate all buffers at startup.
  3. The code comments note that ternary weights should be packed to 2 bits per weight for memory-constrained devices. The current format uses 1 byte per weight value.
- Recommended minimum RAM: For the TinyStories model (dim=256, depth=6):
  - Model weights: approximately 256 × 256 × 4 (attn) + 256 × 1024 × 2 (MLP) × 6 layers + embeddings ≈ ~5 MB for weights alone.
  - Activations and intermediate buffers: proportional to `max_seq_len × dim`. For seq_len=128, dim=256: 128 × 256 × 4 bytes × ~8 buffers ≈ ~1 MB.
  - Total: target at least 8 MB RAM for the small model.

---

### Recommended Hardware Requirements

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Training TinyStories model | 8 GB RAM, any CPU | Apple M-series or NVIDIA GPU, 16 GB RAM |
| Training instruct model (12.5M) | 16 GB RAM + GPU | Apple M2/M3 or NVIDIA RTX 3080+, 32 GB RAM |
| C++ inference (CPU) | Any 64-bit CPU, ~16 MB RAM | ARM Cortex-A or x86-64, 64 MB RAM |
| C++ inference (Metal) | Apple Silicon Mac | Apple M1 or later |
| C++ inference (CUDA) | NVIDIA GPU (any compute capability supported by installed CUDA toolkit) | NVIDIA GPU with 4+ GB VRAM |

---

### Performance Tuning Tips

**Training:**
- Gradient accumulation in `train_instruct.py` (`GRAD_ACCUM_STEPS = 8`) allows a large effective batch size with a small physical batch. Increase this if GPU memory is the bottleneck and you want to train with a larger effective batch without increasing physical batch size.
- The cosine learning rate schedule with warmup (`WARMUP_STEPS = 1000`) in `train_instruct.py` stabilizes early training. Do not reduce warmup steps below ~500 for fresh training runs.
- Checkpoints are saved every 1000 steps to `weights/<name>_final.pt`. Adjust this interval in the training script if disk I/O is a concern.

**Export:**
- The exporter uses `strict=False` when loading checkpoints, which means it will silently skip missing keys. Always check the exporter output for `Warning:` lines that indicate key mismatches.
- Exporting does not require a GPU. It runs on CPU and is fast (typically under 10 seconds for any model size in this project).

**C++ Inference:**
- For greedy decoding (deterministic output), use `--temp 0.0`. This skips the softmax and multinomial sampling, running the argmax path directly.
- The runner implements a rolling context window: if `context.size()` exceeds `max_seq_len`, the oldest tokens are evicted from the front. This allows indefinite generation but the model may lose coherence as early context is dropped.
- To measure tokens per second, run with `--steps 100` or more. The TPS figure printed at the end is more accurate with longer runs due to startup overhead.
- The debug print statements in `bit_linear` (inside `atomic_runner.cpp`) fire for the first 5 BitLinear calls. These are written to `stdout` and can be suppressed by setting `debug_count` to a high value or removing the guard in the source.
