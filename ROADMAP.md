# Atomic-1Bit Future Roadmap

## ✅ v1.0 (Completed)
- **Parity-verified ternary inference** (Python ↔ NumPy ↔ C++)
- **INT8 add/sub-only kernel correctness** (zero diff)
- **Embedded-ready C++ runtime** (no dependencies)
- **Reproducible benchmarks**
- **Full command reference** (`docs/COMMANDS.md`)
- **Demo-quality TinyStories checkpoint** (~15k steps)

---

## 🚀 v1.1 — Runtime Acceleration
**Goal:** Make inference faster on existing hardware without changing model semantics.

### Planned
- [ ] **SIMD-optimized ternary matmul** (AVX2 / AVX-512)
- [ ] **Cache-friendly weight layouts**
- [ ] **Reduced heap allocations** in hot paths
- [ ] **Optional KV-cache compression** (INT8 / INT16)

**Target:** 2–4× CPU speedup over v1.0 scalar kernel

---

## ✅ v1.2 — Hardware-Native Backends (Completed)
**Goal:** Prove Atomic-1Bit works beyond general-purpose CPUs.

### Achievements
- [x] **Metal backend** (Apple Silicon) - Zero-copy memory & compute shaders
- [x] **CUDA backend** (NVIDIA) - Shared memory tiled matmul
- [x] **Modular Architecture** - CPU/GPU conditional compilation

---

## 📈 v1.3 — Model Scaling & Quality (In Progress)
**Goal:** Demonstrate that low-bit inference scales in quality, not just efficiency.

### Planned
- [ ] **Train 50M–100M parameter ternary models**
- [ ] **Improved training stability** (scheduler + clipping)
- [ ] **Evaluation suite** (perplexity, repetition, coherence)

**Outcome:** Higher-quality samples suitable for public demos

---

## 🧩 v2.0 — Atomic Inference System
**Goal:** Evolve Atomic-1Bit into a general low-bit inference framework.

### Planned
- [ ] **Support for mixed-precision** (2-bit, 3-bit, hybrid)
- [ ] **Pluggable quantization backends**
- [ ] **Mobile and microcontroller demos** (ESP32-class)
- [ ] **Optional FPGA / ASIC exploration**

**Deliverable:** Research-grade system with real deployment targets

---

## 📌 Guiding Principles
1. **Correctness before speed**
2. **Parity before optimization**
3. **Measured claims only**
4. **Deployment-focused research**
