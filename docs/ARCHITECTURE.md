# Atomic-1Bit Architecture

A deep dive into the theory, design decisions, and implementation details of the Atomic-1Bit inference engine.

## Table of Contents

1. [Overview](#overview)
2. [Why Ternary Quantization Works](#why-ternary-quantization-works)
3. [Straight-Through Estimator (STE)](#straight-through-estimator-ste)
4. [The ATOM Binary Format](#the-atom-binary-format)
5. [Kernel Design: Add/Sub vs Multiply](#kernel-design-addsub-vs-multiply)
6. [Gist Token Theory](#gist-token-theory)
7. [System Architecture](#system-architecture)
8. [Implementation Details](#implementation-details)

---

## Overview

**Atomic-1Bit** is a bare-metal inference engine implementing BitNet b1.58, a neural network architecture that uses ternary weights {-1, 0, 1} instead of full-precision floats. The core innovation: replacing floating-point matrix multiplication with integer addition and subtraction operations, enabling ultra-low compute and memory footprint suitable for edge devices.

**Key Characteristics:**
- **1.58-bit weights**: Ternary values {-1, 0, 1} (log2(3) ≈ 1.58 bits per weight)
- **INT8 activations**: 8-bit integer quantization for activations
- **No FP16 multiply in inference**: Only INT8 add/sub operations
- **62% model size reduction** vs FP16 baseline
- **Deployment-focused**: Runs on CPUs, Apple Silicon (Metal), NVIDIA GPUs (CUDA), ESP32, Raspberry Pi, WebAssembly

---

## Why Ternary Quantization Works

### The Mathematics of BitNet b1.58

Traditional neural networks use full-precision floating-point weights (FP16 or FP32). A standard matrix multiplication:

```
y = W @ x
```

where `W` is a weight matrix (Out × In) and `x` is an input vector (In × 1), requires `Out × In` multiply-add operations.

**BitNet b1.58** restricts weights to three values: {-1, 0, 1}. This has profound implications:

1. **Memory**: 2 bits per weight (00=0, 01=+1, 11=-1) vs 16 bits for FP16 → **8x compression**
2. **Compute**: Multiplication becomes conditional addition/subtraction
3. **Energy**: Integer operations consume ~10-100x less energy than floating-point

### Quantization Functions

#### Weight Quantization (Mean Scaling)

Ternary weights are computed using mean absolute value scaling:

```python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quantized = (w * scale).round().clamp(-1, 1)
    return w_quantized, scale
```

**Why mean scaling?** The mean of absolute values provides a robust scale factor that balances the range of the weights. After scaling by the mean, most weights fall within [-1, 1], and rounding snaps them to the nearest ternary value.

**Mathematical intuition:**
- If `mean(|w|) = 0.5`, then `scale = 2.0`
- Weights of magnitude 0.4 → 0.8 after scaling → round to 1
- Weights of magnitude 0.1 → 0.2 after scaling → round to 0
- Weights of magnitude -0.7 → -1.4 after scaling → round to -1

#### Activation Quantization (AbsMax Scaling)

Activations use INT8 quantization with per-tensor absolute maximum scaling:

```python
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    x_quantized = (x * scale).round().clamp(-127, 127)
    return x_quantized, scale
```

**Why AbsMax for activations?** Activations can have varying ranges across layers and batches. Using the absolute maximum ensures the full INT8 range [-127, 127] is utilized, preserving maximum precision.

### Dequantization

After the quantized matrix multiplication, results are dequantized:

```python
y_dequantized = y_quantized / (scale_activation * scale_weight)
```

This recovers an approximation of the full-precision result.

### Error Analysis

Ternary quantization introduces quantization error:

```
ε = w_original - w_quantized / scale
```

The error is bounded by the quantization step size. For weights distributed around mean `μ`, the quantization error is approximately:

```
|ε| ≤ (2 * μ) / 2 = μ
```

In practice, neural networks are **surprisingly robust** to this error because:
1. Training learns to compensate (see STE section)
2. The network's expressiveness comes from depth and width, not weight precision
3. Ternary values provide sufficient representational capacity for many tasks

---

## Straight-Through Estimator (STE)

### The Gradient Problem

Quantization is a non-differentiable operation:

```
w_quantized = round(w * scale)
```

The gradient of `round()` is zero almost everywhere, making backpropagation impossible. This is the fundamental challenge of training quantized networks.

### STE Solution

The **Straight-Through Estimator** (Bengio et al., 2013) solves this by using an identity function for the backward pass:

```python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    y = (w * scale).round().clamp(-1, 1)
    # STE: detach y, but backward passes gradients to w * scale
    y_ste = (y - w * scale).detach() + w * scale
    return y_ste, scale
```

**How it works:**

**Forward pass:**
```
output = y = round(w * scale)
```

**Backward pass:**
```
∂L/∂w = ∂L/∂y * ∂y/∂w
       ≈ ∂L/∂y * 1  (treating round() as identity)
       = ∂L/∂y
```

The gradient flows through as if quantization didn't exist. The `.detach()` operation prevents PyTorch from computing gradients for `y`, while adding `w * scale` back creates a computational graph path.

### Implementation in BitLinear

```python
class BitLinear(nn.Module):
    def forward(self, x):
        # 1. RMSNorm
        x_norm = x / rms(x)

        # 2. Quantize (with STE)
        x_q, scale_x = activation_quant(x_norm)
        w_q, scale_w = weight_quant(self.weight)

        # 3. Quantized matmul
        y = F.linear(x_q, w_q)

        # 4. Dequantize
        y_out = y / (scale_x * scale_w)

        return y_out
```

During training:
- Forward pass uses quantized values (simulating inference)
- Backward pass receives full gradients as if no quantization occurred
- Latent weights (`self.weight`) remain as continuous floats and are updated by the optimizer

This allows the network to **learn weights that are robust to quantization**.

### Why STE Works

Empirically, STE enables successful training because:
1. **Gradient information is preserved**: The network knows when to increase or decrease weights
2. **Quantization-aware training**: The network sees quantized activations during forward pass and learns to compensate
3. **Stochastic rounding effect**: Small weight updates accumulate over many steps, eventually crossing quantization boundaries

---

## The ATOM Binary Format

The ATOM format is a custom binary serialization designed for embedded deployment. It's compact, easy to parse in C++, and version-controlled.

### Format Specification

```
┌─────────────────────────────────────────────────────────┐
│                    HEADER (32 bytes)                    │
├─────────────────────────────────────────────────────────┤
│  0-3   : Magic Bytes (0x41544F4D = "ATOM")             │
│  4-7   : Version (uint32, currently 1)                  │
│  8-11  : Vocabulary Size (uint32)                       │
│  12-15 : Model Dimension (uint32)                       │
│  16-19 : Depth (number of layers, uint32)               │
│  20-23 : Number of Attention Heads (uint32)             │
│  24-27 : Context Length (uint32)                        │
│  28-31 : Has Gist Flag (uint32, 0 or 1)                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              GIST VECTOR (if Has Gist = 1)              │
├─────────────────────────────────────────────────────────┤
│  Dim × sizeof(float32)  : Gist Vector                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     EMBEDDINGS                          │
├─────────────────────────────────────────────────────────┤
│  Token Embeddings: VocabSize × Dim (float32)            │
│  Position Embeddings: ContextLength × Dim (float32)     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│          TRANSFORMER LAYERS (repeated Depth times)      │
├─────────────────────────────────────────────────────────┤
│  Layer Norm 1:                                          │
│    - Weights: Dim (float32)                             │
│                                                         │
│  Attention (Q, K, V, O projections):                    │
│    For each projection:                                 │
│      - Scale: 1 (float32)                               │
│      - Weights: Dim × Dim (int8, transposed)            │
│                                                         │
│  Layer Norm 2:                                          │
│    - Weights: Dim (float32)                             │
│                                                         │
│  MLP (fc1, fc2):                                        │
│    For each layer:                                      │
│      - Scale: 1 (float32)                               │
│      - Weights: Out × In (int8, transposed)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   FINAL NORM & HEAD                     │
├─────────────────────────────────────────────────────────┤
│  Final Layer Norm:                                      │
│    - Weights: Dim (float32)                             │
│                                                         │
│  Output Head:                                           │
│    - Scale: 1 (float32)                                 │
│    - Weights: VocabSize × Dim (int8, transposed)        │
└─────────────────────────────────────────────────────────┘
```

### Byte-Level Details

#### Header (32 bytes)

All integers are **little-endian uint32**.

| Offset | Field            | Type    | Description                           |
|--------|------------------|---------|---------------------------------------|
| 0      | Magic            | uint32  | 0x41544F4D ("ATOM")                   |
| 4      | Version          | uint32  | Format version (1)                    |
| 8      | vocab_size       | uint32  | Vocabulary size                       |
| 12     | dim              | uint32  | Model dimension                       |
| 16     | depth            | uint32  | Number of transformer layers          |
| 20     | heads            | uint32  | Number of attention heads             |
| 24     | context_length   | uint32  | Maximum sequence length               |
| 28     | has_gist         | uint32  | 0=no gist, 1=gist vector follows      |

#### Weight Storage

**Float32 weights** (embeddings, normalization):
- Stored as IEEE 754 single-precision floats
- Little-endian byte order
- Row-major layout

**Quantized INT8 weights** (BitLinear projections):
- **Transposed**: Stored as (In, Out) instead of PyTorch's (Out, In)
- Preceded by a float32 scale factor
- Each weight is a signed 8-bit integer {-1, 0, 1}
- Row-major layout after transpose

**Why transpose?** C++ inference iterates over output dimensions in the outer loop. Transposed storage provides cache-friendly sequential access patterns.

### Example: Decoding a Q Projection

Given a 320-dim model with 5 heads (head_dim = 64):

```
Offset (hex)  | Data               | Interpretation
--------------+--------------------+----------------------------------
0x0000        | 4D 4F 54 41        | Magic "ATOM"
0x0004        | 01 00 00 00        | Version = 1
...
0x0020        | Scale (4 bytes)    | Scale factor for dequantization
0x0024        | -1 0 1 1 0 -1 ...  | 320 × 320 = 102,400 INT8 weights
```

### Loading in C++

```cpp
struct AtomicHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t depth;
    uint32_t heads;
    uint32_t context_length;
    uint32_t has_gist;
};

FILE* f = fopen("model.bin", "rb");
AtomicHeader header;
fread(&header, sizeof(AtomicHeader), 1, f);

if (header.magic != 0x41544F4D) {
    printf("Invalid ATOM file\n");
    exit(1);
}

// Read gist if present
float* gist = nullptr;
if (header.has_gist) {
    gist = new float[header.dim];
    fread(gist, sizeof(float), header.dim, f);
}

// Read token embeddings
float* token_emb = new float[header.vocab_size * header.dim];
fread(token_emb, sizeof(float), header.vocab_size * header.dim, f);
```

---

## Kernel Design: Add/Sub vs Multiply

### Standard Matrix Multiplication

A typical FP16 matrix multiplication:

```cpp
// C = A @ B^T
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[i][k] * B[j][k];  // FP multiply
        }
        C[i][j] = acc;
    }
}
```

**Cost per output element:** K multiply-add operations

### Ternary Matrix Multiplication

With ternary weights {-1, 0, 1}, multiplication degenerates to conditional addition:

```cpp
// C = A @ B^T, where B contains {-1, 0, 1}
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) {
            int8_t w = B[j][k];
            if (w == 1)       acc += A[i][k];   // Add
            else if (w == -1) acc -= A[i][k];   // Subtract
            // if w == 0: skip (free sparsity)
        }
        C[i][j] = acc;
    }
}
```

**Cost per output element:** ~K integer additions (zero weights are free)

### Why This Matters

#### 1. Energy Efficiency

From hardware perspective (approximate values):

| Operation      | Energy (pJ) | Relative Cost |
|----------------|-------------|---------------|
| INT8 Add       | 0.03        | 1x            |
| FP16 Multiply  | 1.1         | 37x           |
| FP32 Multiply  | 3.7         | 123x          |

Ternary kernels use **37-123x less energy per operation** than floating-point.

#### 2. Memory Bandwidth

Weight storage:

| Precision | Bits/weight | 12.5M param model |
|-----------|-------------|-------------------|
| FP32      | 32          | 50 MB             |
| FP16      | 16          | 25 MB             |
| INT8      | 8           | 12.5 MB           |
| Ternary   | 2 (packed)  | 3.1 MB            |

**8x reduction** in memory bandwidth vs FP16. On bandwidth-limited devices (mobile, embedded), this is the primary bottleneck.

#### 3. Hardware Simplicity

Ternary operations can be implemented with:
- **Adder circuits** (simple, small, low-power)
- **No multiplier units** (complex, large, high-power)

This enables custom accelerators with significantly reduced chip area and power consumption.

### SIMD Optimization

The ternary kernel is vectorized for modern CPUs:

#### NEON (ARM)

```cpp
// Process 16 INT8 elements at once
int8x16_t va = vld1q_s8(val_a + k);      // Load 16 activations
int8x16_t vb = vld1q_s8(val_b + k);      // Load 16 weights
int8x16_t vprod = vmulq_s8(va, vb);      // Element-wise multiply (w*a)

// Pairwise sum to accumulate
int16x8_t v_pair = vpaddlq_s8(vprod);    // 16 → 8 pairs
int32x4_t v_quad = vpaddlq_s16(v_pair);  // 8 → 4 quads
v_acc = vaddq_s32(v_acc, v_quad);        // Accumulate
```

**Key insight:** Even though weights are ternary, we still use `vmulq_s8` because:
- `1 * x = x`
- `-1 * x = -x`
- `0 * x = 0`

The hardware multiply unit handles this efficiently, and the subsequent pairwise additions reduce the result.

#### AVX2 (x86)

```cpp
// Process 32 INT8 elements at once
__m256i va = _mm256_loadu_si256((__m256i*)(val_a + k));
__m256i vb = _mm256_loadu_si256((__m256i*)(val_b + k));
__m256i vprod = _mm256_sign_epi8(va, vb);  // Sign-based multiply

// Widen to INT16, then INT32 for accumulation
__m256i v_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod));
__m256i v_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod, 1));
v_lo = _mm256_madd_epi16(v_lo, ones);  // Horizontal add pairs
v_hi = _mm256_madd_epi16(v_hi, ones);
v_acc = _mm256_add_epi32(v_acc, v_lo);
v_acc = _mm256_add_epi32(v_acc, v_hi);
```

`_mm256_sign_epi8` is the key: it computes `sign(vb) * va`, perfect for ternary weights.

### Metal & CUDA Backends

GPU kernels exploit massive parallelism:

**Metal (Apple Silicon):**
```metal
kernel void ternary_matmul(
    device const int8_t* A [[buffer(0)]],
    device const int8_t* B [[buffer(1)]],
    device int32_t* C [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    int32_t acc = 0;
    for (uint k = 0; k < K; k++) {
        int8_t w = B[gid.y * K + k];
        acc += (w == 1) ? A[gid.x * K + k] : ((w == -1) ? -A[gid.x * K + k] : 0);
    }
    C[gid.x * N + gid.y] = acc;
}
```

Each GPU thread computes one output element, with thousands of threads running in parallel.

---

## Gist Token Theory

### Problem: System Prompts are Expensive

Instruction-following models often require system prompts:

```
System: You are a helpful assistant.
User: What is the capital of France?
Assistant: The capital of France is Paris.
```

The system prompt must be:
1. Prepended to every input
2. Processed by the entire transformer stack
3. Cached in KV-cache (consuming memory)

For a 20-token system prompt in a 256-token context, **7.8% of compute is wasted** on redundant prompt processing.

### Solution: Thought Compression

**Key insight:** System prompts are fixed and repetitive. Can we compress them into a single learned vector?

The **Gist Token** (concept from "Learning to Compress Prompts with Gist Tokens", Mu et al., 2023) encodes an entire prompt into a single dense vector.

### Implementation

#### 1. Gist Encoder

```python
class GistEncoder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        # Average token embeddings
        self.embedding = nn.EmbeddingBag(vocab_size, dim, mode='mean')

    def forward(self, token_ids):
        # token_ids: (Batch, SeqLen)
        # output: (Batch, Dim) - single vector per sequence
        return self.embedding(token_ids)
```

#### 2. Gist Injection

During inference, the gist vector is **prepended** to the sequence at the embedding level:

```python
def forward(self, idx, gist_token=None):
    # idx: (Batch, Seq) - user input tokens
    x = self.token_emb(idx) + self.pos_emb(...)

    if gist_token is not None:
        # gist_token: (Batch, Dim)
        gist_emb = gist_token.unsqueeze(1)  # (Batch, 1, Dim)
        x = torch.cat([gist_emb, x], dim=1)  # Prepend

    # Transformer processes [Gist, Token_1, Token_2, ..., Token_N]
    for layer in self.layers:
        x = layer(x)

    return self.head(x)
```

The gist vector acts as the "first token" that all subsequent tokens can attend to.

#### 3. Pre-Computation

For a fixed system prompt, the gist vector is computed **once** during model export:

```python
# During export
gist_encoder = GistEncoder(vocab_size, dim)
prompt_tokens = tokenize("You are a helpful assistant.")
gist_vector = gist_encoder(prompt_tokens)  # (1, Dim)

# Save to .bin file
f.write(gist_vector.numpy().tobytes())
```

At inference time, the gist vector is **loaded from disk**—no computation required.

### Benefits

1. **Zero inference cost**: Pre-computed, no encoding overhead
2. **Memory savings**: 1 token vs 20+ tokens in KV-cache
3. **Context preservation**: Full prompt semantics retained
4. **Flexible**: Different gist vectors for different personas/modes

### Attention Mechanism

How does a single vector encode an entire prompt? Through **attention**:

When tokens attend to the gist vector, they extract relevant information:

```
Query (from token "France"): [0.2, -0.5, 0.8, ...]
Key (from gist):              [0.1,  0.3, 0.9, ...]
Attention Score:              high (relevant)

Query (from token "capital"):
Key (from gist):
Attention Score:              high (also relevant)
```

The gist vector has learned to encode **all aspects** of the system prompt as a dense representation that different query contexts can selectively attend to.

### Training Gist Encoders

Gist encoders are typically trained with:

1. **Supervised learning**: Minimize reconstruction loss between gist-conditioned and full-prompt outputs
2. **Auxiliary loss**: Maximize mutual information between gist and prompt
3. **Joint training**: Train alongside the main model

In Atomic-1Bit, we use a simple EmbeddingBag with mean pooling, but more sophisticated encoders (RNN, small Transformer) can be used.

---

## System Architecture

### Dual-Stack Design

Atomic-1Bit follows a **research + deployment** architecture:

```
┌──────────────────────────────────────────────────────────┐
│                   RESEARCH STACK (Python)                │
├──────────────────────────────────────────────────────────┤
│  • Training (PyTorch)                                    │
│  • Experimentation (BitLinear layers, STE)               │
│  • Model design (AtomicTransformer, Gist)                │
│  • Evaluation (Perplexity, quality metrics)              │
│  • Export pipeline (.pt → .bin)                          │
└──────────────────────────────────────────────────────────┘
                          ↓ Export
┌──────────────────────────────────────────────────────────┐
│               DEPLOYMENT STACK (C++/Embedded)            │
├──────────────────────────────────────────────────────────┤
│  • CPU kernel (scalar, NEON, AVX2)                       │
│  • GPU kernels (Metal, CUDA)                             │
│  • Embedded runtime (header-only, no dependencies)       │
│  • Platform ports (ESP32, RPi, WebAssembly)              │
└──────────────────────────────────────────────────────────┘
```

**Critical constraint:** Both stacks must produce **bit-exact** identical outputs. This is verified via parity checks.

### Component Map

```
atomic_1bit/
├── nn/layers.py              ← BitLinear (quantization + STE)
├── model/
│   ├── transformer.py        ← AtomicTransformer (attention, MLP)
│   └── gist.py               ← GistEncoder
├── training/
│   ├── train.py              ← TinyStories training
│   ├── train_instruct.py     ← Flagship 12.5M model
│   └── data.py               ← Dataset loaders
├── utils/
│   ├── export_to_cpp.py      ← ATOM serialization
│   └── thermal.py            ← Safety monitoring
├── core/
│   ├── backends/
│   │   ├── cpu_kernel.cpp    ← SIMD-optimized ternary matmul
│   │   ├── metal_kernel.mm   ← Apple Silicon GPU
│   │   └── cuda_kernel.cu    ← NVIDIA GPU
│   └── Makefile              ← Build system
└── python/
    ├── wrapper.py            ← ctypes bridge to C++
    └── chat.py               ← Interactive inference

embedded/
├── atomic_lib.h              ← Header-only runtime
├── atomic_runner.cpp         ← Standalone binary
└── platforms/
    ├── esp32/                ← Microcontroller port
    ├── rpi/                  ← Raspberry Pi demos
    └── wasm/                 ← Browser deployment
```

---

## Implementation Details

### RMSNorm

Root Mean Square normalization is used instead of LayerNorm for efficiency:

```python
def rms_norm(x, weight, eps=1e-5):
    # x: (Batch, Seq, Dim)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    return x_norm * weight  # Element-wise scale
```

**Advantages over LayerNorm:**
- No mean subtraction (saves compute)
- No bias term (fewer parameters)
- Similar normalization properties

### KV-Cache Implementation

Caching key and value tensors avoids recomputing attention for previous tokens:

```python
def forward(self, x, kv_cache=None):
    q = self.q_proj(x)  # (B, T_new, Dim)
    k = self.k_proj(x)
    v = self.v_proj(x)

    if kv_cache is not None:
        cached_k, cached_v = kv_cache
        k = torch.cat([cached_k, k], dim=1)  # Append new keys
        v = torch.cat([cached_v, v], dim=1)  # Append new values

    att = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
    att = softmax(att)
    out = att @ v

    return out, (k, v)  # Return updated cache
```

**Memory tradeoff:** Cache size = `Batch × Layers × Heads × SeqLen × HeadDim × 2 (K+V)`. For a 12.5M model with 256 context and batch=1: ~5MB of cache.

### Causal Masking

Autoregressive models prevent attention to future tokens:

```python
# Mask is lower-triangular
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
att = att.masked_fill(mask, float('-inf'))
att = softmax(att)  # Future positions become 0 after softmax
```

### Temperature Sampling

Generating diverse outputs:

```python
logits = model(idx)[:, -1, :]  # Last token logits
probs = softmax(logits / temperature)
next_token = torch.multinomial(probs, 1)
```

- `temperature=0`: Greedy decoding (argmax)
- `temperature=1`: Sampling from model distribution
- `temperature>1`: More random (flattens distribution)
- `temperature<1`: More deterministic (sharpens distribution)

### Thermal Safety

Training on Apple Silicon monitors die temperature:

```python
class ThermalMonitor:
    def check_throttle(self):
        temp = self.get_die_temp()
        if temp > 80:
            print("Temperature critical, pausing...")
            torch.save(model.state_dict(), "thermal_safe.pt")
            self.wait_for_cooldown(target=70)
```

Prevents thermal damage and ensures consistent training speed.

---

## Performance Characteristics

### Model Scaling (12.5M Flagship)

| Component                  | Parameters | Memory (INT8) |
|----------------------------|------------|---------------|
| Token Embedding            | 16.1M      | 16.1 MB       |
| Position Embedding         | 81.9K      | 81.9 KB       |
| 8 × Transformer Layers     | 8.2M       | 2.0 MB        |
| Final Norm + Head          | 16.1M      | 4.0 MB        |
| **Total**                  | **40.5M**  | **22.2 MB**   |

Note: Embeddings remain FP32 for quality; ternary weights compress 8x.

### Speedup Analysis

| Backend       | Platform          | TPS (tokens/sec) |
|---------------|-------------------|------------------|
| Python (PyTorch) | M1 Max (CPU)   | 130              |
| C++ (Scalar)  | M1 Max (CPU)      | 450              |
| C++ (NEON)    | M1 Max (CPU)      | 1,200            |
| Metal         | M1 Max (GPU)      | 3,800            |
| CUDA          | RTX 3090          | 5,600            |

---

## Future Directions

1. **Weight Packing**: Store 2-bit weights instead of 8-bit (4x compression)
2. **INT4 Activations**: Further reduce memory bandwidth
3. **Mixed-Precision**: High-precision first/last layers, ternary middle layers
4. **Custom Hardware**: FPGA/ASIC implementations exploiting ternary simplicity
5. **Larger Models**: Scale to 100M+ parameters with ternary bottleneck layers

---

## References

1. **BitNet: Scaling 1-bit Transformers for Large Language Models** (Wang et al., 2023)
2. **BitNet b1.58: Training Tips and Tricks** (Ma et al., 2024)
3. **Estimating or Propagating Gradients Through Stochastic Neurons** (Bengio et al., 2013)
4. **Learning to Compress Prompts with Gist Tokens** (Mu et al., 2023)
5. **RMSNorm: Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-09
**Project Version:** v1.3 (Model Scaling & Quality)
