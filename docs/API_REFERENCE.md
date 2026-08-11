# Atomic-1Bit API Reference

This document covers the public API exposed by `embedded/atomic_lib.h`, the header-only C++ inference library for Atomic-1Bit models. It is the primary interface for running ternary transformer inference in embedded or standalone C++ environments.

---

## Data Structures

### `Config`

Holds the architecture parameters for a loaded model.

```cpp
struct Config {
  int vocab_size;    // Number of tokens in the vocabulary
  int dim;           // Model embedding dimension
  int depth;         // Number of transformer layers
  int heads;         // Number of attention heads
  int max_seq_len;   // Maximum sequence length the model was exported for
  int has_gist;      // 1 if a pre-baked Gist vector is present, 0 otherwise
};
```

These values are read directly from the binary file header during `load_model`. They must match the dimensions used when the model was trained and exported.

---

### `AtomicLayer`

Holds all weights for a single transformer block.

```cpp
struct AtomicLayer {
  vector<float>   ln1;              // RMSNorm scale parameters for pre-attention norm
  vector<int8_t>  q_w, k_w, v_w;   // Ternary {-1, 0, 1} attention projection weights
  vector<int8_t>  o_w;             // Ternary output projection weight
  vector<float>   ln2;              // RMSNorm scale parameters for pre-MLP norm
  vector<int8_t>  fc1_w, fc2_w;    // Ternary MLP weights (fc1: dim -> 4*dim, fc2: 4*dim -> dim)
};
```

Attention weight shapes: `[dim, dim]` (flattened). MLP fc1: `[dim, 4*dim]`, fc2: `[4*dim, dim]`.

---

### `AtomicModel`

Top-level model container.

```cpp
struct AtomicModel {
  Config          config;
  vector<float>   token_emb;    // Token embedding table: [vocab_size * dim] float32
  vector<float>   pos_emb;      // Positional embedding table: [max_seq_len * dim] float32
  vector<float>   ln_f;         // Final RMSNorm scale: [dim] float32
  vector<int8_t>  head_w;       // Language model head weight (ternary): [dim * vocab_size]
  vector<AtomicLayer> layers;   // Per-layer weights, length = config.depth
  vector<float>   gist_vector;  // Pre-baked Gist vector: [dim] float32 (empty if has_gist == 0)
};
```

---

## Functions

### `load_model`

```cpp
inline bool load_model(const string &filename, AtomicModel &model);
```

Loads a binary model file produced by `atomic_1bit/utils/export_to_cpp.py` into an `AtomicModel` struct.

**Parameters:**
- `filename` — Path to the `.bin` model file.
- `model` — Output parameter. The struct is populated in-place.

**Returns:** `true` on success, `false` if the file cannot be opened.

**Notes:**
- The binary format does not include a magic-number check in this header version. The runner (`atomic_runner.cpp`) does validate the `ATOM` magic bytes and version before using this path.
- Weights are stored in the following order: config header, optional gist vector, token embeddings, positional embeddings, per-layer weights (ln1, q/k/v/o, ln2, fc1, fc2), final norm, language model head.
- Float tensors (`ln1`, `ln2`, `ln_f`, embeddings) are stored as raw `float32`. Ternary weight tensors are stored as raw `int8_t` with values in `{-1, 0, 1}`.
- The caller is responsible for keeping `model` alive for as long as inference runs, since `forward` holds non-owning pointers into the vectors.

---

### `forward`

```cpp
inline void forward(AtomicModel &model, const vector<int> &tokens, vector<float> &logits);
```

Runs a full forward pass of the model on the given token sequence and writes next-token logits into `logits`.

**Parameters:**
- `model` — A fully loaded `AtomicModel`.
- `tokens` — Input token IDs as a sequence of integers. If the model has `has_gist == 1`, the Gist vector is prepended automatically, consuming one position of the context window.
- `logits` — Output buffer of size `model.config.vocab_size`. Must be pre-allocated by the caller before calling.

**Returns:** `void`. Returns immediately without writing to `logits` if the effective sequence length (including the Gist token if present) exceeds `config.max_seq_len`.

**Computation performed:**
1. Embed each token as `token_emb[id] + pos_emb[position]`.
2. For each transformer layer: pre-norm (RMSNorm), multi-head attention with causal masking, residual add, pre-norm (RMSNorm), MLP with GELU activation, residual add.
3. Apply final RMSNorm to the last token's hidden state.
4. Project to vocabulary via `bit_linear` using `head_w`.

**Memory:** All intermediate buffers are heap-allocated as `std::vector` per call. For microcontroller use, replace these with static or arena-allocated buffers.

---

### `bit_linear`

```cpp
inline void bit_linear(const vector<float> &x, const vector<int8_t> &w, vector<float> &out);
```

Core BitLinear compute kernel. Implements one linear layer of the form: quantize input, integer matrix-multiply against ternary weight, dequantize output.

**Parameters:**
- `x` — Float input vector of size `in_dim`.
- `w` — Ternary weight matrix, flattened as `[in_dim, out_dim]` in row-major order (i.e., `w[i * out_dim + o]`). Values must be in `{-1, 0, 1}`.
- `out` — Output vector. Must be pre-allocated to size `out_dim`.

**Quantization steps:**
1. AbsMax scaling: `scale = 127 / max(|x|, 1e-5)`, clamp to INT8 range `[-127, 127]`.
2. Integer accumulation: `acc[o] += x_q[i] * w[i * out_dim + o]` (uses actual `int32_t` multiplication; no branch-on-sign optimization at this layer).
3. Dequantize: `out[o] = acc[o] / scale`.

Per-tensor `float32` weight scales are read from the binary and applied during dequantization, matching the Python `BitLinear` forward. `atomic_lib.h` and `atomic_runner.cpp` now consume the identical format written by `export_to_cpp.py`.

---

### `rms_norm`

```cpp
inline void rms_norm(const vector<float> &x, const vector<float> &w,
                     vector<float> &out, int dim);
```

Affine RMSNorm. Normalizes `x` by its RMS, then scales element-wise by learned weight `w`.

**Parameters:**
- `x` — Input vector of length `dim`.
- `w` — Learned scale vector of length `dim`.
- `out` — Output vector of length `dim`.
- `dim` — Dimension to normalize over. Must match `x.size()`.

**Formula:** `out[i] = x[i] * (1 / sqrt(mean(x^2) + 1e-5)) * w[i]`

---

### `softmax`

```cpp
inline void softmax(vector<float> &x);
```

Numerically stable in-place softmax. Subtracts the max value before exponentiating.

**Parameters:**
- `x` — Float vector to normalize. Modified in-place.

---

### `gelu`

```cpp
inline float gelu(float x);
```

Scalar approximation of GELU activation using the `tanh` formula.

**Returns:** `0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * x^3)))`

---

## Usage Example

```cpp
#include "atomic_lib.h"
#include <iostream>
#include <vector>

int main() {
    AtomicModel model;

    if (!load_model("atomic_model.bin", model)) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    // Starting token ID (e.g., BOS or a known token)
    std::vector<int> context = {42};

    std::vector<float> logits(model.config.vocab_size);

    // Greedy decode loop
    for (int step = 0; step < 50; ++step) {
        if ((int)context.size() >= model.config.max_seq_len) break;

        forward(model, context, logits);

        // Pick the highest-scoring token
        int best = 0;
        for (int i = 1; i < model.config.vocab_size; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        std::cout << best << " " << std::flush;
        context.push_back(best);
    }
    std::cout << std::endl;
    return 0;
}
```

---

## Memory Management

- All model weights are stored in `std::vector` members of `AtomicModel`. When the `AtomicModel` instance goes out of scope, all weight memory is automatically released.
- `forward` allocates temporary buffers (Q, K, V, attention output, intermediate MLP tensors) on every call using `std::vector`. For performance-sensitive or memory-constrained deployments, these should be moved to pre-allocated workspace buffers as done in `atomic_runner.cpp`'s `Workspace` struct.
- The `logits` output buffer must be pre-allocated by the caller to exactly `model.config.vocab_size` elements before calling `forward`.

---

## Thread Safety

`atomic_lib.h` contains no global mutable state. However, the `forward` function and `bit_linear` allocate thread-local-stack-level `std::vector` objects on the heap on each call, which involves the system allocator. Concurrent calls to `forward` on the same `AtomicModel` are not safe because `forward` reads shared model weight vectors. If concurrent inference is required, load separate `AtomicModel` instances per thread.

---

## Binary File Format

The `.bin` files produced by `export_to_cpp.py` and consumed by `atomic_runner.cpp` follow this layout:

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | `int32` | Magic: `0x41544F4D` ("ATOM") |
| 4 | 4 | `int32` | Version: `1` |
| 8 | 4 | `int32` | `vocab_size` |
| 12 | 4 | `int32` | `dim` |
| 16 | 4 | `int32` | `depth` |
| 20 | 4 | `int32` | `heads` |
| 24 | 4 | `int32` | `max_seq_len` |
| 28 | 4 | `int32` | `has_gist` |
| 32 | `dim * 4` | `float32[]` | Gist vector (only if `has_gist == 1`) |
| ... | `vocab_size * dim * 4` | `float32[]` | Token embeddings |
| ... | `max_seq_len * dim * 4` | `float32[]` | Positional embeddings |
| ... | (repeated `depth` times) | — | Layer weights (see below) |
| ... | `dim * 4` | `float32[]` | Final RMSNorm scale |
| ... | `4 + dim * vocab_size` | `float32 + int8[]` | LM head (scale + ternary weights) |

Each layer block contains:

| Size | Type | Description |
|------|------|-------------|
| `dim * 4` | `float32[]` | `ln1` scale |
| `4 + dim * dim` | `float32 + int8[]` | Q projection (scale + weights) |
| `4 + dim * dim` | `float32 + int8[]` | K projection (scale + weights) |
| `4 + dim * dim` | `float32 + int8[]` | V projection (scale + weights) |
| `4 + dim * dim` | `float32 + int8[]` | O projection (scale + weights) |
| `dim * 4` | `float32[]` | `ln2` scale |
| `4 + dim * 4*dim` | `float32 + int8[]` | FC1 weight (scale + weights) |
| `4 + 4*dim * dim` | `float32 + int8[]` | FC2 weight (scale + weights) |

Note: `atomic_lib.h` and `atomic_runner.cpp` read the same format, including the per-tensor `float32` scale prefix on every quantized weight tensor.
