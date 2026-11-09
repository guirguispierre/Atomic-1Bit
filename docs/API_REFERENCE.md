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
- Float tensors (`ln1`, `ln2`, `ln_f`) are stored as raw `float32`. Ternary weight tensors are packed four per byte on disk and unpacked into `int8_t` with values in `{-1, 0, 1}` at load time. Embedding tables are version-dependent: format v3 stores each row as `int8` codes with one `float32` absmax scale per row, v2 stores raw `float32`; both land in memory as `float32`.
- The caller is responsible for keeping `model` alive for as long as inference runs, since `forward` holds non-owning pointers into the vectors.

---

### `forward`

```cpp
inline void forward(AtomicModel &model, const vector<int> &tokens,
                    vector<float> &logits, KVCache *cache = nullptr);
```

Runs a full forward pass of the model on the given token sequence and writes next-token logits into `logits`.

**Parameters:**
- `model` — A fully loaded `AtomicModel`.
- `tokens` — Input token IDs as a sequence of integers. If the model has `has_gist == 1`, the Gist vector is prepended automatically, consuming one position of the context window.
- `logits` — Output buffer of size `model.config.vocab_size`. Must be pre-allocated by the caller before calling.
- `cache` — Optional `KVCache`. When non-null, every layer's K/V rows are copied into it during the pass and `cache->len` ends at the sequence length, so generation can continue with `forward_one` instead of re-running the prefix. The cache is (re)initialized automatically if its dimensions do not match the model.

**Returns:** `void`. Returns immediately without writing to `logits` if the effective sequence length (including the Gist token if present) exceeds `config.max_seq_len`.

**Computation performed:**
1. Embed each token as `token_emb[id] + pos_emb[position]`, where `position` is the absolute stream index (token `i` sits at index `i`, or `i + 1` when a Gist occupies slot 0).
2. For each transformer layer: pre-norm (RMSNorm), multi-head attention with causal masking, residual add, pre-norm (RMSNorm), MLP with GELU activation, residual add.
3. Apply final RMSNorm to the last token's hidden state.
4. Project to vocabulary via `bit_linear` using `head_w`.

**Memory:** All intermediate buffers are heap-allocated as `std::vector` per call. For microcontroller use, replace these with static or arena-allocated buffers.

---

### `KVCache`

```cpp
struct KVCache {
    vector<float> K;   // [layer][pos][dim]
    vector<float> V;
    int len = -1;      // valid positions; -1 = empty/stale
    int depth, max_seq_len, dim;
    void init(int depth, int max_seq_len, int dim);
    float *k_row(int layer, int pos);
    float *v_row(int layer, int pos);
};
```

Per-layer key/value cache holding one `dim` row per position per layer, sized for `config.max_seq_len` positions. `len` is the number of valid (contiguous, from position 0) rows.

Typical usage: pass it to `forward` during prefill, then call `forward_one` per generated token. When the context window slides, learned position embeddings are rebased and every cached row goes stale — set `cache.len = -1` and refill via `forward` on the truncated context.

---

### `forward_one`

```cpp
inline bool forward_one(AtomicModel &model, int token,
                        vector<float> &logits, KVCache &cache);
```

Decodes a single token at stream position `cache.len`, appending its K/V rows at every layer. Produces **bit-identical** logits to running `forward` on the whole context — it performs the same operations in the same order for the new row.

**Returns:** `true` on success. `false` when the cache is stale (`len < 0`), full (`len >= max_seq_len`), dimensionally mismatched, or `token` is out of vocab range — refill with `forward` in that case.

**Parameters:**
- `model` — A fully loaded `AtomicModel`.
- `token` — The token ID at position `cache.len` (usually the last context token or the previously generated one).
- `logits` — Output buffer of size `model.config.vocab_size`.
- `cache` — A `KVCache` previously filled by `forward`.

---

### `bit_linear`

```cpp
inline void bit_linear(const vector<float> &x, const vector<int8_t> &w,
                       float scale_w, vector<float> &out);
```

Core BitLinear compute kernel. Implements one linear layer of the form: quantize input, integer matrix-multiply against ternary weight, dequantize output.

**Parameters:**
- `x` — Float input vector of size `in_dim`.
- `w` — Ternary weight matrix, flattened as `[in_dim, out_dim]` in row-major order (i.e., `w[i * out_dim + o]`). Values must be in `{-1, 0, 1}`.
- `scale_w` — Per-tensor `float32` weight scale read from the binary.
- `out` — Output vector. Must be pre-allocated to size `out_dim`.

**Quantization steps:**
1. SubLN: compute `rms = sqrt(mean(x^2) + 1e-5)`; the effective input is `x / rms`.
2. AbsMax scaling of the normalized input: `scale_x = 127 / max(|x| / rms, 1e-5)`; quantize `roundf(x_i / rms * scale_x)` clamped to `[-127, 127]`.
3. Integer accumulation: `acc[o] += x_q[i] * w[i * out_dim + o]` in `int32` (zero activations skip the whole row).
4. Dequantize: `out[o] = acc[o] * (scale_w / scale_x)`.

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
    KVCache cache;
    int t_off = model.config.has_gist ? 1 : 0;
    int ctx_cap = model.config.max_seq_len - t_off;

    // Greedy decode loop: full forward on the prompt, one cached step per
    // generated token after that
    for (int step = 0; step < 50; ++step) {
        if ((int)context.size() > ctx_cap) {
            // Window is full: slide it, then rebuild -- learned positions are
            // rebased, so cached rows are stale either way
            int keep = ctx_cap - ctx_cap / 4;
            context.erase(context.begin(),
                          context.begin() + ((int)context.size() - keep));
            cache.len = -1;
        }

        if (cache.len == (int)context.size() - 1 + t_off) {
            if (!forward_one(model, context.back(), logits, cache)) break;
        } else {
            forward(model, context, logits, &cache);
        }

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
| 4 | 4 | `int32` | Version: `3` (v2 files remain readable) |
| 8 | 4 | `int32` | `vocab_size` |
| 12 | 4 | `int32` | `dim` |
| 16 | 4 | `int32` | `depth` |
| 20 | 4 | `int32` | `heads` |
| 24 | 4 | `int32` | `max_seq_len` |
| 28 | 4 | `int32` | `has_gist` |
| 32 | `dim * 4` | `float32[]` | Gist vector (only if `has_gist == 1`) |
| ... | `vocab_size * 4` | `float32[]` | Token embedding row scales (v3 only) |
| ... | `vocab_size * dim` | `int8[]` | Token embedding codes (v3; v2: `vocab_size * dim * 4` `float32[]`) |
| ... | `max_seq_len * 4` | `float32[]` | Positional embedding row scales (v3 only) |
| ... | `max_seq_len * dim` | `int8[]` | Positional embedding codes (v3; v2: `max_seq_len * dim * 4` `float32[]`) |
| ... | (repeated `depth` times) | — | Layer weights (see below) |
| ... | `dim * 4` | `float32[]` | Final RMSNorm scale |
| ... | `4 + ceil(dim * vocab_size / 4)` | `float32 + packed[]` | LM head (scale + ternary weights) |

Each layer block contains:

| Size | Type | Description |
|------|------|-------------|
| `dim * 4` | `float32[]` | `ln1` scale |
| `4 + ceil(dim * dim / 4)` | `float32 + packed[]` | Q projection (scale + weights) |
| `4 + ceil(dim * dim / 4)` | `float32 + packed[]` | K projection (scale + weights) |
| `4 + ceil(dim * dim / 4)` | `float32 + packed[]` | V projection (scale + weights) |
| `4 + ceil(dim * dim / 4)` | `float32 + packed[]` | O projection (scale + weights) |
| `dim * 4` | `float32[]` | `ln2` scale |
| `4 + ceil(dim * 4*dim / 4)` | `float32 + packed[]` | FC1 weight (scale + weights) |
| `4 + ceil(4*dim * dim / 4)` | `float32 + packed[]` | FC2 weight (scale + weights) |

### Ternary weight packing

Version 2 stores four ternary weights per byte. Each weight is written as
`w + 1`, so `{-1, 0, 1}` become codes `{0, 1, 2}` and code `3` never appears.
Codes fill a byte from the low-order pair upward, following the flat row-major
order of the `(in_dim, out_dim)` tensor; the final byte is zero-padded.

To decode weight `i`:

```cpp
int8_t w = (int8_t)((packed[i >> 2] >> (2 * (i & 3))) & 3) - 1;
```

`atomic_lib.h` and `atomic_runner.cpp` unpack at load time, so `AtomicLayer`
still exposes plain `vector<int8_t>` and the inner loop is unchanged. The ESP32
port decodes while streaming from flash instead, which is where the 4x
reduction in reads matters.

### Embedding quantization (version 3)

The embedding tables are the largest tensors in the file, so version 3 stores
them as int8 with per-row scales instead of raw float32. Each row `r` is
quantized independently by its absmax:

```
scale[r] = max(|w[r, :]|) / 127          (clamped to >= 1e-8)
code[r, d] = round(w[r, d] / scale[r])   (clamped to [-127, 127])
```

All `rows` scales are written first, then all `rows * dim` codes in row-major
order. Loaders reconstruct `w[r, d] ~= code[r, d] * scale[r]` into the same
float32 tables v2 used, so the forward pass is identical and runtime RAM does
not change; only the file shrinks (~4x on each table). Reconstruction error is
bounded at `scale[r] / 2` per element. A v2 file keeps the raw float32 layout
and is still accepted.
