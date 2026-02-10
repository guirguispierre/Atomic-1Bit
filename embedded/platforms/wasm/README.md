# Atomic-1Bit WebAssembly Demo

Run a ternary language model directly in your browser. No server required.

## Requirements

- [Emscripten SDK](https://emscripten.org/docs/getting_started/) (emcc compiler)
- A trained Pocket model exported to `.bin` format
- A modern web browser with WebAssembly support

## Build

```bash
# 1. Activate Emscripten environment
source /path/to/emsdk/emsdk_env.sh

# 2. Build the WASM module
cd embedded/platforms/wasm
make

# This produces:
#   atomic.js    - Emscripten JS glue code
#   atomic.wasm  - WebAssembly binary
```

## Run

```bash
# 3. Export a model (if not already done)
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/pocket_final.pt \
  --output embedded/platforms/wasm/atomic_model.bin \
  --dim 256 --depth 4 --heads 4 --vocab_size 4096 --context_len 128

# 4. Start a local HTTP server
cd embedded/platforms/wasm
make serve
# Or: python3 -m http.server 8080

# 5. Open http://localhost:8080/index.html in your browser
# 6. Click "Load .bin Model" and select the .bin file
# 7. Click "Generate" to run inference
```

## Architecture

```
Browser
  ├── index.html      (UI + JS controller)
  ├── atomic.js       (Emscripten glue, loads WASM)
  ├── atomic.wasm     (Compiled C++ inference engine)
  └── *.bin           (Model file, loaded via file picker)
```

### How It Works

1. **Model Loading**: The user selects a `.bin` file through the browser file picker.
   The binary is copied to the WASM heap and parsed (ATOM header + embeddings + weights).

2. **Inference**: Each `generate_token()` call runs a full forward pass through the
   ternary transformer (RMSNorm, BitLinear, multi-head attention, GELU MLP).

3. **Sampling**: Greedy decoding runs in WASM. Temperature sampling reads the logits
   buffer from WASM memory and samples in JavaScript.

4. **Output**: Token IDs are displayed in real-time with TPS measurement.

## Memory Budget

| Component | Size (Pocket Model) |
|-----------|-------------------|
| Token embeddings | 4096 * 256 * 4 = 4 MB |
| Positional embeddings | 128 * 256 * 4 = 128 KB |
| Layer weights (4 layers) | ~2 MB (INT8) |
| Working buffers | ~1 MB |
| **Total** | **~7 MB** |

The WASM module is configured with 64 MB initial memory and up to 256 MB maximum,
which is well within browser limits.

## Controls

| Control | Description |
|---------|-------------|
| **Load .bin Model** | Select an ATOM-format model binary |
| **Generate** | Start autoregressive token generation |
| **Stop** | Stop generation early |
| **Clear** | Clear output and reset context |
| **Tokens** | Maximum tokens to generate |
| **Temp** | Sampling temperature (0 = greedy) |
| **Start** | Initial token ID to seed generation |

## Limitations

- Token IDs are displayed (not decoded text) since the tokenizer runs in Python
- No SIMD optimization in WASM (scalar ternary matmul only)
- Performance depends on browser and device (~5-20 TPS for Pocket model)
- Model file must be loaded each time (not cached between sessions)
