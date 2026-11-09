# Atomic-1Bit: ESP32 Porting Guide

## 1. Memory Analysis

### Storage (Flash/SD)
Format v3 packs ternary weights four per byte and stores each embedding row as
int8 codes with a float32 absmax scale, so the exported `.bin` is what you
flash directly. Measured sizes:

| Config | Exported | Fits 8MB flash | Fits 16MB flash |
|:---|---:|:---|:---|
| `pocket_4k` (5.28M params) | 2.2 MB | yes | yes |
| `flagship_12m` (12.5M params) | 4.2 MB | yes | yes |
| `stories_base` (6.85M params, 4K vocab) | 2.6 MB | yes | yes |

`main.cpp` reads either layout -- v3 quantized embeddings or v2 raw float32 --
and dequantizes to float32 tables in PSRAM at load time, so the RAM budget
below is unchanged; only the flash footprint shrank.

### RAM (Runtime)
- **Activations**: 
  - `dim = 256`. 
  - `context = 128`.
  - `float` buffer (4 bytes).
  - `256 * 128 * 4` = ~131KB.
  - **Result**: Fits easily in ESP32 SRAM (512KB).
- **Weights (Loading)**:
  - Still too large for SRAM even packed.
  - **Strategy**: **Stream weights layer-by-layer.**
  - Load Layer 0 -> Compute -> Discard -> Load Layer 1 -> ...
  - `platforms/esp32/src/main.cpp` reads one packed weight row per call, so a
    row of `out_dim` weights costs `out_dim / 4` bytes of flash read.

## 2. Porting Strategy

### Step 1: `atomic_lib.h` Integration
Copy `atomic_lib.h` into your Arduino project.

### Step 2: Implement `Stream` Loading
Modify `bit_linear` to read weights from a `File*` (SD Card) or `const uint8_t*` (Flash pointer) on demand, instead of `vector<int8_t>`.

```cpp
// Pseudocode for Streaming Linear
void bit_linear_stream(vector<float>& x, File& weight_file, int out_dim) {
    int dim = x.size();
    vector<int8_t> weight_row(dim); // 256 bytes buffer
    
    for(int o=0; o<out_dim; ++o) {
        weight_file.read(weight_row.data(), dim);
        // ... compute dot product ...
    }
}
```

### Step 3: Packing (Already Done)
Format v2 already packs 4 ternary weights into 1 byte (`(byte >> (i*2)) & 0x03`
decodes them), and v3 quantizes the embedding tables to int8 with per-row
scales. `main.cpp` and `atomic_lib.h` decode both at load.

## Recommendation
Start with **ESP32-S3 + SD Card**.
- Simple code (no packing needed yet).
- Huge storage (can run larger models).
- Use `atomic_lib.h` but replace `vector<AtomicLayer>` with `load_layer(i)` function.
