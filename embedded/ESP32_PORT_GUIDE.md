# Atomic-1Bit: ESP32 Porting Guide

## 1. Memory Analysis

### Storage (Flash/SD)
Format v2 packs ternary weights four per byte, so the exported `.bin` is what
you flash directly. Measured sizes:

| Config | Exported | Fits 8MB flash | Fits 16MB flash |
|:---|---:|:---|:---|
| `pocket_4k` (5.28M params) | 5.4 MB | yes | yes |
| `flagship_12m` (12.5M params) | 8.4 MB | no | yes |
| `stories_base` (30.5M params, 50257 vocab) | 56.0 MB | no | no (use an SD card) |

What remains is the float32 token embedding table, which is now the larger half
of the file: `vocab_size * dim * 4` bytes. Quantizing it is the next lever if
you need to go smaller.

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
  - `platforms/esp32/main.cpp` reads one packed weight row per call, so a
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

### Step 3: Packing (Advanced)
To fit on Flash without SD:
1. Update `export_to_cpp.py` to pack 4 weights into 1 byte.
2. Update `atomic_lib.h` to unpack `(byte >> (i*2)) & 0x03` on the fly.

## Recommendation
Start with **ESP32-S3 + SD Card**.
- Simple code (no packing needed yet).
- Huge storage (can run larger models).
- Use `atomic_lib.h` but replace `vector<AtomicLayer>` with `load_layer(i)` function.
