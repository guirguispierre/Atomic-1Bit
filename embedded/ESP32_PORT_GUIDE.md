# Atomic-1Bit: ESP32 Porting Guide

## 1. Memory Analysis
Your model has **~28 Million Parameters**.

### Storage (Flash/SD)
- **Current (Int8)**: 28MB.
- **Max ESP32 Flash**: 16MB (Typical S3 is 8MB or 16MB).
- **Conclusion**: You CANNOT store the raw Int8 model on internal Flash.
- **Solution A**: Use an SD Card (Supports 32GB+).
- **Solution B (BitNet)**: 1.58-bit packing.
  - 1 weight = 2 bits (00, 01, 10, 11).
  - 28M * 2 bits = 56M bits = **7MB**.
  - **Result**: Packed model FITS on 8MB/16MB Flash!

### RAM (Runtime)
- **Activations**: 
  - `dim = 256`. 
  - `context = 128`.
  - `float` buffer (4 bytes).
  - `256 * 128 * 4` = ~131KB.
  - **Result**: Fits easily in ESP32 SRAM (512KB).
- **Weights (Loading)**:
  - You cannot load all 28MB (or 7MB) into RAM.
  - **Strategy**: **Stream weights layer-by-layer.**
  - Load Layer 0 -> Compute -> Discard -> Load Layer 1 -> ...

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
