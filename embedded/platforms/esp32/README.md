# Atomic-1Bit ESP32 Demo

Run a ternary language model on an ESP32 microcontroller.

## Requirements

- ESP32 DevKit with **PSRAM** (e.g., ESP32-WROVER)
- [PlatformIO](https://platformio.org/) installed
- A trained Pocket model exported to `.bin` format

## Memory Budget

| Component | Size |
|-----------|------|
| ESP32 SRAM | 520 KB |
| PSRAM (optional) | 4-8 MB |
| Token embeddings | 4096 * 256 * 4 = 4 MB |
| Positional embeddings | 128 * 256 * 4 = 128 KB |
| Weight layers | Streamed from flash |

**Strategy**: Embeddings in PSRAM, weights streamed from SPI flash (SPIFFS).

## Workflow

### 1. Export the Pocket Model

```bash
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/pocket_final.pt \
  --output embedded/platforms/esp32/data/atomic_model.bin \
  --dim 256 --depth 4 --heads 4 --vocab_size 4096 --context_len 128
```

### 2. Upload Model to SPIFFS

Place the `.bin` file in the `data/` directory, then:

```bash
cd embedded/platforms/esp32
pio run --target uploadfs
```

### 3. Build and Flash

```bash
pio run --target upload
```

### 4. Monitor Output

```bash
pio device monitor
```

## Flash-Streaming Architecture

Instead of loading all weights into RAM, this demo reads weight blocks
from SPI flash on demand:

1. **Embeddings** are loaded into PSRAM at startup (4 MB)
2. **Layer weights** are read from SPIFFS during inference
3. Each `bit_linear` call seeks to the weight offset and streams data

This trades latency for memory: inference is slower than full-RAM loading,
but fits within the ESP32's memory constraints.

## Future: Packed 2-bit Weights

With 2-bit weight packing (4 weights per byte), the Pocket model shrinks
from ~2 MB to ~0.5 MB, making it feasible to load entirely into PSRAM.
