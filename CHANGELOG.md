# Changelog

All notable changes to Atomic-1Bit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-04-30

### Fixed
- **C++ deployment parity** — `embedded/atomic_lib.h` previously read 6 ints in the wrong order, skipping the `ATOM` magic and version, treating `magic` as `vocab_size`, and never consuming the per-tensor `float32` weight scale written by the exporter. Rewritten to read the full 8-int header, validate magic/version, consume weight scales, and apply them in `bit_linear` via SubLN-aware dequantization that matches the Python `BitLinear` forward exactly. (`atomic_runner.cpp` was already correct; this brings the documented header-only API in line.)
- **Trainer crash on temperature logging** — `train_instruct.py` called a nonexistent `thermal_monitor.get_current_temp()`. Added the public method to `ThermalMonitor` and made the trainer format `None` defensively when sensors are unavailable.
- **`chat.py` produced garbage on most checkpoints** — hardcoded `vocab=50257, dim=256, depth=4` only matched one model. Now auto-infers `vocab / dim / depth / context_length` from the checkpoint's `state_dict`, with CLI overrides.
- **`gen_gist.py` was locked to flagship dims** — added `--dim / --depth / --heads / --vocab_size / --context_length` CLI flags.

### Added
- **Top-p (nucleus) and top-k sampling** in the C++ runner alongside the existing temperature sampling. New flags: `--top_k`, `--top_p`.
- **Multi-token prompts** — `--prompt 1,2,3` now accepts a comma-separated starting sequence, overriding `--start_token`.
- **Streaming controls** — `--stream` (default) prints tokens as generated; `--no-stream` buffers until end.
- **Runner help** — `--help` / `-h` flag prints all options. Unknown flags now error instead of being silently ignored.
- **Export/load roundtrip test** (`tests/test_export_roundtrip.py`) — parses the exported binary using the C++ loader's exact expected layout and asserts every field arrives where the loader expects it. Catches future drift between exporter and loader.

### Changed
- `chat.py`: cleaner device auto-selection (`cuda` → `mps` → `cpu`), explicit error messages on missing checkpoint, `--max_new_tokens` flag.
- Documentation (`docs/USAGE.md`, `docs/COMMANDS.md`) updated to cover the new sampling and prompt flags.

### Verification
- 132 pytest tests pass (was 129; +3 roundtrip).
- CPU kernel parity (NEON SIMD path on Apple Silicon) passes against NumPy reference across 9 size configurations.

## [1.3.0] - 2026-03

### Added
- 12.5M flagship configuration (`flagship_12m`): 320d, 8 layers, 5 heads, 256 ctx.
- Evaluation harness — perplexity, repetition rate, coherence, diversity metrics.
- KV-cache for autoregressive generation.
- YAML config system with `pocket_4k`, `stories_base`, `flagship_12m`, `mixed_precision` presets.
- Tokenizer abstraction layer.
- ESP32 deployment guide and reference port (`embedded/platforms/esp32/`).
- Raspberry Pi NEON benchmark (`benchmarks/platforms/rpi/`).
- WebAssembly port (`embedded/platforms/wasm/`).
- CI/CD pipeline (GitHub Actions) with performance regression detection.

## [1.2.0] - 2026-02

### Added
- Metal backend (Apple Silicon GPU) with verified parity vs CPU reference.
- CUDA backend (NVIDIA GPU) with verified parity vs CPU reference.
- Modular backend selection via `make BACKEND=CPU|METAL|CUDA`.
- SIMD acceleration in the CPU backend: NEON (ARM) and AVX2 (x86_64) paths with scalar fallback.

## [1.0.0] - 2026-01

### Added
- Bit-exact Python ↔ C++ kernel parity (zero numerical divergence vs NumPy reference).
- Standalone C++ inference runtime (`embedded/atomic_runner.cpp`) with zero external dependencies.
- BitLinear layer with Straight-Through Estimator (STE) for training.
- Activation quantization (INT8) and weight quantization (ternary {-1, 0, 1}).
- Gist tokens — pre-computed thought vectors injected at zero inference cost.
- Training stack: TinyStories and Pocket model variants.
- Exporter (`atomic_1bit/utils/export_to_cpp.py`) for PyTorch → `.bin` conversion.
- Benchmark suite vs FP16 baselines.

### Fixed
- Kernel layout mismatch — Python wrapper passed `(K, N)` weights to a kernel expecting `(N, K)`. Parity now 100%.
- Memory alignment handling in the C++ loader.
- Export dimension validation now requires explicit flags to prevent silent shape mismatches.

[1.4.0]: https://github.com/guirguispierre/Atomic-1Bit/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/guirguispierre/Atomic-1Bit/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/guirguispierre/Atomic-1Bit/compare/v1.0.0...v1.2.0
[1.0.0]: https://github.com/guirguispierre/Atomic-1Bit/releases/tag/v1.0.0
