# Changelog

All notable changes to Atomic-1Bit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-11-08

### Changed
- **Binary format v3: embeddings quantized to int8 with per-row scales.** Each
  embedding row is stored as int8 codes plus one `float32` absmax scale
  (`scale = max(|row|) / 127`), replacing the raw float32 tables that were
  72-78% of a v2 export. The Quick Start 6.85M model drops 5.8 MB -> 2.6 MB,
  `pocket_4k` 5.4 MB -> 2.2 MB, and `flagship_12m` 8.4 MB -> 4.2 MB -- all
  three presets now fit 8 MB ESP32 flash. Ternary packing was rejected for
  embeddings: they are not QAT-trained, and {-1,0,1} rounding would zero most
  of the table.
- All four loaders (`atomic_runner.cpp`, `atomic_lib.h`, `platforms/esp32`,
  `platforms/wasm`) accept both v2 and v3 and dequantize embeddings to float32
  at load, so inference math and RAM usage are unchanged.
- `export_model` takes a `format_version` parameter (default 3); pass 2 for
  the legacy layout.

### Added
- **KV cache in the C++ runner.** Decode now appends one K/V row per layer
  per step instead of recomputing the prefix: ~3.1-3.5k TPS at 100 generated
  tokens on the Quick Start config where the O(n^2) path managed ~83 TPS
  (measured on Apple Silicon; output is bit-identical for a fixed seed).
- **ARM dot-product (sdot) path in the CPU kernel** plus a NEON-safe fallback
  and branch-free scalar tail -- ~83 GMAC/s on Apple Silicon. Kernel parity
  remains bit-exact.
- **A fresh clone now runs end to end**: `wrapper.py` builds `libatomic.so`
  via make on first import when it is missing, `setup.py` exists for
  `pip install .`, a root `Makefile` offers `make verify`, and
  `benchmarks/run_suite.py` builds the runner and exports a model itself.
- **Real parity check**: `tools/parity_check.py` now diffs the runner's
  dumped activations against PyTorch numerically and exits nonzero on
  failure, and `checkpoints/tiny_stories.pt` (a 2000-step TinyStories
  fixture, 2.65 MB) ships so parity and evals run out of the box.
- `tests/test_trained_parity.py`, `benchmarks/kernel_micro.py`, and a
  kernel parity test for odd tail dimensions.
- `tools/train_fixture_checkpoint.py` -- the script that produced the
  checked-in checkpoint, with an offline fallback corpus.

### Fixed
- **Metal backend**: `default.metallib` was located via compile-time
  `__FILE__` (a relative path under the Makefile), so any process outside
  `atomic_1bit/core/` hit `exit(1)` and died. The library now resolves next
  to the dylib via `dladdr`, warns and uses a scalar CPU fallback instead of
  exiting, and copies into real Metal buffers when host memory is not
  page-aligned for zero-copy wraps. 168 tests pass under `BACKEND=METAL`.
- **AVX2 bit-exactness**: `_mm256_sign_epi8` wrapped activation -128 x
  weight -1 back to -128 instead of +128 (off by 256). Replaced with a
  widen-to-int16 multiply path, verified on x86_64 under emulation.
- `make -C atomic_1bit/core clean` removes the generated metallib from the
  directory it is actually written to.
- **CardputerChat sketch** rewritten for format v2/v3 -- it previously read
  a layout that predated the packed format entirely (no header, no scales,
  unpacked weights) and could not load any export the tool produces.
- ESP32 PlatformIO project: source moved to `src/main.cpp` so `pio run`
  builds it; verified compiling for `esp32dev` (22.5 KB RAM, 318 KB flash).

## [1.5.0] - 2026-08-10

### Changed
- **Binary format v2: ternary weights packed four per byte.** Weights are
  written as `w + 1`, so `{-1, 0, 1}` become 2-bit codes `{0, 1, 2}`. This cuts
  the ternary portion of an export by 4x: `pocket_4k` 8.5 MB -> 5.4 MB,
  `flagship_12m` 16.7 MB -> 8.4 MB, and the Quick Start 6.85M model
  10.1 MB -> 5.8 MB. `pocket_4k` and `flagship_12m` now fit ESP32 flash.
  Both C++ loaders unpack at load, so the inner loop and throughput are
  unchanged. `platforms/esp32/main.cpp` decodes while streaming, replacing one
  flash seek per weight with one sequential read per weight row.
- Loaders reject v1 files with a message pointing at the exporter. Re-export
  any existing `.bin`.
- `platforms/wasm` and `platforms/esp32` now validate the format version, which
  neither did before.

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
