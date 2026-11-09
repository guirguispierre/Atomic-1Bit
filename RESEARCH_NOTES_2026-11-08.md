# Research notes, 2026-11-08: follow-up verification run

Everything below was measured on macOS arm64 (Apple Silicon), clang 21,
torch 2.8.0, Python 3.9 venv. This run closes out the findings from
2026-11-07 and re-verifies the whole pipeline after the changes.

## Yesterday's findings -> today

| Finding (2026-11-07) | Status now |
|---|---|
| 1. Fresh clone does not run (3 manual steps) | Fixed: kernel auto-builds on import, `make verify` covers clone->proof, benchmark suite builds the runner and exports the model itself |
| 2. Embeddings are 72-78% of the export | Fixed: format v3 stores int8 codes + per-row f32 scales; Quick Start export 5.78 MB -> 2.55 MB (-55.8%) |
| 3. `--parity` never compares | Fixed: `tools/parity_check.py` diffs runner probes vs PyTorch numerically, exits nonzero on fail |
| next: SIMD kernel | Done: sdot path at 84.85 GMAC/s; AVX2 -128 wraparound bug found + fixed |
| next: ESP32 target build | Done: `pio run` for `esp32dev` compiles -- 22,492 B RAM (6.9%), 318,308 B flash (10.1%) |
| next: trained checkpoint in repo | Done: `checkpoints/tiny_stories.pt` (2.65 MB), used by `tools/parity_check.py` and `tests/test_trained_parity.py` |

## New things verified this run

- `make verify` from zero artifacts: kernel builds, kernel parity
  `>> SUCCESS` (NumPy bit-exact), runner smoke test, quickstart demo -- exit 0.
- `pytest tests/ -q`: **185 passed**.
- `tools/parity_check.py` on the shipped checkpoint: all probes PASS at the
  5e-2 bound, argmax next-token agrees, exit 0. Worst probe rel diff 2.98e-2
  (documented int8-boundary fork behavior).
- Trained checkpoint through the C++ engine: v3 export 532.88 KB,
  `--steps 60 --temp 0.8` -> **12,233 TPS**, real English out
  (TinyStories register; quality is honest for 2k steps).
- Metal backend (was silently broken): `make BACKEND=METAL` + metallib
  resolve via dladdr -> **all 168 tests at the time passed bit-exact on
  GPU**, and it no longer kills the host process without the metallib.
- Benchmark suite, self-bootstrapped (runner + bin deleted first):

  | Path | TPS | Size |
  |---|---|---|
  | Python ternary | 190.96 | 5.05 MB ckpt |
  | FP16 PyTorch | 1,201.43 | 5.05 MB |
  | Atomic C++ runner | **12,724.90** | **0.52 MB v3 bin** |

  The C++ engine is ~10.6x the fp16 PyTorch baseline and ~67x the Python
  ternary path on this machine, at a tenth of yesterday's file size.
  (Linux x86 numbers from 2026-11-07 were 14.09 / 115.11 / 123.31 --
  different hardware; the KV cache and sdot kernel did not exist then.)
- Kernel micro-benchmark (`benchmarks/kernel_micro.py`): 84.85 GMAC/s,
  0.203 ms for 1x4096x4096 vs 55.58 ms scalar -- ~255x. AVX2 path verified
  on x86_64 under Rosetta: 493 parity cases, 0 failures.
- KV cache identical-output proof: 25/25 generation cases byte-identical
  pre/post change (greedy, sampled, top-p/top-k, multi-token prompts,
  gist model, context-overflow rebuild).
- ESP32 firmware build (xtensa-esp-elf-g++ 14.2): SUCCESS, v3 loader
  included.
- WASM verified end to end: built with Emscripten (emsdk 4.x) and driven
  from Node against the exported model -- the greedy 30-token stream is
  byte-identical to the native runner's.

## Round 2 findings -> fixes

- **Gist position parity bug (latent, both engines)**: PyTorch indexed
  prompt tokens from 0 with a gist occupying slot 0 while its decode
  path used stream indices; `atomic_lib.h` copied the same mistake.
  Prefill and decode disagreed by one position on every prompt token
  under a gist. Both now use stream indices; a v2-export parity test
  holds the two engines to <1e-3.
- **Gist models dead-generated past context**: the runner's slide cap
  was `max_seq_len` even with `has_gist`, so a full window skipped the
  forward pass and sampled uniformly forever. Cap is now `max_seq - 1`
  under a gist.
- **atomic_lib.h now has the KV cache** (`KVCache` + `forward_one`):
  bit-identical to full `forward()`, proven by a compiled harness that
  runs both paths per step over plain and gist exports.
- **`--vocab` text output** in the runner (`tools/make_vocab.py` makes
  the hex-per-line file; `checkpoints/tiny_stories.vocab` ships) and
  malformed `.vocab` files now fail loudly.
- **27 runner-CLI tests** added (`tests/test_runner_cli.py`), locking
  determinism, context-crossing slides, `--parity`, `--vocab`, arg
  validation, and gist exports.
- **Lint**: the Python tree was carrying ~700 flake8 violations; it is
  now clean under the exact CI command (black + isort + noqa markers
  only on `sys.path`-dependent imports). 217 tests pass.
- **`--threads` worker pool**: chunked column claiming keeps P and E
  cores busy; output is byte-identical to serial (int32 accumulators
  partition exactly). Measured on the merged build: 15.4k TPS at
  `--threads 4` vs 14.2k serial on the fixture (memory-bound), and the
  agent measured ~5.3-5.6k vs ~3.3k on the larger Quick Start config.
- **`--gist` parity coverage** in `tools/parity_check.py`, plus a WARN
  band (1-4x tol) so quantization boundary excursions report without
  failing the engine.
- **Cardputer**: same stream-index position fix applied (the ESP32
  loader is a proof-of-concept stub with no forward pass to fix);
  `pio run` now compiles the firmware in CI.

## Round 3: fixture quality jump + interactive features

- **Fixture retrained 2000 -> 14,000 steps** (`tools/continue_fixture.py`,
  TinyStories train[:10%] on MPS, ~50 min): held-out perplexity on the
  identical `validation[:1000]` harness dropped **115.5 -> 42.0** (fp16
  stored artifact; the 2000-step model reproduced 115.5 exactly, so the
  protocol comparison is apples-to-apples). C++-sampled output went from
  word-fragments to grammatical text with dialogue. The checkpoint was
  repackaged fp16 with `config` + `training` metadata so parity and tests
  consume it identically.
- **`--prompt_text` + `--repeat_penalty`** in the runner: text prompts
  tokenize via the `.vocab` map (greedy longest-match), and a presence
  penalty curbs greedy loops.
- **`export_model(prompt=...)` repaired**: it called a nonexistent
  `model.gist_encoder.encode()`; now tokenizes via `vocab_map` and
  mean-pools `token_emb` (the same mechanism `gen_gist.py` uses).
- **`examples/generate.py` works out of the box** on the fixture:
  infers config from checkpoint tensor shapes and applies the vocab map.
- **WASM demo + thread-identity tests**: `index.html` decodes tokens to
  text through an optional `.vocab`; `TestThreads` locks byte-identical
  output across `--threads` counts. 230 tests pass total.

## Caveats kept honest

- KV cache still rebuilds once the window exceeds context_length
  (learned absolute positions rebase), but the slide now erases a
  quarter-window chunk: ~1 rebuild per `ctx/4` tokens instead of every
  token. Measured on the trained 1.32M model (ctx 64, dim 128):
  14,174 TPS for 60 in-context steps vs 5,628 TPS for 200 steps
  (136 of them post-window). The earlier pin-at-cap slide rebuilt the
  prefix on every post-window token.
- The Metal path needs `default.metallib` next to the dylib; without it
  (or without a Metal device) it now warns once and runs a scalar CPU
  fallback rather than exiting.
- CardputerChat.ino was rewritten for v2/v3 but not hardware-tested; its
  char-as-token-id stub is a demo shim, not a tokenizer.
- v3 embedding quant is lossy at load (<=scale/2 per element, ~1e-3 on
  logits) -- measured well under the 5e-2 parity gate.
