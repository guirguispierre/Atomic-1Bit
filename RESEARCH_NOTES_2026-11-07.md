# Research notes, 2026-11-07: independent verification run

Everything below was measured, not read from the README. Environment: Linux
x86_64 VM, torch 2.15.0+cpu, g++ -O3. Absolute speeds are this machine, not
Apple silicon and not an ESP32. Weights were random (untrained) unless noted:
speed and size numbers are valid, quality numbers do not exist yet.

## What works, verified

- Quickstart passes end to end: model builds, forward pass clean, tokens
  generate. Measured sparsity 31.1% zero weights. Packed size 450.3 KB vs
  7204.5 KB fp32 on the demo config (94 percent smaller).
- Test suite green: `pytest -m "not slow"` gives 146 passed, 15 skipped,
  8 deselected (slow), exit 0.
- C++ runner compiles clean (`g++ -O3 -std=c++17`), loads an exported model,
  generates 50 tokens, exit 0.

## Benchmark config numbers (dim 128, depth 4, heads 4, vocab 2048, ctx 64, 1,320,064 params)

From `benchmarks/results.json` after building what the suite assumes exists:

| Path | TPS | Size |
|---|---|---|
| Python ternary | 14.09 | 5.05 MB (fp32 checkpoint) |
| FP16 baseline (PyTorch) | 115.11 | 5.05 MB |
| Atomic C++ runner | 123.31 | 1.29 MB exported bin |

The C++ engine beats the fp16 PyTorch baseline on this CPU and is about 9x
the Python ternary path. The product claim lives in the C++ engine; the
Python path is a reference, and its numbers should never be quoted as the
engine's.

## Finding 1: a fresh clone does not run

The Python package hard-requires `atomic_1bit/core/libatomic.so`, and nothing
in the install builds it. Every entry point (quickstart, tests, parity tool,
benchmarks) fails with FileNotFoundError until you run `make` in
`atomic_1bit/core/` by hand. Same pattern twice more: the benchmark suite
assumes `embedded/runner` and `embedded/atomic_model.bin` already exist, and
`tools/parity_check.py` assumes `weights/stories_final.pt`. Three undocumented
manual steps between clone and any number at all.

## Finding 2: the embeddings are the model

In the exported bin for the benchmark config (1,348,228 bytes total):

- token embedding, stored fp32: 1,048,576 bytes (78 percent)
- position embedding, stored fp32: 32,768 bytes
- all transformer block weights plus head, 2-bit packed: about 262 KB

Effective size is about 7.8 bits per parameter, not 1.58, because the
embeddings ship unquantized. The flagship config has the same shape:
4096 x 256 fp32 token embedding is 4.19 MB of the 5.8 MB headline size
(about 72 percent). Packing or quantizing the embeddings is the single
biggest lever in the project: on the benchmark config it takes the bin from
1.32 MB to roughly 0.35 MB.

## Finding 3: parity is a trace, not a check

`./runner --parity` dumps per-layer activations and final logits for one
step. It never compares them against the Python model numerically, so it
cannot fail. The kernel parity tests inside the pytest suite do compare and
pass. Trained-model parity is still unverified: the checkpoint the parity
tool wants is not in the repo.

## Next steps, proof first (in order)

1. Quantize the embeddings in the export format (ternary pack like the rest,
   or int8 as a middle step) and re-measure size and parity. Biggest win,
   smallest change.
2. Wire the kernel build into the install so a fresh clone runs: build
   libatomic.so from setup, and have the benchmark suite build the runner
   and export a model itself.
3. SIMD in `backends/cpu_kernel.cpp` (it is plain scalar loops today), with
   the kernel parity tests as the gate.
4. ESP32-S3 target build via the PlatformIO project in
   `embedded/platforms/esp32/`, so the port is proven to compile; on-device
   tokens per second still needs the physical board.
5. Check in one small trained checkpoint so parity and quality evals run
   out of the box.

Raw logs from this run are kept with the workspace copy (results, quickstart,
pytest, parity, benchmark outputs), available on request.
