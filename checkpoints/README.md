# Checked-in fixture checkpoint

`tiny_stories.pt` is a **small trained model** (2.65 MB) shipped so that
parity and quality evals run out of the box -- `weights/` and `*.pt` are
gitignored, so without this file the parity tool and the trained-model
tests have nothing to run on. It is a pipeline proof, not a quality model.

## What it is

| Field | Value |
|---|---|
| Config | vocab 2048, dim 128, depth 4, heads 4, ctx 64 (1,320,064 params) |
| Training | 14,000 steps total, batch 32, AdamW lr 1e-3 (2,000 via `train_fixture_checkpoint.py` + 12,000 via `tools/continue_fixture.py` on MPS) |
| Data | TinyStories (HF `roneneldan/TinyStories`); vocab map unchanged |
| Final train loss | ~3.5 |
| Held-out perplexity | **42.0** vs 2048 uniform, measured on the stored fp16 artifact with the same `validation[:1000]` eval harness (the 2,000-step version scored 115.5) |
| Stored dtype | **float16** state dict (keeps the file under ~3 MB; `load_state_dict` upcasts to float32, and Python/C++ see identical stored weights) |

Uniform-perplexity baseline for this vocab is 2048; 42.0 means the
checkpoint produces genuinely readable TinyStories text -- e.g. temp-0.8
sampling gives *"Once upon a time, there was a little bird loved to be a
noise too. One day, Lucy ... He was so excited to the ghost."* It is
still a 1.3M-parameter fixture, not a storyteller.

## How it was produced

```bash
python3 tools/train_fixture_checkpoint.py --steps 2000
python3 tools/continue_fixture.py --steps 12000   # resumes, non-interactive
```

`tiny_stories_vocab.json` is the gpt2-id -> model-id frequency map the same
script writes (schema matches `weights/vocab_map_*.json`); decode runner
output through it. If TinyStories cannot be downloaded the script falls
back to a deterministic local corpus and says so in its output and in the
checkpoint metadata (`training.dataset`).

`tiny_stories.vocab` is the detokenizer file for the C++ runner's `--vocab`
flag -- one line per model token id holding the lowercase hex of its raw
bytes (regenerate with `tools/make_vocab.py`). With it the runner prints
decoded text instead of bare ids:

```bash
./embedded/runner --model trained.bin --vocab checkpoints/tiny_stories.vocab
```

## What consumes it

- `python3 tools/parity_check.py` -- exports it to `.bin`, runs the C++
  runner's `--parity` dump, and compares every probe numerically.
- `tests/test_trained_parity.py` -- same check under pytest, plus a
  trained-not-random gate.
- `tests/test_runner_cli.py` -- exercises `--vocab` against
  `tiny_stories.vocab` end to end.
- `benchmarks/run_suite.py` loads a bare state dict from
  `weights/benchmark_model.pt` at this exact config; adapt with
  `torch.save(torch.load("checkpoints/tiny_stories.pt")["model_state_dict"],
  "weights/benchmark_model.pt")`.
