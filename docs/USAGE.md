# Usage Guide

This guide covers the main workflows: training models, running inference, exporting to C++, and evaluating quality.

## Training

### TinyStories (Default)

Train a 1.33M parameter model on the TinyStories dataset:

```bash
python3 atomic_1bit/training/train.py
```

- Saves checkpoints to `weights/`
- Default config: 256d, 6 layers, 4 heads, 128 context length
- Includes thermal safety monitoring (auto-pauses if CPU > 80C)

### Instruction Model (12.5M Parameters)

Train the flagship model on Alpaca-cleaned:

```bash
python3 atomic_1bit/training/train_instruct.py
```

- Uses gradient accumulation (effective batch size 256)
- Cosine learning rate schedule with warmup
- Saves to `weights/instruct_final.pt`

### Pocket Model (ESP32 Optimized)

Train a tiny model with 4K vocabulary for embedded deployment:

```bash
python3 atomic_1bit/training/train_pocket.py
```

- Vocabulary filtered to 4096 tokens
- Optimized for ESP32-S3 memory constraints
- Saves to `weights/pocket_final.pt`

### Using YAML Configs

Load model configuration from YAML files:

```python
from atomic_1bit.config import load_config, config_to_atomic

config = config_to_atomic(load_config("configs/flagship_12m.yaml"))
```

Available presets in `configs/`:
- `stories_base.yaml` -- 1.33M params, development/testing
- `flagship_12m.yaml` -- 12.5M params, quality demos
- `pocket_4k.yaml` -- Tiny model for microcontrollers
- `mixed_precision.yaml` -- Experimental hybrid quantization

## Inference

### Python Chat Interface

Interactive text generation:

```bash
python3 atomic_1bit/python/chat.py
```

### Kernel Verification

Verify C++ and Python produce identical output:

```bash
python3 atomic_1bit/python/inference.py
```

### C++ Bare-Metal Inference

After exporting a model (see below):

```bash
cd embedded
./runner --model atomic_model.bin --steps 100 --temp 0.7 --top_p 0.9 --seed 42 --prompt 1,42,58
```

Flags:
- `--steps` -- Number of tokens to generate
- `--temp` -- Temperature (0.0 = greedy, 0.7 = creative)
- `--top_k` -- Keep top-K logits before sampling (0 = disabled)
- `--top_p` -- Nucleus sampling cutoff (0.0 = disabled, 0.9 = typical)
- `--seed` -- Random seed for reproducibility
- `--start_token` -- Single starting token ID (default 42)
- `--prompt 1,2,3` -- Comma-separated starting token IDs (overrides `--start_token`)
- `--stream` / `--no-stream` -- Stream tokens as generated, or buffer until end
- `--help` -- Show all flags

## Exporting Models

Convert a PyTorch checkpoint to C++ binary format:

```bash
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --dim 256 --depth 6 --heads 4 --vocab_size 4096 --context_len 128
```

The `--dim`, `--depth`, `--heads`, `--vocab_size`, and `--context_len` flags **must match** the config used during training. Mismatched dimensions will produce garbage output.

### Exporting with Gist Tokens

To embed a pre-computed system prompt:

```bash
# 1. Generate the gist vector
python3 atomic_1bit/utils/gen_gist.py \
  --model weights/stories_final.pt \
  --prompt "You are a helpful storyteller."

# 2. Export with gist
python3 atomic_1bit/utils/export_to_cpp.py \
  --model weights/stories_final.pt \
  --output embedded/atomic_model.bin \
  --gist weights/gist.bin \
  --dim 256 --depth 6 --heads 4 --vocab_size 4096 --context_len 128
```

## Evaluation

### Run the Evaluation Suite

```bash
python3 atomic_1bit/evaluation/run_eval.py
```

This measures:
- **Perplexity** -- Language model quality
- **Repetition rate** -- Degenerate output detection
- **Coherence** -- Semantic consistency
- **Diversity** -- Vocabulary and n-gram variety

### Benchmarking

Compare against FP16 baselines:

```bash
python3 benchmarks/run_suite.py
```

Generate benchmark plots:

```bash
python3 scripts/generate_plots.py
```

Results are saved to `benchmarks/results.json` and charts to `assets/`.

## ESP32 Deployment

See [embedded/ESP32_PORT_GUIDE.md](../embedded/ESP32_PORT_GUIDE.md) for instructions on running models on ESP32-S3 microcontrollers, including memory analysis, streaming weight loading, and bit-packing strategies.

## Thermal Safety

Long training runs include automatic thermal monitoring:

- **Auto-pause** when system temperature exceeds 80C
- **Auto-resume** when temperature drops below 70C
- **Checkpoint save** before any thermal pause

The monitor disables itself gracefully on systems where temperature sensors aren't accessible.
