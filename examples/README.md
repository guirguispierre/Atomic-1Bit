# Examples

Runnable example scripts for Atomic-1Bit.

## Scripts

### `quickstart.py`

End-to-end demo: creates a model, runs a forward pass, quantizes weights, and shows how the ternary kernel works. No trained checkpoint needed.

```bash
python3 examples/quickstart.py
```

### `generate.py`

Loads a trained checkpoint and generates text. Requires a trained model in `weights/`.

```bash
# First, train a model:
python3 atomic_1bit/training/train.py

# Then generate:
python3 examples/generate.py
```

## Requirements

All examples use the same dependencies as the main project:

```bash
pip install -r requirements.txt
```
