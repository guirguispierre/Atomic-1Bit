"""
Atomic-1Bit Quickstart Demo

Demonstrates the core concepts without needing a trained checkpoint:
1. Creates a small ternary transformer
2. Shows how BitLinear quantizes weights to {-1, 0, 1}
3. Runs a forward pass
4. Generates sample tokens

Usage:
    python3 examples/quickstart.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.nn.layers import BitLinear, weight_quant


def main():
    print("=" * 60)
    print("Atomic-1Bit Quickstart Demo")
    print("=" * 60)

    # --- 1. Create a small model ---
    config = AtomicConfig(
        vocab_size=4096,
        dim=128,
        depth=4,
        heads=4,
        context_length=64,
    )

    model = AtomicTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel created: {param_count:,} parameters")
    print(f"  Dimensions: {config.dim}d, {config.depth} layers, {config.heads} heads")
    print(f"  Context length: {config.context_length}")
    print(f"  Vocab size: {config.vocab_size}")

    # --- 2. Show weight quantization ---
    print("\n--- Weight Quantization Demo ---")
    sample_weight = model.layers[0].attn.q_proj.weight.data
    quantized, scale = weight_quant(sample_weight)

    unique_vals = torch.unique(quantized.round())
    print(f"Original weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")
    print(f"Quantized to: {sorted([int(v.item()) for v in unique_vals])}")
    print(f"Scale factor: {scale.item():.4f}")

    # Count ternary distribution
    total = quantized.numel()
    n_neg = (quantized.round() == -1).sum().item()
    n_zero = (quantized.round() == 0).sum().item()
    n_pos = (quantized.round() == 1).sum().item()
    print(f"Distribution: -1={n_neg/total:.1%}, 0={n_zero/total:.1%}, +1={n_pos/total:.1%}")
    print(f"Sparsity (zero weights): {n_zero/total:.1%}")

    # --- 3. Forward pass ---
    print("\n--- Forward Pass ---")
    tokens = torch.randint(0, config.vocab_size, (1, 16))  # batch=1, seq=16
    print(f"Input shape: {tokens.shape}")

    with torch.no_grad():
        logits = model(tokens)

    print(f"Output logits shape: {logits.shape}")
    print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")

    # --- 4. Generate tokens ---
    print("\n--- Token Generation ---")
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    print(f"Prompt tokens: {prompt[0].tolist()}")

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)

    new_tokens = generated[0, 4:].tolist()
    print(f"Generated {len(new_tokens)} tokens: {new_tokens}")

    # --- 5. Model size estimate ---
    print("\n--- Size Comparison ---")
    fp32_size = param_count * 4  # 4 bytes per float32 param
    ternary_size = param_count * 2 / 8  # 2 bits per ternary weight
    print(f"FP32 size:    {fp32_size / 1024:.1f} KB")
    print(f"Ternary size: {ternary_size / 1024:.1f} KB (packed)")
    print(f"Reduction:    {(1 - ternary_size / fp32_size) * 100:.0f}%")

    print("\n" + "=" * 60)
    print("Done. To train a real model, run:")
    print("  python3 atomic_1bit/training/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
