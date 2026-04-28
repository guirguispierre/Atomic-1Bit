"""
Atomic-1Bit Text Generation Example

Loads a trained checkpoint and generates text interactively.
Requires a trained model -- run `python3 atomic_1bit/training/train.py` first.

Usage:
    python3 examples/generate.py
    python3 examples/generate.py --checkpoint weights/stories_final.pt --temp 0.7
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tiktoken
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig


def main():
    parser = argparse.ArgumentParser(description="Generate text with Atomic-1Bit")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/stories_final.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Sampling temperature (0=greedy)"
    )
    parser.add_argument(
        "--tokens", type=int, default=100, help="Number of tokens to generate"
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train a model first: python3 atomic_1bit/training/train.py")
        sys.exit(1)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract config from checkpoint or use defaults
    config = AtomicConfig(
        vocab_size=checkpoint.get("vocab_size", 4096),
        dim=checkpoint.get("dim", 256),
        depth=checkpoint.get("depth", 6),
        heads=checkpoint.get("heads", 4),
        context_length=checkpoint.get("context_len", 128),
    )

    model = AtomicTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters")

    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Interactive generation
    print(f"\nGenerating {args.tokens} tokens at temperature {args.temp}")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.strip().lower() in ("quit", "exit", "q"):
            break

        if not prompt.strip():
            continue

        tokens = enc.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        # Truncate to context length
        if input_ids.shape[1] > config.context_length - args.tokens:
            input_ids = input_ids[:, -(config.context_length - args.tokens) :]

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=args.tokens, temperature=args.temp
            )

        generated = output[0].tolist()
        text = enc.decode(generated)
        print(f"\n{text}\n")

    print("Done.")


if __name__ == "__main__":
    main()
