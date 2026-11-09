"""
Atomic-1Bit Text Generation Example

Loads a trained checkpoint and generates text interactively. Works out of
the box with the shipped fixture -- no training run needed:

    python3 examples/generate.py
    python3 examples/generate.py --checkpoint weights/stories_final.pt --temp 0.7

Checkpoints trained with a remapped vocabulary (e.g. the TinyStories
fixture) need the matching vocab map so gpt2 ids translate to model ids;
``--vocab_map`` defaults to the fixture map when it exists.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tiktoken  # noqa: E402
import torch  # noqa: E402

from atomic_1bit.model.transformer import AtomicConfig, AtomicTransformer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def infer_config(checkpoint):
    """Build AtomicConfig from a checkpoint dict.

    Prefer an embedded ``config`` mapping; otherwise infer shapes from the
    state dict (embedding rows -> vocab, columns -> dim, pos rows -> ctx,
    layer key count -> depth). Heads cannot be inferred from shapes, so a
    stored value wins and 4 is the fallback.
    """
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        c = checkpoint["config"]
        return AtomicConfig(
            vocab_size=c.get("vocab_size", 4096),
            dim=c.get("dim", 256),
            depth=c.get("depth", 6),
            heads=c.get("heads", 4),
            context_length=c.get("context_length", 128),
        )
    state = checkpoint.get("model_state_dict", checkpoint)
    vocab, dim = state["token_emb.weight"].shape
    ctx = state["pos_emb.weight"].shape[0]
    depth = sum(1 for k in state if k.startswith("layer") and k.endswith("ln1.weight"))
    return AtomicConfig(
        vocab_size=vocab,
        dim=dim,
        depth=depth,
        heads=checkpoint.get("heads", 4),
        context_length=ctx,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate text with Atomic-1Bit")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(ROOT, "checkpoints", "tiny_stories.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--vocab_map",
        type=str,
        default=os.path.join(ROOT, "checkpoints", "tiny_stories_vocab.json"),
        help="JSON with token_map/reverse_map for remapped-vocab models",
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Sampling temperature (0=greedy)"
    )
    parser.add_argument(
        "--tokens", type=int, default=80, help="Number of tokens to generate"
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train a model first: python3 atomic_1bit/training/train.py")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = infer_config(checkpoint)
    print(
        f"  Config: vocab={config.vocab_size} dim={config.dim} "
        f"depth={config.depth} heads={config.heads} ctx={config.context_length}"
    )

    model = AtomicTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    enc = tiktoken.get_encoding("gpt2")
    token_map = reverse_map = None
    if os.path.exists(args.vocab_map):
        with open(args.vocab_map) as f:
            data = json.load(f)
        token_map = {int(k): v for k, v in data["token_map"].items()}
        reverse_map = {int(k): v for k, v in data["reverse_map"].items()}
        print(f"  Vocab map: {len(token_map)} entries ({args.vocab_map})")
    else:
        print("  No vocab map found -- using raw gpt2 ids")
    unk_id = config.vocab_size - 1

    # Learned positions hard-limit the window: prompt + generation must fit.
    max_new = max(1, config.context_length - 1)
    if args.tokens > max_new:
        print(f"  Clamping --tokens {args.tokens} -> {max_new} (context limit)")
        args.tokens = max_new

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

        ids = enc.encode(prompt)
        if token_map is not None:
            ids = [token_map.get(g, unk_id) for g in ids]
        input_ids = torch.tensor([ids], dtype=torch.long)
        if input_ids.shape[1] > config.context_length - args.tokens:
            input_ids = input_ids[:, -(config.context_length - args.tokens) :]

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=args.tokens, temperature=args.temp
            )

        generated = output[0].tolist()
        if reverse_map is not None:
            generated = [reverse_map.get(t, enc.eot_token) for t in generated]
        print(f"\n{enc.decode(generated)}\n")

    print("Done.")


if __name__ == "__main__":
    main()
