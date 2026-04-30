import sys
import os
import torch
import torch.nn.functional as F
import tiktoken
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig


def top_k_sampling(logits, k=50, temperature=1.0):
    logits = logits / temperature
    top_k_vals, top_k_inds = torch.topk(logits, k)
    probs = F.softmax(top_k_vals, dim=-1)
    idx = torch.multinomial(probs, 1)
    return top_k_inds[0, idx[0]]


def infer_config_from_state_dict(sd):
    """Best-effort dim/depth/heads inference from a saved state_dict.

    Reads:
      - vocab_size and dim from token_emb.weight
      - context_length from pos_emb.weight
      - depth from the highest layer index in keys
      - heads from BitAttention.heads (not in sd) — falls back to caller default
    """
    if "token_emb.weight" not in sd or "pos_emb.weight" not in sd:
        return None
    vocab_size, dim = sd["token_emb.weight"].shape
    context_length = sd["pos_emb.weight"].shape[0]

    depth = 0
    for k in sd.keys():
        if k.startswith("layers."):
            try:
                idx = int(k.split(".")[1])
                depth = max(depth, idx + 1)
            except (ValueError, IndexError):
                continue
    if depth == 0:
        return None

    return dict(
        vocab_size=int(vocab_size),
        dim=int(dim),
        depth=int(depth),
        context_length=int(context_length),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weights/final.pt", help="Path to weights")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=None, help="Override vocab size (auto-inferred if omitted)")
    parser.add_argument("--dim", type=int, default=None, help="Override model dim (auto-inferred if omitted)")
    parser.add_argument("--depth", type=int, default=None, help="Override model depth (auto-inferred if omitted)")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--context_length", type=int, default=None, help="Override context length (auto-inferred if omitted)")
    parser.add_argument("--device", type=str, default=None, help="cpu, mps, or cuda (auto-detected if omitted)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model from {args.checkpoint} on {device}...")

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    raw = torch.load(args.checkpoint, map_location=device)
    state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

    inferred = infer_config_from_state_dict(state_dict) or {}
    config = AtomicConfig(
        vocab_size=args.vocab_size or inferred.get("vocab_size", 4096),
        dim=args.dim or inferred.get("dim", 256),
        depth=args.depth or inferred.get("depth", 6),
        heads=args.heads,
        context_length=args.context_length or inferred.get("context_length", 128),
    )
    print(
        f"Config: vocab={config.vocab_size} dim={config.dim} "
        f"depth={config.depth} heads={config.heads} ctx={config.context_length}"
    )

    if config.dim % config.heads != 0:
        print(f"Error: dim ({config.dim}) must be divisible by heads ({config.heads}).")
        sys.exit(1)

    model = AtomicTransformer(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    enc = tiktoken.get_encoding("gpt2")

    print("\n--- Atomic-1Bit Chat (type 'quit' to exit) ---")

    while True:
        try:
            prompt = input("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break

        print("AI: ", end="", flush=True)

        ids = enc.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long).to(device)

        for _ in range(args.max_new_tokens):
            if x.size(1) >= config.context_length:
                break

            with torch.no_grad():
                logits = model(x)
                next_token_logits = logits[:, -1, :]
                next_token = top_k_sampling(next_token_logits, k=args.k, temperature=args.temp)

            word = enc.decode([next_token.item()])
            print(word, end="", flush=True)

            next_token = next_token.view(1, 1)
            x = torch.cat([x, next_token], dim=1)
        print()


if __name__ == "__main__":
    main()
