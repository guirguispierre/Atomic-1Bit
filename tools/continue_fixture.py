"""Non-interactive continuation training for checkpoints/tiny_stories.pt.

train.py is interactive and defaults to the stories_base config (4096 vocab,
dim 256); this script reuses its PocketStoriesDataset to extend the 2,000-step
fixture checkpoint (2048 vocab, dim 128) without prompts.

Usage:
    python tools/continue_fixture.py --steps 15000
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

from atomic_1bit.model.transformer import AtomicConfig, AtomicTransformer  # noqa: E402
from atomic_1bit.training.train import PocketStoriesDataset  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CKPT = os.path.join(ROOT, "checkpoints", "tiny_stories.pt")
VOCAB = os.path.join(ROOT, "checkpoints", "tiny_stories_vocab.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000,
                    help="additional steps to train")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=CKPT,
                    help="checkpoint output path (default overwrites fixture)")
    args = ap.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    ds = PocketStoriesDataset(context_length=64, vocab_file=VOCAB)
    config = AtomicConfig(
        vocab_size=2048, dim=128, depth=4, heads=4, context_length=64
    )
    model = AtomicTransformer(config).to(device)

    ckpt = torch.load(CKPT, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    start_step = int(ckpt.get("step", 0))
    print(f"Resuming fixture at step {start_step}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as e:
            print(f"  optimizer state not loaded ({e}); starting fresh")

    model.train()
    target = start_step + args.steps
    for step in range(start_step, target):
        x, y = ds.get_batch(args.batch_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}/{target} | Loss: {loss.item():.4f}", flush=True)
        if (step + 1) % 1000 == 0:
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                args.out,
            )
            print(f"  saved -> {args.out}", flush=True)

    torch.save(
        {
            "step": target,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args.out,
    )
    print(f"Done. Final checkpoint -> {args.out} (step {target})")


if __name__ == "__main__":
    main()
