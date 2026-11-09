"""Train the checked-in fixture checkpoint (checkpoints/tiny_stories.pt).

Produces the small trained model that tools/parity_check.py,
tests/test_trained_parity.py, and the eval harness consume out of the box.
It exists to prove the pipeline end to end -- it is NOT a quality model.

Config is the benchmark config (dim 128, depth 4, heads 4, vocab 2048,
ctx 64; 1,320,064 params) so the artifact doubles as a benchmark_model.pt.
The state dict is stored in float16 to keep the file under ~3 MB; loading
upcasts to float32, and every consumer (Python forward, C++ export) sees
the same stored weights, so parity is unaffected.

Data: TinyStories via HF `datasets` when reachable; otherwise a small
deterministic template corpus (the script prints which it used and writes
it into the checkpoint metadata).

Usage:
    python3 tools/train_fixture_checkpoint.py [--steps 2000] [--batch-size 32]
        [--out checkpoints/tiny_stories.pt] [--seed 0]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

from atomic_1bit.model.transformer import (  # noqa: E402
    AtomicConfig,
    AtomicTransformer,
)

# --- Fixture config: matches benchmarks/run_suite.py and train_tiny.py ---
VOCAB_SIZE = 2048
DIM = 128
DEPTH = 4
HEADS = 4
CONTEXT_LEN = 64

UNK_ID = 0  # unknown gpt2 ids collapse onto id 0, same as benchmarks/train_tiny.py


def load_training_texts(n_stories, seed):
    """Return (train_texts, eval_texts, source_name).

    TinyStories when the Hub is reachable, else a deterministic local
    corpus so the fixture can still be produced offline.
    """
    try:
        from datasets import load_dataset

        train = load_dataset("roneneldan/TinyStories", split=f"train[:{n_stories}]")
        val = load_dataset("roneneldan/TinyStories", split="validation[:1000]")
        return list(train["text"]), list(val["text"]), "tinystories-hf"
    except Exception as e:  # offline / rate-limited / dataset schema change
        print(
            f"[data] TinyStories unavailable ({e}); "
            f"using the deterministic local corpus instead"
        )
        rng = np.random.RandomState(seed)
        names = ["Lily", "Tom", "Ben", "Anna", "Max", "Sara", "Jack", "Ella"]
        things = ["ball", "cat", "dog", "kite", "book", "cake", "boat", "bird"]
        places = ["park", "garden", "beach", "school", "forest", "house"]
        acts = ["played", "ran", "jumped", "sang", "slept", "read"]
        tmpl = [
            "Once upon a time, {n} found a {t} in the {p}. {n} was happy.",
            "One day, {n} went to the {p} with a {t}. They {a} all day.",
            "{n} had a little {t}. The {t} liked to {a} in the {p}.",
            "There was a {t} named Spot. {n} and the {t} went to the {p}.",
            '{n} said, "Look at my {t}!" Then they {a} in the {p}.',
        ]
        texts = []
        for i in range(n_stories):
            t = tmpl[rng.randint(len(tmpl))]
            texts.append(
                t.format(
                    n=names[rng.randint(len(names))],
                    t=things[rng.randint(len(things))],
                    p=places[rng.randint(len(places))],
                    a=acts[rng.randint(len(acts))],
                )
            )
        # deterministic split: last 10% held out
        n_eval = max(1, len(texts) // 10)
        return texts[:-n_eval], texts[-n_eval:], "local-template-fallback"


def build_vocab(texts, enc, n_scan):
    """Frequency remap of gpt2 ids into [0, VOCAB_SIZE), eot pinned to
    VOCAB_SIZE-1 -- identical scheme to benchmarks/train_tiny.py."""
    counter = Counter()
    for text in texts[:n_scan]:
        counter.update(enc.encode(text))

    most_common = counter.most_common(VOCAB_SIZE - 2)
    token_map = {gpt_id: i for i, (gpt_id, _) in enumerate(most_common)}
    token_map[enc.eot_token] = VOCAB_SIZE - 1
    reverse_map = {v: k for k, v in token_map.items()}
    return token_map, reverse_map


def encode_stories(texts, enc, token_map, ctx):
    """Remap each story into model ids, pad/truncate to ctx+1."""
    eot_model = token_map[enc.eot_token]
    rows = []
    for text in texts:
        ids = [token_map.get(g, UNK_ID) for g in enc.encode(text)]
        ids.append(eot_model)
        if len(ids) < ctx + 1:
            ids = ids + [eot_model] * (ctx + 1 - len(ids))
        else:
            ids = ids[: ctx + 1]
        rows.append(ids)
    return np.asarray(rows, dtype=np.int64)


def mean_nll(model, data, vocab_size, n_batches, batch_size, rng):
    """Mean next-token cross-entropy over fixed batches from `data`."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            idx = rng.randint(0, len(data), batch_size)
            batch = torch.tensor(data[idx], dtype=torch.long)
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            total_nll += F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum"
            ).item()
            total_tok += y.numel()
    return total_nll / total_tok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stories", type=int, default=20000, help="TinyStories train slice size"
    )
    p.add_argument(
        "--vocab-scan",
        type=int,
        default=3000,
        help="stories scanned to build the frequency vocab",
    )
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--out", default="checkpoints/tiny_stories.pt")
    p.add_argument(
        "--vocab-out", default=None, help="default: <out minus .pt>_vocab.json"
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    t0 = time.time()
    train_texts, eval_texts, source = load_training_texts(args.stories, args.seed)
    print(f"[data] source={source} train={len(train_texts)} eval={len(eval_texts)}")

    token_map, reverse_map = build_vocab(train_texts, enc, args.vocab_scan)
    print(f"[data] vocab: {len(token_map)} mapped gpt2 ids -> {VOCAB_SIZE} slots")

    train_ids = encode_stories(train_texts, enc, token_map, CONTEXT_LEN)
    eval_ids = encode_stories(eval_texts, enc, token_map, CONTEXT_LEN)

    config = AtomicConfig(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        context_length=CONTEXT_LEN,
    )
    model = AtomicTransformer(config)
    n_params = sum(q.numel() for q in model.parameters())
    print(
        f"[model] {n_params:,} params "
        f"(vocab={VOCAB_SIZE} dim={DIM} depth={DEPTH} heads={HEADS} ctx={CONTEXT_LEN})"
    )

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    nll0 = mean_nll(
        model,
        eval_ids,
        VOCAB_SIZE,
        args.eval_batches // 4,
        args.batch_size,
        np.random.RandomState(args.seed + 1),
    )
    print(
        f"[eval] untrained held-out NLL {nll0:.4f} "
        f"(ppl {np.exp(nll0):.1f}; uniform = {VOCAB_SIZE})"
    )

    model.train()
    t_train = time.time()
    losses = []
    for step in range(args.steps):
        idx = rng.randint(0, len(train_ids), args.batch_size)
        batch = torch.tensor(train_ids[idx], dtype=torch.long)
        x, y = batch[:, :-1], batch[:, 1:]

        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == args.steps - 1:
            avg = np.mean(losses[-100:])
            print(
                f"[train] step {step}/{args.steps} "
                f"loss {loss.item():.4f} avg100 {avg:.4f}"
            )
    train_s = time.time() - t_train
    final_loss = float(np.mean(losses[-100:]))

    # Perplexity of the exact artifact being checked in: downcast to fp16,
    # reload, then measure so the reported number describes the stored file.
    fp16_sd = {k: v.detach().half().clone() for k, v in model.state_dict().items()}
    model_fp16 = AtomicTransformer(config)
    model_fp16.load_state_dict(fp16_sd)
    nll = mean_nll(
        model_fp16,
        eval_ids,
        VOCAB_SIZE,
        args.eval_batches,
        args.batch_size,
        np.random.RandomState(args.seed + 1),
    )
    ppl = float(np.exp(nll))
    print(
        f"[eval] held-out NLL {nll:.4f} -> perplexity {ppl:.1f} "
        f"(uniform = {VOCAB_SIZE})"
    )

    ckpt = {
        "step": args.steps,
        "config": {
            "vocab_size": VOCAB_SIZE,
            "dim": DIM,
            "depth": DEPTH,
            "heads": HEADS,
            "context_length": CONTEXT_LEN,
        },
        "model_state_dict": fp16_sd,
        "training": {
            "script": "tools/train_fixture_checkpoint.py",
            "dataset": source,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_seconds": round(train_s, 1),
            "final_loss_avg100": round(final_loss, 4),
            "untrained_eval_nll": round(nll0, 4),
            "eval_nll": round(nll, 4),
            "eval_ppl": round(ppl, 1),
            "eval_tokens": args.eval_batches * args.batch_size * CONTEXT_LEN,
            "storage_dtype": "float16 (load_state_dict upcasts to float32)",
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[save] {args.out} ({size_mb:.2f} MB)")

    vocab_path = args.vocab_out or args.out.rsplit(".pt", 1)[0] + "_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({"token_map": token_map, "reverse_map": reverse_map}, f)
    print(f"[save] {vocab_path}")

    print(
        f"[done] total {time.time() - t0:.0f}s "
        f"(train {train_s:.0f}s) | final loss {final_loss:.4f} | "
        f"held-out ppl {ppl:.1f} vs uniform {VOCAB_SIZE}"
    )


if __name__ == "__main__":
    main()
