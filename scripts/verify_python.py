import os
import sys

import tiktoken
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from atomic_1bit.model.transformer import AtomicConfig, AtomicTransformer  # noqa: E402
from scripts.generate_samples import (  # noqa: E402
    CTX,
    DEPTH,
    DIM,
    HEADS,
    VOCAB,
    generate,
    load_vocab_map,
)


def run():
    enc = tiktoken.get_encoding("gpt2")
    token_map, reverse_map = load_vocab_map("weights/vocab_map_stories.json")
    config = AtomicConfig(
        vocab_size=VOCAB, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CTX
    )
    model = AtomicTransformer(config)
    model.load_state_dict(
        torch.load("weights/stories_final.pt", map_location="cpu")["model_state_dict"]
    )

    text = generate(
        model, "Once upon a time", enc, token_map, reverse_map, length=100, seed=42
    )
    print(f"PYTHON OUT: {text}")


if __name__ == "__main__":
    run()
