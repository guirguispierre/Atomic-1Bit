"""Write a .vocab detokenizer file for the C++ runner.

A .vocab file has one line per model token id; each line is the lowercase
hex of that token's raw bytes (empty line = empty token). The runner's
`--vocab` flag loads it and prints decoded text instead of token ids.

Input is a vocab-map json like checkpoints/tiny_stories_vocab.json whose
"reverse_map" maps model token id -> source tokenizer id. The source
tokenizer must be a tiktoken encoding (default gpt2, matching
training/data.py's remapped TinyStories vocabulary).

Usage:
    python3 tools/make_vocab.py checkpoints/tiny_stories_vocab.json \
        --out checkpoints/tiny_stories.vocab
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vocab_map", help="vocab-map json with token_map/reverse_map")
    ap.add_argument("--out", required=True, help="output .vocab path")
    ap.add_argument(
        "--encoding",
        default="gpt2",
        help="tiktoken encoding the ids come from (default gpt2)",
    )
    args = ap.parse_args()

    import tiktoken  # local dep, kept out of module scope

    with open(args.vocab_map) as f:
        data = json.load(f)
    reverse_map = {int(k): int(v) for k, v in data["reverse_map"].items()}
    n = max(reverse_map) + 1
    enc = tiktoken.get_encoding(args.encoding)

    with open(args.out, "w") as f:
        for i in range(n):
            src_id = reverse_map.get(i)
            if src_id is None:
                f.write("\n")
                continue
            try:
                raw = enc.decode_single_token_bytes(src_id)
            except KeyError:
                raw = b""
            f.write(raw.hex() + "\n")

    print(f"wrote {n} token byte-strings to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
