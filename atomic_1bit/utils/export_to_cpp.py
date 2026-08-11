import torch
import struct
import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.model.gist import GistEncoder, GistConfig

def quantize_and_transpose_weight(w):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_q = (w * scale).round().clamp(-1, 1).to(torch.int8)
    return w_q.t().contiguous().cpu().numpy()


def pack_ternary(w_q):
    """Pack ternary weights four per byte, low-order pair first.

    Values are stored as w + 1, so {-1, 0, 1} map to {0, 1, 2} and the unused
    code 3 never appears. The flat row-major order of w_q is preserved; the
    final byte is zero-padded when the element count is not a multiple of 4.
    """
    flat = w_q.reshape(-1).astype(np.int8)
    codes = (flat + 1).astype(np.uint8)
    pad = (-codes.size) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    q = codes.reshape(-1, 4)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)


def unpack_ternary(packed, count):
    """Inverse of pack_ternary, for tests and tooling."""
    b = np.frombuffer(packed, dtype=np.uint8)
    codes = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], axis=1)
    return (codes.reshape(-1)[:count].astype(np.int8) - 1)


def export_model(model: AtomicTransformer, filename: str, prompt: str = None, gist_file: str = None):
    print(f"Exporting model to {filename} (Gist: {prompt if prompt else 'None'})...")

    # Gist Vector
    has_gist = 0
    gist_vector = None

    if gist_file:
         print(f"  Loading Gist Vector from file: {gist_file}")
         if not os.path.exists(gist_file):
             print("Error: Gist file not found.")
             has_gist = 0
         else:
             import numpy as np
             # Read raw float32 bytes
             try:
                 gist_vector_np = np.fromfile(gist_file, dtype=np.float32)
                 if len(gist_vector_np) != model.config.dim:
                     print(f"Error: Gist dimension mismatch. Expected {model.config.dim}, got {len(gist_vector_np)}")
                 else:
                     gist_vector = torch.tensor(gist_vector_np)
                     has_gist = 1
             except Exception as e:
                 print(f"Error reading gist file: {e}")

    elif prompt:
        has_gist = 1
        print(f"  Computing Gist for prompt: '{prompt}'")
        model.eval()
        with torch.no_grad():
            # Assumes model.gist_encoder exists and is trained
            gist_vector = model.gist_encoder.encode(prompt).squeeze(0) # (Dim,)

    with open(filename, "wb") as f:
        # 1. Header with Magic Bytes and Version
        # Magic: 'ATOM' = 0x41544F4D
        # guirguispierre 2026-08-10 - v2 packs ternary weights four per byte
        magic = 0x41544F4D
        version = 2

        # vocab_size, dim, depth, heads, max_seq_len, has_gist (6 * 4 = 24 bytes)
        c = model.config

        # Struct: Magic(I), Ver(I), Vocab(I), Dim(I), Depth(I), Heads(I), CTX(I), Gist(I)
        header = struct.pack("iiiiiiii", magic, version, c.vocab_size, c.dim, c.depth, c.heads, c.context_length, has_gist)
        f.write(header)

        # 1.5 Write Gist Vector if present
        if has_gist == 1:
            if gist_vector is not None:
                f.write(gist_vector.detach().cpu().numpy().astype('float32').tobytes())

        # Helper to write tensor
        def write_tensor(name, tensor, dtype='f', needs_quant=False):
            # dtype: 'f' for float32, 'b' for int8
            # needs_quant: If True, quantize float->int8{-1,0,1} and transpose

            if needs_quant:
                 # Compute scale (mean_abs)
                 # w_f32 ~= w_q * scale
                 # scale = mean(abs(w))
                 scale_val = tensor.abs().mean().clamp(min=1e-5).item()
                 packed = struct.pack("f", scale_val)
                 import binascii
                 hex_str = binascii.hexlify(packed).decode('utf-8')
                 f.write(packed)
                 print(f"  Wrote scale: {scale_val:.6f} [Hex: {hex_str}]")

                 data = quantize_and_transpose_weight(tensor)
                 blob = pack_ternary(data)
                 f.write(blob.tobytes())
                 print(f"  Wrote {name}: {data.shape} -> {blob.size} bytes (2-bit packed)")
                 return

            data = tensor.detach().cpu().numpy()
            if dtype == 'f':
                data = data.astype('float32') # Ensure float32
                f.write(data.tobytes())
            elif dtype == 'b':
                data = data.astype('int8')
                f.write(data.tobytes())
            print(f"  Wrote {name}: {data.shape} ({dtype})")

        # 2. Embeddings
        write_tensor("token_emb", model.token_emb.weight, 'f')
        write_tensor("pos_emb", model.pos_emb.weight, 'f')

        # 3. Layers
        for i, layer in enumerate(model.layers):
            print(f"  [Layer {i}]")
            # LN1
            write_tensor(f"layer{i}.ln1", layer.ln1.weight, 'f')

            # Attn
            # We need to grab the weights from BitLinear
            # Q, K, V, O
            write_tensor(f"layer{i}.attn.q", layer.attn.q_proj.weight, 'b', True)
            write_tensor(f"layer{i}.attn.k", layer.attn.k_proj.weight, 'b', True)
            write_tensor(f"layer{i}.attn.v", layer.attn.v_proj.weight, 'b', True)
            write_tensor(f"layer{i}.attn.o", layer.attn.o_proj.weight, 'b', True)

            # LN2
            write_tensor(f"layer{i}.ln2", layer.ln2.weight, 'f')

            # MLP
            write_tensor(f"layer{i}.mlp.fc1", layer.mlp.fc1.weight, 'b', True)
            write_tensor(f"layer{i}.mlp.fc2", layer.mlp.fc2.weight, 'b', True)

        # 4. Final Norm
        write_tensor("ln_f", model.ln_f.weight, 'f')

        # 5. Head
        write_tensor("head", model.head.weight, 'b', True)

    print(f"Export complete. File size: {os.path.getsize(filename) / 1024:.2f} KB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Atomic-1Bit Model to C++ Binary")
    parser.add_argument("--model", type=str, default="weights/final.pt", help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="atomic_model.bin", help="Output binary file")
    parser.add_argument("--prompt", type=str, default=None, help="System Prompt to bake into Gist (Optional)")
    parser.add_argument("--gist_file", type=str, default=None, help="Path to pre-computed .gist file (Overrides prompt)")
    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--depth", type=int, default=4, help="Model depth")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--context_len", type=int, default=64, help="Context length")
    parser.add_argument("--vocab_size", type=int, default=2048, help="Vocab size")

    args = parser.parse_args()

    # Init config
    config = AtomicConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        context_length=args.context_len
    )

    print(f"Initialized Model: Dim={args.dim}, Depth={args.depth}, Heads={args.heads}")
    model = AtomicTransformer(config)

    # Load Checkpoint
    if args.model:
        if os.path.exists(args.model):
            print(f"Loading checkpoint from {args.model}...")
            state_dict = torch.load(args.model, map_location="cpu")
            try:
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

                model.load_state_dict(state_dict, strict=False)
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with random weights (WARNING)...")
        else:
            print(f"Model file {args.model} not found. using random weights.")

    # Export
    export_model(model, args.output, prompt=args.prompt, gist_file=args.gist_file)
