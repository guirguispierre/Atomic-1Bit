import torch
import struct
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.model.gist import GistEncoder, GistConfig
import torch

def quantize_and_transpose_weight(w_f32):
    """
    Takes float weights (Out, In), quantizes to {-1, 0, 1}, and transposes to (In, Out).
    Returns int8 numpy array.
    """
    scale = 1.0 / w_f32.abs().mean().clamp(min=1e-5)
    w_q = (w_f32 * scale).round().clamp(-1, 1).to(torch.int8)
    # Transpose to (In, Out) for C++ kernel
    return w_q.t().cpu().numpy()


def export_model(model: AtomicTransformer, filename: str, prompt: str = None, gist_file: str = None):
    print(f"Exporting model to {filename} (Gist Source: {'File: '+gist_file if gist_file else (prompt if prompt else 'None')})...")
    
    # Gist Vector
    has_gist = 0
    gist_vector = None
    
    if gist_file:
         print(f"  Loading Gist Vector from file: {gist_file}")
         if not os.path.exists(gist_file):
             print("Error: Gist file not found.")
             # We shouldn't exit here inside a function maybe? Or just warn.
             # Let's assume critical failure if gist missing but requested.
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
        # Create minimal GistEncoder on the fly
        # NOTE: This uses RANDOM weights for the GistEncoder unless we loaded them.
        # But `AtomicTransformer` usually contains `gist_encoder`.
        # If we use `model.gist_encoder`, it has the correct weights!
        
        print(f"  Computing Gist for prompt: '{prompt}'")
        model.eval()
        with torch.no_grad():
            # Assumes model.gist_encoder exists and is trained
            gist_vector = model.gist_encoder.encode(prompt).squeeze(0) # (Dim,)
            
    with open(filename, "wb") as f:
        # 1. Header
        # vocab_size, dim, depth, heads, max_seq_len, has_gist (6 * 4 = 24 bytes)
        c = model.config
        header = struct.pack("iiiiii", c.vocab_size, c.dim, c.depth, c.heads, c.context_length, has_gist)
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
                 data = quantize_and_transpose_weight(tensor)
                 f.write(data.tobytes())
                 print(f"  Wrote {name}: {data.shape} (b - quantized)")
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
    import os

    parser = argparse.ArgumentParser(description="Export Atomic-1Bit Model to C++ Binary")
    parser.add_argument("--model", type=str, default="weights/final.pt", help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="atomic_model.bin", help="Output binary file")
    # Gist Options
    parser.add_argument("--prompt", type=str, default=None, help="System Gist Prompt (will compute gist vector from model's gist_encoder)")
    parser.add_argument("--gist_file", type=str, default=None, help="Path to .gist file (Raw Float32 Vector) - overrides --prompt")
    
    # Model Config (Must match training!)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8) # Updated default for instruct
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=50257)
    args = parser.parse_args()
    config = AtomicConfig(
        vocab_size=args.vocab_size, 
        dim=args.dim, 
        depth=args.depth, 
        heads=args.heads, 
        context_length=args.context_len
    )
    model = AtomicTransformer(config)
    print(f"Initialized Model: Dim={args.dim}, Depth={args.depth}, Heads={args.heads}")

    # 2. Load Checkpoint
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            # map_location='cpu' to be safe
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            # Strict=False if gist encoder missing in old ckpt
            try:
                if "model_state_dict" in state_dict:
                    print("  (Detected Checkpoint Wrapper - unwrapping)")
                    state_dict = state_dict["model_state_dict"]
                
                model.load_state_dict(state_dict, strict=False) 
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with random weights (WARNING)...")
        else:
            print(f"Checkpoint {args.checkpoint} not found! Using random weights.")
    else:
        print("No checkpoint specified. Using random weights.")

    # 3. Export
    export_model(model, args.output, prompt=args.prompt, gist_file=args.gist_file)
