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


def export_model(model: AtomicTransformer, filename: str, prompt: str = None):
    print(f"Exporting model to {filename} (Gist: {prompt if prompt else 'None'})...")
    
    # Gist Vector
    has_gist = 0
    gist_vector = None
    
    if prompt:
        has_gist = 1
        # Create minimal GistEncoder on the fly
        gist_conf = GistConfig(vocab_size=model.config.vocab_size, dim=model.config.dim)
        gist_enc = GistEncoder(gist_conf)
        # In real usage, we'd load trained weights. Here we use random init for demo.
        
        # Tokenize (Dummy: map chars to int or just random for demo)
        # Length of prompt
        dummy_ids = torch.tensor([[hash(w) % model.config.vocab_size for w in prompt.split()]], dtype=torch.long)
        
        with torch.no_grad():
            gist_vector = gist_enc(dummy_ids).squeeze(0) # (Dim,)
            
    with open(filename, "wb") as f:
        # 1. Header
        # vocab_size, dim, depth, heads, max_seq_len, has_gist (6 * 4 = 24 bytes)
        c = model.config
        header = struct.pack("iiiiii", c.vocab_size, c.dim, c.depth, c.heads, c.context_length, has_gist)
        f.write(header)
        
        # 1.5 Write Gist Vector if present
        if has_gist == 1:
            print(f"  Writing Gist Vector for prompt: '{prompt}'")
            f.write(gist_vector.detach().numpy().astype('float32').tobytes())
        
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
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint (e.g. weights/final.pt)")
    parser.add_argument("--output", type=str, default="atomic_model.bin", help="Output binary file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="System Gist Prompt")
    
    # Model Architecture (Defaults match training/train.py)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=50257)

    args = parser.parse_args()

    # 1. Initialize Model
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
            # If state dict has 'module.' prefix (from DataParallel), strip it? 
            # Usually not needed for single GPU training but good practice.
            # model.load_state_dict(state_dict)
            
            # Strict=False might be needed if there are extra buffers? 
            # Our BitLinear has no extra buffers, just weight/bias.
            try:
                model.load_state_dict(state_dict)
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with random weights (WARNING)...")
        else:
            print(f"Checkpoint {args.checkpoint} not found! Using random weights.")
    else:
        print("No checkpoint specified. Using random weights.")

    # 3. Export
    export_model(model, args.output, prompt=args.prompt)

