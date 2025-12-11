import torch
import struct
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

from atomic_1bit.model.gist import GistEncoder, GistConfig

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
        def write_tensor(name, tensor, dtype='f'):
            # dtype: 'f' for float32, 'b' for int8
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
            write_tensor(f"layer{i}.attn.q", layer.attn.q_proj.weight, 'b')
            write_tensor(f"layer{i}.attn.k", layer.attn.k_proj.weight, 'b')
            write_tensor(f"layer{i}.attn.v", layer.attn.v_proj.weight, 'b')
            write_tensor(f"layer{i}.attn.o", layer.attn.o_proj.weight, 'b')
            
            # LN2
            write_tensor(f"layer{i}.ln2", layer.ln2.weight, 'f')
            
            # MLP
            write_tensor(f"layer{i}.mlp.fc1", layer.mlp.fc1.weight, 'b')
            write_tensor(f"layer{i}.mlp.fc2", layer.mlp.fc2.weight, 'b')
            
        # 4. Final Norm
        write_tensor("ln_f", model.ln_f.weight, 'f')
        
        # 5. Head
        write_tensor("head", model.head.weight, 'b')
        
    print(f"Export complete. File size: {os.path.getsize(filename) / 1024:.2f} KB")

if __name__ == "__main__":
    # Create a Dummy Model and export it
    config = AtomicConfig(vocab_size=1000, dim=64, depth=2, heads=4, context_length=32)
    model = AtomicTransformer(config)
    
    # Prompt user or default
    gist_prompt = "You are a helpful assistant" 
    
    export_model(model, "atomic_model.bin", prompt=gist_prompt)
