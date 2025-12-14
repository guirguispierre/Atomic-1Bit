import sys
import os
import argparse
import json
import torch
import tiktoken
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# Defaults for the Instruct Model (Flagship Efficient)
DIM = 320
DEPTH = 8
HEADS = 5
VOCAB = 4096
CTX = 256

def load_vocab_map(vocab_file):
    if not os.path.exists(vocab_file):
        print(f"Error: Vocab map {vocab_file} not found.")
        print("Please run train_instruct.py first to generate it.")
        sys.exit(1)
        
    print(f"Loading vocab map from {vocab_file}...")
    with open(vocab_file, 'r') as f:
        data = json.load(f)
        # JSON keys are strings, convert to int for mapping
        token_map = {int(k): v for k, v in data["token_map"].items()}
    return token_map

def main():
    parser = argparse.ArgumentParser(description="Atomic-1Bit Personality Swapper (Gist Generator)")
    parser.add_argument("--prompt", type=str, required=True, help="System prompt to encode (e.g. 'You are a coder')")
    parser.add_argument("--output", type=str, default="personality.gist", help="Output .gist file")
    parser.add_argument("--model_path", type=str, default="weights/instruct_final.pt", help="Path to trained model weights")
    parser.add_argument("--vocab_map", type=str, default="weights/vocab_map_instruct.json", help="Path to vocab map")
    
    args = parser.parse_args()
    
    # 1. Load Vocab Map
    token_map = load_vocab_map(args.vocab_map)
    UNK_ID = VOCAB - 1
    
    # 2. Tokenize Prompt
    enc = tiktoken.get_encoding("gpt2")
    gpt_ids = enc.encode(args.prompt)
    
    # Remap tokens
    ids = [token_map.get(gid, UNK_ID) for gid in gpt_ids]
    print(f"Encoded Prompt: '{args.prompt}' -> {len(ids)} tokens")
    
    if len(ids) == 0:
        print("Error: Prompt is empty or all tokens mapped to UNK.")
        return

    # 3. Load Model
    print(f"Loading Model Config from {args.model_path}...")
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    
    config = AtomicConfig(vocab_size=VOCAB, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CTX)
    model = AtomicTransformer(config).to(device)
    
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False) # Strict=False in case of minor diffs
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        print("Error: Model weights not found.")
        return

    model.eval()
    
    # 4. Generate Gist (Mean Pooling of Model Embeddings)
    # We use the model's OWN embedding layer so the Gist is in the correct latent space.
    input_tensor = torch.tensor([ids], dtype=torch.long).to(device) # (1, Seq)
    
    with torch.no_grad():
        # Look up embeddings: (1, Seq, Dim)
        embeds = model.token_emb(input_tensor) 
        
        # Mean Pool: (1, Dim)
        gist_vec = torch.mean(embeds, dim=1)
        
    print(f"Gist Vector Shape: {gist_vec.shape}")
    
    # 5. Save
    gist_np = gist_vec.cpu().numpy().astype("float32")
    gist_np.tofile(args.output)
    
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
