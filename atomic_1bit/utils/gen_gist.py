import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import atomic_1bit # Ensure package is importable if needed
# Wait, the remote code imported:
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# Defaults for the Instruct Model
DIM = 256
DEPTH = 8
HEADS = 4
VOCAB = 50257
CTX = 256

def main():
    parser = argparse.ArgumentParser(description="Atomic-1Bit Personality Swapper (Gist Generator)")
    parser.add_argument("--prompt", type=str, required=True, help="System prompt to encode (e.g. 'You are a coder')")
    parser.add_argument("--output", type=str, default="personality.gist", help="Output .gist file")
    parser.add_argument("--model_path", type=str, default="weights/instruct_final.pt", help="Path to trained model weights")
    
    args = parser.parse_args()
    
    print(f"Loading Model Config from {args.model_path}...")
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    
    # Init Model
    config = AtomicConfig(vocab_size=VOCAB, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CTX)
    model = AtomicTransformer(config).to(device)
    
    # Load weights
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("Error: Model weights not found. Cannot encode Gist without trained GistEncoder.")
        return

    model.eval()
    
    # Encode
    print(f"Encoding Prompt: '{args.prompt}'")
    with torch.no_grad():
        # Encode Gist
        # GistEncoder expects full text, returns (1, Dim)
        # Using model.gist_encoder.encode which should be available
        gist_vec = model.gist_encoder.encode(args.prompt) # (1, Dim)
        
    print(f"Gist Vector Shape: {gist_vec.shape}")
    
    # Save to file
    gist_np = gist_vec.cpu().numpy().astype("float32")
    gist_np.tofile(args.output)
    
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
