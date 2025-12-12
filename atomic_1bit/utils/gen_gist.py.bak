import torch
import argparse
import sys
import os

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.gist import GistEncoder, GistConfig
from atomic_1bit.model.transformer import AtomicConfig # For default dims

def main():
    parser = argparse.ArgumentParser(description="Generate Gist Vector (.gist) from text prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Personality prompting text")
    parser.add_argument("--output", type=str, required=True, help="Output filename (e.g. coder.gist)")
    parser.add_argument("--dim", type=int, default=256, help="Model Dimension")
    parser.add_argument("--vocab_size", type=int, default=50257)
    
    args = parser.parse_args()
    
    print(f"Generating Gist for: '{args.prompt}'")
    
    # Init Encoder (Random weights? No, GistEncoder is distinct. 
    # Wait, GistEncoder mimics EmbeddingBag. 
    # If we want the *same* gist representation as training, we need the *trained* GistEncoder weights.
    # But `AtomicTransformer` *contains* the embeddings used by GistEncoder (it shares token_emb usually? 
    # In my implementation `GistEncoder` was standalone with its own EmbeddingBag?
    # Let's check `atomic_1bit/model/gist.py`.
    
    # If GistEncoder has its own weights, we must load the trained model checkpoint to get them!
    # Otherwise we generate random vectors which is useless.
    
    # Re-checking implementation plan... 
    # It says "Runs it through the GistEncoder".
    # I need to load the trained model to get the correct embeddings.
    
    parser.add_argument("--checkpoint", type=str, default="weights/instruct_final.pt", help="Path to trained model checkpoint")
    args = parser.parse_args() # Re-parse to get checkpoint
    
    # Load Main Model to extract Embeddings for Gist
    # (Assuming GistEncoder uses the same embeddings or is saved in the checkpoint)
    # Actually `GistEncoder` in `model/gist.py` has `self.emb = nn.EmbeddingBag(vocab_size, dim, mode='mean')`.
    # It is a *separate* set of weights unless we tied them.
    # In `AtomicTransformer`, `self.gist_encoder = GistEncoder(gist_config)`.
    # So the weights ARE in the checkpoint under `gist_encoder.emb.weight`.
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found. Cannot generate valid Gist.")
        print("Please train the model first.")
        # For testing, we might allow random, but for real usage, we need trained weights.
        sys.exit(1)

    print(f"Loading weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    
    # Extract Gist Weights
    # Key: "gist_encoder.emb.weight"
    if "gist_encoder.emb.weight" not in state_dict:
        print("Error: Checkpoint does not contain 'gist_encoder.emb.weight'. Is Gist enabled?")
        sys.exit(1)
        
    gist_weights = state_dict["gist_encoder.emb.weight"]
    
    # Init Encoder
    config = GistConfig(vocab_size=args.vocab_size, dim=args.dim)
    encoder = GistEncoder(config)
    encoder.emb.weight.data = gist_weights # Load weights
    
    # Encode
    gist_vec = encoder.encode(args.prompt) # (1, Dim)
    
    # Save
    print(f"Saving to {args.output}...")
    with open(args.output, "wb") as f:
        f.write(gist_vec.detach().numpy().astype("float32").tobytes())
        
    print("Done.")

if __name__ == "__main__":
    main()
