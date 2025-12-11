import sys
import os
import torch
import torch.nn.functional as F
import tiktoken
import argparse

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

def top_k_sampling(logits, k=50, temperature=1.0):
    # Apply temp
    logits = logits / temperature
    # Top K
    top_k_vals, top_k_inds = torch.topk(logits, k)
    # Softmax
    probs = F.softmax(top_k_vals, dim=-1)
    # Sample
    idx = torch.multinomial(probs, 1)
    return top_k_inds[0, idx[0]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weights/final.pt", help="Path to weights")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=40)
    args = parser.parse_args()

    # Config (Must match training)
    # If you changed training config, update this!
    config = AtomicConfig(vocab_size=50257, dim=256, depth=4, heads=4, context_length=128)
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = AtomicTransformer(config).to(device)
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded!")
    else:
        print("Checkpoint not found! Using random weights (Gibberish warning).")

    model.eval()
    enc = tiktoken.get_encoding("gpt2")

    print("\n--- Atomic-1Bit Chat (Type 'quit' to exit) ---")
    
    while True:
        try:
            prompt = input("\nUser: ")
            if prompt.lower() in ["quit", "exit"]: break
            
            print("AI: ", end="", flush=True)
            
            # Context
            ids = enc.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long).to(device)
            
            # Generation Loop
            for _ in range(100): # Max new tokens
                if x.size(1) >= config.context_length: break
                
                with torch.no_grad():
                    logits = model(x)
                    next_token_logits = logits[:, -1, :]
                    
                    next_token = top_k_sampling(next_token_logits, k=args.k, temperature=args.temp)
                    
                    # Print token
                    word = enc.decode([next_token.item()])
                    print(word, end="", flush=True)
                    
                    # Append
                    # next_token is 0-d or 1-d from top_k_sampling.
                    # We want (1, 1) to match x (1, Seq)
                    next_token = next_token.view(1, 1)
                    x = torch.cat([x, next_token], dim=1)
            print() # Newline
            
        except KeyboardInterrupt:
            print("\n")
            break

if __name__ == "__main__":
    main()
