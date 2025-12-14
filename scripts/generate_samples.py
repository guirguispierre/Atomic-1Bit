import sys
import os
import torch
import torch.nn.functional as F
import argparse
import tiktoken
import json

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# Pocket Stories Config (Must match train.py)
DIM = 256
DEPTH = 6
HEADS = 4
VOCAB = 4096
CTX = 128

def load_vocab_map(vocab_file):
    with open(vocab_file, 'r') as f:
        data = json.load(f)
        token_map = {int(k): v for k, v in data["token_map"].items()}
        reverse_map = {int(k): v for k, v in data["reverse_map"].items()}
    return token_map, reverse_map

def generate(model, prompt, enc, token_map, reverse_map, length=200, temp=0.7, seed=42):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    
    # Encode
    gpt_ids = enc.encode(prompt)
    pocket_ids = [token_map.get(gid, VOCAB-1) for gid in gpt_ids]
    x = torch.tensor([pocket_ids], dtype=torch.long).to(device)
    
    out_ids = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            if x.size(1) >= CTX:
                x = x[:, -CTX:] # Rolling window
                
            logits = model(x)
            last_logits = logits[:, -1, :] / temp
            probs = F.softmax(last_logits, dim=-1)
            
            next_token = torch.multinomial(probs, 1)
            pid = next_token.item()
            out_ids.append(pid)
            
            x = torch.cat([x, next_token], dim=1)
            
    # Decode
    gpt_out_ids = [reverse_map.get(pid, enc.eot_token) for pid in out_ids]
    text = enc.decode(gpt_out_ids)
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="weights/stories_final.pt")
    parser.add_argument("--vocab_map", default="weights/vocab_map_stories.json")
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print("Model not found.")
        return

    # Load Data
    enc = tiktoken.get_encoding("gpt2")
    token_map, reverse_map = load_vocab_map(args.vocab_map)
    
    # Load Model
    config = AtomicConfig(vocab_size=VOCAB, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CTX)
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    
    model = AtomicTransformer(config).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    print(f"Loaded model from {args.model_path} on {device}")
    
    prompts = ["Once upon a time", "In a small town", "The robot said"]
    seeds = [42, 123, 999]
    
    print("-" * 40)
    for p in prompts:
        print(f"PROMPT: {p}")
        for s in seeds:
            text = generate(model, p, enc, token_map, reverse_map, length=args.length, seed=s)
            print(f"  [Seed {s}]: {p}{text}...")
            print("   ---")
        print("=" * 40)

if __name__ == "__main__":
    main()
