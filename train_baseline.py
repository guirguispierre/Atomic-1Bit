import torch
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
import numpy as np
import os
import json
from datasets import load_dataset
from baseline_fp16 import FP16Transformer, FP16Config

# --- CONFIG ---
BATCH_SIZE = 16
CONTEXT_LEN = 64
DIM = 128
DEPTH = 4
HEADS = 4
VOCAB_SIZE = 2048
LR = 1e-3
STEPS = 200

def train_baseline():
    print(f"--- FP16 BASELINE TRAINING ---")
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"Device: {device}")
    
    # Reuse vocab from Atomic run if possible for fair comparison
    if os.path.exists("weights/benchmark_vocab.json"):
        with open("weights/benchmark_vocab.json", "r") as f:
            data = json.load(f)
            token_map = {int(k): v for k, v in data["token_map"].items()}
            # Handle string keys if json loaded them as strings
            if not isinstance(list(token_map.keys())[0], int):
                 token_map = {int(k): v for k, v in data["token_map"].items()}
    else:
        print("Error: Run train_tiny.py first to generate vocab!")
        return

    dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
    enc = tiktoken.get_encoding("gpt2")
    
    def get_batch():
        indices = np.random.randint(0, len(dataset), BATCH_SIZE)
        batch_x, batch_y = [], []
        for i in indices:
            gpt_ids = enc.encode(dataset[i]["text"])
            my_ids = [token_map.get(g, 0) for g in gpt_ids] 
            if len(my_ids) > CONTEXT_LEN + 1:
                start = np.random.randint(0, len(my_ids) - CONTEXT_LEN - 1)
                chunk = my_ids[start : start + CONTEXT_LEN + 1]
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])
        if not batch_x: return get_batch()
        return torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)

    config = FP16Config(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = FP16Transformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    model.train()
    for step in range(STEPS):
        x, y = get_batch()
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}/{STEPS} Loss: {loss.item():.4f}")
            
    out_path = "weights/baseline_model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    train_baseline()
