import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
import numpy as np
import json
from collections import Counter
from datasets import load_dataset
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# --- CONFIG ---
BATCH_SIZE = 16
CONTEXT_LEN = 64
DIM = 128
DEPTH = 4
HEADS = 4
VOCAB_SIZE = 2048
LR = 1e-3
STEPS = 200 # Very short training for benchmark purposes

def train_tiny():
    print(f"--- Atomic-1Bit BENCHMARK TRAINING ---")
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"Device: {device}")
    
    # 1. Dataset (Rapid/Mock)
    print("Loading TinyStories subset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]") # Small subset
    enc = tiktoken.get_encoding("gpt2")
    
    # Simple frequent vocab
    counter = Counter()
    print("Building micro-vocab...")
    for text in dataset["text"][:500]:
        counter.update(enc.encode(text))
    
    most_common = counter.most_common(VOCAB_SIZE - 2)
    token_map = {k: i for i, (k, _) in enumerate(most_common)}
    token_map[enc.eot_token] = VOCAB_SIZE - 1
    reverse_map = {v: k for k, v in token_map.items()}
    
    def get_batch():
        indices = np.random.randint(0, len(dataset), BATCH_SIZE)
        batch_x, batch_y = [], []
        for i in indices:
            gpt_ids = enc.encode(dataset[i]["text"])
            my_ids = [token_map.get(g, 0) for g in gpt_ids] # 0 = UNK (approx)
            if len(my_ids) > CONTEXT_LEN + 1:
                start = np.random.randint(0, len(my_ids) - CONTEXT_LEN - 1)
                chunk = my_ids[start : start + CONTEXT_LEN + 1]
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])
        
        # Pad if needed (lazy padding for now, purely for benchmark model creation)
        # Actually simplest to just skip short ones or repeat
        if not batch_x: return get_batch() 
        return torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)

    # 2. Model
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. Train Loop
    model.train()
    for step in range(STEPS):
        try:
            x, y = get_batch()
            if x.shape[0] == 0: continue
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}/{STEPS} Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Skipping step {step}: {e}")
            
    # 4. Save
    os.makedirs("weights", exist_ok=True)
    out_path = "weights/benchmark_model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved to {out_path}")
    
    # Save Vocab ID map for inference consistency (used by Python runner)
    with open("weights/benchmark_vocab.json", "w") as f:
        json.dump({"token_map": token_map, "reverse_map": reverse_map}, f)

if __name__ == "__main__":
    train_tiny()
