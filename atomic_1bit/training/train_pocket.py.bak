import sys
import os
import glob
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
import numpy as np
from datasets import load_dataset
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# --- POCKET CONFIG (For 8MB RAM Cardputer) ---
BATCH_SIZE = 32 
CONTEXT_LEN = 128 
DIM = 256
DEPTH = 6  
HEADS = 4
VOCAB_SIZE = 4096 # Reduced from 50257
LR = 1e-3

# Dataset customized for Pocket
class PocketAlpacaDataset:
    def __init__(self, split="train", context_length=128):
        print(f"Loading Alpaca (Pocket Mode - Modulo {VOCAB_SIZE})...")
        self.dataset = load_dataset("yahma/alpaca-cleaned", split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        print(f"Loaded {len(self.dataset)} samples.")
        
    def format_prompt(self, sample):
        text = f"### Instruction: {sample['instruction']}\n"
        if sample.get("input", ""):
            text += f"### Input: {sample['input']}\n"
        text += f"### Response: {sample['output']}"
        return text

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.dataset), batch_size)
        rows = self.dataset.select(indices)
        
        batch_input_ids = []
        batch_targets = []
        
        for i in range(len(rows)):
            row = rows[i]
            text = self.format_prompt(row)
            
            # Encode
            ids = self.enc.encode(text)
            
            # --- POCKET HACK: Modulo Token IDs ---
            # Restricted vocab for embedded device
            ids = [t % VOCAB_SIZE for t in ids]
            
            # Add EOT (mapped to 0 or VOCAB_SIZE-1? Let's use 0)
            eot = self.enc.eot_token % VOCAB_SIZE
            ids.append(eot) 
            
            # Handling length
            if len(ids) < self.context_length + 1:
                ids = ids + [eot] * (self.context_length + 1 - len(ids))
            elif len(ids) > self.context_length + 1:
                ids = ids[:self.context_length + 1]
                
            batch_input_ids.append(ids[:-1])
            batch_targets.append(ids[1:])
            
        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_targets, dtype=torch.long)
        return x, y

def generate_demo_pocket(model, enc, instruction="Count to 5."):
    model.eval()
    device = next(model.parameters()).device
    
    prompt = f"### Instruction: {instruction}\n### Response:"
    ids = enc.encode(prompt)
    # Apply Modulo
    ids = [t % VOCAB_SIZE for t in ids]
    
    x = torch.tensor([ids], dtype=torch.long).to(device)
    
    print(f"\n[POCKET DEMO INPUT] {instruction}")
    print("[POCKET DEMO OUTPUT] ", end="", flush=True)
    
    eot = enc.eot_token % VOCAB_SIZE
    
    for _ in range(50):
        if x.size(1) >= CONTEXT_LEN: break
        with torch.no_grad():
            logits = model(x)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            tok_id = next_token.item()
            
            # Decode is tricky because we lost info with modulo.
            # We can't really decoded nicely back to text unless we had a specific reduced vocab map.
            # But 'tiktoken.decode' expects full IDs.
            # We will just print the ID, or try to decode as-is (might be wrong word, but gives an idea).
            # ACTUALLY: Since we trained on `id % 4096`, the model output `id`. 
            # This `id` corresponds to `original_id % 4096`. 
            # There are many original_ids that map to this. We decode using the original_id = id
            # (assuming the most frequent ones are in 0-4096 range effectively).
            
            try:
                word = enc.decode([tok_id]) 
                # Note: This might produce garbage if ID 5 maps to "apple" but we mean "bear" (5030%4096=5 ?).
                # But GPT2 tokenizer is byte-level BPE, low IDs are frequent?
                # Actually GPT2 low IDs are bytes/chars.
                print(word, end="", flush=True)
            except:
                print("?", end="", flush=True)
            
            x = torch.cat([x, next_token], dim=1)
            if tok_id == eot: break
            
    print("\n")
    model.train()

def train():
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    print("--- Atomic-1Bit POCKET Training (Vocab=4096) ---")
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"Device: {device}")
    
    ds = PocketAlpacaDataset(context_length=CONTEXT_LEN)
    
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config).to(device)
    
    start_step = 0
    ckpt_path = os.path.join(weights_dir, "pocket_final.pt")
    
    if os.path.exists(ckpt_path):
        print(f">> Found checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "step" in checkpoint:
                start_step = checkpoint["step"]
                print(f"   Resuming from STEP {start_step}")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(">> Starting Fresh POCKET Model")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    steps_input = input(f"Current Step: {start_step}. How many ADDITIONAL steps to train? (Def: 5000): ")
    additional_steps = int(steps_input) if steps_input.strip() else 5000
    total_steps_target = start_step + additional_steps
    
    print(f"Training from {start_step} to {total_steps_target}...")
    
    try:
        for step in range(start_step, total_steps_target):
            x, y = ds.get_batch(BATCH_SIZE)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}/{total_steps_target} | Loss: {loss.item():.4f}")
                
            if step % 500 == 0:
                generate_demo_pocket(model, ds.enc, "What is AI?")
                
            if step > 0 and step % 1000 == 0:
                save_dict = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(save_dict, ckpt_path) 
                print(f"Saved checkpoint to {ckpt_path}")

        save_dict = {
            "step": total_steps_target,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(save_dict, ckpt_path)
        print("Done.")

    except KeyboardInterrupt:
        print("Training Interrupted.")
        save_dict = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(save_dict, ckpt_path)
        print("Progress saved.")

if __name__ == "__main__":
    train()
