import sys
import os
import glob
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.training.data import TinyStoriesDataset

# Config
BATCH_SIZE = 16 
CONTEXT_LEN = 128
DIM = 256
DEPTH = 4
HEADS = 4
VOCAB_SIZE = 50257 
LR = 1e-3

def generate_demo(model, enc, prompt="Once upon a time"):
    model.eval()
    # Fix: Get device from model
    device = next(model.parameters()).device
    ids = enc.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    
    # Generate 20 tokens
    for _ in range(20):
        if x.size(1) >= CONTEXT_LEN: break
        with torch.no_grad():
            logits = model(x) # (B, T, V)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token], dim=1)
            
    text = enc.decode(x[0].tolist())
    print(f"\n[DEMO] {text}\n")
    model.train()

def train():
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    print("--- Atomic-1Bit Training v2 ---")

    # 1. Interactive Step Count
    default_steps = 5000
    try:
        user_input = input(f"How many steps do you want to train for? (Default: {default_steps}): ")
        TOTAL_STEPS = int(user_input) if user_input.strip() else default_steps
    except ValueError:
        print("Invalid input. Using default.")
        TOTAL_STEPS = default_steps

    # Device Setup
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Data
    ds = TinyStoriesDataset(context_length=CONTEXT_LEN, subset_ratio=0.10) # Increased subset slightly
    
    # Model
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config).to(device)
    
    # 2. Resume Capability
    checkpoint_path = None
    final_path = os.path.join(weights_dir, "final.pt")
    
    # Check for final.pt first
    if os.path.exists(final_path):
        checkpoint_path = final_path
    else:
        # Check for latest ckpt
        list_of_files = glob.glob(os.path.join(weights_dir, "ckpt_*.pt")) 
        if list_of_files:
            checkpoint_path = max(list_of_files, key=os.path.getctime)
            
    if checkpoint_path:
        print(f">> Resuming from checkpoint: {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print("   Weights loaded successfully.")
        except Exception as e:
            print(f"   Error loading weights: {e}. Starting fresh.")
    else:
        print(">> Starting fresh (Random Initialization)...")

    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    print(f"Starting training for {TOTAL_STEPS} steps...")
    
    # 3. Safe Exit Loop
    try:
        for step in range(TOTAL_STEPS):
            x, y = ds.get_batch(BATCH_SIZE)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Loss
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}/{TOTAL_STEPS} | Loss: {loss.item():.4f}")
                
            if step % 100 == 0:
                generate_demo(model, ds.enc)
                
            if step > 0 and step % 500 == 0:
                 ckpt_name = os.path.join(weights_dir, f"ckpt_{step}.pt")
                 torch.save(model.state_dict(), ckpt_name)
                 print(f"Checkpoint saved: {ckpt_name}")

        # Save final
        torch.save(model.state_dict(), final_path)
        print("Training Complete. Saved to weights/final.pt")

    except KeyboardInterrupt:
        print("\n\n>> Training Interrupted! (Ctrl+C detected)")
        save_path = os.path.join(weights_dir, "interrupted.pt")
        torch.save(model.state_dict(), save_path)
        print(f"   Progress saved safely to: {save_path}")
        print("   Exiting...")

if __name__ == "__main__":
    train()
