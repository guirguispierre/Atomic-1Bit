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
from atomic_1bit.training.data import AlpacaDataset

# Config (Matches Instruction Tuning better?)
# Keep Dim/Depth same for "Potato" hardware compat.
BATCH_SIZE = 16 
CONTEXT_LEN = 256 # Increased for Instructions
DIM = 256
DEPTH = 8 # Deeper for logic? Original user request said Depth 8.
HEADS = 4
VOCAB_SIZE = 50257 
LR = 1e-3

def generate_demo_instruct(model, enc, instruction="Count to 5."):
    model.eval()
    device = next(model.parameters()).device
    
    prompt = f"### Instruction: {instruction}\n### Response:"
    ids = enc.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    
    print(f"\n[DEMO INPUT] {instruction}")
    print("[DEMO OUTPUT] ", end="", flush=True)
    
    for _ in range(50):
        if x.size(1) >= CONTEXT_LEN: break
        with torch.no_grad():
            logits = model(x)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Print
            word = enc.decode([next_token.item()])
            print(word, end="", flush=True)
            
            x = torch.cat([x, next_token], dim=1)
            if next_token.item() == enc.eot_token: break
            
    print("\n")
    model.train()

def train():
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    print("--- Atomic-1Bit INSTRUCT Training ---")
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"Device: {device}")
    
    ds = AlpacaDataset(context_length=CONTEXT_LEN)
    
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config).to(device)
    
    start_step = 0
    
    ckpt_path = os.path.join(weights_dir, "instruct_final.pt")
    
    # Check for existing checkpoint
    if os.path.exists(ckpt_path):
        print(f">> Found checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # New Format Detection
        if "model_state_dict" in checkpoint:
            print("   (Format: Standard Checkpoint wrapper)")
            model.load_state_dict(checkpoint["model_state_dict"])
            if "step" in checkpoint:
                start_step = checkpoint["step"]
                print(f"   Resuming from STEP {start_step}")
        else:
            # Legacy format (raw state dict)
            print("   (Format: Legacy Raw State Dict)")
            try:
                model.load_state_dict(checkpoint)
                print("   Weights loaded. Starting step count at 0 (Legacy).")
            except Exception as e:
                print(f"   Error loading legacy weights: {e}")
                
    else:
        print(">> Starting Fresh Instruct Model (Step 0)")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # Ideally load optimizer state too if available, but for now we skip to keep it simple/robust against errors
    # if "optimizer_state_dict" in checkpoint: optimizer.load_state_dict(...)

    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Interactive Step Count
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
                
            if step % 200 == 0:
                generate_demo_instruct(model, ds.enc, "What is the capital of France?")
                
            if step > 0 and step % 1000 == 0:
                # Save Wrapper
                save_dict = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(save_dict, ckpt_path) 
                print(f"Saved checkpoint to {ckpt_path} (Step {step})")

        # Final Save
        save_dict = {
            "step": total_steps_target,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(save_dict, ckpt_path)
        print("Done.")

    except KeyboardInterrupt:
        print("Training Interrupted.")
        # Save emergency
        save_dict = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(save_dict, ckpt_path)
        print("Progress saved.")

if __name__ == "__main__":
    train()
