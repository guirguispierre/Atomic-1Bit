import sys
import os
import glob
import json
import argparse
import yaml
from collections import Counter

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
import numpy as np
from datasets import load_dataset
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# --- POCKET STORIES CONFIG (defaults, may be overridden by --config) ---
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", type=str, default=None)
_args, _ = _parser.parse_known_args()

_cfg = {}
_config_path = _args.config or os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "stories_base.yaml"
)
if os.path.exists(_config_path):
    with open(_config_path, "r") as _f:
        _cfg = yaml.safe_load(_f)
    print(f"[Config] Loaded from {_config_path}")
else:
    print(f"[Config] File not found: {_config_path}, using hardcoded defaults")

_model = _cfg.get("model", {})
_training = _cfg.get("training", {})

BATCH_SIZE  = _training.get("batch_size",      32)
CONTEXT_LEN = _model.get("context_length",     128)
DIM         = _model.get("dim",                256)
DEPTH       = _model.get("depth",              6)
HEADS       = _model.get("heads",              4)
VOCAB_SIZE  = _model.get("vocab_size",         4096)
UNK_ID      = VOCAB_SIZE - 1
LR          = _training.get("learning_rate",   1e-3)

class PocketStoriesDataset:
    def __init__(self, split="train", context_length=128, vocab_file="weights/vocab_map_stories.json"):
        print(f"Loading TinyStories ({split})...")
        # Use streaming/subset to keep it fast
        self.dataset = load_dataset("roneneldan/TinyStories", split=f"{split}[:10%]")
        self.enc = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        self.vocab_file = vocab_file
        
        self.token_map = {}   
        self.reverse_map = {} 
        
        self._init_vocab()
        
    def _init_vocab(self):
        # 1. Try Load
        if os.path.exists(self.vocab_file):
            print(f"Loading vocab map from {self.vocab_file}...")
            with open(self.vocab_file, 'r') as f:
                data = json.load(f)
                token_map = {int(k): v for k, v in data["token_map"].items()}
                reverse_map = {int(k): v for k, v in data["reverse_map"].items()}

            # guirguispierre 2026-08-10 - the cache is not keyed by VOCAB_SIZE, so
            # a map built for a larger vocab yields ids past the embedding table
            if len(token_map) > VOCAB_SIZE - 1:
                print(f"  Cached map has {len(token_map)} tokens but VOCAB_SIZE "
                      f"is {VOCAB_SIZE}; rebuilding.")
            else:
                self.token_map = token_map
                self.reverse_map = reverse_map
                print(f"Loaded {len(self.token_map)} mapped tokens.")
                return

        # 2. Build Frequency Map
        print("Building Frequency-Based Vocab (Scanning first 20k samples)...")
        counter = Counter()
        
        scan_limit = min(20000, len(self.dataset))
        rows = self.dataset.select(range(scan_limit))
        
        for text in rows["text"]:
            ids = self.enc.encode(text)
            counter.update(ids)
            
        eot = self.enc.eot_token
        
        # Reserve UNK, ensure EOT
        most_common = counter.most_common(VOCAB_SIZE - 2)
        
        new_id = 0
        valid_gpt_ids = [k for k, v in most_common]
        if eot not in valid_gpt_ids:
            valid_gpt_ids.append(eot)
            
        valid_gpt_ids = valid_gpt_ids[:VOCAB_SIZE - 1]
        
        for gpt_id in valid_gpt_ids:
            self.token_map[gpt_id] = new_id
            self.reverse_map[new_id] = gpt_id
            new_id += 1
            
        self.unk_token = UNK_ID
        
        print(f"Saving vocab map to {self.vocab_file}...")
        save_data = {
            "token_map": self.token_map,
            "reverse_map": self.reverse_map
        }
        os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
        with open(self.vocab_file, 'w') as f:
            json.dump(save_data, f)

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.dataset), batch_size)
        rows = self.dataset.select(indices)
        
        batch_input_ids = []
        batch_targets = []
        
        for text in rows["text"]:
            gpt_ids = self.enc.encode(text)
            gpt_ids.append(self.enc.eot_token)
            
            # REMAP
            pocket_ids = [self.token_map.get(gid, UNK_ID) for gid in gpt_ids]
            
            if len(pocket_ids) < self.context_length + 1:
                eot_mapped = self.token_map.get(self.enc.eot_token, UNK_ID)
                pocket_ids = pocket_ids + [eot_mapped] * (self.context_length + 1 - len(pocket_ids))
            
            if len(pocket_ids) > self.context_length + 1:
                start = np.random.randint(0, len(pocket_ids) - self.context_length - 1)
                pocket_ids = pocket_ids[start : start + self.context_length + 1]
                
            batch_input_ids.append(pocket_ids[:-1])
            batch_targets.append(pocket_ids[1:])
            
        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_targets, dtype=torch.long)
        return x, y

def generate_demo(model, ds, start_text="Once upon a time"):
    model.eval()
    device = next(model.parameters()).device
    
    gpt_ids = ds.enc.encode(start_text)
    ids = [ds.token_map.get(gid, UNK_ID) for gid in gpt_ids]
    x = torch.tensor([ids], dtype=torch.long).to(device)
    
    print(f"\n[INPUT] {start_text}")
    print("[OUTPUT] ", end="", flush=True)
    
    eot_mapped = ds.token_map.get(ds.enc.eot_token, UNK_ID)
    
    for _ in range(60):
        if x.size(1) >= CONTEXT_LEN: break
        with torch.no_grad():
            logits = model(x)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            pocket_id = next_token.item()
            
            gpt_id = ds.reverse_map.get(pocket_id, ds.enc.eot_token)
            try:
                word = ds.enc.decode([gpt_id])
                print(word, end="", flush=True)
            except (UnicodeDecodeError, ValueError, KeyError):
                pass
            
            x = torch.cat([x, next_token], dim=1)
            if pocket_id == eot_mapped: break
            
    print("\n")
    model.train()

def train():
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    print(f"--- Atomic-1Bit POCKET STORIES (Vocab={VOCAB_SIZE}) ---")
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"Device: {device}")
    
    ds = PocketStoriesDataset(context_length=CONTEXT_LEN)
    
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config).to(device)
    
    start_step = 0
    ckpt_path = os.path.join(weights_dir, "stories_final.pt")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    if os.path.exists(ckpt_path):
        print(f">> Found checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        if missing:
            print(f"   Warning: {len(missing)} missing key(s) in checkpoint: {sorted(missing)}")
        if unexpected:
            print(f"   Warning: {len(unexpected)} unexpected key(s) in checkpoint: {sorted(unexpected)}")
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"   load_state_dict result — missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")
        if "step" in checkpoint: start_step = checkpoint["step"]

    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    steps_input = input(f"Current Step: {start_step}. How many ADDITIONAL steps to train? (Def: 5000): ")
    additional_steps = int(steps_input) if steps_input.strip() else 5000
    total_steps_target = start_step + additional_steps
    
    # Init Thermal Monitor
    from atomic_1bit.utils.thermal import ThermalMonitor
    thermal_monitor = ThermalMonitor(high_threshold=80.0, resume_threshold=70.0)
    
    print(f"Training from {start_step} to {total_steps_target}...")
    
    try:
        for step in range(start_step, total_steps_target):
            # Thermal Check
            thermal_monitor.check_and_pause(
                step=step, 
                model=model, 
                optimizer=optimizer, 
                save_path=ckpt_path.replace(".pt", "_thermal_safe.pt")
            )
            
            x, y = ds.get_batch(BATCH_SIZE)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}/{total_steps_target} | Loss: {loss.item():.4f}")
                
            if step % 500 == 0:
                generate_demo(model, ds, "One day,")
                
            if step > 0 and step % 1000 == 0:
                save_dict = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(save_dict, ckpt_path) 
                print(f"Saved checkpoint.")

        save_dict = {
            "step": total_steps_target,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(save_dict, ckpt_path)
        print("Done.")

    except KeyboardInterrupt:
        print("Training Interrupted.")
        torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)

if __name__ == "__main__":
    train()
