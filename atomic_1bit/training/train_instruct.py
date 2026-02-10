import sys
import os
import glob
import json
import math
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

import csv
import time
from atomic_1bit.utils.thermal import ThermalMonitor

# --- EFFICIENT INSTRUCT CONFIG (v1.3) ---
# "Flagship" 12.5M Parameter Model.
# High Quality, Low RAM.
BATCH_SIZE = 32         # Physical batch size
GRAD_ACCUM_STEPS = 8    # Effective batch = 32 * 8 = 256 (scaled up from 4)
CONTEXT_LEN = 256       # Standard for Instructions
DIM = 320               # Sweet Spot
DEPTH = 8
HEADS = 5               # 320/5 = 64
VOCAB_SIZE = 4096       # Compressed
UNK_ID = VOCAB_SIZE - 1
LR = 3e-4               # Scaled for larger effective batch (was 6e-4 with eff=128)
WARMUP_STEPS = 1000     # 5% of 20k steps (increased from 500)
CLIP_GRAD = 1.0         # Gradient Clipping Threshold
WEIGHT_DECAY = 0.1      # AdamW weight decay for non-norm parameters
DROPOUT = 0.0           # Configurable dropout (set >0 to experiment)

class EfficientInstructDataset:
    def __init__(self, split="train", context_length=256, vocab_file="weights/vocab_map_instruct.json"):
        print(f"Loading Alpaca Cleaned ({split})...")
        raw_dataset = load_dataset("yahma/alpaca-cleaned", split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        self.vocab_file = vocab_file

        # --- 1. STRICT FILTERING ---
        print(f"Filtering dataset (Max Tokens: {context_length}, Min Tokens: 10)...")

        def filter_fn(sample):
            text = self.format_prompt(sample)
            ids = self.enc.encode(text)
            # Filter: min 10 tokens, max context_length (inclusive of EOT)
            return 10 <= len(ids) + 1 <= context_length

        self.dataset = raw_dataset.filter(filter_fn)
        print(f"Kept {len(self.dataset)}/{len(raw_dataset)} clean samples.")

        self.token_map = {}
        self.reverse_map = {}

        self._init_vocab()

    def _init_vocab(self):
        # 1. Try Load
        if os.path.exists(self.vocab_file):
            print(f"Loading vocab map from {self.vocab_file}...")
            with open(self.vocab_file, 'r') as f:
                data = json.load(f)
                self.token_map = {int(k): v for k, v in data["token_map"].items()}
                self.reverse_map = {int(k): v for k, v in data["reverse_map"].items()}
            print(f"Loaded {len(self.token_map)} mapped tokens.")
            return

        # 2. Build Frequency Map
        print("Building Frequency-Based Vocab (Scanning first 20k filtered samples)...")
        counter = Counter()

        scan_limit = min(20000, len(self.dataset))
        for i in range(scan_limit):
            row = self.dataset[i]
            text = self.format_prompt(row)
            ids = self.enc.encode(text)
            counter.update(ids)

        eot = self.enc.eot_token

        # Reserve UNK, ensure EOT
        most_common = counter.most_common(VOCAB_SIZE - 2)

        new_id = 0
        valid_gpt_ids = [k for k, v in most_common]
        if eot not in valid_gpt_ids:
            valid_gpt_ids.append(eot)

        # Ensure we don't exceed limit
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
            gpt_ids = self.enc.encode(text)
            gpt_ids.append(self.enc.eot_token)

            # REMAP
            pocket_ids = [self.token_map.get(gid, UNK_ID) for gid in gpt_ids]

            # Pad
            if len(pocket_ids) < self.context_length + 1:
                eot_mapped = self.token_map.get(self.enc.eot_token, UNK_ID)
                pocket_ids = pocket_ids + [eot_mapped] * (self.context_length + 1 - len(pocket_ids))

            batch_input_ids.append(pocket_ids[:-1])
            batch_targets.append(pocket_ids[1:])

        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_targets, dtype=torch.long)
        return x, y

def generate_demo(model, ds, instruction="Count to 5."):
    model.eval()
    device = next(model.parameters()).device

    prompt = f"### Instruction: {instruction}\n### Response:"
    gpt_ids = ds.enc.encode(prompt)

    ids = [ds.token_map.get(gid, UNK_ID) for gid in gpt_ids]
    x = torch.tensor([ids], dtype=torch.long).to(device)

    print(f"\n[INPUT] {instruction}")
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
            except:
                pass

            x = torch.cat([x, next_token], dim=1)
            if pocket_id == eot_mapped: break

    print("\n")
    model.train()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Linear warmup followed by cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def init_weights(model):
    """Apply scaled Kaiming initialization to BitLinear latent weights."""
    from atomic_1bit.nn.layers import BitLinear
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            torch.nn.init.kaiming_normal_(module.weight, a=math.sqrt(5))


def train():
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    print(f"--- Atomic-1Bit FLAGSHIP INSTRUCT (12.5M Params, Vocab={VOCAB_SIZE}) ---")

    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"Device: {device}")

    ds = EfficientInstructDataset(context_length=CONTEXT_LEN)

    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config).to(device)

    start_step = 0
    ckpt_path = os.path.join(weights_dir, "instruct_final.pt")

    # Separate weight decay: apply to non-norm, non-bias, non-embedding parameters
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ln' in name or 'bias' in name or 'emb' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=LR)

    # Default total steps (may be overridden if resuming)
    additional_steps = 5000

    if os.path.exists(ckpt_path):
        print(f">> Found checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "step" in checkpoint:
                start_step = checkpoint["step"]
                print(f"   Resuming from STEP {start_step}")
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("   Optimizer state restored.")
                except:
                    print("   Warning: Could not load optimizer state")
            if "rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["rng_state"])
                if "np_rng_state" in checkpoint:
                    np.random.set_state(checkpoint["np_rng_state"])
                print("   RNG state restored.")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(">> Starting Fresh Model")
        init_weights(model)
        print("   Applied scaled Kaiming initialization to BitLinear weights.")

    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- v1.3 STABILITY & MONITORING ---
    thermal_monitor = ThermalMonitor(high_threshold=80.0, resume_threshold=70.0, check_interval_steps=50)

    log_file = os.path.join(weights_dir, "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("step,loss,lr,temp,step_time_ms\n")

    steps_input = input(f"Current Step: {start_step}. How many ADDITIONAL steps to train? (Def: 5000): ")
    additional_steps = int(steps_input) if steps_input.strip() else 5000
    total_steps_target = start_step + additional_steps

    # Cosine schedule with warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps_target)

    # If resuming, advance scheduler to current step
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()
        # Also try to restore scheduler state if available
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("   Scheduler state restored.")
                except:
                    print("   Warning: Could not restore scheduler, using re-computed state.")

    print(f"Training from {start_step} to {total_steps_target} (Grad Accum: {GRAD_ACCUM_STEPS}, Eff Batch: {BATCH_SIZE * GRAD_ACCUM_STEPS})...")

    optimizer.zero_grad()

    try:
        step_start_time = time.time()

        for step in range(start_step, total_steps_target):
            # 1. Thermal Safety Check
            thermal_monitor.check_and_pause(step, model=model, optimizer=optimizer, scheduler=scheduler, save_path=ckpt_path + ".safe")
            current_temp = thermal_monitor.get_current_temp()

            # Gradient Accumulation
            loss_accum = 0.0
            for _ in range(GRAD_ACCUM_STEPS):
                x, y = ds.get_batch(BATCH_SIZE)
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                loss = loss / GRAD_ACCUM_STEPS # Normalize
                loss.backward()
                loss_accum += loss.item()

            # 2. Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Timing
            step_end_time = time.time()
            step_duration_ms = (step_end_time - step_start_time) * 1000
            step_start_time = step_end_time

            # Logging
            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {step}/{total_steps_target} | Loss: {loss_accum:.4f} | LR: {current_lr:.2e} | Temp: {current_temp:.1f}C | Time: {step_duration_ms:.1f}ms")

                with open(log_file, "a") as f:
                    f.write(f"{step},{loss_accum:.5f},{current_lr:.5e},{current_temp:.1f},{step_duration_ms:.1f}\n")

            if step % 500 == 0 and step > 0:
                generate_demo(model, ds, "What is AI?")

            if step > 0 and step % 1000 == 0:
                save_dict = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "np_rng_state": np.random.get_state(),
                    "config": {
                        "vocab_size": VOCAB_SIZE, "dim": DIM, "depth": DEPTH,
                        "heads": HEADS, "context_length": CONTEXT_LEN,
                    },
                }
                torch.save(save_dict, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        save_dict = {
            "step": total_steps_target,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            "config": {
                "vocab_size": VOCAB_SIZE, "dim": DIM, "depth": DEPTH,
                "heads": HEADS, "context_length": CONTEXT_LEN,
            },
        }
        torch.save(save_dict, ckpt_path)
        print("Done.")

    except KeyboardInterrupt:
        print("Training Interrupted.")
        save_dict = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
        }
        torch.save(save_dict, ckpt_path)
        print("Progress saved.")

if __name__ == "__main__":
    train()
