import sys
import os
import torch
import tiktoken
import json
import subprocess
import argparse

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# Config (Pocket)
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

def run_python_gen(model_path, vocab_map_path, prompts, seeds, length=200):
    print("\n--- 1. PYTHON GENERATION ---")
    enc = tiktoken.get_encoding("gpt2")
    token_map, reverse_map = load_vocab_map(vocab_map_path)
    
    config = AtomicConfig(vocab_size=VOCAB, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CTX)
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    
    model = AtomicTransformer(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()
    
    from scripts.generate_samples import generate
    
    results = {}
    
    for p in prompts:
        results[p] = {}
        for s in seeds:
            text = generate(model, p, enc, token_map, reverse_map, length=length, seed=s)
            results[p][s] = text
            print(f"P: '{p}' | S: {s} | Len: {len(text)}")
            
    return results

def run_embedded_gen(vocab_map_path, prompts, seeds, length=200):
    print("\n--- 2. EMBEDDED GENERATION ---")
    enc = tiktoken.get_encoding("gpt2")
    token_map, _ = load_vocab_map(vocab_map_path)
    
    results = {}
    
    for p in prompts:
        results[p] = {}
        # Get start token
        gpt_ids = enc.encode(p)
        start_id_gpt = gpt_ids[0]
        start_id = token_map.get(start_id_gpt, VOCAB-1)
        
        for s in seeds:
            cmd = [
                "./embedded/runner",
                "--model", "embedded/atomic_model.bin",
                "--steps", str(length),
                "--temp", "0.7",
                "--seed", str(s),
                "--start_token", str(start_id)
            ]
            
            try:
                out = subprocess.check_output(cmd, text=True)
                # Parse "Generating: 1 2 3 ..."
                for line in out.splitlines():
                    if line.startswith("Generating:"):
                        raw_ids = line.replace("Generating:", "").strip()
                        # Decode
                        tokens = [int(x) for x in raw_ids.split()]
                        # Decode back to text
                        # We need to map Model IDs -> GPT IDs
                        # Luckily the decode.py logic does this? Wait decode.py uses gpt2 decode directly...
                        # Ah, decode.py assumes the Tokens are GPT2 tokens?
                        # Wait, the runner outputs MODEL tokens (0-4095).
                        # We must reverse map them BEFORE decoding with tiktoken.
                        
                        _, reverse_map = load_vocab_map(vocab_map_path)
                        gpt_out_ids = [reverse_map.get(pid, enc.eot_token) for pid in tokens]
                        text = enc.decode(gpt_out_ids)
                        
                        # Add the start word which runner consumes but maybe prints?
                        # Runner prints `context.push_back(start_token)` then generates.
                        # Does it print the start token?
                        # "cout << best_token" is inside the loop.
                        # So it prints generated tokens. The start prompt word is implicit.
                        # We should verify this.
                        
                        full_text = text # Just generated part
                        results[p][s] = full_text
            except Exception as e:
                print(f"Error running embedded: {e}")
                
    return results

def compare(py_res, emb_res):
    print("\n--- 3. COMPARISON ---")
    
    # Just show one key example
    p = "Once upon a time"
    s = 42
    
    print(f"PROMPT: {p} (Seed {s})")
    
    py_text = py_res[p][s]
    emb_text = emb_res[p][s]
    
    print("\n[PYTHON]:")
    print(f"{p}{py_text[:300]}...")
    
    print("\n[EMBEDDED (Decoded from Tokens)]:")
    # We suspect embedded might miss the first word in output since it consumes it as context
    print(f"(Start Token) {emb_text[:300]}...")

if __name__ == "__main__":
    prompts = ["Once upon a time", "In a small town", "The robot said"]
    seeds = [42, 123, 999]
    
    py_res = run_python_gen("weights/stories_final.pt", "weights/vocab_map_stories.json", prompts, seeds, length=200)
    emb_res = run_embedded_gen("weights/vocab_map_stories.json", prompts, seeds, length=200)
    
    compare(py_res, emb_res)
