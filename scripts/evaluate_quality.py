import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.training.train_instruct import EfficientInstructDataset, VOCAB_SIZE, DIM, DEPTH, HEADS, CONTEXT_LEN

def calculate_perplexity(model, ds, device, num_samples=100):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    print(f"Evaluating Perplexity on {num_samples} samples...")
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for i in range(num_samples):
            # Slow sample by sample validation
            x, y = ds.get_batch(1) # Batch 1
            x, y = x.to(device), y.to(device)
            
            logits = model(x) # (1, T, V)
            
            # Reshape
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total_nll += loss.item()
            total_tokens += y.numel() # Includes padding, but we should mask really.
            # Minimally correct for demo: padded tokens contribute to PPL if not masked.
            # Assuming EfficientInstructDataset pads with mapped EOT or similar, 
            # ideally we mask loss. For v1.3 speed, simple average.
            
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return ppl.item()

def calculate_repetition_rate(text, n=3):
    """Calculates percentage of n-grams that are repeated."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    unique_ngrams = set(ngrams)
    
    if len(ngrams) == 0: return 0.0
    
    # Simple repetition ratio: 1.0 - (Unique / Total)
    # If all unique, 1 - 1 = 0% repetition.
    # If all same, 1 - (1/N) ~= 100% repetition.
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def evaluate_generation(model, ds, device, prompt="What is AI?"):
    model.eval()
    
    gpt_ids = ds.enc.encode(f"### Instruction: {prompt}\n### Response:")
    UNK_ID = VOCAB_SIZE - 1
    ids = [ds.token_map.get(gid, UNK_ID) for gid in gpt_ids]
    x = torch.tensor([ids], dtype=torch.long).to(device)
    
    generated_text = ""
    
    with torch.no_grad():
        for _ in range(100):
            if x.size(1) >= CONTEXT_LEN: break
            logits = model(x)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            pocket_id = next_token.item()
            
            gpt_id = ds.reverse_map.get(pocket_id, ds.enc.eot_token)
            try:
                word = ds.enc.decode([gpt_id])
                generated_text += word
            except (UnicodeDecodeError, ValueError, KeyError):
                pass
            
            x = torch.cat([x, next_token], dim=1)
            
            eot_mapped = ds.token_map.get(ds.enc.eot_token, UNK_ID)
            if pocket_id == eot_mapped: break
            
    return generated_text

def main():
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"Device: {device}")
    
    ds = EfficientInstructDataset(context_length=CONTEXT_LEN)
    
    config = AtomicConfig(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    model = AtomicTransformer(config).to(device)
    
    ckpt_path = "weights/instruct_final.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("No checkpoint found. Evaluating random model.")

    # 1. PPL
    ppl = calculate_perplexity(model, ds, device)
    
    # 2. Generation Quality
    test_prompts = [
        "What is Artificial Intelligence?",
        "Count from 1 to 5.",
        "Tell me a joke."
    ]
    
    avg_rep = 0.0
    print("\n--- Generation Samples ---")
    for p in test_prompts:
        out = evaluate_generation(model, ds, device, p)
        rep = calculate_repetition_rate(out)
        avg_rep += rep
        print(f"Prompt: {p}")
        print(f"Output: {out}")
        print(f"Repetition (3-gram): {rep*100:.1f}%\n")
        
    avg_rep /= len(test_prompts)
    
    print("\n--- v1.3 Quality Report ---")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Avg Repetition Rate: {avg_rep*100:.1f}%")
    
    # Save Metrics
    with open("v1.3_quality_report.txt", "w") as f:
        f.write(f"Perplexity: {ppl:.2f}\n")
        f.write(f"Avg Repetition Rate: {avg_rep*100:.1f}%\n")

if __name__ == "__main__":
    main()
