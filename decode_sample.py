import torch
import json
import tiktoken
from baseline_fp16 import FP16Transformer, FP16Config
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

# --- CONFIG ---
VOCAB_SIZE = 2048
DIM = 128
DEPTH = 4
HEADS = 4
CONTEXT_LEN = 64

def get_vocab():
    with open("weights/benchmark_vocab.json", "r") as f:
        data = json.load(f)
        reverse_map = {int(k): v for k, v in data["reverse_map"].items()}
    return reverse_map

def decode(tokens, reverse_map, enc):
    text = ""
    for t in tokens:
        gpt_id = reverse_map.get(t, enc.eot_token)
        try:
            word = enc.decode([gpt_id])
            text += word
        except:
            pass
    return text

def run():
    reverse_map = get_vocab()
    enc = tiktoken.get_encoding("gpt2")
    
    # 1. Atomic C++ Output (Manual)
    cpp_tokens = [494, 305, 313, 393, 1447, 793, 305, 480, 2017, 1095, 1121, 1970, 718, 194, 507, 477, 872, 1195, 383, 2023, 966, 305, 1140, 1662, 299, 1619, 1212, 1273, 2023, 971, 1388, 909, 1712, 305, 495, 1747, 1242, 1273, 1863, 1662, 739, 1891, 150, 2043, 966, 1212, 1103, 2043, 315, 1801]
    # Correcting the list syntax error in my copy-paste above (missing comma after 1891)
    cpp_tokens = [494, 305, 313, 393, 1447, 793, 305, 480, 2017, 1095, 1121, 1970, 718, 194, 507, 477, 872, 1195, 383, 2023, 966, 305, 1140, 1662, 299, 1619, 1212, 1273, 2023, 971, 1388, 909, 1712, 305, 495, 1747, 1242, 1273, 1863, 1662, 739, 1891, 150, 2043, 966, 1212, 1103, 2043, 315, 1801]
    
    atomic_text = decode(cpp_tokens, reverse_map, enc)
    print("--- ATOMIC SAMPLE ---")
    print(atomic_text)
    
    # 2. FP16 Baseline Output (Generate new)
    fp_config = FP16Config(vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, context_length=CONTEXT_LEN)
    fp_model = FP16Transformer(fp_config)
    fp_model.load_state_dict(torch.load("weights/baseline_model.pt", map_location="cpu"))
    fp_model.eval()
    
    # Start with token 42
    x = torch.tensor([[42]], dtype=torch.long)
    
    gen_tokens = []
    with torch.no_grad():
        for _ in range(50):
            logits = fp_model(x)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            gen_tokens.append(next_token.item())
            x = torch.cat([x, next_token], dim=1)
            x = x[:, -CONTEXT_LEN:]
            
    fp_text = decode(gen_tokens, reverse_map, enc)
    print("\n--- FP16 SAMPLE ---")
    print(fp_text)

if __name__ == "__main__":
    run()
