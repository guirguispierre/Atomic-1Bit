import subprocess
import os
import sys
import json
import tiktoken

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def get_vocab():
    base_path = os.path.dirname(__file__)
    vocab_path = os.path.join(base_path, "../weights/benchmark_vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} not found. Run benchmark suite first.")
        return {}
    with open(vocab_path, "r") as f:
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

def run_cpp_inference(length=100, temp=0.7, seed=42):
    runner_path = os.path.join(os.path.dirname(__file__), "../embedded/runner")
    if not os.path.exists(runner_path):
        print("Error: Runner binary not found.")
        return []

    cmd = [runner_path, str(length), str(temp), str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(runner_path))
    
    if result.returncode != 0:
        print(f"Runner failed: {result.stderr}")
        return []

    # Parse output: "Generating: 123 456 ..."
    output = result.stdout
    tokens = []
    for line in output.split("\n"):
        if line.startswith("Generating:"):
            parts = line.replace("Generating:", "").strip().split()
            tokens = [int(x) for x in parts if x.isdigit()]
            break
    return tokens

def main():
    print("--- Atomic-1Bit Quality Evaluation ---")
    reverse_map = get_vocab()
    if not reverse_map: return
    enc = tiktoken.get_encoding("gpt2")

    target_len = 100
    
    # 1. Greedy (Baseline)
    print(f"\n[1] Greedy (Temp=0.0) | Len={target_len}")
    tokens = run_cpp_inference(length=target_len, temp=0.0, seed=42)
    print(decode(tokens, reverse_map, enc))

    # 2. Temp=0.7
    print(f"\n[2] Sampling (Temp=0.7) | Len={target_len}")
    tokens = run_cpp_inference(length=target_len, temp=0.7, seed=42)
    print(decode(tokens, reverse_map, enc))

    # 3. Temp=0.9
    print(f"\n[3] Sampling (Temp=0.9) | Len={target_len}")
    tokens = run_cpp_inference(length=target_len, temp=0.9, seed=123)
    print(decode(tokens, reverse_map, enc))

if __name__ == "__main__":
    main()
