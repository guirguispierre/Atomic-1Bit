"""
Evaluation runner for Atomic-1Bit models.

Usage:
    python3 atomic_1bit/evaluation/run_eval.py --model weights/instruct_final.pt --output eval_results.json
"""
import sys
import os
import json
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.evaluation.perplexity import calculate_perplexity
from atomic_1bit.evaluation.repetition import calculate_repetition_rate, calculate_repetition_rates
from atomic_1bit.evaluation.diversity import diversity_metrics, vocabulary_utilization
from atomic_1bit.evaluation.coherence import sentence_completion_accuracy, top_k_accuracy


def generate_text(model, input_ids, max_new_tokens=100, temperature=0.8):
    """Generate text autoregressively."""
    model.eval()
    device = next(model.parameters()).device
    x = input_ids.to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= model.config.context_length:
                break
            logits = model(x)
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token.item())
            x = torch.cat([x, next_token], dim=1)

    return generated


def run_evaluation(model_path, output_path=None, device=None):
    """Run full evaluation suite on a trained model."""
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Device: {device}")
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        config = AtomicConfig(
            vocab_size=cfg["vocab_size"],
            dim=cfg["dim"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            context_length=cfg["context_length"],
        )
    else:
        # Default flagship config
        config = AtomicConfig(vocab_size=4096, dim=320, depth=8, heads=5, context_length=256)

    model = AtomicTransformer(config).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    step = checkpoint.get("step", "unknown")
    print(f"Model loaded (step: {step}, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M)")

    # Load dataset for PPL and coherence evaluation
    from atomic_1bit.training.train_instruct import EfficientInstructDataset, CONTEXT_LEN
    ds = EfficientInstructDataset(context_length=config.context_length)

    results = {"model_path": model_path, "step": step, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. Perplexity
    print("\n--- Perplexity ---")
    ppl = calculate_perplexity(model, ds, config.vocab_size, device, num_samples=200)
    results["perplexity"] = round(ppl, 2)
    print(f"Perplexity: {ppl:.2f}")

    # 2. Coherence
    print("\n--- Coherence ---")
    acc_top1 = sentence_completion_accuracy(model, ds, config.vocab_size, device, num_samples=100)
    acc_top5 = top_k_accuracy(model, ds, config.vocab_size, device, k=5, num_samples=100)
    results["coherence"] = {
        "top1_accuracy": round(acc_top1, 4),
        "top5_accuracy": round(acc_top5, 4),
    }
    print(f"Top-1 Accuracy: {acc_top1:.2%}")
    print(f"Top-5 Accuracy: {acc_top5:.2%}")

    # 3. Generation Quality
    print("\n--- Generation Samples ---")
    test_prompts = [
        "What is Artificial Intelligence?",
        "Count from 1 to 5.",
        "Tell me a joke.",
        "Explain the water cycle.",
    ]

    generation_results = []
    all_generated_texts = []

    for prompt_text in test_prompts:
        prompt = f"### Instruction: {prompt_text}\n### Response:"
        gpt_ids = ds.enc.encode(prompt)
        from atomic_1bit.training.train_instruct import UNK_ID
        ids = [ds.token_map.get(gid, UNK_ID) for gid in gpt_ids]
        input_tensor = torch.tensor([ids], dtype=torch.long)

        tokens = generate_text(model, input_tensor, max_new_tokens=100)

        # Decode
        text = ""
        for t in tokens:
            gpt_id = ds.reverse_map.get(t, ds.enc.eot_token)
            try:
                text += ds.enc.decode([gpt_id])
            except (UnicodeDecodeError, ValueError, KeyError):
                pass

        rep_rates = calculate_repetition_rates(text)
        div = diversity_metrics(text)

        result = {
            "prompt": prompt_text,
            "output": text[:500],
            "length_tokens": len(tokens),
            "repetition_rates": {str(k): round(v, 4) for k, v in rep_rates.items()},
            "diversity": {k: round(v, 4) for k, v in div.items()},
        }
        generation_results.append(result)
        all_generated_texts.append(text)

        print(f"\nPrompt: {prompt_text}")
        print(f"Output: {text[:200]}")
        print(f"3-gram Repetition: {rep_rates[3]*100:.1f}%")

    results["generations"] = generation_results

    # 4. Average Metrics
    avg_rep = sum(g["repetition_rates"]["3"] for g in generation_results) / len(generation_results)
    avg_diversity = sum(g["diversity"]["bigram_ratio"] for g in generation_results) / len(generation_results)
    results["summary"] = {
        "avg_3gram_repetition": round(avg_rep, 4),
        "avg_bigram_diversity": round(avg_diversity, 4),
    }

    print(f"\n--- Summary ---")
    print(f"Perplexity: {results['perplexity']}")
    print(f"Avg 3-gram Repetition: {avg_rep*100:.1f}%")
    print(f"Avg Bigram Diversity: {avg_diversity*100:.1f}%")
    print(f"Top-1 Accuracy: {results['coherence']['top1_accuracy']:.2%}")
    print(f"Top-5 Accuracy: {results['coherence']['top5_accuracy']:.2%}")

    # Save
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Atomic-1Bit Model")
    parser.add_argument("--model", type=str, default="weights/instruct_final.pt")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    run_evaluation(args.model, args.output)
