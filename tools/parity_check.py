import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

def print_tensor_stats(name, tensor, num=10):
    t_flat = tensor.detach().cpu().flatten()
    vals = t_flat[:num].tolist()
    print(f"[{name}] Shape: {tensor.shape}")
    print(f"  Values: {['%.4f' % v for v in vals]}")

def hook_fn(name):
    def hook(module, input, output):
        # input is tuple
        print_tensor_stats(f"{name}_input", input[0])
        print_tensor_stats(f"{name}_output", output)
    return hook

def run_parity_check():
    print("--- Python Parity Check ---")
    
    # 1. Load Model
    config = AtomicConfig(vocab_size=4096, dim=256, depth=6, heads=4, context_length=128)
    model = AtomicTransformer(config)
    
    model_path = "weights/stories_final.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 2. Register Hooks for Debugging
    # We want to see: 
    # - Embeddings output
    # - Layer 0 RMSNorm
    # - Layer 0 Attn Output
    # - Layer 0 MLP Output
    
    # We can manually inspect submodules
    
    # Input ID for "Once" (Token 54)
    input_ids = torch.tensor([[54]], dtype=torch.long)
    
    print(f"Input: {input_ids}")
    
    # Run Forward Step by Step manually to replicate C++ logic exactly
    
    with torch.no_grad():
        x = model.token_emb(input_ids)
        pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        p = model.pos_emb(pos)
        x = x + p
        
        print_tensor_stats("Embedding_Sum", x)
        
        for i, layer in enumerate(model.layers):
            print(f"\n--- Layer {i} ---")
            
            # LN1
            rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-5)
            x_norm = x.float() / rms
            # Apply weight
            x_norm = x_norm * layer.ln1.weight
            
            print_tensor_stats(f"Layer{i}_LN1", x_norm)
            
            # Probe Q details
            # Need to access internal quant logic to verify
            # Reimplement activation quant manually for debug
            from atomic_1bit.nn.layers import activation_quant, weight_quant
            
            x_q, scale_x = activation_quant(x_norm)
            w_q_f, scale_w = weight_quant(layer.attn.q_proj.weight)
            # w_q_f is float (STE). Convert to int for comparison
            w_q = w_q_f.round().long() # [-1, 0, 1]
            
            print(f"Layer{i}_X_Q Sample: {x_q[0, 0, :10].int().tolist()}")
            if hasattr(layer.attn.q_proj, 'weight'):
                # Weight shape (Out, In)
                print(f"Layer{i}_W_Q Sample Col 0 (Row 0 of Transposed?):")
                # We want weights contributing to Output 0.
                # Output 0 = dot(Input, Weight[0, :]).
                # So we want Weight[0, :].
                print(f"  W[0][:10]: {w_q[0, :10].tolist()}")
                
                # Manual Dot Product Check
                acc = (x_q[0, 0].float() * w_q[0].float()).sum().item()
                dequant = acc / (scale_x * scale_w).item()
                print(f"  Manual Acc: {acc}")
                print(f"  Manual Dequant: {dequant}")
            
            q_out = layer.attn.q_proj(x_norm)
            attn_out, _ = layer.attn(x_norm)
            print_tensor_stats(f"Layer{i}_Attn_Out", attn_out)

            x = x + attn_out
            
            # LN2
            rms2 = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-5)
            x_norm2 = x.float() / rms2 * layer.ln2.weight
            
            print_tensor_stats(f"Layer{i}_LN2", x_norm2)
            
            # MLP
            mlp_out = layer.mlp(x_norm2)
            print_tensor_stats(f"Layer{i}_MLP_Out", mlp_out)
            
            x = x + mlp_out
            
        print("\n--- Final ---")
        # Final Norm
        rms_f = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-5)
        x_f = x.float() / rms_f * model.ln_f.weight
        
        print_tensor_stats("Final_LN", x_f)
        
        # Head
        logits = model.head(x_f)
        print_tensor_stats("Final_Logits", logits, num=20)
        
        # Top 5
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, 5)
        print("Top 5 Tokens:", topk.indices.tolist())
        print("Top 5 Probs: ", topk.values.tolist())

if __name__ == "__main__":
    run_parity_check()
