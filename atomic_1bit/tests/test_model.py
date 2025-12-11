import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig

def test_atomic_transformer():
    print("--- Atomic-1Bit: Model Verification ---")
    
    # 1. Setup "Baby" Config
    config = AtomicConfig(
        vocab_size=1000,
        dim=64,
        depth=2,
        heads=4,
        context_length=128
    )
    print(f"Config: {config}")
    
    model = AtomicTransformer(config)
    print("Model initialized.")
    
    # 2. Inspect Parameters
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    
    # Check if BitLinear weights are INT8
    # We iterate and find one weight from a BitLinear layer
    found_int8 = False
    for name, param in model.named_parameters():
        if 'weight' in name and param.dtype == torch.int8:
            print(f"Verified INT8 Weight: {name} | Shape: {param.shape}")
            found_int8 = True
            break
            
    if not found_int8:
        print("WARNING: Could not find INT8 weights! Check layers.")
    else:
        print("SUCCESS: INT8 weights detected.")

    # 3. Forward Pass
    print("\nRunning Forward Pass...")
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    
    try:
        logits = model(idx)
        print(f"Output Shape: {logits.shape}")
        
        expected_shape = (B, T, config.vocab_size)
        if logits.shape == expected_shape:
            print(">> SUCCESS: Output shape matches expected.")
        else:
            print(f">> FAILURE: Expected {expected_shape}, got {logits.shape}")
            
    except Exception as e:
        print(f">> CRITICAL FAILURE: Model crashed during forward pass.\n{e}")

if __name__ == "__main__":
    test_atomic_transformer()
