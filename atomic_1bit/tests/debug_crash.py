import torch
import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atomic_1bit.nn.layers import BitLinear
from atomic_1bit.model.transformer import BitAttention, AtomicConfig

def debug():
    print("--- Debugging ---")
    
    # 1. Test BitLinear
    print("\n1. Testing BitLinear...")
    try:
        layer = BitLinear(64, 64)
        x = torch.randn(2, 32, 64)
        y = layer(x)
        print(f"BitLinear Output: {y.shape}")
    except:
        traceback.print_exc()
        return

    # 2. Test BitAttention
    print("\n2. Testing BitAttention...")
    try:
        config = AtomicConfig(dim=64, heads=4)
        attn = BitAttention(config)
        x = torch.randn(2, 32, 64)
        y = attn(x)
        print(f"BitAttention Output: {y.shape}")
    except:
        traceback.print_exc()
        return

if __name__ == "__main__":
    debug()
