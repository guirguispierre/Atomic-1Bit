import torch
import os

model_path = "weights/stories_final.pt"
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    
    # Layer 0 Q weight
    key = "layers.0.attn.q_proj.weight"
    if key in sd:
        w = sd[key]
        print(f"Weight Shape: {w.shape}")
        print(f"Weight[0,0]: {w[0,0].item()}")
        print(f"Weight[0,1]: {w[0,1].item()}")
        
        # Calculate quant
        scale = 1.0 / w.abs().mean().clamp(min=1e-5)
        w_q = (w * scale).round().clamp(-1, 1).long()
        print(f"Quant Scale: {scale.item()}")
        print(f"Quant[0,0]: {w_q[0,0].item()}")
        print(f"Quant[0,1]: {w_q[0,1].item()}") # Out=0, In=1
        
        # Transposed Logic
        # T[0,0] -> In=0, Out=0 -> W[0,0]
        # T[0,1] -> In=0, Out=1 -> W[1,0]
        print(f"Quant[1,0]: {w_q[1,0].item()}") # Out=1, In=0
    else:
        print(f"Key {key} not found")
else:
    print("File not found")
