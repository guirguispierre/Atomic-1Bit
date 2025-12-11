import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Import our wrapper
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
from wrapper import ternary_matmul

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        """
        BitNet b1.58 Linear Layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: If True, adds a learnable bias. (BitNet usually no bias, but we support for compat)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights: INT8 {-1, 0, 1}
        # We store them as standard torch.int8 parameters.
        # Random initialization for demo: Map 0->-1, 1->0, 2->1
        raw_w = torch.randint(0, 3, (in_features, out_features), dtype=torch.int8)
        self.weight = nn.Parameter(raw_w - 1, requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        self.eps = 1e-5

    def forward(self, x):
        """
        Forward pass with quantization and C++ Kernel.
        
        x: Float Tensor (Batch, In_Features)
        """
        # 1. RMSNorm (Standard in BitNet)
        # x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # But wait, usually RMSNorm is a separate layer usually before the Linear. 
        # But the prompt requested: "Include a simple RMSNorm before quantization"
        
        # RMSNorm
        x_f32 = x.float() # Ensure float
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x_f32 / rms
        
        # 2. Quantization (AbsMax to INT8)
        # We need to map range to -127..127
        # Scale factor
        # s = 127 / max(|x|)
        max_val = x_norm.abs().max(dim=-1, keepdim=True)[0]
        # Avoid div by zero
        max_val = torch.clamp(max_val, min=1e-5) 
        
        scale = 127.0 / max_val
        x_quant = torch.clamp(torch.round(x_norm * scale), -127, 127).to(torch.int8)
        
        # 3. Execution (The Magic Kernel)
        # Need to move to CPU numpy for our kernel
        # This is where we pay the "Python Tax" currently, but it proves the architecture.
        x_np = x_quant.cpu().numpy()
        w_np = self.weight.cpu().numpy() # This is already int8
        
        # Handle Multi-dimensional input (Batch, Seq, Dim) -> (Batch*Seq, Dim)
        original_shape = x_np.shape
        if len(original_shape) > 2:
            x_flat = x_np.reshape(-1, original_shape[-1])
        else:
            x_flat = x_np
            
        # Run Kernel
        # shape: (M, Out)
        y_int32 = ternary_matmul(x_flat, w_np)
        
        # Reshape back if needed
        if len(original_shape) > 2:
            y_int32 = y_int32.reshape(*original_shape[:-1], -1)
        
        # 4. Dequantization
        # y_float = y_int32 / scale
        # We need to convert back to tensor
        y_tensor = torch.from_numpy(y_int32).float().to(x.device)
        
        # Rescale
        # Note: scale has shape (Batch, 1) or (Batch, Seq, 1). 
        # y_tensor (Batch, Seq, Out) / scale (Batch, Seq, 1) -> Broadcasting
        y_out = y_tensor / scale.cpu()
        
        if self.bias is not None:
            y_out += self.bias
            
        return y_out
