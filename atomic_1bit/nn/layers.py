import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Import our wrapper (still useful for verification if needed, but not used in training forward)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
from wrapper import ternary_matmul

def activation_quant(x):
    """
    Quantize activation to INT8 using AbsMax scaling with STE.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    y = (x * scale).round().clamp(-127, 127)
    # STE: detach y, but backward passes gradients to x
    y_ste = (y - x * scale).detach() + x * scale
    return y_ste, scale

def weight_quant(w):
    """
    Quantize weights to {-1, 0, 1} using Mean scaling with STE.
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    y = (w * scale).round().clamp(-1, 1)
    y_ste = (y - w * scale).detach() + w * scale
    return y_ste, scale

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        """
        BitNet b1.58 Linear Layer.
        Now supports Training via Straight-Through Estimator (STE).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Latent Float Weights (Out, In) - Standard PyTorch shape
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.eps = 1e-5

    def forward(self, x):
        """
        Forward pass with On-the-fly Quantization (STE).
        """
        # 1. RMSNorm
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x_f32 / rms
        
        # 2. Quantize Activation
        # x_q_ste is effectively (x_int8 / scale_x) * scale_x ... wait.
        # activation_quant returns `y_ste` which is close to `x * scale` in value (approx int8 range).
        # We want the input to F.linear to be the "Simulated Float" version or the "Int" version?
        # F.linear(input, weight) -> input * weight^T.
        # If we use quantized values:
        # y = linear(x_quant_int, w_quant_int)
        # Result y is int32-like (large).
        # Then we dequantize: y / (scale_x * scale_w).
        
        x_quant_ste, scale_x = activation_quant(x_norm)
        
        # 3. Quantize Weights
        w_quant_ste, scale_w = weight_quant(self.weight)
        
        # 4. Linear Operation
        # We use the scaled versions (values ~127 and ~1) so the matrix mul results are large
        # but the gradients are correct due to STE logic in the helper functions.
        # Wait, `activation_quant` returns `(y - x*s).detach() + x*s`.
        # So it returns values in range [-127, 127] (roughly).
        # `weight_quant` returns values in range [-1, 1] (roughly).
        
        # If we pass these to F.linear, we compute Transpose(w) * x.
        y = F.linear(x_quant_ste, w_quant_ste)
        
        # 5. Dequantize
        # y is approx (x * s_x) * (w * s_w).
        # We want x * w.
        # So y_out = y / (s_x * s_w).
        
        y_out = y / (scale_x * scale_w)
        
        if self.bias is not None:
            y_out += self.bias
            
        return y_out

