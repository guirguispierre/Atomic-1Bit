import torch
import torch.nn as nn
import math
from dataclasses import dataclass
import sys
import os

# Import BitLinear
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from atomic_1bit.nn.layers import BitLinear

@dataclass
class AtomicConfig:
    vocab_size: int = 50257
    dim: int = 512
    depth: int = 8
    heads: int = 8
    context_length: int = 1024
    
class BitAttention(nn.Module):
    def __init__(self, config: AtomicConfig):
        super().__init__()
        assert config.dim % config.heads == 0, "Dim must be divisible by heads"
        self.dim = config.dim
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        
        # 1-Bit Projections for Q, K, V
        self.q_proj = BitLinear(config.dim, config.dim, bias=False)
        self.k_proj = BitLinear(config.dim, config.dim, bias=False)
        self.v_proj = BitLinear(config.dim, config.dim, bias=False)
        self.o_proj = BitLinear(config.dim, config.dim, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Project Q, K, V using BitLinear
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for Multi-Head Attention
        # (B, T, Heads, HeadDim) -> (B, Heads, T, HeadDim)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Standard Scaled Dot-Product Attention (Calculated in Float)
        # att = (q @ k.T) * (1.0 / sqrt(head_dim))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal Mask (if needed, usually passed in or built here)
        # For simplicity in this dummy model, we rely mainly on structure, 
        # but let's add a basic causal mask for correctness vibe.
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        att = att.masked_fill(mask, float('-inf'))
        
        att = torch.nn.functional.softmax(att, dim=-1)
        
        # Aggregate
        y = att @ v # (B, Heads, T, HeadDim)
        
        # Reshape back
        # (B, T, Heads, HeadDim) -> (B, T, Dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output Projection (BitLinear)
        return self.o_proj(y)

class BitFeedForward(nn.Module):
    def __init__(self, config: AtomicConfig):
        super().__init__()
        # Expansion factor 4
        hidden_dim = 4 * config.dim
        self.fc1 = BitLinear(config.dim, hidden_dim, bias=False)
        self.fc2 = BitLinear(hidden_dim, config.dim, bias=False)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class AtomicBlock(nn.Module):
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.dim, eps=1e-5)
        self.attn = BitAttention(config)
        self.ln2 = nn.RMSNorm(config.dim, eps=1e-5)
        self.mlp = BitFeedForward(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class AtomicTransformer(nn.Module):
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.config = config
        
        # Embedding (Keep high precision)
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        # Simple learned pos emb
        self.pos_emb = nn.Embedding(config.context_length, config.dim)
        
        # Layers
        self.layers = nn.ModuleList([AtomicBlock(config) for _ in range(config.depth)])
        
        # Final Norm
        self.ln_f = nn.RMSNorm(config.dim, eps=1e-5)
        
        # Head (BitLinear for final projection to vocab)
        self.head = BitLinear(config.dim, config.vocab_size, bias=False)
        
    def forward(self, idx, gist_token=None):
        """
        idx: (Batch, Seq)
        gist_token: Optional (Batch, Dim) - The "Thought Vector" to prepend.
        """
        B, T = idx.shape
        
        # Check sequence length limit
        # If gist is present, total len is T + 1
        seq_len = T + (1 if gist_token is not None else 0)
        assert seq_len <= self.config.context_length, "Sequence too long"
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Inject Gist if present
        if gist_token is not None:
             # gist_token is (Batch, Dim) -> (Batch, 1, Dim)
             gist_emb = gist_token.unsqueeze(1)
             # Prepend: [Gist, Token1, Token2...]
             x = torch.cat([gist_emb, x], dim=1)
             # Note: Positional embeddings for Gist? 
             # Simpler to assume Gist takes Pos 0? Or is Pos Independent?
             # If Gist is "System Prompt", it conceptually sits at Pos -1 or 0.
             # Here we effectively shift user tokens to Pos 0..T (Wait, `pos` var above is 0..T).
             # So Gist has NO positional embedding added here (it's raw vector). 
             # That seems valid for a "Gist".
        
        # Transformer Blocks
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        
        # Logits
        # If gist was added, x is (Batch, T+1, Dim). Output logits should probably match user tokens?
        # Usually we want logits for the last token.
        logits = self.head(x)
        return logits
