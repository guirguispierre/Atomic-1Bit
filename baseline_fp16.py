import torch
import torch.nn as nn
import torch.nn.functional as F

class FP16Config:
    def __init__(self, vocab_size=2048, dim=128, depth=4, heads=4, context_length=64):
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.context_length = context_length

class FP16Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.context_length, config.dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(config.dim),
                "attn": nn.MultiheadAttention(config.dim, config.heads, batch_first=True),
                "ln2": nn.LayerNorm(config.dim),
                "mlp": nn.Sequential(
                    nn.Linear(config.dim, 4 * config.dim),
                    nn.GELU(),
                    nn.Linear(4 * config.dim, config.dim)
                )
            }) for _ in range(config.depth)
        ])
        
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device)
        
        x = self.token_emb(x) + self.pos_emb(pos)
        
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        for layer in self.layers:
            # LN1 + Attn
            normed = layer["ln1"](x)
            attn_out, _ = layer["attn"](normed, normed, normed, attn_mask=mask, need_weights=False)
            x = x + attn_out
            
            # LN2 + MLP
            normed = layer["ln2"](x)
            mlp_out = layer["mlp"](normed)
            x = x + mlp_out
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
