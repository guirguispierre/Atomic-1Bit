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

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: (B, T, C) input tensor
            kv_cache: Optional tuple (cached_k, cached_v) each (B, Heads, T_prev, HeadDim)

        Returns:
            output: (B, T, C)
            new_kv_cache: tuple (k, v) for caching
        """
        B, T, C = x.shape

        # Project Q, K, V using BitLinear
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for Multi-Head Attention
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Append to KV cache if present
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)

        # Attention: Q is (B, H, T, D), K/V are (B, H, T_total, D)
        T_total = k.shape[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Causal Mask
        # When using KV-cache, Q has T new tokens, K/V have T_total tokens
        # Each new query at position i can attend to all positions up to T_prev + i
        mask = torch.ones(T, T_total, device=x.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=T_total - T + 1)
        att = att.masked_fill(mask, float('-inf'))

        att = torch.nn.functional.softmax(att, dim=-1)

        # Aggregate
        y = att @ v

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output Projection (BitLinear)
        return self.o_proj(y), new_kv_cache

class BitFeedForward(nn.Module):
    def __init__(self, config: AtomicConfig):
        super().__init__()
        # Expansion factor 4
        hidden_dim = 4 * config.dim
        self.fc1 = BitLinear(config.dim, hidden_dim, bias=False)
        self.fc2 = BitLinear(hidden_dim, config.dim, bias=False)
        # guirguispierre 2026-08-10 - must match the tanh gelu in
        # embedded/atomic_runner.cpp or python and c++ output diverge
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class AtomicBlock(nn.Module):
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.dim, eps=1e-5)
        self.attn = BitAttention(config)
        self.ln2 = nn.RMSNorm(config.dim, eps=1e-5)
        self.mlp = BitFeedForward(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv_cache

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

    def forward(self, idx, gist_token=None, kv_cache=None):
        """
        Args:
            idx: (Batch, Seq) token indices
            gist_token: Optional (Batch, Dim) - The "Thought Vector" to prepend.
            kv_cache: Optional list of (k, v) tuples per layer for cached inference.

        Returns:
            logits: (Batch, Seq, VocabSize)
            If kv_cache was provided, also returns new_kv_cache list.
        """
        B, T = idx.shape

        # Determine position offset from cache
        if kv_cache is not None and kv_cache[0] is not None:
            pos_offset = kv_cache[0][0].shape[2]  # T_prev from first layer's cached K
        else:
            pos_offset = 0

        # Check sequence length limit
        seq_len = T + pos_offset + (1 if gist_token is not None and pos_offset == 0 else 0)
        assert seq_len <= self.config.context_length, "Sequence too long"

        # Embeddings
        pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        # Inject Gist if present (only on first pass, not when using cache)
        if gist_token is not None and pos_offset == 0:
            gist_emb = gist_token.unsqueeze(1)
            x = torch.cat([gist_emb, x], dim=1)

        # Transformer Blocks
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = layer(x, kv_cache=layer_cache)
            new_kv_cache.append(new_cache)

        x = self.ln_f(x)

        # Logits
        logits = self.head(x)

        if kv_cache is not None:
            return logits, new_kv_cache
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, use_cache=True):
        """
        Autoregressive generation with optional KV-cache.

        Args:
            idx: (Batch, Seq) initial token indices
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            use_cache: Whether to use KV-cache for acceleration

        Returns:
            (Batch, Seq + max_new_tokens) generated token indices
        """
        self.eval()
        kv_cache = None

        if use_cache:
            # Initial pass: process all input tokens at once
            logits, kv_cache = self.forward(idx, kv_cache=[None] * self.config.depth)

            for _ in range(max_new_tokens):
                if idx.shape[1] >= self.config.context_length:
                    break

                # Get next token from last position
                last_logits = logits[:, -1, :]
                if temperature > 0:
                    probs = torch.nn.functional.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = last_logits.argmax(dim=-1, keepdim=True)

                idx = torch.cat([idx, next_token], dim=1)

                # Only process the new token using cache
                logits, kv_cache = self.forward(next_token, kv_cache=kv_cache)
        else:
            # No cache: recompute everything each step
            for _ in range(max_new_tokens):
                if idx.shape[1] >= self.config.context_length:
                    break

                logits = self.forward(idx)
                last_logits = logits[:, -1, :]
                if temperature > 0:
                    probs = torch.nn.functional.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = last_logits.argmax(dim=-1, keepdim=True)

                idx = torch.cat([idx, next_token], dim=1)

        return idx
