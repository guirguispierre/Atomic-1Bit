import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class GistConfig:
    vocab_size: int = 50257
    dim: int = 512

class GistEncoder(nn.Module):
    def __init__(self, config: GistConfig):
        super().__init__()
        # Simple averaging usage: mode='mean'
        # In a real scenario, this could be an LSTM or small Transformer.
        self.embedding = nn.EmbeddingBag(config.vocab_size, config.dim, mode='mean')
        
    def forward(self, input_ids):
        """
        input_ids: (Batch, Seq)
        Returns: (Batch, Dim)
        """
        # EmbeddingBag expects inputs differently depending on usage, 
        # but for fixed length batch, usually it needs 1D input + offsets 
        # OR 2D input.
        # If input is 2D (Batch, Seq), EmbeddingBag handles it if we don't pass offsets?
        # Actually EmbeddingBag with 2D input and no offsets works if include_last_offset=False?
        # Let's verify standard usage.
        # "If input is 2D of shape (B, N), it will be treated as B bags (sequences) each of fixed length N."
        return self.embedding(input_ids)
