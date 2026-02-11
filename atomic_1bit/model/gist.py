import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class GistConfig:
    vocab_size: int = 50257
    dim: int = 512

class GistEncoder(nn.Module):
    """Encodes a token sequence into a single "gist" vector via mean pooling.

    Used to compress system prompts into a fixed-size embedding that can be
    injected into the attention stream at zero inference cost.
    """

    def __init__(self, config: GistConfig):
        super().__init__()
        self.embedding = nn.EmbeddingBag(config.vocab_size, config.dim, mode='mean')

    def forward(self, input_ids):
        """
        Args:
            input_ids: (Batch, Seq) token indices

        Returns:
            (Batch, Dim) mean-pooled embedding vector
        """
        return self.embedding(input_ids)
