"""
Unified tokenizer interface for Atomic-1Bit.

Decouples the model from specific tokenizer implementations,
making it easy to experiment with different vocabularies.
"""
import json
import os
from typing import List
from abc import ABC, abstractmethod


class AtomicTokenizer(ABC):
    """Abstract base class for all Atomic-1Bit tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...

    @property
    @abstractmethod
    def eot_token(self) -> int:
        """End-of-text token ID."""
        ...


class TiktokenWrapper(AtomicTokenizer):
    """Wrapper around tiktoken (GPT-2 encoding, 50257 vocab)."""

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)

    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token


class PocketTokenizer(AtomicTokenizer):
    """
    Frequency-filtered vocabulary tokenizer (e.g., 4096 vocab).

    Maps GPT-2 token IDs to a compact pocket vocabulary built from
    the most frequent tokens in the training data.
    """

    def __init__(self, vocab_file: str, unk_id: int = None):
        """
        Args:
            vocab_file: Path to vocab_map JSON file with token_map and reverse_map
            unk_id: Token ID for unknown tokens (defaults to vocab_size - 1)
        """
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        with open(vocab_file, 'r') as f:
            data = json.load(f)

        self._token_map = {int(k): v for k, v in data["token_map"].items()}
        self._reverse_map = {int(k): v for k, v in data["reverse_map"].items()}
        self._vocab_size = max(self._reverse_map.keys()) + 2  # +1 for UNK, +1 for 0-indexed
        self._unk_id = unk_id if unk_id is not None else self._vocab_size - 1

    def encode(self, text: str) -> List[int]:
        gpt_ids = self._enc.encode(text)
        return [self._token_map.get(gid, self._unk_id) for gid in gpt_ids]

    def decode(self, tokens: List[int]) -> str:
        gpt_ids = [self._reverse_map.get(t, self._enc.eot_token) for t in tokens]
        return self._enc.decode(gpt_ids)

    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eot_token(self) -> int:
        return self._token_map.get(self._enc.eot_token, self._unk_id)

    @property
    def unk_token(self) -> int:
        return self._unk_id
