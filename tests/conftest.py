import sys
import os
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.model.gist import GistEncoder, GistConfig
from atomic_1bit.nn.layers import BitLinear


@pytest.fixture
def tiny_config():
    """Minimal config for fast unit tests."""
    return AtomicConfig(
        vocab_size=64,
        dim=32,
        depth=2,
        heads=4,
        context_length=16,
    )


@pytest.fixture
def small_config():
    """Slightly larger config for integration tests."""
    return AtomicConfig(
        vocab_size=256,
        dim=64,
        depth=4,
        heads=4,
        context_length=32,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """A small AtomicTransformer for testing."""
    model = AtomicTransformer(tiny_config)
    model.eval()
    return model


@pytest.fixture
def small_model(small_config):
    """A slightly larger AtomicTransformer for testing."""
    model = AtomicTransformer(small_config)
    model.eval()
    return model


@pytest.fixture
def sample_input(tiny_config):
    """Sample input token IDs for tiny model."""
    batch_size = 2
    seq_len = 8
    return torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))


@pytest.fixture
def gist_config():
    """Config for GistEncoder testing."""
    return GistConfig(vocab_size=64, dim=32)
