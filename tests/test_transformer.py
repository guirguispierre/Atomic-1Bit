import pytest
import torch

from atomic_1bit.model.transformer import (
    AtomicTransformer,
    AtomicConfig,
    BitAttention,
    BitFeedForward,
    AtomicBlock,
)


class TestAtomicConfig:
    def test_defaults(self):
        config = AtomicConfig()
        assert config.vocab_size == 50257
        assert config.dim == 512
        assert config.depth == 8
        assert config.heads == 8
        assert config.context_length == 1024

    def test_custom_config(self):
        config = AtomicConfig(vocab_size=4096, dim=320, depth=8, heads=5, context_length=256)
        assert config.vocab_size == 4096
        assert config.dim == 320


class TestBitAttention:
    def test_output_shape(self, tiny_config):
        attn = BitAttention(tiny_config)
        x = torch.randn(2, 8, tiny_config.dim)
        y, _ = attn(x)
        assert y.shape == x.shape

    def test_single_token(self, tiny_config):
        attn = BitAttention(tiny_config)
        x = torch.randn(1, 1, tiny_config.dim)
        y, _ = attn(x)
        assert y.shape == (1, 1, tiny_config.dim)

    def test_causal_mask(self, tiny_config):
        """Attention should be causal (no future information leakage)."""
        attn = BitAttention(tiny_config)
        attn.eval()
        x = torch.randn(1, 4, tiny_config.dim)
        with torch.no_grad():
            y_full, _ = attn(x)
            y_partial, _ = attn(x[:, :1, :])
        assert torch.allclose(y_full[:, 0, :], y_partial[:, 0, :], atol=1e-5)

    def test_kv_cache(self, tiny_config):
        """KV-cache should produce same output as full forward pass."""
        attn = BitAttention(tiny_config)
        attn.eval()
        x = torch.randn(1, 4, tiny_config.dim)
        with torch.no_grad():
            y_full, _ = attn(x)
            # Process first 3 tokens, cache, then process 4th
            y_prefix, cache = attn(x[:, :3, :])
            y_last, _ = attn(x[:, 3:4, :], kv_cache=cache)
        assert torch.allclose(y_full[:, 3, :], y_last[:, 0, :], atol=1e-5)


class TestBitFeedForward:
    def test_output_shape(self, tiny_config):
        ff = BitFeedForward(tiny_config)
        x = torch.randn(2, 8, tiny_config.dim)
        y = ff(x)
        assert y.shape == x.shape

    def test_expansion_factor(self, tiny_config):
        """Hidden dim should be 4x model dim."""
        ff = BitFeedForward(tiny_config)
        assert ff.fc1.out_features == 4 * tiny_config.dim
        assert ff.fc2.in_features == 4 * tiny_config.dim


class TestAtomicBlock:
    def test_output_shape(self, tiny_config):
        block = AtomicBlock(tiny_config)
        x = torch.randn(2, 8, tiny_config.dim)
        y, _ = block(x)
        assert y.shape == x.shape

    def test_residual_connection(self, tiny_config):
        """Output should differ from input (residual adds attention/mlp)."""
        block = AtomicBlock(tiny_config)
        x = torch.randn(2, 8, tiny_config.dim)
        with torch.no_grad():
            y, _ = block(x)
        assert not torch.allclose(x, y)


class TestAtomicTransformer:
    def test_forward_shape(self, tiny_config, sample_input):
        model = AtomicTransformer(tiny_config)
        model.eval()
        with torch.no_grad():
            logits = model(sample_input)
        B, T = sample_input.shape
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_single_token_inference(self, tiny_model, tiny_config):
        x = torch.tensor([[0]])
        with torch.no_grad():
            logits = tiny_model(x)
        assert logits.shape == (1, 1, tiny_config.vocab_size)

    def test_parameter_count(self, tiny_config):
        model = AtomicTransformer(tiny_config)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_context_length_limit(self, tiny_model, tiny_config):
        """Should raise error if input exceeds context length."""
        x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length + 1))
        with pytest.raises(AssertionError):
            tiny_model(x)

    def test_max_context_length(self, tiny_model, tiny_config):
        """Should accept input exactly at context length."""
        x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length))
        with torch.no_grad():
            logits = tiny_model(x)
        assert logits.shape == (1, tiny_config.context_length, tiny_config.vocab_size)

    def test_with_gist_token(self, tiny_model, tiny_config):
        """Forward pass with gist token should increase sequence length by 1."""
        x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length - 1))
        gist = torch.randn(1, tiny_config.dim)
        with torch.no_grad():
            logits = tiny_model(x, gist_token=gist)
        # Output should have T+1 tokens (gist prepended)
        assert logits.shape == (1, tiny_config.context_length, tiny_config.vocab_size)

    def test_deterministic_eval(self, tiny_model, tiny_config):
        """Same input should produce same output in eval mode."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            y1 = tiny_model(x)
            y2 = tiny_model(x)
        assert torch.allclose(y1, y2)

    def test_flagship_config(self):
        """Verify flagship 12.5M model config is valid."""
        config = AtomicConfig(
            vocab_size=4096, dim=320, depth=8, heads=5, context_length=256
        )
        model = AtomicTransformer(config)
        total_params = sum(p.numel() for p in model.parameters())
        # Should be roughly 12.5M params
        assert 10_000_000 < total_params < 15_000_000

    def test_generate_with_cache(self, tiny_model, tiny_config):
        """KV-cache generation should produce valid tokens."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 2))
        result = tiny_model.generate(x, max_new_tokens=5, temperature=0, use_cache=True)
        assert result.shape[1] == 7  # 2 input + 5 generated
        assert (result >= 0).all() and (result < tiny_config.vocab_size).all()

    def test_generate_without_cache(self, tiny_model, tiny_config):
        """Non-cached generation should produce valid tokens."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 2))
        result = tiny_model.generate(x, max_new_tokens=5, temperature=0, use_cache=False)
        assert result.shape[1] == 7
        assert (result >= 0).all() and (result < tiny_config.vocab_size).all()

    def test_cache_matches_no_cache(self, tiny_model, tiny_config):
        """Cached and non-cached generation should produce identical greedy outputs."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 2))
        cached = tiny_model.generate(x.clone(), max_new_tokens=5, temperature=0, use_cache=True)
        no_cache = tiny_model.generate(x.clone(), max_new_tokens=5, temperature=0, use_cache=False)
        assert torch.equal(cached, no_cache)
