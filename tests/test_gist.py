import pytest
import torch

from atomic_1bit.model.gist import GistEncoder, GistConfig
from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig


class TestGistConfig:
    def test_defaults(self):
        config = GistConfig()
        assert config.vocab_size == 50257
        assert config.dim == 512


class TestGistEncoder:
    def test_output_shape(self, gist_config):
        encoder = GistEncoder(gist_config)
        input_ids = torch.randint(0, gist_config.vocab_size, (2, 10))
        output = encoder(input_ids)
        assert output.shape == (2, gist_config.dim)

    def test_single_token(self, gist_config):
        encoder = GistEncoder(gist_config)
        input_ids = torch.randint(0, gist_config.vocab_size, (1, 1))
        output = encoder(input_ids)
        assert output.shape == (1, gist_config.dim)

    def test_variable_length_compression(self, gist_config):
        """Different length inputs should still produce same-shaped output."""
        encoder = GistEncoder(gist_config)
        short = torch.randint(0, gist_config.vocab_size, (1, 3))
        long = torch.randint(0, gist_config.vocab_size, (1, 20))
        out_short = encoder(short)
        out_long = encoder(long)
        assert out_short.shape == out_long.shape == (1, gist_config.dim)

    def test_different_inputs_different_outputs(self, gist_config):
        """Different inputs should generally produce different gist vectors."""
        encoder = GistEncoder(gist_config)
        x1 = torch.tensor([[0, 1, 2]])
        x2 = torch.tensor([[3, 4, 5]])
        with torch.no_grad():
            g1 = encoder(x1)
            g2 = encoder(x2)
        assert not torch.allclose(g1, g2)


class TestGistInjection:
    def test_gist_changes_output(self):
        """Injecting a gist token should change the model output."""
        config = AtomicConfig(vocab_size=64, dim=32, depth=2, heads=4, context_length=16)
        model = AtomicTransformer(config)
        model.eval()

        x = torch.randint(0, 64, (1, 8))
        gist = torch.randn(1, 32)

        with torch.no_grad():
            logits_no_gist = model(x)
            logits_with_gist = model(x, gist_token=gist)

        # Shapes differ (gist prepends 1 token)
        assert logits_no_gist.shape[1] == 8
        assert logits_with_gist.shape[1] == 9
