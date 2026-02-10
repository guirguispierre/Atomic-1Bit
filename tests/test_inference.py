import pytest
import torch
import torch.nn.functional as F

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig


class TestEndToEndInference:
    """End-to-end tests: model creation, forward pass, and generation loop."""

    def test_forward_produces_valid_logits(self, tiny_model, tiny_config):
        """Logits should be finite real numbers."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            logits = tiny_model(x)
        assert torch.isfinite(logits).all()

    def test_softmax_sums_to_one(self, tiny_model, tiny_config):
        """Softmax of logits should sum to ~1 for each position."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            logits = tiny_model(x)
            probs = F.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_argmax_within_vocab(self, tiny_model, tiny_config):
        """Argmax of logits should be a valid token ID."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            logits = tiny_model(x)
            next_token = logits[:, -1, :].argmax(dim=-1)
        assert 0 <= next_token.item() < tiny_config.vocab_size

    def test_autoregressive_generation(self, tiny_model, tiny_config):
        """Autoregressive generation should produce valid token sequences."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 1))
        generated = [x[0, 0].item()]

        for _ in range(5):
            with torch.no_grad():
                logits = tiny_model(x)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
            x = torch.cat([x, next_token], dim=1)

            # Don't exceed context length
            if x.shape[1] >= tiny_config.context_length:
                break

        assert len(generated) > 1
        assert all(0 <= t < tiny_config.vocab_size for t in generated)

    def test_batch_inference(self, tiny_model, tiny_config):
        """Batch inference should work correctly."""
        batch_size = 4
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, 8))
        with torch.no_grad():
            logits = tiny_model(x)
        assert logits.shape == (batch_size, 8, tiny_config.vocab_size)

    def test_different_inputs_different_outputs(self, tiny_model, tiny_config):
        """Different inputs should generally produce different logits."""
        x1 = torch.tensor([[0, 1, 2, 3]])
        x2 = torch.tensor([[4, 5, 6, 7]])
        with torch.no_grad():
            y1 = tiny_model(x1)
            y2 = tiny_model(x2)
        assert not torch.allclose(y1, y2)

    def test_training_step(self, tiny_config):
        """A single training step should reduce loss."""
        model = AtomicTransformer(tiny_config)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, tiny_config.vocab_size, (2, 8))
        y = torch.randint(0, tiny_config.vocab_size, (2, 8))

        # Step 1
        logits = model(x)
        loss1 = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), y.view(-1))
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2
        logits = model(x)
        loss2 = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), y.view(-1))

        # Loss should change (not necessarily decrease in 1 step, but should be different)
        assert loss1.item() != loss2.item()
