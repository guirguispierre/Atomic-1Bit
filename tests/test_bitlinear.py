import pytest
import torch
import torch.nn as nn

from atomic_1bit.nn.layers import BitLinear, activation_quant, weight_quant


class TestActivationQuant:
    def test_output_range(self):
        """Quantized activations should be in [-127, 127]."""
        x = torch.randn(2, 4, 32)
        y, scale = activation_quant(x)
        assert y.max() <= 127.0
        assert y.min() >= -127.0

    def test_scale_positive(self):
        """Scale must always be positive."""
        x = torch.randn(2, 4, 32)
        _, scale = activation_quant(x)
        assert (scale > 0).all()

    def test_zero_input(self):
        """Zero input should produce zero output without error."""
        x = torch.zeros(1, 1, 16)
        y, scale = activation_quant(x)
        assert torch.allclose(y, torch.zeros_like(y))

    def test_ste_gradient_flows(self):
        """Straight-Through Estimator should allow gradients to flow."""
        x = torch.randn(2, 4, 32, requires_grad=True)
        y, scale = activation_quant(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestWeightQuant:
    def test_output_ternary(self):
        """Quantized weights should be approximately {-1, 0, 1}."""
        w = torch.randn(64, 32)
        y, scale = weight_quant(w)
        # After rounding, values should be close to -1, 0, or 1
        rounded = y.round()
        assert rounded.max() <= 1.0
        assert rounded.min() >= -1.0

    def test_scale_positive(self):
        """Scale must always be positive."""
        w = torch.randn(64, 32)
        _, scale = weight_quant(w)
        assert scale > 0

    def test_ste_gradient_flows(self):
        """Straight-Through Estimator should allow gradients to flow."""
        w = torch.randn(64, 32, requires_grad=True)
        y, scale = weight_quant(w)
        loss = y.sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape


class TestBitLinear:
    def test_output_shape(self):
        """Output shape should be (batch, seq, out_features)."""
        layer = BitLinear(32, 64)
        x = torch.randn(2, 8, 32)
        y = layer(x)
        assert y.shape == (2, 8, 64)

    def test_single_token(self):
        """Should work with single-token input."""
        layer = BitLinear(32, 64)
        x = torch.randn(1, 1, 32)
        y = layer(x)
        assert y.shape == (1, 1, 64)

    def test_no_bias_default(self):
        """Default should be no bias."""
        layer = BitLinear(32, 64)
        assert layer.bias is None

    def test_with_bias(self):
        """Should support optional bias."""
        layer = BitLinear(32, 64, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (64,)
        x = torch.randn(1, 4, 32)
        y = layer(x)
        assert y.shape == (1, 4, 64)

    def test_gradient_flow(self):
        """Gradients should flow through BitLinear during training."""
        layer = BitLinear(32, 64)
        x = torch.randn(2, 4, 32, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_weight_shape(self):
        """Weight should be (out_features, in_features)."""
        layer = BitLinear(32, 64)
        assert layer.weight.shape == (64, 32)

    def test_deterministic_eval(self):
        """Same input should produce same output in eval mode."""
        layer = BitLinear(32, 64)
        layer.eval()
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x)
        assert torch.allclose(y1, y2)
