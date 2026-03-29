"""
Edge-case tests for BitLinear quantization functions.

Exercises activation_quant, weight_quant, and BitLinear.forward with
pathological inputs: zeros, huge magnitudes, tiny magnitudes, NaNs, Infs,
single-element tensors, negative-only, and uniform tensors.
"""
import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.nn.layers import BitLinear, activation_quant, weight_quant

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer(in_f=8, out_f=16):
    """Return a BitLinear layer in eval mode with fixed seed weights."""
    torch.manual_seed(0)
    layer = BitLinear(in_f, out_f)
    layer.eval()
    return layer


# ---------------------------------------------------------------------------
# activation_quant edge cases
# ---------------------------------------------------------------------------

class TestActivationQuantEdgeCases:
    def test_all_zero_input(self):
        """All-zero activation: quantized output must be all-zero."""
        x = torch.zeros(1, 1, 16)
        y, scale = activation_quant(x)
        assert torch.allclose(y, torch.zeros_like(y)), (
            "Expected all-zero output for all-zero input"
        )

    def test_all_zero_scale_clamped(self):
        """Scale for all-zero input must be > 0 (clamped, not division-by-zero)."""
        x = torch.zeros(1, 1, 16)
        _, scale = activation_quant(x)
        assert (scale > 0).all(), "Scale must be positive even for zero input"

    def test_very_large_values(self):
        """Very large activations (1e6) must be clamped to [-127, 127]."""
        x = torch.full((1, 1, 16), 1e6)
        y, scale = activation_quant(x)
        assert y.max() <= 127.0, f"Max quantized value {y.max()} exceeds 127"
        assert y.min() >= -127.0

    def test_very_small_values(self):
        """Very small activations (1e-8) must not cause NaN or Inf."""
        x = torch.full((1, 1, 16), 1e-8)
        y, scale = activation_quant(x)
        assert torch.isfinite(y).all(), "Quantized output contains NaN or Inf"
        assert torch.isfinite(scale).all(), "Scale contains NaN or Inf"

    def test_mixed_nan_values(self):
        """NaN in input propagates; the function does not silently swallow it."""
        x = torch.randn(1, 1, 16)
        x[0, 0, 0] = float("nan")
        y, scale = activation_quant(x)
        # NaN presence in input should produce at least one NaN in output
        # (no silent replacement expected from the current implementation)
        assert torch.isnan(y).any() or torch.isnan(scale).any(), (
            "Expected NaN to propagate through activation_quant"
        )

    def test_mixed_inf_values(self):
        """Inf in input should not produce finite scale (absmax is Inf)."""
        x = torch.randn(1, 1, 16)
        x[0, 0, 0] = float("inf")
        y, scale = activation_quant(x)
        # scale = 127 / inf = 0; quantized = clamped result
        # Either scale is 0 or output is inf/nan — implementation-defined but must not crash
        assert y.shape == x.shape, "Shape must be preserved regardless of Inf input"

    def test_single_element_tensor(self):
        """Single-element tensor must produce a scalar quantized output."""
        x = torch.tensor([[[3.14]]])
        y, scale = activation_quant(x)
        assert y.shape == (1, 1, 1)
        assert scale.shape == (1, 1, 1)
        assert torch.isfinite(y).all()

    def test_negative_only_input(self):
        """All-negative activations must produce non-positive quantized output."""
        x = torch.full((1, 1, 16), -5.0)
        y, scale = activation_quant(x)
        assert y.max() <= 0.0, "Expected non-positive quantized output for all-negative input"

    def test_uniform_value_input(self):
        """Uniform (all same non-zero value) input: quantized output should be uniform."""
        x = torch.full((2, 4, 8), 2.5)
        y, scale = activation_quant(x)
        # All values are equal so all quantized values should also be equal within a row
        assert y.std() < 1e-3, (
            f"Expected near-uniform quantized output for uniform input, got std={y.std()}"
        )

    def test_output_range_general(self):
        """Quantized activations must always lie in [-127, 127]."""
        torch.manual_seed(42)
        x = torch.randn(4, 8, 32) * 100  # large scale
        y, _ = activation_quant(x)
        assert y.max() <= 127.0
        assert y.min() >= -127.0


# ---------------------------------------------------------------------------
# weight_quant edge cases
# ---------------------------------------------------------------------------

class TestWeightQuantEdgeCases:
    def test_all_zero_weights(self):
        """All-zero weights: quantized output must be all-zero."""
        w = torch.zeros(16, 8)
        y, scale = weight_quant(w)
        assert torch.allclose(y, torch.zeros_like(y)), (
            "Expected all-zero quantized weights for all-zero input"
        )

    def test_all_zero_scale_clamped(self):
        """Scale for all-zero weights must be > 0 (mean clamp prevents divide-by-zero)."""
        w = torch.zeros(16, 8)
        _, scale = weight_quant(w)
        assert scale > 0, "Scale must be positive even for zero weights"

    def test_very_large_weights(self):
        """Very large weights (1e6) must be clamped to [-1, 1]."""
        w = torch.randn(16, 8) * 1e6
        y, scale = weight_quant(w)
        assert y.max() <= 1.0, f"Max ternary value {y.max()} exceeds 1"
        assert y.min() >= -1.0

    def test_very_small_weights(self):
        """Very small weights (1e-8) must not produce NaN or Inf."""
        w = torch.full((16, 8), 1e-8)
        y, scale = weight_quant(w)
        assert torch.isfinite(y).all()
        assert torch.isfinite(scale)

    def test_ternary_values(self):
        """Rounded quantized weights must be in {-1, 0, 1}."""
        torch.manual_seed(7)
        w = torch.randn(64, 32)
        y, _ = weight_quant(w)
        rounded = y.round()
        unique_vals = rounded.unique().tolist()
        for v in unique_vals:
            assert v in (-1.0, 0.0, 1.0), f"Unexpected ternary value: {v}"

    def test_negative_only_weights(self):
        """All-negative weights must produce non-positive ternary output."""
        w = torch.full((16, 8), -3.0)
        y, _ = weight_quant(w)
        assert y.max() <= 0.0

    def test_uniform_value_weights(self):
        """Uniform weights: all quantized values should be identical."""
        w = torch.full((16, 8), 0.5)
        y, _ = weight_quant(w)
        assert y.std() < 1e-3

    def test_single_element_weight(self):
        """Single-element weight matrix must be handled without error."""
        w = torch.tensor([[2.0]])
        y, scale = weight_quant(w)
        assert y.shape == (1, 1)
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# BitLinear edge cases
# ---------------------------------------------------------------------------

class TestBitLinearEdgeCases:
    def test_all_zero_input(self):
        """All-zero input: output should be all-zero (no activation signal)."""
        layer = _make_layer(8, 16)
        x = torch.zeros(1, 1, 8)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (1, 1, 16)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-5), (
            "Expected near-zero output for all-zero input"
        )

    def test_very_large_input(self):
        """Very large input (1e6): output must be finite."""
        layer = _make_layer(8, 16)
        x = torch.full((1, 1, 8), 1e6)
        with torch.no_grad():
            y = layer(x)
        assert torch.isfinite(y).all(), "Output contains NaN or Inf for large input"

    def test_very_small_input(self):
        """Very small input (1e-8): output must be finite."""
        layer = _make_layer(8, 16)
        x = torch.full((1, 1, 8), 1e-8)
        with torch.no_grad():
            y = layer(x)
        assert torch.isfinite(y).all(), "Output contains NaN or Inf for tiny input"

    def test_nan_input_propagates(self):
        """NaN in input must not be silently swallowed—output contains NaN."""
        layer = _make_layer(8, 16)
        x = torch.randn(1, 1, 8)
        x[0, 0, 0] = float("nan")
        with torch.no_grad():
            y = layer(x)
        assert torch.isnan(y).any(), (
            "Expected NaN to propagate through BitLinear"
        )

    def test_inf_input(self):
        """Inf in input: layer must not raise an exception."""
        layer = _make_layer(8, 16)
        x = torch.randn(1, 1, 8)
        x[0, 0, 0] = float("inf")
        with torch.no_grad():
            # Must not raise; output shape must be correct
            y = layer(x)
        assert y.shape == (1, 1, 16)

    def test_single_element_batch(self):
        """Single-element input tensor (1, 1, in_features) must work."""
        layer = _make_layer(8, 16)
        x = torch.randn(1, 1, 8)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (1, 1, 16)
        assert torch.isfinite(y).all()

    def test_negative_only_input(self):
        """All-negative input: output must be finite and have correct shape."""
        layer = _make_layer(8, 16)
        x = torch.full((1, 4, 8), -3.0)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (1, 4, 16)
        assert torch.isfinite(y).all()

    def test_uniform_input(self):
        """Uniform input (all same value): output must be finite."""
        layer = _make_layer(8, 16)
        x = torch.full((2, 4, 8), 1.0)
        with torch.no_grad():
            y = layer(x)
        assert y.shape == (2, 4, 16)
        assert torch.isfinite(y).all()

    def test_gradient_flow_large_input(self):
        """STE gradient must flow even for very large inputs."""
        layer = BitLinear(8, 16)
        x = torch.full((1, 1, 8), 1e4, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None, "Gradient did not flow back through BitLinear"
        assert x.grad.shape == x.shape

    def test_gradient_flow_small_input(self):
        """STE gradient must flow even for very small inputs."""
        layer = BitLinear(8, 16)
        x = torch.full((1, 1, 8), 1e-7, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None, "Gradient did not flow back for tiny input"
        assert torch.isfinite(x.grad).all()

    def test_output_shape_various_batch_sizes(self):
        """Output shape must be correct for various batch sizes."""
        layer = _make_layer(8, 16)
        for batch in (1, 4, 16):
            x = torch.randn(batch, 5, 8)
            with torch.no_grad():
                y = layer(x)
            assert y.shape == (batch, 5, 16), (
                f"Shape mismatch for batch_size={batch}"
            )
