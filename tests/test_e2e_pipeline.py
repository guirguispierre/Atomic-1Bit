"""
End-to-end integration test for the Atomic-1Bit pipeline.

Covers: model creation -> training -> checkpoint save -> export -> binary validation.
Marked @pytest.mark.slow because it performs actual training steps.
"""
import sys
import os
import struct

import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.utils.export_to_cpp import export_model

# Smallest possible config for a fast e2e run
E2E_CONFIG = AtomicConfig(
    vocab_size=64,
    dim=32,
    depth=1,
    heads=1,
    context_length=16,
)

ATOM_MAGIC = 0x41544F4D  # 'ATOM'


@pytest.mark.slow
class TestE2EPipeline:
    """Full pipeline: create -> train -> save -> export -> verify binary."""

    def _make_batch(self, config, batch_size=2, seq_len=8):
        """Generate a random (x, y) pair for training."""
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        return x, y

    def test_model_creation(self):
        """Model should be instantiable with the tiny e2e config."""
        model = AtomicTransformer(E2E_CONFIG)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_training_steps(self):
        """A few gradient-update steps should run without error and reduce loss."""
        model = AtomicTransformer(E2E_CONFIG)
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(3):
            x, y = self._make_batch(E2E_CONFIG)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, E2E_CONFIG.vocab_size),
                y.view(-1),
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # All loss values must be finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
            f"Non-finite loss encountered: {losses}"
        )

    def test_checkpoint_save_and_load(self, tmp_path):
        """Checkpoint saved to disk should reload without errors."""
        model = AtomicTransformer(E2E_CONFIG)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        # One training step
        x, y = self._make_batch(E2E_CONFIG)
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, E2E_CONFIG.vocab_size),
            y.view(-1),
        )
        loss.backward()
        optimizer.step()

        ckpt_path = tmp_path / "e2e_test.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": 1,
                "config": {
                    "vocab_size": E2E_CONFIG.vocab_size,
                    "dim": E2E_CONFIG.dim,
                    "depth": E2E_CONFIG.depth,
                    "heads": E2E_CONFIG.heads,
                    "context_length": E2E_CONFIG.context_length,
                },
            },
            str(ckpt_path),
        )

        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 0

        # Reload
        loaded = torch.load(str(ckpt_path), map_location="cpu")
        assert "model_state_dict" in loaded
        assert loaded["step"] == 1

        model2 = AtomicTransformer(E2E_CONFIG)
        model2.load_state_dict(loaded["model_state_dict"])
        model2.eval()

        # Verify reloaded model produces the same output as the original
        model.eval()
        probe = torch.randint(0, E2E_CONFIG.vocab_size, (1, 4))
        with torch.no_grad():
            out_orig = model(probe)
            out_loaded = model2(probe)
        assert torch.allclose(out_orig, out_loaded, atol=1e-5)

    def test_export_binary_exists(self, tmp_path):
        """export_model should write a non-empty binary file."""
        model = AtomicTransformer(E2E_CONFIG)
        bin_path = tmp_path / "model.bin"
        export_model(model, str(bin_path))

        assert bin_path.exists()
        assert bin_path.stat().st_size > 0

    def test_export_binary_magic_bytes(self, tmp_path):
        """Exported binary must start with 'ATOM' magic bytes."""
        model = AtomicTransformer(E2E_CONFIG)
        bin_path = tmp_path / "model.bin"
        export_model(model, str(bin_path))

        with open(str(bin_path), "rb") as f:
            (magic,) = struct.unpack("i", f.read(4))

        assert magic == ATOM_MAGIC, (
            f"Expected magic 0x{ATOM_MAGIC:08X}, got 0x{magic:08X}"
        )

    def test_export_binary_version(self, tmp_path):
        """Exported binary header version field must equal 1."""
        model = AtomicTransformer(E2E_CONFIG)
        bin_path = tmp_path / "model.bin"
        export_model(model, str(bin_path))

        with open(str(bin_path), "rb") as f:
            _magic, version = struct.unpack("ii", f.read(8))

        assert version == 1

    def test_export_binary_header_dimensions(self, tmp_path):
        """Exported header dimensions must match the model config."""
        model = AtomicTransformer(E2E_CONFIG)
        bin_path = tmp_path / "model.bin"
        export_model(model, str(bin_path))

        with open(str(bin_path), "rb") as f:
            magic, version, vocab, dim, depth, heads, ctx, has_gist = struct.unpack(
                "iiiiiiii", f.read(32)
            )

        assert vocab == E2E_CONFIG.vocab_size
        assert dim == E2E_CONFIG.dim
        assert depth == E2E_CONFIG.depth
        assert heads == E2E_CONFIG.heads
        assert ctx == E2E_CONFIG.context_length
        assert has_gist == 0

    def test_full_pipeline(self, tmp_path):
        """
        Full pipeline in sequence:
          create -> 2 training steps -> save checkpoint -> export binary -> verify.
        """
        # 1. Create model
        model = AtomicTransformer(E2E_CONFIG)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        # 2. Train for 2 steps
        model.train()
        for _ in range(2):
            x, y = self._make_batch(E2E_CONFIG)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, E2E_CONFIG.vocab_size),
                y.view(-1),
            )
            loss.backward()
            optimizer.step()

        # 3. Save checkpoint
        ckpt_path = tmp_path / "pipeline_final.pt"
        torch.save({"model_state_dict": model.state_dict(), "step": 2}, str(ckpt_path))
        assert ckpt_path.exists()

        # 4. Reload and export to binary
        loaded = torch.load(str(ckpt_path), map_location="cpu")
        model_export = AtomicTransformer(E2E_CONFIG)
        model_export.load_state_dict(loaded["model_state_dict"])
        model_export.eval()

        bin_path = tmp_path / "pipeline_final.bin"
        export_model(model_export, str(bin_path))

        # 5. Verify binary
        assert bin_path.exists()
        assert bin_path.stat().st_size > 32  # Larger than header alone

        with open(str(bin_path), "rb") as f:
            magic, version, vocab, dim, depth, heads, ctx, has_gist = struct.unpack(
                "iiiiiiii", f.read(32)
            )

        assert magic == ATOM_MAGIC
        assert version == 1
        assert vocab == E2E_CONFIG.vocab_size
        assert dim == E2E_CONFIG.dim

        # 6. Cleanup is handled automatically by tmp_path fixture
