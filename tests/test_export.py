import pytest
import struct
import os
import tempfile
import torch

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.utils.export_to_cpp import export_model


class TestExportModel:
    def test_export_creates_file(self, tiny_model):
        """Export should create a binary file."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            export_model(tiny_model, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_atom_header(self, tiny_model):
        """Binary should start with ATOM magic bytes and version."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            export_model(tiny_model, path)
            with open(path, "rb") as f:
                magic, version = struct.unpack("ii", f.read(8))
            assert magic == 0x41544F4D  # 'ATOM'
            assert version == 1
        finally:
            os.unlink(path)

    def test_header_dimensions(self, tiny_model, tiny_config):
        """Header should contain correct model dimensions."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            export_model(tiny_model, path)
            with open(path, "rb") as f:
                vals = struct.unpack("iiiiiiii", f.read(32))
            magic, version, vocab, dim, depth, heads, ctx, has_gist = vals
            assert vocab == tiny_config.vocab_size
            assert dim == tiny_config.dim
            assert depth == tiny_config.depth
            assert heads == tiny_config.heads
            assert ctx == tiny_config.context_length
            assert has_gist == 0
        finally:
            os.unlink(path)

    def test_no_gist_by_default(self, tiny_model):
        """Export without gist should set has_gist=0."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            export_model(tiny_model, path)
            with open(path, "rb") as f:
                vals = struct.unpack("iiiiiiii", f.read(32))
            assert vals[7] == 0  # has_gist
        finally:
            os.unlink(path)

    def test_file_size_reasonable(self, tiny_config):
        """Exported file size should be reasonable for model size."""
        model = AtomicTransformer(tiny_config)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            export_model(model, path)
            file_size = os.path.getsize(path)
            # Should be > header size (32 bytes) and < 10MB for tiny model
            assert file_size > 32
            assert file_size < 10 * 1024 * 1024
        finally:
            os.unlink(path)

    def test_different_configs_different_sizes(self):
        """Larger model should produce larger binary."""
        small = AtomicConfig(vocab_size=64, dim=32, depth=2, heads=4, context_length=16)
        large = AtomicConfig(vocab_size=256, dim=64, depth=4, heads=4, context_length=32)
        model_small = AtomicTransformer(small)
        model_large = AtomicTransformer(large)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f2:
            path1, path2 = f1.name, f2.name
        try:
            export_model(model_small, path1)
            export_model(model_large, path2)
            assert os.path.getsize(path2) > os.path.getsize(path1)
        finally:
            os.unlink(path1)
            os.unlink(path2)
