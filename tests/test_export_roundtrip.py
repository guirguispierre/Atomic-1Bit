"""End-to-end format check: exporter output matches the C++ loader's expectation.

The C++ loader (embedded/atomic_lib.h, embedded/atomic_runner.cpp) reads:
  header   : 8 x int32  (magic, version, vocab, dim, depth, heads, ctx, has_gist)
  if has_gist: dim x float32
  token_emb: vocab*dim x float32
  pos_emb  : ctx*dim x float32
  per layer:
    ln1:        dim x float32
    q/k/v/o:    each prefixed by float32 scale, then dim*dim x int8
    ln2:        dim x float32
    fc1:        float32 scale + (dim * 4*dim) int8
    fc2:        float32 scale + (4*dim * dim) int8
  ln_f:       dim x float32
  head:       float32 scale + (dim * vocab) int8

This test parses an exported binary using exactly that layout and asserts
that every field arrives where the C++ loader expects it. If the exporter
or loader drift again, this test catches the drift.
"""
import os
import struct
import tempfile
import math

import pytest
import torch

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.utils.export_to_cpp import export_model


def _read_floats(f, n):
    return struct.unpack(f"{n}f", f.read(4 * n))


def _read_int8s(f, n):
    return f.read(n)


def parse_binary(path):
    """Parse the exported .bin using the C++ loader's expected layout.

    Returns a dict so individual assertions can target specific fields.
    Raises on any layout mismatch (truncated reads, wrong sizes).
    """
    out = {}
    with open(path, "rb") as f:
        header = f.read(32)
        assert len(header) == 32, "header truncated"
        magic, version, vocab, dim, depth, heads, ctx, has_gist = struct.unpack("iiiiiiii", header)
        out["magic"] = magic
        out["version"] = version
        out["vocab"] = vocab
        out["dim"] = dim
        out["depth"] = depth
        out["heads"] = heads
        out["ctx"] = ctx
        out["has_gist"] = has_gist

        if has_gist:
            out["gist"] = _read_floats(f, dim)

        out["token_emb"] = _read_floats(f, vocab * dim)
        out["pos_emb"] = _read_floats(f, ctx * dim)

        layers = []
        for i in range(depth):
            layer = {}
            layer["ln1"] = _read_floats(f, dim)

            for name in ("q", "k", "v", "o"):
                (scale,) = struct.unpack("f", f.read(4))
                weights = _read_int8s(f, dim * dim)
                assert len(weights) == dim * dim, f"layer {i} {name}_w truncated"
                layer[f"{name}_scale"] = scale
                layer[f"{name}_w_len"] = len(weights)

            layer["ln2"] = _read_floats(f, dim)

            (fc1_scale,) = struct.unpack("f", f.read(4))
            fc1_w = _read_int8s(f, dim * 4 * dim)
            assert len(fc1_w) == dim * 4 * dim, f"layer {i} fc1_w truncated"
            layer["fc1_scale"] = fc1_scale
            layer["fc1_w_len"] = len(fc1_w)

            (fc2_scale,) = struct.unpack("f", f.read(4))
            fc2_w = _read_int8s(f, 4 * dim * dim)
            assert len(fc2_w) == 4 * dim * dim, f"layer {i} fc2_w truncated"
            layer["fc2_scale"] = fc2_scale
            layer["fc2_w_len"] = len(fc2_w)

            layers.append(layer)
        out["layers"] = layers

        out["ln_f"] = _read_floats(f, dim)

        (head_scale,) = struct.unpack("f", f.read(4))
        head_w = _read_int8s(f, dim * vocab)
        assert len(head_w) == dim * vocab, "head_w truncated"
        out["head_scale"] = head_scale
        out["head_w_len"] = len(head_w)

        # Make sure there are no unexpected trailing bytes; the C++ loader
        # would silently leave them, but their presence means the writer
        # produced something the loader doesn't know about.
        trailing = f.read()
        out["trailing_bytes"] = len(trailing)

    return out


@pytest.fixture
def tmp_bin():
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _make_model():
    cfg = AtomicConfig(vocab_size=32, dim=16, depth=2, heads=4, context_length=8)
    torch.manual_seed(0)
    return AtomicTransformer(cfg), cfg


def test_layout_matches_loader(tmp_bin):
    """Every byte the loader expects is present in the right place."""
    model, cfg = _make_model()
    export_model(model, tmp_bin)
    parsed = parse_binary(tmp_bin)

    assert parsed["magic"] == 0x41544F4D
    assert parsed["version"] == 1
    assert parsed["vocab"] == cfg.vocab_size
    assert parsed["dim"] == cfg.dim
    assert parsed["depth"] == cfg.depth
    assert parsed["heads"] == cfg.heads
    assert parsed["ctx"] == cfg.context_length
    assert parsed["has_gist"] == 0
    assert parsed["trailing_bytes"] == 0


def test_per_tensor_scales_are_finite_and_positive(tmp_bin):
    """Quantized-tensor scales must be finite, positive, and not the magic
    number / version field (a common drift symptom).
    """
    model, _ = _make_model()
    export_model(model, tmp_bin)
    parsed = parse_binary(tmp_bin)

    suspicious = {0x41544F4D, 1}
    for i, layer in enumerate(parsed["layers"]):
        for key in ("q_scale", "k_scale", "v_scale", "o_scale", "fc1_scale", "fc2_scale"):
            s = layer[key]
            assert math.isfinite(s), f"layer {i} {key} not finite: {s}"
            assert s > 0, f"layer {i} {key} non-positive: {s}"
            # If we accidentally read magic-as-scale, this catches it.
            assert int(s) not in suspicious, f"layer {i} {key} looks like a header field"

    assert math.isfinite(parsed["head_scale"])
    assert parsed["head_scale"] > 0


def test_byte_count_matches_expected(tmp_bin):
    """The total bytes written must equal the loader's read budget exactly."""
    model, cfg = _make_model()
    export_model(model, tmp_bin)

    actual = os.path.getsize(tmp_bin)
    d = cfg.dim
    v = cfg.vocab_size
    L = cfg.depth
    ctx = cfg.context_length
    hidden = 4 * d

    # Header
    expected = 32
    # token_emb + pos_emb (float32)
    expected += (v * d + ctx * d) * 4
    # Per layer
    per_layer = (
        d * 4                    # ln1
        + 4 * (4 + d * d)        # q/k/v/o: scale + int8 weights
        + d * 4                  # ln2
        + (4 + d * hidden)       # fc1
        + (4 + hidden * d)       # fc2
    )
    expected += L * per_layer
    # ln_f + head
    expected += d * 4 + (4 + d * v)

    assert actual == expected, (
        f"file size {actual} != computed {expected}; "
        "exporter and loader format are out of sync"
    )
