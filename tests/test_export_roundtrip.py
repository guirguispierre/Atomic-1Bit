"""End-to-end format check: exporter output matches the C++ loader's expectation.

The C++ loader (embedded/atomic_lib.h, embedded/atomic_runner.cpp) reads:
  header   : 8 x int32  (magic, version, vocab, dim, depth, heads, ctx, has_gist)
  if has_gist: dim x float32
  token_emb: v3 -> vocab x float32 scales + vocab*dim x int8 codes
             v2 -> vocab*dim x float32
  pos_emb  : v3 -> ctx x float32 scales + ctx*dim x int8 codes
             v2 -> ctx*dim x float32
  per layer:
    ln1:        dim x float32
    q/k/v/o:    each prefixed by float32 scale, then ceil(dim*dim / 4) bytes
    ln2:        dim x float32
    fc1:        float32 scale + ceil(dim * 4*dim / 4) bytes
    fc2:        float32 scale + ceil(4*dim * dim / 4) bytes
  ln_f:       dim x float32
  head:       float32 scale + ceil(dim * vocab / 4) bytes

Ternary weights are packed four per byte as w + 1 (format version 2 onward).
Format version 3 quantizes each embedding row to int8 with a per-row float32
absmax scale; loaders dequantize to float32 at load time. Both versions are
still accepted by every loader, so this file parses and checks each.

This test parses an exported binary using exactly that layout and asserts
that every field arrives where the C++ loader expects it. If the exporter
or loader drift again, this test catches the drift.
"""
import os
import struct
import tempfile
import math

import numpy as np
import pytest
import torch

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.utils.export_to_cpp import (
    export_model,
    pack_ternary,
    unpack_ternary,
    quantize_embedding_int8,
    dequantize_embedding_int8,
)


def _read_floats(f, n):
    return struct.unpack(f"{n}f", f.read(4 * n))


def _packed_bytes(n):
    return (n + 3) // 4


def _read_packed(f, n):
    return f.read(_packed_bytes(n))


def _read_embedding(f, rows, dim, version):
    """Read one embedding block the way the C++ loaders do.

    v3: rows float32 absmax scales, then rows*dim int8 codes; the loader
    dequantizes to float32 at load time. v2: rows*dim float32 raw.
    Returns (scales, codes, floats); the unused entries are None.
    """
    if version >= 3:
        scales = np.frombuffer(f.read(4 * rows), dtype=np.float32)
        assert scales.size == rows, "embedding scales truncated"
        codes = np.frombuffer(f.read(rows * dim), dtype=np.int8)
        assert codes.size == rows * dim, "embedding codes truncated"
        deq = dequantize_embedding_int8(scales, codes.reshape(rows, dim))
        return scales, codes, deq.reshape(-1)
    return None, None, _read_floats(f, rows * dim)


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
        assert 2 <= version <= 3, f"unexpected format version {version}"
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

        tok_scales, tok_codes, tok_emb = _read_embedding(f, vocab, dim, version)
        pos_scales, pos_codes, pos_emb = _read_embedding(f, ctx, dim, version)
        out["token_emb"] = tok_emb
        out["pos_emb"] = pos_emb
        out["token_emb_scales"] = tok_scales
        out["token_emb_codes"] = tok_codes
        out["pos_emb_scales"] = pos_scales
        out["pos_emb_codes"] = pos_codes

        layers = []
        for i in range(depth):
            layer = {}
            layer["ln1"] = _read_floats(f, dim)

            for name in ("q", "k", "v", "o"):
                (scale,) = struct.unpack("f", f.read(4))
                weights = _read_packed(f, dim * dim)
                assert len(weights) == _packed_bytes(dim * dim), (
                    f"layer {i} {name}_w truncated"
                )
                layer[f"{name}_scale"] = scale
                layer[f"{name}_w_len"] = len(weights)

            layer["ln2"] = _read_floats(f, dim)

            (fc1_scale,) = struct.unpack("f", f.read(4))
            fc1_w = _read_packed(f, dim * 4 * dim)
            assert len(fc1_w) == _packed_bytes(dim * 4 * dim), (
                f"layer {i} fc1_w truncated"
            )
            layer["fc1_scale"] = fc1_scale
            layer["fc1_w_len"] = len(fc1_w)

            (fc2_scale,) = struct.unpack("f", f.read(4))
            fc2_w = _read_packed(f, 4 * dim * dim)
            assert len(fc2_w) == _packed_bytes(4 * dim * dim), (
                f"layer {i} fc2_w truncated"
            )
            layer["fc2_scale"] = fc2_scale
            layer["fc2_w_len"] = len(fc2_w)

            layers.append(layer)
        out["layers"] = layers

        out["ln_f"] = _read_floats(f, dim)

        (head_scale,) = struct.unpack("f", f.read(4))
        head_w = _read_packed(f, dim * vocab)
        assert len(head_w) == _packed_bytes(dim * vocab), "head_w truncated"
        out["head_scale"] = head_scale
        out["head_w_len"] = len(head_w)
        out["head_w"] = head_w

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


@pytest.mark.parametrize("format_version", [2, 3])
def test_layout_matches_loader(tmp_bin, format_version):
    """Every byte the loader expects is present in the right place."""
    model, cfg = _make_model()
    export_model(model, tmp_bin, format_version=format_version)
    parsed = parse_binary(tmp_bin)

    assert parsed["magic"] == 0x41544F4D
    assert parsed["version"] == format_version
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

    suspicious = {0x41544F4D, 2}
    for i, layer in enumerate(parsed["layers"]):
        for key in ("q_scale", "k_scale", "v_scale", "o_scale", "fc1_scale", "fc2_scale"):
            s = layer[key]
            assert math.isfinite(s), f"layer {i} {key} not finite: {s}"
            assert s > 0, f"layer {i} {key} non-positive: {s}"
            # If we accidentally read magic-as-scale, this catches it.
            assert int(s) not in suspicious, f"layer {i} {key} looks like a header field"

    assert math.isfinite(parsed["head_scale"])
    assert parsed["head_scale"] > 0


@pytest.mark.parametrize("format_version", [2, 3])
def test_byte_count_matches_expected(tmp_bin, format_version):
    """The total bytes written must equal the loader's read budget exactly."""
    model, cfg = _make_model()
    export_model(model, tmp_bin, format_version=format_version)

    actual = os.path.getsize(tmp_bin)
    d = cfg.dim
    v = cfg.vocab_size
    L = cfg.depth
    ctx = cfg.context_length
    hidden = 4 * d

    # Header
    expected = 32
    if format_version >= 3:
        # token_emb + pos_emb: rows x float32 scales + rows*dim int8 codes
        expected += (v * 4 + v * d) + (ctx * 4 + ctx * d)
    else:
        # token_emb + pos_emb (float32)
        expected += (v * d + ctx * d) * 4
    # Per layer
    per_layer = (
        d * 4                                      # ln1
        + 4 * (4 + _packed_bytes(d * d))           # q/k/v/o: scale + weights
        + d * 4                                    # ln2
        + (4 + _packed_bytes(d * hidden))          # fc1
        + (4 + _packed_bytes(hidden * d))          # fc2
    )
    expected += L * per_layer
    # ln_f + head
    expected += d * 4 + (4 + _packed_bytes(d * v))

    assert actual == expected, (
        f"file size {actual} != computed {expected}; "
        "exporter and loader format are out of sync"
    )


class TestTernaryPacking:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 63, 64, 65, 4096])
    def test_roundtrip_is_lossless(self, n):
        rng = np.random.default_rng(n)
        w = rng.integers(-1, 2, size=n).astype(np.int8)
        assert np.array_equal(unpack_ternary(pack_ternary(w).tobytes(), n), w)

    @pytest.mark.parametrize("n", [1, 3, 4, 5, 4096])
    def test_uses_a_quarter_byte_per_weight(self, n):
        w = np.zeros(n, dtype=np.int8)
        assert pack_ternary(w).size == (n + 3) // 4

    def test_unused_code_never_appears(self):
        # guirguispierre 2026-08-10 - w+1 spans 0..2, so code 3 in the stream
        # means a weight escaped the [-1,1] clamp
        rng = np.random.default_rng(0)
        w = rng.integers(-1, 2, size=10000).astype(np.int8)
        b = np.frombuffer(pack_ternary(w).tobytes(), dtype=np.uint8)
        codes = np.concatenate([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3])
        assert not (codes == 3).any()

    def test_padding_bits_are_zero(self):
        w = np.array([1, 1], dtype=np.int8)
        (byte,) = pack_ternary(w)
        assert byte >> 4 == 0

    def test_exported_weights_are_only_ternary(self, tmp_bin):
        model, cfg = _make_model()
        export_model(model, tmp_bin)
        parsed = parse_binary(tmp_bin)

        n = cfg.dim * cfg.vocab_size
        head = unpack_ternary(parsed["head_w"], n)
        assert head.size == n
        assert set(np.unique(head).tolist()) <= {-1, 0, 1}


class TestEmbeddingQuantization:
    """v3 stores each embedding row as int8 codes plus one float32 absmax
    scale per row; the loaders dequantize to float32 at load time."""

    def test_v3_payload_is_bit_exact(self, tmp_bin):
        """The int8 codes and scales in the file are exactly what
        quantize_embedding_int8 produces from the model weights."""
        model, _ = _make_model()
        export_model(model, tmp_bin)
        parsed = parse_binary(tmp_bin)
        assert parsed["version"] == 3

        for name, weight in (
            ("token_emb", model.token_emb.weight),
            ("pos_emb", model.pos_emb.weight),
        ):
            scales, codes = quantize_embedding_int8(weight)
            np.testing.assert_array_equal(
                parsed[f"{name}_scales"], scales, err_msg=f"{name} scales"
            )
            np.testing.assert_array_equal(
                parsed[f"{name}_codes"], codes.reshape(-1),
                err_msg=f"{name} codes",
            )

    def test_v3_dequantized_rows_match_python(self, tmp_bin):
        """The float32 table a loader reconstructs equals the Python-side
        dequantization, element for element."""
        model, _ = _make_model()
        export_model(model, tmp_bin)
        parsed = parse_binary(tmp_bin)

        for name, weight in (
            ("token_emb", model.token_emb.weight),
            ("pos_emb", model.pos_emb.weight),
        ):
            scales, codes = quantize_embedding_int8(weight)
            expected = dequantize_embedding_int8(scales, codes).reshape(-1)
            np.testing.assert_array_equal(
                np.asarray(parsed[name], dtype=np.float32), expected,
                err_msg=f"{name} dequantized",
            )

    def test_v3_error_is_within_half_a_step(self, tmp_bin):
        """Per-row absmax int8 bounds the reconstruction error at scale/2."""
        model, _ = _make_model()
        export_model(model, tmp_bin)
        parsed = parse_binary(tmp_bin)

        for name, weight in (
            ("token_emb", model.token_emb.weight),
            ("pos_emb", model.pos_emb.weight),
        ):
            w = weight.detach().cpu().numpy().astype(np.float32)
            deq = np.asarray(parsed[name], dtype=np.float32).reshape(w.shape)
            scales = parsed[f"{name}_scales"]
            err = np.abs(deq - w)
            bound = (scales / 2.0).reshape(-1, 1) + 1e-8
            assert (err <= bound).all(), (
                f"{name} error {err.max():.3e} exceeds scale/2 bound "
                f"{bound.max():.3e}"
            )
            # and the row absmax element itself survives essentially exactly
            per_row_max_err = err.max(axis=1)
            assert (per_row_max_err <= scales / 2 + 1e-8).all()

    def test_v3_codes_hit_the_int8_range(self, tmp_bin):
        """The absmax element of every nonzero row quantizes to +/-127."""
        model, _ = _make_model()
        export_model(model, tmp_bin)
        parsed = parse_binary(tmp_bin)

        codes = parsed["token_emb_codes"].reshape(
            parsed["vocab"], parsed["dim"]
        )
        assert codes.min() >= -127 and codes.max() <= 127
        assert (np.abs(codes).max(axis=1) == 127).all()

    def test_v3_zero_row_roundtrips_exactly(self, tmp_bin):
        """An all-zero embedding row must not produce NaN scales or codes."""
        model, cfg = _make_model()
        with torch.no_grad():
            model.token_emb.weight[0].zero_()
        export_model(model, tmp_bin)
        parsed = parse_binary(tmp_bin)

        assert np.isfinite(parsed["token_emb_scales"]).all()
        row = parsed["token_emb_codes"][: cfg.dim]
        assert (row == 0).all()
        deq_row = np.asarray(parsed["token_emb"][: cfg.dim], dtype=np.float32)
        assert (deq_row == 0.0).all()

    def test_v2_embeddings_stay_raw_float32(self, tmp_bin):
        """format_version=2 keeps the legacy layout: no scales, weights land
        bit-exact as float32."""
        model, cfg = _make_model()
        export_model(model, tmp_bin, format_version=2)
        parsed = parse_binary(tmp_bin)
        assert parsed["version"] == 2
        assert parsed["token_emb_scales"] is None
        assert parsed["pos_emb_scales"] is None

        w = model.token_emb.weight.detach().cpu().numpy().astype(np.float32)
        np.testing.assert_array_equal(
            np.asarray(parsed["token_emb"], dtype=np.float32), w.reshape(-1)
        )
        p = model.pos_emb.weight.detach().cpu().numpy().astype(np.float32)
        np.testing.assert_array_equal(
            np.asarray(parsed["pos_emb"], dtype=np.float32), p.reshape(-1)
        )
