"""CLI contract tests for the embedded C++ runner.

Locks the subprocess-visible behavior of embedded/runner: exit codes,
stderr messages, generation-stream framing, seeded determinism, the
KV-cache context-slide path, --parity probe output, --vocab
detokenization, and gist loading. Model .bin files are exported to
tmp_path at test time and never checked in. The whole module skips when
the runner binary is not built, the same convention as
test_cpp_runtime_parity.py.
"""

import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.model.transformer import (  # noqa: E402
    AtomicConfig,
    AtomicTransformer,
)
from atomic_1bit.utils.export_to_cpp import export_model  # noqa: E402

RUNNER = os.path.join(os.path.dirname(__file__), "..", "embedded", "runner")
REAL_VOCAB = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "tiny_stories.vocab"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(RUNNER),
    reason="C++ runner not built (cd embedded && g++ -O3 -std=c++17 "
    "atomic_runner.cpp -o runner)",
)

# Same shape as test_cpp_runtime_parity.CFG: small enough that a run
# which crosses context_length still finishes in well under a second.
CFG = AtomicConfig(vocab_size=256, dim=64, depth=4, heads=4, context_length=32)

STEPS_WITHIN_CTX = 16  # stays inside the 32-token window
STEPS_CROSSING_CTX = CFG.context_length + 8  # forces the slide+rebuild path


def _run_runner(model_bin, *args):
    return subprocess.run(
        [RUNNER, "--model", str(model_bin), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _stream_text(res):
    """Raw stream between 'Generating:' and 'Done.', prompt echo included."""
    return res.stdout.split("Generating:", 1)[1].split("Done.")[0]


def _generated_tokens(res, n_prompt=1):
    body = _stream_text(res)
    fields = body.split()
    for tok in fields:
        assert tok.lstrip(
            "-"
        ).isdigit(), f"non-numeric token {tok!r} in generation stream: {body!r}"
    # the runner echoes the prompt before generating
    return [int(t) for t in fields[n_prompt:]]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One seeded model exported twice: plain and with a gist slot."""
    torch.manual_seed(1234)
    model = AtomicTransformer(CFG)
    model.eval()
    out_dir = tmp_path_factory.mktemp("cli")

    plain = out_dir / "model.bin"
    export_model(model, str(plain))

    # gist_file path of export_model(): raw float32 vector of length dim
    gist_path = out_dir / "test.gist"
    gist_path.write_bytes(
        np.random.RandomState(0).randn(CFG.dim).astype(np.float32).tobytes()
    )
    gist = out_dir / "model_gist.bin"
    export_model(model, str(gist), gist_file=str(gist_path))

    return plain, gist


class TestDeterminism:
    def test_same_seed_is_byte_identical(self, exported):
        plain, _ = exported
        args = ("--steps", "20", "--temp", "0.7", "--seed", "42", "--start_token", "1")
        a = _run_runner(plain, *args)
        b = _run_runner(plain, *args)
        assert a.returncode == 0 and b.returncode == 0
        # compare only the stream: the TPS line legitimately varies
        assert _stream_text(a) == _stream_text(b)
        assert _generated_tokens(a) == _generated_tokens(b)

    def test_different_seed_diverges(self, exported):
        plain, _ = exported
        a = _run_runner(
            plain,
            "--steps",
            "20",
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
        )
        b = _run_runner(
            plain, "--steps", "20", "--temp", "0.7", "--seed", "7", "--start_token", "1"
        )
        assert _generated_tokens(a) != _generated_tokens(b), (
            "seeds 42 and 7 produced the same 20-token stream -- the "
            "--seed flag is not reaching the sampler"
        )

    def test_temp_zero_ignores_seed(self, exported):
        # temp 0 is argmax; the rng is never drawn from, so the stream
        # must not depend on --seed at all
        plain, _ = exported
        a = _run_runner(
            plain, "--steps", "12", "--temp", "0", "--seed", "1", "--start_token", "1"
        )
        b = _run_runner(
            plain, "--steps", "12", "--temp", "0", "--seed", "999", "--start_token", "1"
        )
        assert _generated_tokens(a) == _generated_tokens(b)

    def test_no_stream_emits_same_tokens(self, exported):
        plain, _ = exported
        streamed = _run_runner(
            plain, "--steps", "8", "--temp", "0.7", "--seed", "42", "--start_token", "1"
        )
        buffered = _run_runner(
            plain,
            "--steps",
            "8",
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
            "--no-stream",
        )
        assert buffered.returncode == 0
        # --no-stream frames output as 'Generated: t0 t1 ...' with no
        # prompt echo and nothing before it
        assert "Generating:" not in buffered.stdout
        body = buffered.stdout.split("Generated:", 1)[1].split("Done.")[0]
        buffered_tokens = [int(t) for t in body.split()]
        assert buffered_tokens == _generated_tokens(streamed)


class TestKVCachePaths:
    # steps <= ctx runs prefill once then decodes via forward_one();
    # steps > ctx additionally drops the cache and rebuilds it every
    # step once the window slides. Both must stay valid + deterministic.
    def test_within_context_run_is_valid(self, exported):
        plain, _ = exported
        res = _run_runner(
            plain,
            "--steps",
            str(STEPS_WITHIN_CTX),
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
        )
        assert res.returncode == 0, res.stderr[-1000:]
        assert len(_generated_tokens(res)) == STEPS_WITHIN_CTX

    def test_context_crossing_run_is_valid(self, exported):
        plain, _ = exported
        res = _run_runner(
            plain,
            "--steps",
            str(STEPS_CROSSING_CTX),
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
        )
        assert res.returncode == 0, res.stderr[-1000:]
        assert len(_generated_tokens(res)) == STEPS_CROSSING_CTX

    def test_context_crossing_is_deterministic(self, exported):
        plain, _ = exported
        args = (
            "--steps",
            str(STEPS_CROSSING_CTX),
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
        )
        a = _run_runner(plain, *args)
        b = _run_runner(plain, *args)
        assert _generated_tokens(a) == _generated_tokens(b)

    def test_slide_does_not_rewrite_emitted_tokens(self, exported):
        # positions rebase when the window slides, but tokens already
        # emitted are history: a short run must be an exact prefix of a
        # run that later crosses the window
        plain, _ = exported
        short = _run_runner(
            plain,
            "--steps",
            str(STEPS_WITHIN_CTX),
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
        )
        long = _run_runner(
            plain,
            "--steps",
            str(STEPS_CROSSING_CTX),
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
        )
        long_tokens = _generated_tokens(long)
        assert long_tokens[:STEPS_WITHIN_CTX] == _generated_tokens(short)


class TestParityMode:
    def test_parity_exits_zero_and_emits_probes(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--parity", "--start_token", "7")
        assert res.returncode == 0, res.stderr[-1000:]
        assert "--- C++ Parity Check ---" in res.stdout
        # the probe contract tools/parity_check.py parses
        for probe in ("Embedding_Sum", "Final_LN", "Final_Logits"):
            assert f"[{probe}]" in res.stdout
        for i in range(CFG.depth):
            for suffix in ("LN1", "Attn_Out", "MLP_Out"):
                assert f"[Layer{i}_{suffix}]" in res.stdout

    def test_parity_forces_single_step(self, exported):
        # --parity overrides --steps to 1 regardless of what was passed
        plain, _ = exported
        res = _run_runner(plain, "--parity", "--steps", "50", "--start_token", "7")
        assert res.returncode == 0
        assert "| Steps: 1 |" in res.stdout


class TestVocab:
    def test_stream_decodes_to_text(self, exported, tmp_path):
        # synthetic detok: token i -> "t{i}", so the stream is
        # machine-checkable rather than eyeballed
        vocab = tmp_path / "test.vocab"
        vocab.write_text(
            "".join(f"t{i}".encode().hex() + "\n" for i in range(CFG.vocab_size))
        )
        plain, _ = exported
        args = ("--steps", "8", "--temp", "0", "--start_token", "1")
        text_res = _run_runner(plain, "--vocab", str(vocab), *args)
        id_res = _run_runner(plain, *args)

        assert text_res.returncode == 0, text_res.stderr[-1000:]
        # vocab mode drops the 'Generating:' framing entirely
        assert "Generating:" not in text_res.stdout

        # layout: prompt echo line, then one line of concatenated text
        lines = [
            line
            for line in text_res.stdout.split("Done.")[0].splitlines()
            if line.strip()
        ]
        assert lines[-2] == "t1", "prompt token was not detokenized"
        gen_line = lines[-1]
        assert re.fullmatch(
            r"(t\d+)+", gen_line
        ), f"expected pure decoded text, got {gen_line!r}"
        decoded = [int(m) for m in re.findall(r"t(\d+)", gen_line)]
        assert decoded == _generated_tokens(id_res)

    @pytest.mark.skipif(
        not os.path.exists(REAL_VOCAB), reason="checkpoints/tiny_stories.vocab missing"
    )
    def test_real_vocab_file_produces_text(self, exported):
        plain, _ = exported
        res = _run_runner(
            plain,
            "--vocab",
            REAL_VOCAB,
            "--steps",
            "12",
            "--temp",
            "0",
            "--start_token",
            "1",
        )
        assert res.returncode == 0, res.stderr[-1000:]
        assert "Generating:" not in res.stdout
        # loader diagnostics live on stderr, so stdout is the banner line,
        # the config line, then decoded text, then Done/TPS
        body = res.stdout.split("Done.")[0]
        stream = "\n".join(body.splitlines()[2:])
        assert re.search(
            r"[a-zA-Z]", stream
        ), f"expected decoded text in stream, got {stream!r}"
        assert not re.search(
            r"\d+ \d+ \d+ \d+", stream
        ), f"stream still looks like bare token ids: {stream!r}"

    def test_missing_vocab_exits_1(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--vocab", "/nonexistent/path.vocab", "--steps", "2")
        assert res.returncode == 1
        assert "could not load vocab" in res.stderr

    def test_empty_vocab_exits_1(self, exported, tmp_path):
        # load_vocab() fails only when it yields zero entries; an empty
        # file is the 'garbage' case that is actually rejected
        vocab = tmp_path / "empty.vocab"
        vocab.write_text("")
        plain, _ = exported
        res = _run_runner(plain, "--vocab", str(vocab), "--steps", "2")
        assert res.returncode == 1
        assert "could not load vocab" in res.stderr


class TestCLIValidation:
    def test_unknown_flag_exits_2(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--bogus")
        assert res.returncode == 2
        assert "Unknown arg" in res.stderr

    def test_flag_missing_value_exits_2(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--steps")
        assert res.returncode == 2
        assert "requires a value" in res.stderr

    def test_missing_model_exits_1(self):
        res = _run_runner("/nonexistent/model.bin", "--steps", "2")
        assert res.returncode == 1
        assert "Could not load model file" in res.stderr

    def test_out_of_range_prompt_exits_1(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--prompt", "99999", "--steps", "2")
        assert res.returncode == 1
        assert "out of vocab range" in res.stderr
        assert "99999" in res.stderr

    def test_out_of_range_start_token_exits_1(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--start_token", "99999", "--steps", "2")
        assert res.returncode == 1
        assert "out of vocab range" in res.stderr

    def test_negative_prompt_token_exits_1(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--prompt", "-5", "--steps", "2")
        assert res.returncode == 1
        assert "out of vocab range" in res.stderr

    def test_prompt_csv_is_echoed_then_generated(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--prompt", "1,2,3", "--steps", "4", "--temp", "0")
        assert res.returncode == 0, res.stderr[-1000:]
        fields = _stream_text(res).split()
        assert [int(t) for t in fields[:3]] == [1, 2, 3]
        assert len(_generated_tokens(res, n_prompt=3)) == 4

    def test_prompt_overrides_start_token(self, exported):
        plain, _ = exported
        res = _run_runner(
            plain,
            "--prompt",
            "5,6",
            "--start_token",
            "9",
            "--steps",
            "2",
            "--temp",
            "0",
        )
        fields = _stream_text(res).split()
        assert [int(t) for t in fields[:2]] == [5, 6]
        assert "9" not in fields[:2]

    def test_non_integer_prompt_piece_warns_and_skips(self, exported):
        plain, _ = exported
        res = _run_runner(plain, "--prompt", "7,abc,9", "--steps", "2", "--temp", "0")
        assert res.returncode == 0
        assert "skipping non-integer token 'abc'" in res.stderr
        fields = _stream_text(res).split()
        assert [int(t) for t in fields[:2]] == [7, 9]


class TestGist:
    def test_gist_model_loads_and_reports_it(self, exported):
        _, gist = exported
        res = _run_runner(gist, "--steps", "4", "--temp", "0", "--start_token", "1")
        assert res.returncode == 0, res.stderr[-1000:]
        assert "HasGist=1" in res.stdout

    def test_gist_generation_is_deterministic(self, exported):
        _, gist = exported
        args = ("--steps", "20", "--temp", "0.7", "--seed", "42", "--start_token", "1")
        a = _run_runner(gist, *args)
        b = _run_runner(gist, *args)
        assert a.returncode == 0 and b.returncode == 0
        assert _generated_tokens(a) == _generated_tokens(b)

    def test_gist_changes_the_stream(self, exported):
        # same weights, same flags: only the baked-in gist slot differs,
        # so identical output would mean the gist never reaches forward()
        plain, gist = exported
        args = ("--steps", "20", "--temp", "0.7", "--seed", "42", "--start_token", "1")
        no_gist = _run_runner(plain, *args)
        with_gist = _run_runner(gist, *args)
        assert _generated_tokens(no_gist) != _generated_tokens(with_gist)

    def test_gist_past_context_still_deterministic(self, exported):
        # the gist slot consumes one position, so the window fills one
        # token earlier; past that the runner falls back to zeroed
        # logits + uniform sampling -- still seeded, still deterministic
        _, gist = exported
        args = (
            "--steps",
            str(STEPS_CROSSING_CTX),
            "--temp",
            "0.7",
            "--seed",
            "42",
            "--start_token",
            "1",
        )
        a = _run_runner(gist, *args)
        b = _run_runner(gist, *args)
        assert a.returncode == 0 and b.returncode == 0
        assert len(_generated_tokens(a)) == STEPS_CROSSING_CTX
        assert _generated_tokens(a) == _generated_tokens(b)
