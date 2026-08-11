import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atomic_1bit.model.transformer import AtomicTransformer, AtomicConfig
from atomic_1bit.utils.export_to_cpp import export_model

RUNNER = os.path.join(os.path.dirname(__file__), "..", "embedded", "runner")

pytestmark = pytest.mark.skipif(
    not os.path.exists(RUNNER),
    reason="C++ runner not built (cd embedded && g++ -O3 -std=c++17 "
           "atomic_runner.cpp -o runner)",
)

CFG = AtomicConfig(vocab_size=256, dim=64, depth=4, heads=4, context_length=32)

# guirguispierre 2026-08-10 - loose because the int8 quantizer rounds, so one
# ULP of float difference can flip a value across a .5 boundary and move an
# accumulator a whole step; this catches gross mismatches, not single flips
AGREEMENT_TOL = 5e-2


def _run_runner(model_bin, *args):
    return subprocess.run(
        [RUNNER, "--model", str(model_bin), *args],
        capture_output=True, text=True, timeout=120,
    )


def _generated_tokens(res, n_prompt=1):
    body = res.stdout.split("Generating:", 1)[1].split("Done.")[0]
    fields = body.split()
    for tok in fields:
        assert tok.lstrip("-").isdigit(), (
            f"non-numeric token {tok!r} in generation stream: {body!r}"
        )
    # guirguispierre 2026-08-10 - runner echoes the prompt before generating
    return [int(t) for t in fields[n_prompt:]]


def _cpp_logits(model_bin, start_token):
    res = _run_runner(model_bin, "--parity", "--temp", "0",
                      "--start_token", str(start_token))
    for line in res.stdout.splitlines():
        if line.startswith("[Final_Logits]"):
            vals = line.split("] n=", 1)[1].split()[1:]
            return torch.tensor([float(v) for v in vals])
    raise AssertionError(
        f"runner emitted no Final_Logits.\nstdout:\n{res.stdout[-2000:]}\n"
        f"stderr:\n{res.stderr[-2000:]}"
    )


def _py_logits(model, start_token):
    with torch.no_grad():
        return model(torch.tensor([[start_token]], dtype=torch.long))[0, -1, :]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(1234)
    model = AtomicTransformer(CFG)
    model.eval()
    path = tmp_path_factory.mktemp("parity") / "model.bin"
    export_model(model, str(path))
    return model, path


class TestActivationContract:
    # guirguispierre 2026-08-10 - the numeric tests below cannot catch a wrong
    # activation: erf-vs-tanh gelu perturbs logits by the same ~1e-2 that one
    # quantizer boundary flip does, and x86 and arm already differ by that much
    def test_mlp_uses_tanh_gelu(self):
        act = AtomicTransformer(CFG).layers[0].mlp.act
        assert isinstance(act, nn.GELU)
        assert act.approximate == "tanh", (
            "embedded/atomic_runner.cpp implements the tanh gelu; exact-erf "
            "nn.GELU() here diverges by ~4e-4 per layer and compounds"
        )

    def test_tanh_and_erf_gelu_actually_differ(self):
        x = torch.linspace(-3, 3, 64)
        delta = (nn.GELU()(x) - nn.GELU(approximate="tanh")(x)).abs().max()
        assert delta > 1e-5


class TestCppRuntimeParity:
    def test_output_stream_is_not_polluted(self, exported):
        _, path = exported
        res = _run_runner(path, "--steps", "8", "--temp", "0",
                          "--start_token", "7")
        tokens = _generated_tokens(res)
        assert len(tokens) == 8, f"expected 8 generated tokens, got {tokens!r}"

    @pytest.mark.parametrize("start_token", [0, 7, 42, 128, 255])
    def test_logits_agree(self, exported, start_token):
        model, path = exported
        cpp = _cpp_logits(path, start_token)
        py = _py_logits(model, start_token)

        assert cpp.shape == py.shape
        rel = ((cpp - py).abs().max() / py.abs().max()).item()
        assert rel < AGREEMENT_TOL, (
            f"relative logit error {rel:.3e} exceeds {AGREEMENT_TOL:.0e} -- an "
            f"op has diverged between atomic_1bit/ and embedded/"
        )

    @pytest.mark.parametrize("start_token", [0, 7, 42, 128, 255])
    def test_disagreement_is_only_ever_a_tie_break(self, exported, start_token):
        # guirguispierre 2026-08-10 - asserting equal argmax would be flaky, so
        # require instead that pytorch scores the token c++ picked as good as
        # its own; a genuinely wrong engine fails this, a near-tie does not
        model, path = exported
        cpp = _cpp_logits(path, start_token)
        py = _py_logits(model, start_token)

        delta = (cpp - py).abs().max().item()
        shortfall = (py.max() - py[int(cpp.argmax())]).item()
        assert shortfall <= 2 * delta + 1e-6, (
            f"C++ chose token {int(cpp.argmax())}, which PyTorch scores "
            f"{shortfall:.4f} below its own pick {int(py.argmax())} -- more "
            f"than the {delta:.3e} numerical difference explains"
        )


class TestRunnerCLI:
    def test_help_exits_cleanly(self):
        res = subprocess.run([RUNNER, "--help"], capture_output=True,
                             text=True, timeout=60)
        assert res.returncode == 0
        assert "--model" in res.stdout and "--steps" in res.stdout

    def test_unknown_flag_is_rejected(self):
        res = subprocess.run([RUNNER, "--not-a-flag"], capture_output=True,
                             text=True, timeout=60)
        assert res.returncode != 0
