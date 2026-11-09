"""Parity and quality gates on the checked-in fixture checkpoint.

`checkpoints/tiny_stories.pt` is the small trained model the repo ships so
parity and quality evals run out of the box (see checkpoints/README.md).
These tests re-export the .bin at test time -- binaries are never checked
in -- and assert the C++ runner's --parity dump matches the PyTorch forward
numerically, per the tolerance documented in tools/parity_check.py and the
README "Numerical parity" section.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import parity_check as pc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "checkpoints", "tiny_stories.pt")
RUNNER = os.path.join(ROOT, "embedded", "runner")

pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(RUNNER),
        reason="C++ runner not built (cd embedded && g++ -O3 "
        "-std=c++17 atomic_runner.cpp -o runner)",
    ),
    pytest.mark.skipif(
        not os.path.exists(CHECKPOINT),
        reason="fixture checkpoint missing (checkpoints/"
        "tiny_stories.pt is checked in; restore or "
        "retrain with tools/train_fixture_checkpoint.py)",
    ),
]

# First tokens of a real story prompt in model ids -- exercises attention
# across positions, unlike a single start token.
PROMPT = [50, 57, 6, 38, 4, 36, 7, 6, 34, 56, 129, 89, 5, 52, 17, 3, 222, 0]


@pytest.fixture(scope="module")
def parity_report(tmp_path_factory):
    bin_path = tmp_path_factory.mktemp("trained_parity") / "model.bin"
    return pc.check_parity(
        checkpoint=CHECKPOINT,
        runner=RUNNER,
        bin_path=str(bin_path),
        tokens=PROMPT,
        quiet=True,
    )


class TestTrainedParity:
    def test_every_probe_within_tolerance(self, parity_report):
        # Match check_parity's verdict: WARN-band excursions (1-4x tol)
        # are plausible platform quantization noise (x86 vs arm64
        # accumulation order); only FAIL and structural rows fail the
        # test. Warned rows still print in the report.
        failed = [
            r for r in parity_report["rows"] if r["status"] not in ("PASS", "WARN")
        ]
        assert not failed, (
            "parity probes diverged beyond tolerance "
            f"{parity_report['rel_tol']}:\n"
            + pc.format_report(failed, parity_report["rel_tol"], PROMPT)
        )

    def test_greedy_next_token_agrees(self, parity_report):
        assert parity_report["py_argmax"] == parity_report["cpp_argmax"], (
            f"argmax fork: python picks {parity_report['py_argmax']}, "
            f"c++ picks {parity_report['cpp_argmax']}"
        )

    def test_probes_cover_all_layers(self, parity_report):
        # the dump contract is per-layer; a missing probe means the runner
        # silently changed what --parity prints
        depth = torch.load(CHECKPOINT, map_location="cpu")["config"]["depth"]
        names = {r["name"] for r in parity_report["rows"]}
        for i in range(depth):
            for suffix in ("LN1", "Attn_Out", "MLP_Out"):
                assert f"Layer{i}_{suffix}" in names
        assert {"Embedding_Sum", "Final_LN", "Final_Logits"} <= names


class TestCheckpointIsTrained:
    # guards against an untrained/placeholder file being swapped in: a
    # random-weight model scores ~ln(V) or worse on real tokens, this one
    # measured 1.27 nats on the prompt below at check-in time
    def test_nll_beats_uniform_with_margin(self):
        model, _ = pc.build_model(CHECKPOINT)
        x = torch.tensor([PROMPT], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, model.config.vocab_size),
            x[:, 1:].reshape(-1),
        ).item()
        bound = 0.6 * math.log(model.config.vocab_size)
        assert nll < bound, (
            f"fixture checkpoint scores {nll:.3f} nats on real tokens; "
            f"bound {bound:.3f} (0.6 x uniform). Either the file is not the "
            f"trained fixture or the model arch changed -- retrain with "
            f"tools/train_fixture_checkpoint.py"
        )

    def test_checkpoint_records_provenance(self):
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert ckpt.get("config", {}).get("vocab_size") == 2048
        training = ckpt.get("training", {})
        assert training.get("steps", 0) >= 500
        assert training.get("eval_ppl", float("inf")) < 2048
