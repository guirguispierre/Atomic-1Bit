"""Smoke tests for examples/*.py -- these scripts are the documented
onboarding path, so they must keep working with the shipped fixture."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
GENERATE = os.path.join(ROOT, "examples", "generate.py")
FIXTURE = os.path.join(ROOT, "checkpoints", "tiny_stories.pt")

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE), reason="fixture checkpoint missing"
)


def _run_example(script, *args, stdin=""):
    return subprocess.run(
        [sys.executable, script, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=ROOT,
    )


class TestGenerateExample:
    def test_loads_fixture_and_generates(self):
        res = _run_example(
            GENERATE, "--tokens", "8", "--temp", "0", stdin="One day\nquit\n"
        )
        assert res.returncode == 0, res.stderr[-1500:]
        # config was inferred from the fixture shapes, not the defaults
        assert "vocab=2048 dim=128" in res.stdout
        assert "Vocab map: 2047 entries" in res.stdout

    def test_missing_checkpoint_exits_1(self):
        res = _run_example(
            GENERATE, "--checkpoint", "/nonexistent/model.pt", stdin="quit\n"
        )
        assert res.returncode == 1
        assert "Checkpoint not found" in res.stdout
