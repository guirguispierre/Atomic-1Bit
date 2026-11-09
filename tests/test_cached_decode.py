"""Cached-decode correctness.

Two contracts locked here:

1. embedded/atomic_lib.h's KV-cache path: forward() followed by forward_one()
   must produce bit-identical logits to a full forward() on the grown context
   (same ops, same order). A small harness is compiled on the fly and run
   against a real exported model, for both plain and gist exports.

2. Position-id convention under a gist: PyTorch assigns prompt tokens
   positions 1..T (the gist owns slot 0) and the C++ runner uses the same
   stream-index convention. Verified by comparing final logits on a gist
   export; before the transformer.py fix the two engines differed by one
   position id on every prompt token.
"""

import os
import shutil
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

REPO = os.path.join(os.path.dirname(__file__), "..")
RUNNER = os.path.join(REPO, "embedded", "runner")

CFG = AtomicConfig(vocab_size=256, dim=64, depth=4, heads=4, context_length=32)
AGREEMENT_TOL = 5e-2

CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")

pytestmark = pytest.mark.skipif(not CXX, reason="no C++ compiler available")


HARNESS = r"""
#include "atomic_lib.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: harness model.bin [steps]\n"); return 2; }
  AtomicModel model;
  if (!load_model(argv[1], model)) { fprintf(stderr, "load failed\n"); return 1; }
  int steps = argc > 2 ? atoi(argv[2]) : 8;
  int t_off = model.config.has_gist ? 1 : 0;

  vector<int> ctx = {3, 1, 4, 1, 5, 9, 2, 6};
  for (int &t : ctx) t = ((t % model.config.vocab_size) + model.config.vocab_size)
                          % model.config.vocab_size;

  KVCache cache;
  vector<float> logits_cached, logits_ref;
  double max_diff = 0.0;
  for (int step = 0; step < steps; ++step) {
    if (cache.len == (int)ctx.size() - 1 + t_off) {
      if (!forward_one(model, ctx.back(), logits_cached, cache)) {
        fprintf(stderr, "forward_one refused at step %d\n", step);
        return 3;
      }
    } else {
      forward(model, ctx, logits_cached, &cache);
    }

    forward(model, ctx, logits_ref);  // uncached reference
    if (logits_cached.size() != logits_ref.size()) {
      fprintf(stderr, "logit size mismatch %zu vs %zu\n",
              logits_cached.size(), logits_ref.size());
      return 4;
    }
    for (size_t i = 0; i < logits_ref.size(); ++i)
      max_diff = std::max(max_diff,
                          (double)fabs(logits_cached[i] - logits_ref[i]));

    int next = 0;
    for (size_t i = 1; i < logits_ref.size(); ++i)
      if (logits_ref[i] > logits_ref[next]) next = (int)i;
    ctx.push_back(next);
    if ((int)ctx.size() + t_off > model.config.max_seq_len) break;
  }
  printf("steps=%d max_diff=%g\n", steps, max_diff);
  return max_diff == 0.0 ? 0 : 5;
}
"""


@pytest.fixture(scope="module")
def lib_harness(tmp_path_factory):
    d = tmp_path_factory.mktemp("libcache")
    src = d / "harness.cpp"
    src.write_text(HARNESS)
    exe = d / "harness"
    res = subprocess.run(
        [
            CXX,
            "-O2",
            "-std=c++17",
            "-I",
            os.path.join(REPO, "embedded"),
            str(src),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, f"harness compile failed:\n{res.stderr[-2000:]}"
    return exe


@pytest.fixture(scope="module")
def plain_bin(tmp_path_factory):
    torch.manual_seed(1234)
    model = AtomicTransformer(CFG)
    model.eval()
    path = tmp_path_factory.mktemp("bin") / "plain.bin"
    export_model(model, str(path))
    return path


@pytest.fixture(scope="module")
def gist_bin(tmp_path_factory):
    torch.manual_seed(1234)
    model = AtomicTransformer(CFG)
    model.eval()
    d = tmp_path_factory.mktemp("gist")
    rng = np.random.RandomState(0)
    gist_np = (rng.randn(CFG.dim) * 0.2).astype(np.float32)
    gist_path = d / "gist.f32"
    gist_np.tofile(str(gist_path))
    path = d / "gist.bin"
    # v2 keeps fp32 embeddings, so any residual mismatch is convention drift,
    # not v3 int8 quantization noise -- the strict check below relies on that
    export_model(model, str(path), gist_file=str(gist_path), format_version=2)
    return model, path, gist_np


class TestLibKVCache:
    @pytest.mark.parametrize("steps", [1, 4, 12])
    def test_forward_one_matches_full_forward(self, lib_harness, plain_bin, steps):
        res = subprocess.run(
            [str(lib_harness), str(plain_bin), str(steps)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert (
            res.returncode == 0
        ), f"cached decode diverged from full forward:\n{res.stdout}\n{res.stderr}"
        assert "max_diff=0" in res.stdout

    def test_gist_model_cached_decode(self, lib_harness, gist_bin):
        _, path, _ = gist_bin
        res = subprocess.run(
            [str(lib_harness), str(path), "8"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert (
            res.returncode == 0
        ), f"gist cached decode diverged:\n{res.stdout}\n{res.stderr}"


class TestGistPositionParity:
    """PyTorch vs C++ runner must agree on positions under a gist.

    Both engines now use stream-index positions (gist owns slot 0, prompt
    tokens start at 1). The old PyTorch convention indexed prompt tokens from
    0 while decode used stream indices -- this test fails if either side
    regresses.
    """

    @pytest.mark.skipif(not os.path.exists(RUNNER), reason="C++ runner not built")
    def test_gist_logits_agree_with_python(self, gist_bin):
        model, path, gist_np = gist_bin
        res = subprocess.run(
            [
                RUNNER,
                "--model",
                str(path),
                "--parity",
                "--temp",
                "0",
                "--start_token",
                "7",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert res.returncode == 0, res.stderr[-500:]
        cpp = None
        for line in res.stdout.splitlines():
            if line.startswith("[Final_Logits]"):
                vals = line.split("] n=", 1)[1].split()[1:]
                cpp = torch.tensor([float(v) for v in vals])
        assert cpp is not None, f"no Final_Logits probe:\n{res.stdout[-1500:]}"

        gist = torch.from_numpy(gist_np.copy()).unsqueeze(0)
        with torch.no_grad():
            py = model(torch.tensor([[7]], dtype=torch.long), gist_token=gist)[0, -1, :]

        # v2 export: fp32 embeddings are exact, so a correct position
        # convention lands near machine precision; a one-position drift shows
        # an O(1) mismatch instead
        rel = ((cpp - py).abs().max() / py.abs().max()).item()
        assert rel < 1e-3, (
            f"gist logit mismatch {rel:.3e} -- position convention drifted "
            f"between the PyTorch model and the C++ engine"
        )
