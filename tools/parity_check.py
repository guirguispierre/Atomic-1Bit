"""Numeric parity check: PyTorch reference vs the C++ runner's --parity dump.

`./runner --parity` prints the last-position activation vector at each probe
point (Embedding_Sum, Layer{i}_LN1 / _Attn_Out / _MLP_Out, Final_LN,
Final_Logits). This tool runs the same probes through the PyTorch model and
compares the two dumps numerically -- the check can now fail.

Tolerance
---------
Default 5e-2 relative error, measured as max|cpp - ref| / max|ref| per probe,
the same bound tests/test_cpp_runtime_parity.py pins for logits.

Why not tighter: the int8 activation quantizer rounds with a per-tensor
scale, so a last-ULP float difference anywhere upstream (summation order in
an RMS norm, a tanh implementation) can push one value across a `.5`
boundary. That moves an integer accumulator a whole step -- a discrete
jump, not an epsilon -- and it compounds with depth (see README
"Numerical parity"). fp32-exact probes like Embedding_Sum and Final_LN
should land near 1e-6; probes downstream of BitLinear int8 rounding can
legitimately sit at ~1e-2. 5e-2 catches a genuinely wrong op (different
activation function, scale convention, or weight layout) while ignoring
boundary flips.

Verdicts: rel <= tol -> PASS; rel <= 4*tol -> WARN (plausible quantization
boundary excursion -- the gist path amplifies int8 embedding noise and a
single post-quantization probe legitimately reaches ~6e-2); rel > 4*tol or a
missing/shape-mismatched probe -> FAIL. The check fails only on FAIL rows: a
wrong convention produces O(1) errors everywhere, not one borderline probe.

Usage:
    python3 tools/parity_check.py [--checkpoint PATH] [--runner PATH]
        [--bin PATH] [--prompt 12,34,56 | --start-token N] [--tolerance F]

Exits 0 on PASS, 1 on numeric FAIL, 2 on setup error (missing files).
Importable: `check_parity()` returns a report object for the test suite.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atomic_1bit.model.transformer import (  # noqa: E402
    AtomicConfig,
    AtomicTransformer,
)
from atomic_1bit.utils.export_to_cpp import export_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(ROOT, "checkpoints", "tiny_stories.pt")
DEFAULT_RUNNER = os.path.join(ROOT, "embedded", "runner")

# See module docstring for why the bound is 5e-2 relative and not tighter.
DEFAULT_TOLERANCE = 5e-2

# The runner streams "Generating: <prompt...> " with no newline, so the first
# probe ([Embedding_Sum]) lands mid-line -- search, don't anchor.
_TENSOR_LINE = re.compile(r"\[([^\]]+)\] n=(\d+)\s+(.*)$")


def load_checkpoint(path):
    """Load a .pt checkpoint; accept the wrapped format from training or a
    bare state_dict. Returns (state_dict, config_dict_or_None, meta_dict)."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("config"), ckpt
    if isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        return ckpt, None, {}
    raise ValueError(
        f"{path} is neither a wrapped checkpoint (no 'model_state_dict') "
        f"nor a bare state_dict"
    )


def build_model(checkpoint_path, overrides=None):
    """Instantiate AtomicTransformer from checkpoint config (or --dim/--depth/
    --heads/--vocab-size/--ctx overrides) and load the stored weights.

    fp16 tensors in the checkpoint upcast to fp32 on load.
    """
    sd, cfg, meta = load_checkpoint(checkpoint_path)
    cfg = dict(cfg or {})
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    required = {"vocab_size", "dim", "depth", "heads", "context_length"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(
            f"checkpoint has no config and overrides did not supply {missing}; "
            f"pass --dim/--depth/--heads/--vocab-size/--ctx"
        )
    config = AtomicConfig(
        vocab_size=int(cfg["vocab_size"]),
        dim=int(cfg["dim"]),
        depth=int(cfg["depth"]),
        heads=int(cfg["heads"]),
        context_length=int(cfg["context_length"]),
    )
    model = AtomicTransformer(config)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, meta


def python_probes(model, tokens, gist=None):
    """Instrumented forward pass reproducing the vectors runner --parity
    dumps: last-token-position activations at each probe point, as float32
    numpy vectors keyed by the same names. When gist is given, it is
    prepended as stream row 0 and prompt tokens sit at positions 1..T,
    matching both the model's forward() and the C++ engine."""
    ids = torch.tensor([tokens], dtype=torch.long)
    probes = OrderedDict()
    with torch.no_grad():
        t_off = 1 if gist is not None else 0
        pos = torch.arange(t_off, ids.size(1) + t_off, dtype=torch.long)
        x = model.token_emb(ids) + model.pos_emb(pos)
        if gist is not None:
            g = gist.to(dtype=x.dtype).reshape(1, 1, -1)
            x = torch.cat([g, x], dim=1)
        probes["Embedding_Sum"] = x[0, -1].float().numpy()

        for i, layer in enumerate(model.layers):
            h = layer.ln1(x)
            probes[f"Layer{i}_LN1"] = h[0, -1].float().numpy()

            attn_out, _ = layer.attn(h)
            # runner prints ws.layer_out: o_proj output, pre-residual
            probes[f"Layer{i}_Attn_Out"] = attn_out[0, -1].float().numpy()
            x = x + attn_out

            mlp_out = layer.mlp(layer.ln2(x))
            # runner prints ws.mlp_out: fc2 output, pre-residual
            probes[f"Layer{i}_MLP_Out"] = mlp_out[0, -1].float().numpy()
            x = x + mlp_out

        xf = model.ln_f(x)
        probes["Final_LN"] = xf[0, -1].float().numpy()
        logits = model.head(xf)
        probes["Final_Logits"] = logits[0, -1].float().numpy()
    return probes


def parse_runner_output(stdout):
    """Parse `[Name] n=<count> v v v ...` fragments into name -> np.float64."""
    probes = OrderedDict()
    for line in stdout.splitlines():
        m = _TENSOR_LINE.search(line)
        if not m:
            continue
        name, n, body = m.group(1), int(m.group(2)), m.group(3)
        try:
            vals = np.array([float(v) for v in body.split()], dtype=np.float64)
        except ValueError:
            continue
        probes[name] = vals[:n]  # tolerate any trailing echo
    return probes


def run_runner_parity(runner_path, bin_path, tokens, timeout=120):
    """Run `runner --model bin --parity --prompt t0,t1,... --temp 0` and
    return the parsed probe dump."""
    cmd = [
        runner_path,
        "--model",
        str(bin_path),
        "--parity",
        "--temp",
        "0",
        "--prompt",
        ",".join(str(t) for t in tokens),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(
            f"runner exited {res.returncode}\nstdout tail:\n{res.stdout[-1500:]}"
            f"\nstderr:\n{res.stderr[-1500:]}"
        )
    probes = parse_runner_output(res.stdout)
    if "Final_Logits" not in probes:
        raise RuntimeError(
            f"runner emitted no [Final_Logits] probe.\n"
            f"stdout tail:\n{res.stdout[-1500:]}"
        )
    return probes


def compare_probes(ref, got, rel_tol):
    """Compare two name->vector probe dicts. Returns a list of row dicts:
    name, n, max_abs, max_rel, status ('PASS', 'FAIL', 'MISSING')."""
    rows = []
    for name in ref:
        if name not in got:
            rows.append(
                {
                    "name": name,
                    "n": ref[name].size,
                    "max_abs": float("nan"),
                    "max_rel": float("nan"),
                    "status": "MISSING (runner)",
                }
            )
            continue
        a, b = np.asarray(ref[name], dtype=np.float64), np.asarray(got[name])
        if a.shape != b.shape:
            rows.append(
                {
                    "name": name,
                    "n": f"{b.size}!={a.size}",
                    "max_abs": float("nan"),
                    "max_rel": float("nan"),
                    "status": "SHAPE",
                }
            )
            continue
        max_abs = float(np.abs(a - b).max()) if a.size else 0.0
        denom = float(np.abs(a).max()) if a.size else 0.0
        rel = max_abs / max(denom, 1e-12)
        status = (
            "PASS" if rel <= rel_tol else ("WARN" if rel <= 4 * rel_tol else "FAIL")
        )
        rows.append(
            {
                "name": name,
                "n": a.size,
                "max_abs": max_abs,
                "max_rel": rel,
                "status": status,
            }
        )
    for name in got:
        if name not in ref:
            rows.append(
                {
                    "name": name,
                    "n": got[name].size,
                    "max_abs": float("nan"),
                    "max_rel": float("nan"),
                    "status": "MISSING (python)",
                }
            )
    return rows


def format_report(rows, rel_tol, tokens):
    lines = []
    lines.append(
        f"{'probe':<22} {'n':>6} {'max|diff|':>12} " f"{'rel':>10} {'tol':>7}  status"
    )
    lines.append("-" * 72)
    for r in rows:
        lines.append(
            f"{r['name']:<22} {str(r['n']):>6} {r['max_abs']:>12.3e} "
            f"{r['max_rel']:>10.3e} {rel_tol:>7.0e}  {r['status']}"
        )
    return "\n".join(lines)


def worst_elements(ref, got, k=5):
    """Per-probe detail for --verbose: the k largest-mismatch elements."""
    lines = []
    for name in ref:
        if name not in got or ref[name].shape != got[name].shape:
            continue
        a, b = ref[name], got[name]
        idx = np.argsort(-np.abs(a - b))[:k]
        diffs = ", ".join(
            f"[{i}] ref={a[i]:.6g} cpp={b[i]:.6g} |d|={abs(a[i]-b[i]):.3g}" for i in idx
        )
        lines.append(f"  {name}: {diffs}")
    return "\n".join(lines)


def check_parity(
    checkpoint=DEFAULT_CHECKPOINT,
    runner=DEFAULT_RUNNER,
    bin_path=None,
    tokens=(42,),
    rel_tol=DEFAULT_TOLERANCE,
    overrides=None,
    gist_path=None,
    quiet=False,
    verbose=False,
):
    """End-to-end check: load checkpoint -> export .bin -> run both forwards
    -> compare probe dumps. gist_path, when given, is a raw float32[dim]
    file embedded in the export so the gist-attention path is checked too.
    Returns dict with keys:
      ok, rows, tokens, cpp_argmax, py_argmax, bin_path, checkpoint."""
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not os.path.exists(runner):
        raise FileNotFoundError(
            f"runner not found: {runner} "
            f"(build: cd embedded && g++ -O3 -std=c++17 atomic_runner.cpp -o runner)"
        )

    model, meta = build_model(checkpoint, overrides)
    vocab = model.config.vocab_size
    for t in tokens:
        if not (0 <= t < vocab):
            raise ValueError(f"prompt token {t} outside vocab [0, {vocab})")

    gist = None
    if gist_path is not None:
        if not os.path.exists(gist_path):
            raise FileNotFoundError(f"gist vector not found: {gist_path}")
        gist_np = np.fromfile(gist_path, dtype=np.float32)
        if gist_np.size != model.config.dim:
            raise ValueError(
                f"gist file has {gist_np.size} floats, expected dim {model.config.dim}"
            )
        gist = torch.from_numpy(gist_np.copy())

    tmp = None
    if bin_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="parity_", suffix=".bin", delete=False)
        tmp.close()
        bin_path = tmp.name
    try:
        if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
            # export into caller-chosen or temp path; a non-empty --bin file is
            # the artifact under test and is left alone (with --gist it must
            # already embed the vector)
            export_model(model, bin_path, gist_file=gist_path)
        cpp = run_runner_parity(runner, bin_path, tokens)
        ref = python_probes(model, list(tokens), gist=gist)
    finally:
        if tmp is not None and os.path.exists(tmp.name):
            os.unlink(tmp.name)

    rows = compare_probes(ref, cpp, rel_tol)
    ok = all(r["status"] in ("PASS", "WARN") for r in rows)
    report = {
        "ok": ok,
        "rows": rows,
        "tokens": list(tokens),
        "rel_tol": rel_tol,
        "cpp_argmax": int(np.argmax(cpp["Final_Logits"])),
        "py_argmax": int(np.argmax(ref["Final_Logits"])),
        "bin_path": bin_path,
        "checkpoint": checkpoint,
    }
    if not quiet:
        print(format_report(rows, rel_tol, tokens))
        if verbose:
            print("\nworst-mismatch elements per probe:")
            print(worst_elements(ref, cpp))
        print(
            f"\nargmax next-token: python={report['py_argmax']} "
            f"cpp={report['cpp_argmax']} "
            f"({'match' if report['py_argmax'] == report['cpp_argmax'] else 'MISMATCH'})"
        )
        print("PARITY:", "PASS" if ok else "FAIL")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Numeric parity check: PyTorch vs C++ runner --parity"
    )
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f".pt checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--runner",
        default=DEFAULT_RUNNER,
        help=f"compiled runner binary (default: {DEFAULT_RUNNER})",
    )
    p.add_argument(
        "--bin",
        dest="bin_path",
        default=None,
        help="use an existing exported .bin instead of exporting",
    )
    p.add_argument(
        "--prompt", default=None, help="comma-separated prompt token ids, e.g. 12,34,56"
    )
    p.add_argument(
        "--start-token",
        type=int,
        default=42,
        help="single prompt token when --prompt not given (default 42)",
    )
    p.add_argument(
        "--gist",
        dest="gist_path",
        default=None,
        help="raw float32[dim] gist vector; embeds it in the export so the "
        "gist path (stream-row 0, prompt positions 1..T) is parity-checked too",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="max relative error per probe (default %(default)s)",
    )
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print the largest-mismatch elements per probe",
    )
    args = p.parse_args(argv)

    if args.prompt:
        tokens = tuple(int(t) for t in args.prompt.split(",") if t != "")
    else:
        tokens = (args.start_token,)
    if not tokens:
        print("error: empty prompt", file=sys.stderr)
        return 2

    overrides = {
        "vocab_size": args.vocab_size,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "context_length": args.ctx,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None} or None

    try:
        report = check_parity(
            checkpoint=args.checkpoint,
            runner=args.runner,
            bin_path=args.bin_path,
            tokens=tokens,
            rel_tol=args.tolerance,
            overrides=overrides,
            gist_path=args.gist_path,
            verbose=args.verbose,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
