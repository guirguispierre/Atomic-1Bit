#!/usr/bin/env python3
"""Micro-benchmark for the ternary matmul CPU kernel.

Loads one or more builds of libatomic.so directly over ctypes (bypassing
python/wrapper.py so the transpose/copy overhead is not measured) and
times ternary_matmul() across a spread of shapes, including odd tails.

Compare two builds side by side:

    python benchmarks/kernel_micro.py \
        --lib /tmp/libatomic_scalar.so --lib atomic_1bit/core/libatomic.so

The first --lib is the baseline column that speedups are computed against.
"""

import argparse
import ctypes
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIB = os.path.join(ROOT, "atomic_1bit", "core", "libatomic.so")

# (M, N, K) — covers single-token decode, small layers, batch, odd tails.
DEFAULT_SHAPES = [
    (1, 4096, 4096),
    (1, 1024, 1024),
    (1, 256, 256),
    (8, 1024, 512),
    (3, 4095, 127),
]

I8 = ctypes.POINTER(ctypes.c_int8)
I32 = ctypes.POINTER(ctypes.c_int32)


def load_kernel(path):
    lib = ctypes.CDLL(path)
    lib.ternary_matmul.argtypes = [
        I8,
        I8,
        I32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ternary_matmul.restype = None
    return lib


def make_inputs(M, N, K, seed=0):
    rng = np.random.default_rng(seed)
    # Full int8 activation range, ternary weights {-1, 0, 1}.
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B_t = np.ascontiguousarray(rng.integers(-1, 2, size=(N, K), dtype=np.int8))
    C = np.zeros((M, N), dtype=np.int32)
    return (
        A.ctypes.data_as(I8),
        B_t.ctypes.data_as(I8),
        C.ctypes.data_as(I32),
        C,
        (A, B_t),  # keep arrays alive
    )


def sanity_check(lib, lib_name):
    """Small parity check so we never time a broken kernel."""
    A_ptr, B_ptr, C_ptr, C, keep = make_inputs(2, 40, 40, seed=7)
    lib.ternary_matmul(A_ptr, B_ptr, C_ptr, 2, 40, 40)
    ref = keep[0].astype(np.int32) @ keep[1].astype(np.int32).T
    if not np.array_equal(C, ref):
        print(
            f"WARNING: {lib_name} failed parity check "
            f"(max |diff| = {np.abs(C - ref).max()})",
            file=sys.stderr,
        )
        return False
    return True


def time_shape(lib, M, N, K, budget_s):
    a, b, c, C, keep = make_inputs(M, N, K)
    call = lib.ternary_matmul

    # Warmup + calibration: find reps so one timed run ~= budget_s.
    call(a, b, c, M, N, K)
    t0 = time.perf_counter()
    call(a, b, c, M, N, K)
    est = max(time.perf_counter() - t0, 1e-9)
    reps = max(1, int(budget_s / est))

    # Three timed runs, take the best (min) ms/call.
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            call(a, b, c, M, N, K)
        dt = (time.perf_counter() - t0) / reps
        best = min(best, dt)
    return best


def parse_shape(s):
    try:
        m, n, k = (int(x) for x in s.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad shape '{s}', want MxNxK")
    return (m, n, k)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--lib",
        action="append",
        default=[],
        help="path to libatomic.so (repeat to compare builds)",
    )
    ap.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=[],
        help="MxNxK (repeatable); overrides the default list",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="seconds per timed run (default 0.25)",
    )
    args = ap.parse_args()

    libs = args.lib or [DEFAULT_LIB]
    shapes = args.shape or DEFAULT_SHAPES

    kernels = []
    names = []
    for path in libs:
        if not os.path.exists(path):
            sys.exit(f"missing library: {path}")
        lib = load_kernel(path)
        name = os.path.basename(path.rstrip("/"))
        sanity_check(lib, name)
        kernels.append(lib)
        names.append(name)

    headers = [f"{n} ms" for n in names] + [f"vs {names[0]}"] * (len(names) - 1)
    print(f"{'shape (M,N,K)':<20}" + "".join(f"{h:>16}" for h in headers))
    for M, N, K in shapes:
        times = [time_shape(lib, M, N, K, args.budget) for lib in kernels]
        row = f"({M}, {N}, {K})".ljust(20)
        row += "".join(f"{t * 1e3:>16.3f}" for t in times)
        for t in times[1:]:
            row += f"{times[0] / t:>15.2f}x"
        print(row)

    # Aggregate throughput for the first shape set is per-shape; report a
    # GMAC/s (add/sub "MACs") line per lib for the largest shape instead.
    M, N, K = shapes[0]
    print(
        "\nThroughput on first shape "
        f"(MAC = one ternary add/sub term, {M*N*K} MACs/call):"
    )
    for name, lib in zip(names, kernels):
        t = time_shape(lib, M, N, K, args.budget)
        print(f"  {name:<24}{M * N * K / t / 1e9:>8.2f} GMAC/s")


if __name__ == "__main__":
    main()
