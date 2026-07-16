"""
How VOCD's runtime scales with series length, on each backend.

This measures the *implementation*, not the method: the answer must not change
when the speed does, so the two backends are compared for equality at every
length before any timing is reported.

Why length matters
------------------
The synthetic benchmark is T=500.  Real records are not: an ISFET deployment is
weeks of samples, a gas-sensor record is months.  A per-step cost that creeps
up with T is invisible at 500 and fatal at 10^6, so the quantity to watch here
is **us/step**, which should be flat.  A rising line means something in the
pipeline is O(T) per step and the whole run is quadratic.

Change-point density is held constant (one every ``--seg`` samples) rather than
fixing the count, so the number of hypotheses grows with T as it would in a
real record.  Fixing the count instead would flatter the long series.

Outputs (into --out-dir):
    scaling.csv   one row per (backend, length)
    scaling.png   runtime vs length, log-log, with an O(T) reference slope

Usage::

    python benchmarks/scaling_benchmark.py
    python benchmarks/scaling_benchmark.py --lengths 500,5000,50000
    python benchmarks/scaling_benchmark.py --out-dir results/synthetic
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import numpy as np

import vocd
from vocd import _backend

DEFAULT_LENGTHS = [500, 1000, 2000, 4000, 8000, 16000, 32000]


def make_series(n: int, seg: int = 250, seed: int = 0) -> np.ndarray:
    """
    Piecewise-constant Gaussian with a fixed change-point density and 2%
    outlier contamination, matching the character of the paper benchmark but
    at arbitrary length.
    """
    rng = np.random.default_rng(seed)
    n_seg = max(2, n // seg)
    parts = [rng.normal(rng.uniform(-6.0, 6.0), 1.0, seg) for _ in range(n_seg + 1)]
    y = np.concatenate(parts)[:n]

    k = int(0.02 * n)
    if k:
        idx = rng.choice(n, k, replace=False)
        y[idx] += rng.choice([-1.0, 1.0], k) * 10.0
    return y


def best_ms(y: np.ndarray, backend: str, repeats: int) -> float:
    """Best-of-N wall time in ms.  The minimum is the least noisy estimate."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        vocd.detect(y, backend=backend)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def plot(rows: List[dict], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("! matplotlib not installed; skipping the plot\n"
              "  pip install matplotlib")
        return False

    backends = sorted({r["backend"] for r in rows})
    colours = {"python": "#377eb8", "cpp": "#e41a1c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for be in backends:
        rs = sorted((r for r in rows if r["backend"] == be),
                    key=lambda r: r["length"])
        L = [r["length"] for r in rs]
        ms = [r["ms"] for r in rs]
        us = [r["us_per_step"] for r in rs]
        c = colours.get(be, None)
        ax1.plot(L, ms, "o-", color=c, label=f"backend={be}")
        ax2.plot(L, us, "o-", color=c, label=f"backend={be}")

    # O(T) reference through the first python point: linear scaling is a
    # straight line of slope 1 on log-log axes.
    ref = sorted((r for r in rows if r["backend"] == backends[0]),
                 key=lambda r: r["length"])
    L0, ms0 = ref[0]["length"], ref[0]["ms"]
    Ls = np.array([r["length"] for r in ref], dtype=float)
    ax1.plot(Ls, ms0 * Ls / L0, "k--", lw=1, alpha=0.6, label="O(T) reference")

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("series length T"); ax1.set_ylabel("runtime (ms)")
    ax1.set_title("Runtime vs series length")
    ax1.grid(alpha=0.3, which="both"); ax1.legend()

    ax2.set_xscale("log")
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("series length T"); ax2.set_ylabel("per-step cost (us)")
    ax2.set_title("Per-step cost (flat = linear scaling)")
    ax2.grid(alpha=0.3, which="both"); ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="VOCD runtime vs series length")
    ap.add_argument("--lengths", default=",".join(str(n) for n in DEFAULT_LENGTHS),
                    help="comma-separated series lengths")
    ap.add_argument("--seg", type=int, default=250,
                    help="samples per segment (fixes change-point density)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", default="results/synthetic")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    backends = ["python"] + (["cpp"] if _backend.available() else [])
    if len(backends) == 1:
        print("! compiled backend not available; timing pure Python only.\n"
              "  build it with:  pip install -e .   (or python setup_ext.py)\n")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"lengths: {lengths}   backends: {', '.join(backends)}   "
          f"1 CP per {args.seg} samples\n")
    print(f"{'T':>8}{'backend':>10}{'ms':>10}{'us/step':>10}{'speedup':>10}")
    print("-" * 48)

    rows: List[dict] = []
    for n in lengths:
        y = make_series(n, seg=args.seg)

        # Correctness before speed: a faster wrong answer is not a speedup.
        if len(backends) == 2:
            a = vocd.detect(y, backend="python")
            b = vocd.detect(y, backend="cpp")
            if not (np.array_equal(a.candidates, b.candidates)
                    and np.allclose(a.p_max, b.p_max, equal_nan=True)):
                raise SystemExit(f"backends disagree at T={n} — timings are "
                                 "meaningless until that is fixed")

        base = None
        for be in backends:
            ms = best_ms(y, be, args.repeats)
            base = base or ms
            rows.append({"backend": be, "length": n, "ms": ms,
                         "us_per_step": ms * 1000.0 / n,
                         "speedup": base / ms})
            print(f"{n:>8}{be:>10}{ms:>10.1f}{ms * 1000 / n:>10.1f}"
                  f"{base / ms:>9.2f}x")
        print("-" * 48)

    csv_path = os.path.join(args.out_dir, "scaling.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["backend", "length", "ms",
                                           "us_per_step", "speedup"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    plot(rows, os.path.join(args.out_dir, "scaling.png"))

    # Flag a rising per-step cost explicitly: it is the failure this script
    # exists to catch, and it is easy to miss in a table.
    for be in backends:
        us = [r["us_per_step"] for r in sorted(
            (r for r in rows if r["backend"] == be), key=lambda r: r["length"])]
        if len(us) > 2 and us[-1] > 2.0 * min(us):
            print(f"\n! backend={be}: per-step cost rose from {min(us):.1f} to "
                  f"{us[-1]:.1f} us across the sweep.\n"
                  "  That indicates super-linear scaling — something is O(T) "
                  "per step.")


if __name__ == "__main__":
    main()
