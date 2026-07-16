"""
Where VOCD spends its time, and what each backend costs.

This is not a benchmark of the *method* -- that is synthetic_benchmark.py.
This measures the *implementation*, which is a separate question: the answer
must not change when the speed does.

    python benchmarks/profile_backends.py
    python benchmarks/profile_backends.py --profile      # per-function breakdown
    python benchmarks/profile_backends.py --n 5000       # scaling with series length
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

import vocd
from vocd import _backend

CPS_TRUE = [100, 200, 300, 350, 400, 450, 600]
MUS = [10, 5, 0, 10, 5, 12, 0]


def paper_series(n: int = 500) -> np.ndarray:
    """The benchmark series, tiled if a longer one is asked for."""
    np.random.seed(3000)
    y = np.zeros(500)
    mu, i = MUS[0], 0
    for t in range(500):
        if t == CPS_TRUE[i]:
            mu = MUS[i + 1]
            i += 1
        y[t] = np.random.normal(mu, 1)
    if n <= 500:
        return y[:n]
    return np.tile(y, int(np.ceil(n / 500)))[:n]


def best_of(fn, repeats: int = 3) -> float:
    """Minimum wall time: least contaminated by scheduler noise."""
    return min(_timed(fn) for _ in range(repeats))


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=500, help="series length")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile", action="store_true",
                    help="per-function breakdown of the pure-Python path")
    args = ap.parse_args()

    y = paper_series(args.n)
    print(f"series length: {y.size}")
    print(f"compiled backend available: {_backend.available()}")
    if not _backend.available():
        print("  build it with:  python setup_ext.py\n")

    backends = ["python"] + (["cpp"] if _backend.available() else [])

    # correctness before speed: a faster wrong answer is worthless
    if len(backends) == 2:
        a = vocd.detect(y, backend="python")
        b = vocd.detect(y, backend="cpp")
        same = (np.array_equal(a.candidates, b.candidates)
                and np.allclose(a.p_max, b.p_max, equal_nan=True))
        print(f"backends produce identical output: {same}")
        if not same:
            raise SystemExit("backends disagree — timings are meaningless")

    print(f"\n{'backend':<10}{'ms/series':>12}{'us/step':>10}{'speedup':>10}")
    print("-" * 42)
    base = None
    for be in backends:
        ms = best_of(lambda: vocd.detect(y, backend=be), args.repeats) * 1000
        base = base or ms
        print(f"{be:<10}{ms:>12.1f}{ms * 1000 / y.size:>10.1f}"
              f"{base / ms:>9.2f}x")
    print("-" * 42)
    print("us/step is the streaming cost: what VOCD.update() must fit inside\n"
          "between two observations.")

    if args.profile:
        print("\nper-function (pure Python path):")
        pr = cProfile.Profile()
        pr.enable()
        vocd.detect(y, backend="python")
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(10)
        body = s.getvalue().split("ncalls")[1]
        print("ncalls" + body[:1200])


if __name__ == "__main__":
    main()
