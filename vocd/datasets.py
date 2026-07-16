"""
Benchmark datasets.

``make_synthetic`` reproduces the piecewise-constant Gaussian benchmark used
in the VOCD paper: 500 samples, 6 true change-points, optional symmetric
outlier contamination.  The contamination is the point of the benchmark --
it is what separates robust methods from fragile ones.

Real-data loaders read from the repository ``data/`` directory.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "SYNTHETIC_CPS",
    "SYNTHETIC_MUS",
    "make_synthetic",
    "contaminate",
    "synthetic_trials",
    "data_dir",
    "load_csv_column",
]

#: True change-point locations in the synthetic benchmark.
SYNTHETIC_CPS: List[int] = [100, 200, 300, 350, 400, 450]

#: Segment means, one per segment (len == len(SYNTHETIC_CPS) + 1).
SYNTHETIC_MUS: List[float] = [10, 5, 0, 10, 5, 12, 0]


def make_synthetic(
    T: int = 500,
    cps: Optional[List[int]] = None,
    mus: Optional[List[float]] = None,
    sigma: float = 1.0,
    seed: int = 3000,
) -> Tuple[np.ndarray, List[int]]:
    """
    Piecewise-constant Gaussian signal.

    Returns (y, true_cps).  ``true_cps`` excludes 0 and T.
    """
    cps = list(SYNTHETIC_CPS if cps is None else cps)
    mus = list(SYNTHETIC_MUS if mus is None else mus)
    if len(mus) != len(cps) + 1:
        raise ValueError(f"need {len(cps) + 1} means for {len(cps)} change-points")

    rng = np.random.default_rng(seed)
    y = np.empty(T, dtype=float)
    bounds = [0] + [c for c in cps if c < T] + [T]
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        y[lo:hi] = rng.normal(mus[i], sigma, hi - lo)
    return y, [c for c in cps if c < T]


def contaminate(
    y: np.ndarray,
    frac: float = 0.02,
    spike: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Add symmetric +/- ``spike`` outliers to a random ``frac`` of samples.

    These are outliers, not change-points: a robust detector must ignore them.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(y, dtype=float).copy()
    k = int(frac * out.size)
    if k == 0:
        return out
    idx = rng.choice(out.size, k, replace=False)
    out[idx] += rng.choice([-1.0, 1.0], size=k) * spike
    return out


def synthetic_trials(
    n_trials: int = 10,
    T: int = 500,
    frac: float = 0.02,
    spike: float = 10.0,
    data_seed: int = 3000,
    contam_seed: int = 54321,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    ``n_trials`` independently contaminated copies of one clean signal.

    The clean signal is fixed; only the contamination varies, which isolates
    robustness from segmentation difficulty.
    """
    clean, true_cps = make_synthetic(T=T, seed=data_seed)
    trials = [
        contaminate(clean, frac=frac, spike=spike, seed=contam_seed + i)
        for i in range(n_trials)
    ]
    return trials, true_cps


def data_dir() -> str:
    """Path to the repository ``data/`` directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data")


def load_csv_column(path: str, column: str | int = -1, skip_header: bool = True) -> np.ndarray:
    """
    Minimal CSV column loader (no pandas dependency).

    ``column`` may be a header name or an integer index.
    """
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    start = 0
    if skip_header:
        header = [h.strip() for h in lines[0].split(",")]
        start = 1
        if isinstance(column, str):
            column = header.index(column)

    if isinstance(column, str):
        raise ValueError("column given by name but skip_header=False")

    vals = []
    for ln in lines[start:]:
        parts = ln.split(",")
        try:
            vals.append(float(parts[column]))
        except (ValueError, IndexError):
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)
