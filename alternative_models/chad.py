"""CHAD — grid-based online wrapper around a two-sample test.

Reference reimplementation in the spirit of

    Moen (2026), "CHAD: online change-point detection by wrapping offline
    tests on a dyadic grid" (working title; align citation with the version
    you cite in the paper).

At each time ``t`` the detector maintains a dyadic grid of window half-widths
``h in {h0, 2 h0, 4 h0, ...}`` and, for each admissible ``h`` (needs
``t >= 2h`` within the current segment), applies an OFFLINE two-sample test to
the two adjacent past blocks
    A = x[t-2h : t-h]      B = x[t-h : t].
Both blocks lie in the past, so the test is strictly causal — CHAD's grid is
exactly what lets an offline test run online with O(log n) work and storage.
We use a distribution-free rank/ECDF two-sample test (Mann-Whitney by default,
Kolmogorov-Smirnov optional) so CHAD enters the panel as a fair distribution-
free competitor rather than a Gaussian test in disguise.  The change estimate
is the block boundary ``t-h`` of the most significant window; the family-wise
budget is split across the grid by Bonferroni.  Not the authors' code.

The wrapped test is an explicit choice (report it as ``CHAD(MW)`` etc.): CHAD's
behaviour is a property of the test it schedules, not of the schedule alone.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp

from ._base import OnlineAlarm


class CHAD(OnlineAlarm):
    def __init__(self, horizon: int, alpha: float, min_seg: int = 10,
                 test: str = "mw", h0: Optional[int] = None):
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self.min_seg = int(min_seg)
        self.h0 = int(h0) if h0 else int(min_seg)
        self.test = test

    def reset(self, seg_start: int = 0) -> None:
        self.seg_start = int(seg_start)
        self._buf = []

    def _grid(self, m: int):
        h = self.h0
        while 2 * h <= m:
            yield h
            h *= 2

    def update(self, x: float) -> Tuple[float, Optional[int]]:
        self._buf.append(float(x))
        m = len(self._buf)
        arr = np.asarray(self._buf)
        hs = list(self._grid(m))
        if not hs:
            return 1.0, None
        best_p, best_loc = 1.0, None
        for h in hs:
            A = arr[m - 2 * h: m - h]
            B = arr[m - h: m]
            try:
                if self.test == "ks":
                    p = float(ks_2samp(A, B, method="asymp").pvalue)
                else:
                    p = float(mannwhitneyu(A, B, alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
            if p < best_p:
                best_p, best_loc = p, self.seg_start + (m - h)
        # Bonferroni across the grid actually tested this step
        p_adj = min(1.0, best_p * len(hs))
        return p_adj, best_loc


def detect(x: np.ndarray, alpha: float = 0.01, min_seg: int = 10,
           test: str = "mw") -> list:
    from ._restart import run_restart
    x = np.asarray(x, float).ravel()
    n = x.size
    return run_restart(lambda: CHAD(n, alpha, min_seg, test=test), x, alpha, min_seg)
