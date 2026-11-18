"""NUNC — Non-parametric UNbounded Changepoint detection in a rolling window.

Reference reimplementation following

    Austin, Romano, Eckley & Fearnhead (2023), "Online non-parametric
    changepoint detection with application to monitoring operational
    performance of network devices", Computational Statistics & Data
    Analysis 177, 107551.

Not the authors' code.

NUNC keeps a sliding window of the last ``W`` observations and, for every
split of that window, scores the change with the non-parametric cost of
Haynes, Fearnhead & Eckley (2017): the empirical CDF is evaluated at K
quantiles and each quantile contributes a binomial (Bernoulli) deviance.
The statistic is the sum over quantiles of the deviance REDUCTION obtained
by splitting the window, maximised over the split point.  This is the
"NUNC Local" variant: the K quantiles are recomputed from the current
window, so no training / probation period is needed.

Threshold calibration.  As in Austin et al., the alarm threshold is set by
Monte Carlo simulation under H0.  Because the statistic depends on the data
only through ranks / eCDF values it is DISTRIBUTION-FREE under the null, so
simulating i.i.d. Gaussian windows calibrates it exactly for any continuous
null distribution.  ``update`` returns the empirical null survival
probability of the observed max-statistic (the fraction of simulated
windows whose max-statistic is at least as large), which already accounts
for the multiplicity over split points; the harness fires when it drops
below the shared ``alpha``.  Below the simulation's resolution the survival
function is extrapolated with an exponential tail fitted to the top 5% of
null draws, so small family-wise budgets on the sweep grid remain
meaningful.  Calibration is done lazily per window length and is part of the
method's cost: a fresh set of null tables is built inside every
``detect`` call, so the calibration is charged to that call's runtime
exactly as every other detector's work is, and nothing is carried over
between series.  Within one call the tables are shared across restarts
(the method would hold them in memory).  ``shared_cache=True`` reuses
tables across calls for quick development iteration only; it is NOT a
fair setting for reported runtimes and the driver defaults it off.

The window size ``W`` is NUNC's dominant tuning parameter (Austin et al.
show performance is strongly window-dependent).  It is fixed A PRIORI and
not tuned on the test series; report it.  The default W=50 equals the total
number of points VOCD's gate inspects (2 x 25), so the two windowed tests
see the same amount of data.  While the current segment is shorter than W
the window simply grows (otherwise NUNC would be blind for the first W
points after every restart, which is unfair under ``min_seg=10`` restarts).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import xlogy

from ._base import OnlineAlarm

_SHARED_NULL_CACHE: Dict[tuple, tuple] = {}
_TAIL_FRAC = 0.95


def _binom_deviance(c: np.ndarray, n: np.ndarray) -> np.ndarray:
    """-2 * log-likelihood of the Bernoulli fit at eCDF value c/n (0 log 0 = 0)."""
    return -2.0 * (xlogy(c, c / n) + xlogy(n - c, (n - c) / n))


def _max_stat_batch(X: np.ndarray, levels: np.ndarray, min_side: int):
    """NUNC Local max-over-splits statistic for every row of ``X`` (B, n).

    Returns ``(stat, j)``: the max statistic per row and the left-block
    length at which it is attained.
    """
    B, n = X.shape
    qs = np.quantile(X, levels, axis=1).T                     # (B, K)
    ind = (X[:, :, None] <= qs[:, None, :]).astype(float)     # (B, n, K)
    csum = np.cumsum(ind, axis=1)                             # left counts
    tot = csum[:, -1, :]                                      # (B, K)
    js = np.arange(min_side, n - min_side + 1)                # left lengths
    cL = csum[:, js - 1, :]                                   # (B, nj, K)
    cR = tot[:, None, :] - cL
    nL = js[None, :, None].astype(float)
    nR = (n - js)[None, :, None].astype(float)
    full = _binom_deviance(tot[:, None, :], float(n))
    red = full - _binom_deviance(cL, nL) - _binom_deviance(cR, nR)
    stat = red.sum(axis=2)                                    # (B, nj)
    jbest = np.argmax(stat, axis=1)
    return stat[np.arange(B), jbest], js[jbest]


def _null_table(n: int, K: int, min_side: int, n_sims: int, seed: int,
                cache: Dict[tuple, tuple]):
    """Sorted null max-statistics for window length n, plus a fitted
    exponential tail (log-survival linear in the statistic) for p-values
    below the simulation's resolution."""
    key = (n, K, min_side, n_sims, seed)
    if key in cache:
        return cache[key]
    rng = np.random.default_rng(seed)
    levels = np.arange(1, K + 1) / (K + 1)
    stats = np.empty(n_sims)
    chunk = max(1, min(n_sims, 4000))                # bound memory (B*n*K)
    for a in range(0, n_sims, chunk):
        b = min(n_sims, a + chunk)
        stats[a:b], _ = _max_stat_batch(rng.standard_normal((b - a, n)),
                                        levels, min_side)
    null = np.sort(stats)
    start = int(_TAIL_FRAC * n_sims)
    top = null[start:]
    sf = (n_sims - np.arange(start, n_sims)) / n_sims      # survival at each
    slope, intercept = np.polyfit(top, np.log(sf), 1)
    out = (null, float(null[start]), float(slope), float(intercept))
    cache[key] = out
    return out


class NUNC(OnlineAlarm):
    def __init__(self, horizon: int, alpha: float, min_seg: int = 10,
                 window: int = 50, n_quantiles: Optional[int] = None,
                 min_side: Optional[int] = None, n_sims: int = 20000,
                 seed: int = 0, cache: Optional[Dict[tuple, tuple]] = None):
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self.min_seg = int(min_seg)
        self.W = int(window)
        # Haynes et al. / Austin et al. default: K = ceil(log(2W - 1)).
        self.K = int(n_quantiles) if n_quantiles else \
            max(3, int(np.ceil(np.log(2 * self.W - 1))))
        # smallest block on either side of a split; = CHAD's h0 for parity
        self.min_side = int(min_side) if min_side else self.min_seg
        if self.W < 2 * self.min_side:
            raise ValueError(
                f"NUNC window W={self.W} can never hold two blocks of min_side="
                f"{self.min_side}: no split would ever be tested. Use window >= "
                f"2*min_side (e.g. max(50, 2*min_seg)) or a smaller min_side.")
        self.n_sims = int(n_sims)
        self.seed = int(seed)
        self._cache = cache if cache is not None else {}
        self._levels = np.arange(1, self.K + 1) / (self.K + 1)

    def reset(self, seg_start: int = 0) -> None:
        self.seg_start = int(seg_start)
        self._buf = []

    def _pvalue(self, S: float, n: int) -> float:
        null, tail_start, slope, intercept = _null_table(
            n, self.K, self.min_side, self.n_sims, self.seed, self._cache)
        if S < tail_start:
            m = null.size
            idx = int(np.searchsorted(null, S, side="left"))
            return float((m - idx + 1) / (m + 1))             # empirical sf
        p = float(np.exp(intercept + slope * S))              # fitted tail
        return float(min(1.0, max(p, 1e-300)))

    def update(self, x: float) -> Tuple[float, Optional[int]]:
        self._buf.append(float(x))
        m = len(self._buf)
        n = min(self.W, m)                       # growing-then-sliding window
        if n < 2 * self.min_side:
            return 1.0, None
        w = np.asarray(self._buf[-n:])[None, :]
        stat, j = _max_stat_batch(w, self._levels, self.min_side)
        p = self._pvalue(float(stat[0]), n)
        # absolute index of the first point of the right block (new regime)
        loc = self.seg_start + (m - n) + int(j[0])
        return p, loc


def detect(x: np.ndarray, alpha: float = 0.01, min_seg: int = 10,
           window: int = 50, n_quantiles: Optional[int] = None,
           n_sims: int = 20000, shared_cache: bool = False,
           cache: Optional[Dict[tuple, tuple]] = None) -> list:
    """``cache``: a caller-owned dict for the null tables.  Pass the same dict
    across the prefix-replay calls of ONE record so the calibration is paid
    once per record evaluation (a deployed NUNC calibrates once per sensor),
    while still being inside the caller's timed region.  Default: a fresh
    cache per call."""
    from ._restart import run_restart
    x = np.asarray(x, float).ravel()
    n = x.size
    # One cache per call unless the caller owns one: null tables are rebuilt
    # for this series and shared only across its own restarts, so calibration
    # is part of the timed cost.
    if cache is None:
        cache = _SHARED_NULL_CACHE if shared_cache else {}
    return run_restart(lambda: NUNC(n, alpha, min_seg, window, n_quantiles,
                                    n_sims=n_sims, cache=cache),
                       x, alpha, min_seg)
