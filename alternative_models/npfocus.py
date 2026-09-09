"""NP-FOCuS — nonparametric FOCuS over an empirical-quantile grid.

Reference reimplementation following

    Romano, Eckley & Fearnhead (2024), "A log-linear nonparametric online
    change-point detection algorithm based on functional pruning", IEEE TSP.

Idea: run a Bernoulli/Gaussian-approx FOCuS recursion on the indicator series
``1{x_t <= q_k}`` for each quantile ``q_k`` on a fixed grid, then combine
across the grid by the maximum statistic.  This is distribution-free: it reacts
to a change in *any* part of the marginal, not only the mean.  The quantiles
are estimated once per segment from the burn-in (past data only), and the
family-wise budget is split across both the horizon and the grid size in
``_base``.  Not the authors' code.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._base import OnlineAlarm, alpha_to_chi2_threshold


class NPFOCuS(OnlineAlarm):
    def __init__(self, horizon: int, alpha: float, min_seg: int = 10,
                 quantiles=(0.25, 0.5, 0.75), burnin: int = 15):
        self.horizon = int(horizon)
        self.qs = tuple(quantiles)
        self.h = alpha_to_chi2_threshold(alpha, horizon, n_tests=len(self.qs))
        self.burnin = int(burnin)

    def reset(self, seg_start: int = 0) -> None:
        self.seg_start = int(seg_start)
        self._buf = []
        self._thr = None                    # quantile thresholds q_k
        self._csum = None                   # per-quantile cumulative sums

    def update(self, x: float) -> Tuple[float, Optional[int]]:
        self._buf.append(float(x))
        k = len(self._buf)
        if k <= self.burnin:
            arr = np.asarray(self._buf)
            self._thr = np.quantile(arr, self.qs)
            # centred Bernoulli increments 1{x<=q} - p_k, rebuilt each burn-in step
            self._csum = [np.zeros(len(self.qs))]
            for v in self._buf:
                ind = (v <= self._thr).astype(float)
                self._csum.append(self._csum[-1] + (ind - self.qs))
            return 1.0, None
        ind = (x <= self._thr).astype(float)
        self._csum.append(self._csum[-1] + (ind - np.asarray(self.qs)))
        S = np.asarray(self._csum)          # (k+1, n_q)
        taus = np.arange(1, k)
        # per-quantile Gaussian-approx GLR, then max across the grid
        num = (S[k] - S[taus]) ** 2         # (len(taus), n_q)
        den = (2.0 * (k - taus))[:, None]
        q = num / den
        qmax_per_tau = q.max(axis=1)
        j = int(np.argmax(qmax_per_tau))
        Q = float(qmax_per_tau[j])
        tau = int(taus[j])
        p = 0.0 if Q >= self.h else 1.0
        return p, self.seg_start + tau


def detect(x: np.ndarray, alpha: float = 0.01, min_seg: int = 10,
           quantiles=(0.25, 0.5, 0.75)) -> list:
    from ._restart import run_restart
    x = np.asarray(x, float).ravel()
    n = x.size
    return run_restart(lambda: NPFOCuS(n, alpha, min_seg, quantiles), x, alpha, min_seg)
