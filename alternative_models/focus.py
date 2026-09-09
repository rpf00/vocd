"""FOCuS — functional-pruning CUSUM for a Gaussian change in mean.

Reference reimplementation following

    Romano, Eckley, Fearnhead & Rigaill (2023), "Fast online change-point
    detection via functional pruning CUSUM statistics", JMLR 24(81).

NOTE ON FIDELITY.  The authors' contribution is an O(log n)-amortised
functional-pruning recursion.  For the T=500 synthetic task we compute the
*same* statistic exactly by an O(t) scan per step (O(n^2) total) — the
detection decisions are identical to the pruned version; only the runtime
differs.  Swap in the authors' pruning if you need the published complexity.
Not the authors' code.

The pre-change mean is estimated online as the current-segment running mean
after a short burn-in; the scale is a per-segment MAD (see ``_base``).  The
statistic is the two-sided page-CUSUM/GLR
    Q_t = max_{seg_start < tau <= t}  (S_t - S_tau)^2 / (2 (t - tau)),
mapped to a p-value against the family-wise threshold in ``_base``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._base import OnlineAlarm, alpha_to_chi2_threshold, robust_scale


class FOCuS(OnlineAlarm):
    def __init__(self, horizon: int, alpha: float, min_seg: int = 10,
                 burnin: int = 10):
        self.horizon = int(horizon)
        self.h = alpha_to_chi2_threshold(alpha, horizon, n_tests=1)
        self.burnin = int(burnin)

    # -- transform applied to each raw sample; overridden by R-FOCuS ----
    def _psi(self, resid: float) -> float:
        return resid

    def reset(self, seg_start: int = 0) -> None:
        self.seg_start = int(seg_start)
        self._buf = []           # raw samples of current segment
        self._csum = [0.0]       # cumulative sum of transformed, centred resid
        self._mu = None
        self._scale = None

    def update(self, x: float) -> Tuple[float, Optional[int]]:
        self._buf.append(float(x))
        k = len(self._buf)
        # burn-in: fix segment location/scale, no test yet
        if k <= self.burnin:
            arr = np.asarray(self._buf)
            self._mu = float(np.median(arr))
            self._scale = robust_scale(arr)
            self._csum = [0.0]
            for v in self._buf:
                self._csum.append(self._csum[-1] + self._psi((v - self._mu) / self._scale))
            return 1.0, None
        r = self._psi((x - self._mu) / self._scale)
        self._csum.append(self._csum[-1] + r)
        S = np.asarray(self._csum)          # length k+1, S[0]=0 .. S[k]
        # Q = max over tau in [1, k-1] of (S[k]-S[tau])^2 / (2 (k - tau))
        taus = np.arange(1, k)
        num = (S[k] - S[taus]) ** 2
        den = 2.0 * (k - taus)
        q = num / den
        j = int(np.argmax(q))
        Q = float(q[j])
        tau = taus[j]
        # p-value against the shared Gaussian-GLR threshold
        p = 0.0 if Q >= self.h else 1.0
        loc = self.seg_start + int(tau)     # absolute index of change
        return p, loc


def detect(x: np.ndarray, alpha: float = 0.01, min_seg: int = 10) -> list:
    from ._restart import run_restart
    x = np.asarray(x, float).ravel()
    n = x.size
    return run_restart(lambda: FOCuS(n, alpha, min_seg), x, alpha, min_seg)
