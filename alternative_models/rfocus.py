"""R-FOCuS — outlier-robust FOCuS via a capped (biweight) score.

Reference reimplementation following the robust-CUSUM idea of

    Fearnhead & Rigaill (2019), "Change-point detection in the presence of
    outliers", JASA 114(525),

grafted onto the FOCuS recursion of Romano et al. (2023).  Identical to
:class:`models.focus.FOCuS` except that each standardised residual is passed
through Tukey's biweight influence function, so a single gross outlier
contributes a bounded amount to the CUSUM instead of an unbounded spike.
Not the authors' code.

The cap ``c`` is in robust-scale (MAD) units; ``c = 4.685`` is the classic
95%-efficiency biweight tuning and is used a priori (not tuned on the series).
"""
from __future__ import annotations

import numpy as np

from .focus import FOCuS


class RFOCuS(FOCuS):
    def __init__(self, horizon: int, alpha: float, min_seg: int = 10,
                 burnin: int = 10, c: float = 4.685):
        super().__init__(horizon, alpha, min_seg, burnin)
        self.c = float(c)

    def _psi(self, resid: float) -> float:
        # Tukey biweight psi (redescending), bounded in magnitude by ~0.385*c
        c = self.c
        if abs(resid) >= c:
            return 0.0
        u = resid / c
        return resid * (1.0 - u * u) ** 2


def detect(x: np.ndarray, alpha: float = 0.01, min_seg: int = 10,
           c: float = 4.685) -> list:
    from ._restart import run_restart
    x = np.asarray(x, float).ravel()
    n = x.size
    return run_restart(lambda: RFOCuS(n, alpha, min_seg, c=c), x, alpha, min_seg)
