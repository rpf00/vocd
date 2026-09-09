"""R-FOCuS — outlier-robust FOCuS via a Huber-clipped score.

Reference reimplementation following the robust-CUSUM idea of

    Fearnhead & Rigaill (2019), "Change-point detection in the presence of
    outliers", JASA 114(525),

grafted onto the FOCuS recursion of Romano et al. (2023).  Identical to
:class:`alternative_models.focus.FOCuS` except that each standardised residual
is passed through Huber's influence function before entering the CUSUM.
Not the authors' code.

WHY HUBER, NOT A REDESCENDING BIWEIGHT.  A robust *detection* score must stay
bounded for isolated outliers yet keep responding to a *sustained* level shift.
Tukey's biweight redescends to exactly zero beyond its cap, so a genuine
5-10 sigma mean change is treated as a run of outliers and contributes ~0 to
the CUSUM — the detector goes blind to the very change it should find (this was
the cause of R-FOCuS's degenerate operating point in earlier runs).  Huber's
psi saturates to +/-c instead of redescending: an isolated spike contributes a
single bounded term, while a sustained shift of any magnitude contributes ~c
per sample and still accumulates to threshold.  The cap ``c = 1.345`` is the
classic 95%-efficiency Huber tuning and is fixed a priori (not tuned on the
series).
"""
from __future__ import annotations

from .focus import FOCuS


class RFOCuS(FOCuS):
    def __init__(self, horizon: int, alpha: float, min_seg: int = 10,
                 burnin: int = 10, c: float = 1.345):
        super().__init__(horizon, alpha, min_seg, burnin)
        self.c = float(c)

    def _psi(self, resid: float) -> float:
        # Huber psi: linear in [-c, c], saturating (not redescending) beyond.
        c = self.c
        if resid > c:
            return c
        if resid < -c:
            return -c
        return resid


def detect(x, alpha: float = 0.01, min_seg: int = 10, c: float = 1.345):
    import numpy as np
    from ._restart import run_restart
    x = np.asarray(x, float).ravel()
    n = x.size
    return run_restart(lambda: RFOCuS(n, alpha, min_seg, c=c), x, alpha, min_seg)
