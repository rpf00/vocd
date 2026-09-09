"""Common interface + shared statistics for the alarm-style detectors."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


class OnlineAlarm:
    """Base class for a strictly-causal, restartable alarm detector.

    A subclass consumes one observation at a time via :meth:`update`, which
    returns ``(pvalue, cp_location)``: the current smallest p-value over all
    within-segment change hypotheses tested so far, and the absolute index of
    the most-likely change (or ``None`` if undefined yet).  The detector uses
    only the samples fed to it — never future data — and :meth:`reset`
    re-initialises it so :func:`_restart.run_restart` can restart after an
    alarm.
    """

    def reset(self, seg_start: int = 0) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, x: float) -> Tuple[float, Optional[int]]:  # pragma: no cover
        raise NotImplementedError


# ── shared helpers ────────────────────────────────────────────────────

def robust_scale(x: np.ndarray) -> float:
    """MAD-based scale estimate (consistent for the normal), floored > 0."""
    if x.size == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    return float(s) if s > 1e-9 else float(np.std(x) + 1e-9) or 1.0


def alpha_to_chi2_threshold(alpha: float, horizon: int, n_tests: int = 1) -> float:
    """Map a family-wise ``alpha`` to a threshold on a GLR statistic.

    The FOCuS-type statistic ``Q`` behaves, per candidate, like ``z**2 / 2`` for
    a two-sided Gaussian test.  Controlling the family-wise error over the
    ``horizon`` steps (and ``n_tests`` parallel tests, e.g. quantiles) by
    Bonferroni gives a per-test level ``alpha / (horizon * n_tests)`` and a
    threshold ``h = 0.5 * z_{level/2}**2``.  ``horizon`` is the series length —
    a scalar known at start, not a peek at the values.
    """
    level = alpha / max(1, horizon) / max(1, n_tests)
    level = min(max(level, 1e-15), 0.5)
    z = norm.isf(level / 2.0)
    return 0.5 * z * z
