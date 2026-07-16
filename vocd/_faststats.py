"""
Fast two-sample tests.

``scipy.stats.mannwhitneyu`` and ``ks_2samp`` spend almost all of their time in
input validation (``_axis_nan_policy_wrapper``), not arithmetic: roughly 1.5 ms
per candidate to compare fifty numbers, against ~10 us of real work.  VOCD
calls them once per candidate, so that overhead is a large share of runtime.

This module computes the same p-values without the wrapper.  SciPy remains the
definition of correctness -- ``tests/test_equivalence.py`` fuzzes these
functions against it and requires agreement to ~1e-12.  ``_verify_reference``
in ``core`` keeps the SciPy path available as a permanent oracle.

Scope of the fast paths
-----------------------
Mann-Whitney: the normal approximation with tie correction and continuity
correction.  SciPy's ``method='auto'`` selects that only when *both* samples
exceed 8 (``if n1 > 8 and n2 > 8``), or when ties are present; otherwise it
uses an exact permutation calculation.  So whenever either sample is 8 or
smaller we defer to SciPy rather than risk the wrong distribution.  With
``verify_w=25`` that only arises at the very edges of a series.

Kolmogorov-Smirnov: the two-sample statistic is computed here and the exact
p-value is obtained from SciPy's own routine, bypassing only the validation
layer.  The asymptotic Kolmogorov series is *not* used: for the window sizes
VOCD tests (n ~ 25-50) SciPy uses the exact distribution, and the two disagree
by up to five orders of magnitude in the far tail -- exactly where VOCD's
thresholds sit.  If the private routine is unavailable the public function is
used instead, so results never depend on it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import ks_2samp as _scipy_ks
from scipy.stats import mannwhitneyu as _scipy_mw

__all__ = ["mannwhitney_p", "ks2samp_p"]

# SciPy's exact two-sample KS routine, minus the validation wrapper.
try:  # pragma: no cover - availability depends on the SciPy build
    from scipy.stats._stats_py import _attempt_exact_2kssamp as _exact_ks
except Exception:  # pragma: no cover
    _exact_ks = None

_MW_MIN_ASYMPTOTIC_N = 9  # SciPy: asymptotic only when both n > 8
_KS_EXACT_MAX_NM = 10000  # SciPy uses exact KS while n1*n2 <= this
_SQRT2 = math.sqrt(2.0)


def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal, via erfc (no SciPy dispatch)."""
    return 0.5 * math.erfc(z / _SQRT2)


def _ranks_and_tie_counts(x: np.ndarray):
    """
    Average ranks (1-based) and the size of each tied group.

    Equivalent to ``scipy.stats.rankdata(x, method='average')`` plus the group
    sizes needed for the tie correction, in one pass.
    """
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]

    starts = np.flatnonzero(np.r_[True, sx[1:] != sx[:-1]])
    ends = np.r_[starts[1:], n]
    counts = ends - starts
    # Mean of the 1-based positions spanned by each tied group.
    avg = (starts + ends - 1) / 2.0 + 1.0

    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat(avg, counts)
    return ranks, counts


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value.  Matches ``scipy.mannwhitneyu``."""
    n1, n2 = a.size, b.size
    if min(n1, n2) < _MW_MIN_ASYMPTOTIC_N:
        # SciPy may use the exact permutation distribution here; defer to it.
        return float(_scipy_mw(a, b, alternative="two-sided")[1])

    pooled = np.concatenate((a, b))
    ranks, counts = _ranks_and_tie_counts(pooled)

    n = n1 + n2
    u1 = float(ranks[:n1].sum()) - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = max(u1, u2)                       # two-sided: SciPy takes the larger

    mu = n1 * n2 / 2.0
    tie = float(np.sum(counts.astype(float) ** 3 - counts))
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1.0)))
    if var <= 0:
        return 1.0

    # Continuity correction: always subtract, since the SF is always used.
    z = (u - mu - 0.5) / math.sqrt(var)
    return min(1.0, 2.0 * _norm_sf(z))


def ks2samp_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided two-sample KS p-value.  Matches ``scipy.ks_2samp``."""
    n1, n2 = a.size, b.size
    if _exact_ks is None or n1 * n2 > _KS_EXACT_MAX_NM:
        return float(_scipy_ks(a, b)[1])

    sa = np.sort(a)
    sb = np.sort(b)
    pooled = np.concatenate((sa, sb))
    cdf_a = np.searchsorted(sa, pooled, side="right") / n1
    cdf_b = np.searchsorted(sb, pooled, side="right") / n2
    d = float(np.max(np.abs(cdf_a - cdf_b)))

    g = math.gcd(n1, n2)
    success, _, prob = _exact_ks(n1, n2, g, d, "two-sided")
    if not success:
        return float(_scipy_ks(a, b)[1])
    return float(np.clip(prob, 0.0, 1.0))
