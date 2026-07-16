"""
Multiple-testing correction over the VOCD candidate set.

VOCD proposes K candidate change-points and tests each one.  Testing K
hypotheses at a fixed per-test level alpha does not control any global error
rate: the expected number of false alarms grows with K.  This module converts
the user's *error budget* into a per-candidate decision rule.

This is the reason VOCD needs no alpha calibration.  The user states what
false-alarm rate they can tolerate; the threshold follows from the data.

Methods
-------
bonferroni           : controls FWER (P(any false alarm) <= alpha).
benjamini_hochberg   : controls FDR under independence or PRDS.
benjamini_yekutieli  : controls FDR under *arbitrary* dependence.

On dependence
-------------
VOCD's tests use windows of half-width ``verify_w`` around each candidate.
When candidates are closer together than ``2 * verify_w`` their windows
overlap, so the p-values are dependent and the PRDS condition that BH relies
on is not established.  Use ``benjamini_yekutieli`` when you need a guarantee
under dependence; it is valid unconditionally at the cost of a log(K) factor.

On post-selection inference
---------------------------
VOCD's candidates are chosen by the Viterbi passes *from the same data* the
tests then use.  The p-values are therefore anti-conservative under the null:
we are testing locations selected precisely because they look like changes.
Realised FDR may exceed the nominal level.  See Jewell, Fearnhead & Witten
(2022, JRSS-B) for the conditional correction in the change-point setting.
Treat the level below as a *target*, not an exact guarantee, and validate
empirically on data with known ground truth.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bonferroni",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "select",
]


def bonferroni(p: np.ndarray, alpha: float) -> np.ndarray:
    """Reject where ``p <= alpha / K``.  Controls FWER under any dependence."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    return p <= alpha / p.size


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """
    Benjamini-Hochberg step-up procedure.  Controls FDR at ``q`` under
    independence or positive regression dependence (PRDS).
    """
    p = np.asarray(p, dtype=float)
    K = p.size
    if K == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p, kind="mergesort")  # stable: deterministic on ties
    p_sorted = p[order]
    thresholds = q * np.arange(1, K + 1) / K
    passed = p_sorted <= thresholds

    reject = np.zeros(K, dtype=bool)
    if passed.any():
        k_max = int(np.nonzero(passed)[0].max())
        reject[order[: k_max + 1]] = True
    return reject


def benjamini_yekutieli(p: np.ndarray, q: float) -> np.ndarray:
    """
    Benjamini-Yekutieli procedure: BH run at ``q / c(K)`` where
    ``c(K) = sum_{i=1..K} 1/i``.  Controls FDR under arbitrary dependence.
    """
    p = np.asarray(p, dtype=float)
    K = p.size
    if K == 0:
        return np.zeros(0, dtype=bool)
    c_K = float(np.sum(1.0 / np.arange(1, K + 1)))
    return benjamini_hochberg(p, q / c_K)


def select(
    p: np.ndarray,
    *,
    fdr: float | None = None,
    fwer: float | None = None,
    alpha: float | None = None,
    method: str = "bh",
) -> np.ndarray:
    """
    Apply exactly one decision rule to a vector of p-values.

    Parameters
    ----------
    p : array of p-values, one per candidate.
    fdr : target false-discovery rate.  Uses ``method`` ('bh' or 'by').
    fwer : target family-wise error rate.  Uses Bonferroni.
    alpha : raw per-test threshold, *no* correction applied.  Provided for
        reproducing fixed-alpha results; does not control any global rate.
    method : 'bh' or 'by', only consulted when ``fdr`` is given.

    Returns
    -------
    Boolean mask of rejections (i.e. accepted change-points).
    """
    given = [x is not None for x in (fdr, fwer, alpha)]
    if sum(given) != 1:
        raise ValueError(
            "specify exactly one of fdr=, fwer=, or alpha= "
            f"(got {sum(given)})"
        )

    p = np.asarray(p, dtype=float)

    if fwer is not None:
        return bonferroni(p, fwer)
    if alpha is not None:
        return p <= alpha
    if method == "bh":
        return benjamini_hochberg(p, fdr)
    if method == "by":
        return benjamini_yekutieli(p, fdr)
    raise ValueError(f"unknown method {method!r}; expected 'bh' or 'by'")
