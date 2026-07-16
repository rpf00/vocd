"""
Evaluation metrics for change-point detection.

Matching is one-to-one and greedy within a tolerance window: each true
change-point may be claimed by at most one detection and vice versa, so
clustered detections around one event count as one hit plus false alarms
rather than several hits.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = ["match", "evaluate"]


def match(detected: Sequence[int], true: Sequence[int], tol: int = 10):
    """
    Greedy one-to-one matching within +/- ``tol``.

    Returns (pairs, unmatched_detected, unmatched_true) where ``pairs`` is a
    list of (detected, true) tuples.
    """
    det = sorted(int(d) for d in detected)
    tru = sorted(int(t) for t in true)

    pairs: List[tuple] = []
    used_t = set()
    used_d = set()

    # Closest pairs first, so a detection is not stolen by a farther truth.
    cands = sorted(
        (
            (abs(d - t), d, t)
            for d in det
            for t in tru
            if abs(d - t) <= tol
        ),
        key=lambda x: (x[0], x[1], x[2]),
    )
    for _, d, t in cands:
        if d in used_d or t in used_t:
            continue
        used_d.add(d)
        used_t.add(t)
        pairs.append((d, t))

    return (
        sorted(pairs, key=lambda p: p[1]),
        [d for d in det if d not in used_d],
        [t for t in tru if t not in used_t],
    )


def evaluate(detected: Sequence[int], true: Sequence[int], tol: int = 10) -> Dict[str, float]:
    """
    Precision (PPV), recall (TPR), F1, and mean absolute detection delay.

    ``delay`` is the mean |detected - true| over matched pairs, in samples.
    """
    pairs, fp, fn = match(detected, true, tol)
    tp = len(pairs)

    ppv = tp / (tp + len(fp)) if (tp + len(fp)) else 0.0
    tpr = tp / (tp + len(fn)) if (tp + len(fn)) else 0.0
    f1 = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) else 0.0
    delay = float(np.mean([abs(d - t) for d, t in pairs])) if tp else float("nan")

    return {
        "n_detected": len(detected),
        "tp": tp,
        "fp": len(fp),
        "fn": len(fn),
        "ppv": ppv,
        "tpr": tpr,
        "f1": f1,
        "delay": delay,
    }
