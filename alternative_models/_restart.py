"""Restart-on-alarm driver: turn an online alarm detector into a segmentation.

This is the single mechanism that makes the alarm-style methods (FOCuS,
R-FOCuS, NP-FOCuS, CHAD) comparable to the segmentation methods on the same
task, WITHOUT letting them see the future:

* the detector is fed x[0], x[1], ... one at a time;
* the moment its statistic crosses the (a-priori) threshold, the current
  most-likely change location is committed as a change-point;
* the detector is reset and *re-fed only the samples already seen* since that
  change (x[cp+1 : t+1]), so no post-change information is discarded and no
  future information is used;
* a minimum segment length guards against immediate re-triggering.

All four alarm detectors share this driver and one ``alpha`` budget, so the
comparison is fair by construction.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from ._base import OnlineAlarm


def run_restart(
    factory: Callable[[], OnlineAlarm],
    x: np.ndarray,
    alpha: float,
    min_seg: int = 10,
) -> List[int]:
    """Drive ``factory()`` over ``x`` causally, returning committed change-points.

    Parameters
    ----------
    factory   : builds a fresh :class:`OnlineAlarm` (given the horizon via a
                closure, if it needs it).
    x         : 1-D signal.
    alpha     : family-wise error budget; each detector maps it to its own
                threshold in :meth:`OnlineAlarm.update` via ``_base``.
    min_seg   : minimum number of samples between consecutive change-points
                (also the per-segment burn-in the detectors use to estimate a
                scale).  Shared across methods for fairness.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    cps: List[int] = []
    seg_start = 0
    det = factory()
    det.reset(seg_start=seg_start)

    t = 0
    while t < n:
        p, loc = det.update(x[t])
        fired = (p < alpha) and (t - seg_start + 1 >= 2 * min_seg)
        if fired:
            cp = loc if (loc is not None and loc > seg_start) else t
            cp = int(min(max(cp, seg_start + 1), n - 1))
            cps.append(cp)
            seg_start = cp
            det.reset(seg_start=seg_start)
            # re-feed only already-seen post-change samples (no lookahead)
            for s in range(cp + 1, t + 1):
                det.update(x[s])
        t += 1

    # unique + sorted
    return sorted(set(int(c) for c in cps))
