"""Locate and import a clone of maltamiranomontero/DSM-bocd (the BOCD baselines).

The Bayesian baselines are intentionally NOT vendored, so they run on their own
terms.  Point at a clone with the ``VOCD_DSM_PATH`` env var or the ``dsm_path``
argument::

    git clone https://github.com/maltamiranomontero/DSM-bocd.git
    export VOCD_DSM_PATH=./DSM-bocd
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Dict, Optional


@lru_cache(maxsize=4)
def load(dsm_path: Optional[str] = None) -> Dict:
    path = dsm_path or os.environ.get("VOCD_DSM_PATH")
    if not path:
        raise RuntimeError(
            "BOCD baselines need a DSM-bocd clone. Set VOCD_DSM_PATH or pass "
            "--dsm-path.\n  git clone https://github.com/maltamiranomontero/DSM-bocd.git"
        )
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise RuntimeError(f"DSM-bocd path not found: {path}")
    if path not in sys.path:
        sys.path.insert(0, path)
    from bocpd import bocpd
    from hazard import ConstantHazard
    from models import DSMGaussian, Gaussian
    from utils.find_cp import find_cp
    return {"bocpd": bocpd, "ConstantHazard": ConstantHazard,
            "Gaussian": Gaussian, "DSMGaussian": DSMGaussian, "find_cp": find_cp}


def find_cp_causal(R, lag: int = 20):
    """Causal change-point read-off from the run-length matrix.

    The upstream ``find_cp`` scans the full matrix and can DELETE a previously
    reported change-point when later rows disagree — i.e. it revises the past
    with the future, which is not online.  This variant commits a change-point
    the first time the MAP run-length implies one and never revises it, using
    only rows up to the current step.  It is the default in the harness so the
    BOCD baselines meet the same no-lookahead bar as every other method; pass
    ``extract="upstream"`` to recover the published behaviour.
    """
    import numpy as np
    n = len(R)
    CPs, last = [], 0
    for i in range(n):
        cand = i - int(np.argmax(R[i, : i + 1]))
        if cand > last + lag and cand not in CPs:
            CPs.append(cand)
            last = cand
    return CPs
