"""BOCD (Adams & MacKay 2007) — standard conjugate-Gaussian baseline.

Thin wrapper over maltamiranomontero/DSM-bocd; priors and hazard match the
synthetic protocol of Altamirano et al. (2023).  Change-points are extracted
causally by default (see ``models._dsm.find_cp_causal``); ``extract="upstream"``
reproduces the published operating point but is not strictly online.  ``hazard_timescale`` (default 300) is exposed only
so the sweep can find this baseline's own best operating point; the DSM-bocd
algorithm itself is unchanged.
"""
from __future__ import annotations

import numpy as np

from ._dsm import load, find_cp_causal

HAZARD_TIMESCALE = 300


def detect(x: np.ndarray, mean0: float = None, dsm_path: str = None,
           extract: str = "causal",
           hazard_timescale: float = HAZARD_TIMESCALE) -> list:
    x = np.asarray(x, float).reshape(-1, 1)
    if mean0 is None:
        mean0 = float(np.mean(x))
    m = load(dsm_path)
    model = m["Gaussian"](mu0=mean0, kappa0=1, alpha0=1, omega0=1)
    R = m["bocpd"](x, m["ConstantHazard"](hazard_timescale), model)
    cps = find_cp_causal(R) if extract == "causal" else list(m["find_cp"](R))
    return sorted(int(c) for c in cps if c > 0)
