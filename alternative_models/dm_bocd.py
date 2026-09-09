"""Dm-BOCD (Altamirano, Briol & Knoblauch 2023) — diffusion-score robust BOCD.

Thin wrapper over maltamiranomontero/DSM-bocd.  The score/weight functions,
priors and omega match the reference notebook ``accuracy_delay.ipynb``.
``omega`` is the method's calibrated robustness parameter (kept at the paper's
value); change-point extraction is causal by default (see ``bocd``).
"""
from __future__ import annotations

import numpy as np

from ._dsm import load, find_cp_causal

HAZARD_TIMESCALE = 300
OMEGA = 0.1


def detect(x: np.ndarray, mean0: float = None, dsm_path: str = None,
           extract: str = "causal", omega: float = OMEGA) -> list:
    x = np.asarray(x, float).reshape(-1, 1)
    if mean0 is None:
        mean0 = float(np.mean(x))
    m = load(dsm_path)

    mean_mu0, var_mu0 = mean0, 1
    mean_Sigma0, var_Sigma0 = 1, 1
    mu0 = np.array([[mean_mu0 / var_mu0], [1 / var_mu0]])
    Sigma0 = np.eye(2)
    Sigma0[0, 0] = mean_Sigma0 / var_Sigma0
    Sigma0[1, 1] = 1 / var_Sigma0

    def mfun(v):
        return np.array([(1 + v ** 2) ** (-1 / 2)])

    def grad_m(v):
        return np.array([[-v / ((1 + v ** 2) ** (3 / 2))]])

    model = m["DSMGaussian"](data=x, m=mfun, grad_m=grad_m,
                             omega=omega, mu0=mu0, Sigma0=Sigma0)
    R = m["bocpd"](x, m["ConstantHazard"](HAZARD_TIMESCALE), model)
    cps = find_cp_causal(R) if extract == "causal" else list(m["find_cp"](R))
    return sorted(int(c) for c in cps if c > 0)
