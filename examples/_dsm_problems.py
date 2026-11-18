"""
Shared helpers for the four DSM-bocd real-data problems (well-log, bond,
cryptocrash, flashcrash).

Two things live here:
  1. load_dsm(dsm_path): imports the AUTHOR's classes from a DSM-bocd clone
     (bocpd, ConstantHazard, find_cp, the DSM* models, the Omega estimators).
     Dm-BOCD is run with the author's PER-PROBLEM configuration (his model,
     priors, b, hazard, and estimated omega) — faithful to his notebooks.
  2. our_panel_1d(...): runs OUR consistent panel on a 1-D series —
     vocd, bocd, np-focus, chad — at the synthetic defaults
     (fwer/alpha = 0.01, min_seg = 10).  'bocd' here is our standard conjugate
     wrapper (kept identical to every other experiment in the paper); Dm-BOCD is
     the author's faithful config, handled per-problem in each script.

There is NO ground truth in these problems: output is the set of change-points
each method predicts (index into the series).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vocd
from alternative_models import REGISTRY

OUR_METHODS = ["vocd", "bocd", "np-focus", "chad"]
DEFAULT_ALPHA = 0.01     # synthetic default (vocd fwer / alarm alpha)
DEFAULT_MIN_SEG = 10     # synthetic default
BOCD_HAZARD = 300        # our standard-BOCD wrapper default (synthetic)


# ── author's m / grad_m (verbatim from his notebooks) ─────────────────

def m1(x):                                  # 1-D (well-log, bond, flashcrash)
    return np.asarray([(1 + x ** 2) ** (-1 / 2)])


def grad_m1(x):
    return np.asarray([[-x / ((1 + x ** 2) ** (3 / 2))]])


def m2(x):                                  # 2-D (cryptocrash)
    M = np.eye(2)
    M[0, 0] = (1 + x[0] ** 2) ** (-1 / 2)
    M[1, 1] = (1 + x[1] ** 2) ** (-1 / 2)
    return M


def grad_m2(x):
    a = np.zeros((2, 2)); a[0, 0] = -x[0] / ((1 + x[0] ** 2) ** (3 / 2))
    b = np.zeros((2, 2)); b[1, 1] = -x[1] / ((1 + x[1] ** 2) ** (3 / 2))
    return np.concatenate((np.asarray([a]), np.asarray([b])), axis=0)


# ── author's library (from a DSM-bocd clone) ──────────────────────────

@lru_cache(maxsize=2)
def load_dsm(dsm_path=None):
    path = dsm_path or os.environ.get("VOCD_DSM_PATH")
    if not path:
        raise SystemExit("need a DSM-bocd clone (--dsm-path / VOCD_DSM_PATH):\n"
                         "  git clone https://github.com/maltamiranomontero/DSM-bocd.git")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise SystemExit(f"DSM-bocd path not found: {path}")
    if path not in sys.path:
        sys.path.insert(0, path)
    from bocpd import bocpd
    from hazard import ConstantHazard
    from models import (Gaussian, MultivariateGaussian,
                        DSMGaussian, DSMGamma, DSMMultivariateGaussian)
    from utils.find_cp import find_cp
    # our CAUSAL extractor (commit-on-arrival, no future revision) — used for
    # Dm-BOCD here so every method is strictly online, unlike upstream find_cp
    from alternative_models._dsm import find_cp_causal
    # NB: the omega estimators need `jax`; imported lazily via load_estimators()
    return dict(bocpd=bocpd, ConstantHazard=ConstantHazard, find_cp=find_cp,
                find_cp_causal=find_cp_causal,
                Gaussian=Gaussian, MultivariateGaussian=MultivariateGaussian,
                DSMGaussian=DSMGaussian, DSMGamma=DSMGamma,
                DSMMultivariateGaussian=DSMMultivariateGaussian)


def load_estimators(dsm_path=None):
    """Author's omega estimators (require `jax` — his dependency)."""
    load_dsm(dsm_path)                       # ensures dsm_path on sys.path
    try:
        from omega_estimator import OmegaEstimatorGaussian, OmegaEstimatorGamma
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"omega estimation needs the author's dependency: {e}. "
            "Install it (pip install jax) or pass --omega to skip estimation.")
    return OmegaEstimatorGaussian, OmegaEstimatorGamma


# ── our panel on a 1-D series ─────────────────────────────────────────

def our_panel_1d(y, dsm_path=None, alpha=DEFAULT_ALPHA, min_seg=DEFAULT_MIN_SEG,
                 methods=OUR_METHODS, bocd_hazard=BOCD_HAZARD):
    """Return {method: sorted changepoint indices} on 1-D array y."""
    y = np.asarray(y, float).ravel()
    out = {}
    for name in methods:
        if name == "vocd":
            try:
                cps = [int(c) for c in vocd.detect(y, backend="auto")
                       .changepoints(fwer=alpha)]
            except Exception:
                cps = []
        elif name == "bocd":
            fn, _ = REGISTRY["bocd"]
            cps = fn(y, mean0=None, dsm_path=dsm_path, extract="causal",
                     hazard_timescale=bocd_hazard)
        else:  # np-focus / chad
            fn, _ = REGISTRY[name]
            cps = fn(y, alpha=alpha, min_seg=min_seg)
        out[name] = sorted(int(c) for c in cps if 0 < c < len(y))
    return out


def print_cps(title, cps_by_method, dates=None):
    print(f"\n=== {title} — predicted change-points (series index) ===")
    for name, cps in cps_by_method.items():
        if dates is not None:
            shown = ", ".join(f"{c}({dates[c]})" for c in cps) if cps else "(none)"
        else:
            shown = ", ".join(str(c) for c in cps) if cps else "(none)"
        print(f"  {name:10s} [{len(cps):2d}] {shown}")
