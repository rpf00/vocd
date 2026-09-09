"""
Synthetic-data benchmark: VOCD vs BOCD, Dm-BOCD, FOCuS, R-FOCuS, NP-FOCuS, CHAD.

Solves the contaminated change-in-mean case of Altamirano, Briol & Knoblauch
(2023) — T=500, six true change-points at [100, 200, 300, 350, 400, 450],
segment means [10, 5, 0, 10, 5, 12, 0], 2% outliers at +/-10, ten trials — with
every method run ONLINE and with no access to the whole series before deciding
a change (see ``models/`` for how each method is held to that bar).

The data generator, contamination procedure and the ``accuracy_delay`` scorer
are reproduced verbatim from that repository's ``notebooks/accuracy_delay.ipynb``
so the baselines are scored on their own terms.  VOCD is the installed package;
the other five come from the local ``alternative_models/`` package.  BOCD / Dm-BOCD need a
clone of maltamiranomontero/DSM-bocd (``--dsm-path`` or ``VOCD_DSM_PATH``); if
it is absent they are skipped and everything else still runs.

Usage::

    python examples/synthetic_data.py --dsm-path ./DSM-bocd
    python examples/synthetic_data.py --alpha 0.01
    python examples/synthetic_data.py --methods vocd,focus,chad
    python examples/synthetic_data.py --bocd-extract upstream   # published, non-causal
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

# make the repo root importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vocd
from alternative_models import REGISTRY

# ── experiment constants (verbatim from accuracy_delay.ipynb) ─────────
T = 500
CPS_TRUE = [100, 200, 300, 350, 400, 450, 600]   # 600 is a sentinel past T
MUS = [10, 5, 0, 10, 5, 12, 0]
WINDOW = 10
N_TRIALS = 10
SEED_DATA = 3000
SEED_CONTAM = 54321
CONTAM_FRAC = 0.02
CONTAM_SPIKE = 10.0


def make_paper_data(n: int = T, seed: int = SEED_DATA):
    np.random.seed(seed)
    data = np.zeros((n, 1))
    mu, i = MUS[0], 0
    for t in range(n):
        if t == CPS_TRUE[i]:
            mu = MUS[i + 1]
            i += 1
        data[t, 0] = np.random.normal(mu, 1)
    return data, float(np.mean(data))


def contaminate(data, frac=CONTAM_FRAC, spike=CONTAM_SPIKE):
    n = data.shape[0]
    idx = np.random.choice(np.arange(0, n, 1), int(frac * n), replace=False)
    out = data.copy()
    sgn = np.random.choice([1, -1], size=len(idx))
    out[idx, 0] = out[idx, 0] + sgn * spike
    return out


def make_trials(data, n_trials=N_TRIALS):
    np.random.seed(SEED_CONTAM)
    return [contaminate(data) for _ in range(n_trials)]


def accuracy_delay(cps_pred: Sequence[int], cps_true=CPS_TRUE, window=WINDOW):
    """Reference scorer, reproduced as-is (many-to-one; skips cps_pred[0])."""
    TP = FP = FN = 0
    delays = []
    for cp in cps_pred[1:]:
        if np.min(np.abs(np.array(cps_true) - cp)) < window:
            TP += 1
            delays.append(np.min(np.abs(np.array(cps_true) - cp)))
        else:
            FP += 1
    for cp in np.array(cps_true)[:-1]:
        if np.min(np.abs(np.array(cps_pred) - cp)) > window:
            FN += 1
    ppv = TP / (TP + FP) if (TP + FP) else 0.0
    tpr = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) else 0.0
    return {"n_detected": len(cps_pred) - 1, "ppv": ppv, "tpr": tpr, "f1": f1,
            "delay": float(np.mean(delays)) if delays else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsm-path", default=None,
                    help="clone of maltamiranomontero/DSM-bocd (for BOCD / Dm-BOCD)")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--methods",
                    default="vocd,bocd,dm-bocd,focus,r-focus,np-focus,chad")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="shared family-wise budget: VOCD FWER and the alarm "
                         "detectors' threshold (a priori, not tuned on data)")
    ap.add_argument("--min-seg", type=int, default=10,
                    help="min samples between change-points / per-segment burn-in")
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"],
                    default="causal",
                    help="'causal' commits change-points online; 'upstream' uses "
                         "DSM-bocd find_cp (revises past with future — published, "
                         "not strictly online)")
    ap.add_argument("--backend", choices=["auto", "python", "cpp"], default="auto")
    ap.add_argument("--out", default="results/synthetic/synthetic_data.csv")
    args = ap.parse_args()

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"vocd"} | set(REGISTRY)
    for m in wanted:
        if m not in known:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(known)}")

    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    needs_dsm = [m for m in wanted if m in REGISTRY and REGISTRY[m][1]]
    if needs_dsm and not dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs_dsm}\n"
              "  git clone https://github.com/maltamiranomontero/DSM-bocd.git\n")
        wanted = [m for m in wanted if m not in needs_dsm]
    if not wanted:
        raise SystemExit("no methods left to run")

    data_clean, mean0 = make_paper_data()
    trials = make_trials(data_clean, args.trials)

    def run_one(method: str, x1d: np.ndarray) -> List[int]:
        if method == "vocd":
            r = vocd.detect(x1d, backend=args.backend)
            return [int(c) for c in r.changepoints(fwer=args.alpha)]
        fn, nd = REGISTRY[method]
        if nd:  # bocd / dm-bocd
            return fn(x1d, mean0=mean0, dsm_path=dsm_path, extract=args.bocd_extract)
        return fn(x1d, alpha=args.alpha, min_seg=args.min_seg)

    print(f"synthetic-data benchmark — T={T}, 6 true CPs at {CPS_TRUE[:-1]}")
    print(f"{CONTAM_FRAC:.0%} outliers @ +/-{CONTAM_SPIKE:g}, {args.trials} trials, "
          f"window={WINDOW}, shared alpha={args.alpha:g}, "
          f"bocd-extract={args.bocd_extract}")
    print(f"methods: {', '.join(wanted)}\n")

    rows: List[dict] = []
    for name in wanted:
        print(f"[{name}] ", end="", flush=True)
        for i, dc in enumerate(trials):
            x1d = dc.ravel()
            t0 = time.perf_counter()
            cps = run_one(name, x1d)
            ms = (time.perf_counter() - t0) * 1000.0
            r = accuracy_delay([0] + sorted(int(c) for c in cps))
            r.update(method=name, trial=i, runtime_ms=ms)
            rows.append(r)
            print(".", end="", flush=True)
        print(" done")

    def agg(name, key):
        v = np.array([r[key] for r in rows if r["method"] == name], float)
        return np.nanmean(v), np.sqrt(np.nanvar(v))

    print(f"\n{'method':<10}{'#CPs':>12}{'PPV':>13}{'TPR':>13}"
          f"{'F1':>13}{'Delay':>12}{'Runtime (ms)':>16}")
    print("-" * 89)
    for name in wanted:
        n, p, t = agg(name, "n_detected"), agg(name, "ppv"), agg(name, "tpr")
        f1, d, rt = agg(name, "f1"), agg(name, "delay"), agg(name, "runtime_ms")
        print(f"{name:<10}{n[0]:>7.1f}±{n[1]:<4.1f}{p[0]:>7.3f}±{p[1]:<5.3f}"
              f"{t[0]:>7.3f}±{t[1]:<5.3f}{f1[0]:>7.3f}±{f1[1]:<5.3f}"
              f"{d[0]:>7.2f}±{d[1]:<4.2f}{rt[0]:>10.1f}±{rt[1]:<5.1f}")
    print("-" * 89)
    print("scorer: reference accuracy_delay (many-to-one). spread = population "
          "std over trials.\nalpha is a single a-priori budget shared by every "
          "method; no per-method tuning on the test series.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["method", "trial", "n_detected", "ppv", "tpr", "f1", "delay", "runtime_ms"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
