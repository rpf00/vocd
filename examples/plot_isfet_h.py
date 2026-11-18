"""
Plot the H+ ISFET drift-corrected curves.

The summary CSVs (isfet_best.csv / isfet_sweep.csv) hold only per-method scalars
(RMSE, beta, runtime, ...), NOT the per-day corrected traces — those were never
written out.  So this script does not "plot the CSV"; it reads each method's
chosen BUDGET from isfet_best.csv and recomputes just the curve for that one
operating point (one evaluation per method, not the whole sweep).  It reuses the
loaders, estimators, detectors and anchored correction from ``isfet_data.py`` so
the drawn curves match the scored numbers exactly.

Panels
------
  top     drift space:  true drift E0(t) vs each method's E0_hat(t) (anchored)
  bottom  correction:   reference pH vs each method's corrected pH_hat(t)
                        (the anchored quantity that RMSE is computed on)
VOCD's change-points are drawn as vertical lines.

Fast by default (oracle/none/two-point/M0/vocd).  Add dm-bocd only if you want
it — it is the slow one (prefix replay of the sampler).

Usage
-----
    python examples/plot_isfet_h.py --data-dir data/isfet \
        --best-csv results/isfet/isfet_best.csv --out results/isfet/isfet_h_curves.png
    python examples/plot_isfet_h.py --methods oracle,none,two-point,M0,vocd,chad ...
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isfet_data import (S_NERNST, BURN_IN, MIN_FIT, load_daily,
                        est_none, est_twopoint, est_M0, est_exp, est_oracle,
                        est_from_cp_sets, detector_cp_sets)

PARAM = {"none": est_none, "two-point": None, "M0": est_M0,
         "exp": est_exp, "oracle": None}   # two-point/oracle need e0


def read_budgets(path):
    b = {}
    if not os.path.exists(path):
        print(f"! {path} not found; detectors will use --alpha fallback")
        return b
    for row in csv.DictReader(open(path)):
        v = row.get("budget", "")
        try:
            b[row["method"]] = float(v)
        except (TypeError, ValueError):
            b[row["method"]] = None
    return b


def build_e0hat(name, dd, vd, e0, budget, args):
    """Return (e0_hat, cp_sets_or_None) for one method at its budget."""
    if name == "none":
        return est_none(dd, vd), None
    if name == "M0":
        return est_M0(dd, vd), None
    if name == "exp":
        return est_exp(dd, vd), None
    if name == "two-point":
        return est_twopoint(dd, vd, e0=e0, period=args.twopoint_period), None
    if name == "oracle":
        return est_oracle(dd, vd, e0=e0), None
    # detector
    shim = SimpleNamespace(_dsm_path=args._dsm_path, bocd_extract="causal",
                           min_seg=args.min_seg)
    if budget is None:
        budget = args.alpha
    sets = detector_cp_sets(name, vd, budget, shim,
                            vocd_backend=("cpp" if args.cpp else "python"))
    return est_from_cp_sets(dd, vd, sets), sets


def anchored(e0_hat, vd, ph_ref, dd, burn_in):
    """pH_hat with the single day-0 anchor, and the same in drift (mV) space."""
    ph_hat = (e0_hat - vd) / S_NERNST
    m = dd >= burn_in
    bias = float(np.mean(ph_hat[m][:MIN_FIT] - ph_ref[m][:MIN_FIT]))
    ph_corr = ph_hat - bias
    e0_anch = e0_hat - bias * S_NERNST      # drift-space estimate, anchored to E0
    return ph_corr, e0_anch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/isfet")
    ap.add_argument("--ion", default="h")
    ap.add_argument("--best-csv", default="results/isfet/isfet_best.csv")
    ap.add_argument("--methods", default="oracle,none,two-point,M0,vocd")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--alpha", type=float, default=0.01, help="budget fallback")
    ap.add_argument("--min-seg", type=int, default=4)
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--twopoint-period", type=float, default=7.0)
    ap.add_argument("--cpp", action="store_true", help="use vocd cpp backend")
    ap.add_argument("--out", default="results/isfet/isfet_h_curves.png")
    args = ap.parse_args()
    args._dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("needs matplotlib:  pip install matplotlib")

    dd, vd, ph_ref, e0 = load_daily(args.data_dir, args.ion)
    budgets = read_budgets(args.best_csv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    fig, (axU, axL) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    m = dd >= args.burn_in

    # truth
    axU.plot(dd, e0, color="k", lw=2.0, label="true drift $E_0$ (reference)")
    axU.plot(dd, vd, color="0.7", lw=1.0, ls=":", label="raw $V_H$")
    axL.plot(dd, ph_ref, color="k", lw=2.0, label="reference pH")

    vocd_cps = None
    for i, name in enumerate(methods):
        e0_hat, sets = build_e0hat(name, dd, vd, e0, budgets.get(name), args)
        ph_corr, e0_anch = anchored(e0_hat, vd, ph_ref, dd, args.burn_in)
        rmse = float(np.sqrt(np.mean(((e0_anch - e0)[m]) ** 2)))
        c = f"C{i}"
        axU.plot(dd, e0_anch, color=c, lw=1.3, alpha=0.9,
                 label=f"{name} $\\hat{{E_0}}$")
        axL.plot(dd, ph_corr, color=c, lw=1.3, alpha=0.9,
                 label=f"{name}  (RMSE {rmse:.1f} mV)")
        if name == "vocd" and sets is not None:
            vocd_cps = [dd[c2] for c2 in sorted(sets[-1]) if 0 < c2 < dd.size]

    if vocd_cps:
        for x in vocd_cps:
            for ax in (axU, axL):
                ax.axvline(x, color="C3", ls="--", lw=0.8, alpha=0.6)
        axU.plot([], [], color="C3", ls="--", lw=0.8, label="VOCD change-points")

    axU.axvspan(dd[0], args.burn_in, color="0.9", alpha=0.5, lw=0)
    axL.axvspan(dd[0], args.burn_in, color="0.9", alpha=0.5, lw=0)
    axU.set_ylabel("drift $E_0$  (mV)")
    axL.set_ylabel("pH")
    axL.set_xlabel("day")
    axU.set_title(f"{args.ion.upper()} ISFET — drift estimate vs truth "
                  f"(grey band = burn-in, unscored)", fontsize=10)
    axL.set_title("corrected pH vs reference (anchored — the scored quantity)",
                  fontsize=10)
    axU.legend(fontsize=8, ncol=2, framealpha=0.6)
    axL.legend(fontsize=8, ncol=2, framealpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
