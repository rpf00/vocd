"""
Plot per-day MAE for the gas-sensor drift-correction benchmark.

Companion to ``examples/gas_data.py`` (same style as the Worner-et-al. figures):
one line per method showing MAE vs the day-1 reference across measurement days,
plus the uncorrected curve and the oracle.

The summary CSV (gas_best.csv) holds only per-method scalars, NOT the per-day
MAE series, so this script does not "plot the CSV" — it reads each method's
chosen BUDGET from gas_best.csv and recomputes just that method's per-day MAE at
its best operating point (one pass per method).  It reuses the loaders,
detector->baseline adapter, correction and scoring from ``gas_data.py`` so the
drawn curves match the scored numbers exactly.

Skips dm-bocd by default (slow sampler); add it explicitly if you want it drawn.

Usage
-----
    python examples/plot_gas.py --sensor 0 --analyte Diacetyl_1ppm \
        --best-csv results/gas/gas_best.csv --out results/gas/fig_mae.png
    python examples/plot_gas.py --methods vocd,bocd,np-focus,r-focus,chad ...
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gas_data import (ANALYTE_LABELS, _DEFAULT_GAS_DIR, load_all_days,
                      base_mask_from_cps, correct_measurement, curve_metrics,
                      measure_cps)

COLORS = {"uncorrected": "#222222", "oracle": "#888888", "vocd": "#2ca02c",
          "bocd": "#ff7f0e", "dm-bocd": "#e377c2", "np-focus": "#1f77b4",
          "r-focus": "#d62728", "chad": "#9467bd"}
LABELS = {"uncorrected": "Uncorrected", "oracle": "Oracle (ground-truth stages)",
          "vocd": "VOCD (ours)", "bocd": "Piecewise BOCD",
          "dm-bocd": "Piecewise Dm-BOCD", "np-focus": "NP-FOCuS",
          "r-focus": "R-FOCuS", "chad": "CHAD"}


class _Args:
    """minimal shim carrying the fields gas_data's detector functions read."""
    def __init__(self, dsm_path, min_seg, bocd_extract):
        self._dsm_path = dsm_path
        self.min_seg = min_seg
        self.bocd_extract = bocd_extract


def read_budgets(path):
    b = {}
    if not os.path.exists(path):
        print(f"! {path} not found; detectors fall back to --alpha")
        return b
    for row in csv.DictReader(open(path)):
        try:
            b[row["method"]] = float(row["budget"])
        except (TypeError, ValueError):
            b[row["method"]] = None
    return b


def per_day_mae(name, days, budget, args, backend):
    """MAE per test day for one method at one budget (matches gas_data logic)."""
    d0, r0, cs0 = days[0]
    if name == "oracle":
        bm0 = cs0 == 1
    else:
        bm0 = base_mask_from_cps(measure_cps(name, r0, budget, args, backend), len(r0))
    if bm0 is None or bm0.sum() < 5:
        bm0 = cs0 == 1
    ref_base_mean, ref_r = float(r0[bm0].mean()), r0
    maes = []
    for _, r, cs in days[1:]:
        if name == "oracle":
            bm = cs == 1
        else:
            bm = base_mask_from_cps(measure_cps(name, r, budget, args, backend), len(r))
        rc = correct_measurement(r, bm, ref_base_mean)
        maes.append(curve_metrics(rc, ref_r)[0])
    return np.array(maes)


def uncorrected_mae(days):
    ref_r = days[0][1]
    return np.array([curve_metrics(r, ref_r)[0] for _, r, _ in days[1:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_GAS_DIR)
    ap.add_argument("--sensor", type=int, default=0)
    ap.add_argument("--analyte", default="Diacetyl_1ppm", choices=list(ANALYTE_LABELS))
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--best-csv", default="results/gas/gas_best.csv")
    ap.add_argument("--methods", default="vocd,bocd,np-focus,r-focus,chad")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--alpha", type=float, default=0.01, help="budget fallback")
    ap.add_argument("--min-seg", type=int, default=30)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    ap.add_argument("--cpp", action="store_true")
    ap.add_argument("--out", default="results/gas/fig_mae.png")
    args = ap.parse_args()
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    shim = _Args(dsm_path, args.min_seg, args.bocd_extract)
    backend = "cpp" if args.cpp else "python"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("needs matplotlib:  pip install matplotlib")

    days = load_all_days(args.data_dir, args.analyte, args.replicate, args.sensor)
    x = np.arange(1, len(days))          # test-day indices (day 1 is reference)
    budgets = read_budgets(args.best_csv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    needs_dsm = any(m in ("bocd", "dm-bocd") for m in methods)
    if needs_dsm and not dsm_path:
        print("! no --dsm-path; dropping bocd/dm-bocd from the plot")
        methods = [m for m in methods if m not in ("bocd", "dm-bocd")]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, uncorrected_mae(days), color=COLORS["uncorrected"], lw=1.6,
            label=LABELS["uncorrected"])
    ax.plot(x, per_day_mae("oracle", days, None, shim, backend),
            color=COLORS["oracle"], lw=1.8, marker="o", ms=3, label=LABELS["oracle"])
    for name in methods:
        b = budgets.get(name)
        if b is None and name != "oracle":
            b = args.alpha
        y = per_day_mae(name, days, b, shim, backend)
        emph = (name == "vocd")
        ax.plot(x, y, color=COLORS.get(name, None), lw=2.6 if emph else 1.3,
                ls="-" if emph else "--", marker="o" if emph else None, ms=4,
                label=LABELS.get(name, name) + (f"  (@{b:.4g})" if b is not None else ""))

    ax.set_xlabel("Measurement day index")
    ax.set_ylabel("MAE vs day-1 reference (Ohm)")
    ax.set_title(f"Sensor R{args.sensor+1} | {ANALYTE_LABELS[args.analyte]}\n"
                 f"Drift correction quality over 12 months  (Worner et al. 2025)")
    ax.legend(ncol=2, fontsize=8, framealpha=0.7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
