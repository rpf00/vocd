"""
Plot change-point results for the DSM-bocd real-data problems.

Reads results/<problem>/changepoints.csv (long format: method,cp_index[,cp_date])
and, if present, results/<problem>/series.csv (index,value[,date] or
index,value_0,value_1[,date] for the bivariate crypto case).  Draws the signal
on top and a per-method change-point raster beneath (one lane per method, a
vertical tick per predicted change-point), then saves the figure into the same
results/<problem>/ directory.

There is no run-length heatmap here on purpose: only BOCD/Dm-BOCD produce a
run-length posterior, so the common, honest view across the 6-method panel is
the change-points themselves marked on the series.

Usage
-----
    python examples/plot_dsm_problem.py --problem well-log
    python examples/plot_dsm_problem.py --problem bond --results-root results
    python examples/plot_dsm_problem.py --problem cryptocrash --value-col value_0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# stable method order + colours (extras fall through to a default cycle)
ORDER = ["vocd", "vocd (cpp)", "bocd", "dm-bocd", "np-focus", "r-focus", "chad"]
COLORS = {"vocd": "#2ca02c", "vocd (cpp)": "#2ca02c", "bocd": "#ff7f0e",
          "dm-bocd": "#e377c2", "np-focus": "#1f77b4", "r-focus": "#d62728",
          "chad": "#9467bd"}


def read_changepoints(path):
    by = {}
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        has_date = "cp_date" in (r.fieldnames or [])
        for row in r:
            by.setdefault(row["method"], []).append(
                (int(row["cp_index"]), row.get("cp_date") if has_date else None))
    return by


def read_series(path, value_col=None):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        cols = r.fieldnames or []
        val_cols = ([value_col] if value_col else
                    [c for c in cols if c == "value" or c.startswith("value_")])
        rows = list(r)
    if not val_cols:
        return None
    idx = [int(x["index"]) for x in rows]
    series = {c: [float(x[c]) for x in rows] for c in val_cols}
    return idx, series


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True,
                    help="results/<problem>/ directory name (e.g. well-log)")
    ap.add_argument("--results-root", default=os.path.join(_REPO, "results"))
    ap.add_argument("--value-col", default=None,
                    help="which series column to plot (default: all value* cols)")
    ap.add_argument("--out", default=None, help="output PNG (default in results dir)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("needs matplotlib:  pip install matplotlib")

    rdir = os.path.join(args.results_root, args.problem)
    cp_path = os.path.join(rdir, "changepoints.csv")
    if not os.path.exists(cp_path):
        sys.exit(f"not found: {cp_path}")
    cps = read_changepoints(cp_path)
    methods = [m for m in ORDER if m in cps] + [m for m in cps if m not in ORDER]
    ser = read_series(os.path.join(rdir, "series.csv"), args.value_col)

    n_m = len(methods)
    fig, (axS, axR) = plt.subplots(
        2, 1, figsize=(12, 3 + 0.45 * n_m), sharex=True,
        gridspec_kw={"height_ratios": [3, max(1.2, 0.5 * n_m)]})

    # top: the signal (if we have it)
    if ser is not None:
        idx, series = ser
        for name, vals in series.items():
            axS.plot(idx, vals, lw=0.9, color="0.25",
                     label=(name if len(series) > 1 else None))
        if len(series) > 1:
            axS.legend(fontsize=8, framealpha=0.6)
        axS.set_ylabel("signal")
        xmax = max(idx)
    else:
        axS.text(0.5, 0.5, "series.csv not found — raster only",
                 ha="center", va="center", transform=axS.transAxes, color="0.5")
        axS.set_yticks([])
        xmax = max((c for cc in cps.values() for c, _ in cc), default=1)
    axS.set_title(f"{args.problem} — predicted change-points by method")

    # bottom: one lane per method, a tick per change-point
    for row, name in enumerate(methods):
        y = n_m - 1 - row
        xs = [c for c, _ in cps[name]]
        c = COLORS.get(name, f"C{row}")
        axR.vlines(xs, y + 0.15, y + 0.85, color=c, lw=1.3)
        axR.text(-0.008 * xmax, y + 0.5, f"{name} ({len(xs)})",
                 ha="right", va="center", fontsize=8, color=c)
        # faint guide lines from ticks up into the signal panel
        for x in xs:
            axS.axvline(x, color=c, lw=0.4, alpha=0.12)
    axR.set_ylim(0, n_m)
    axR.set_yticks([])
    axR.set_xlabel("series index")
    axR.set_xlim(axS.get_xlim())

    fig.tight_layout()
    out = args.out or os.path.join(rdir, f"{args.problem}_changepoints.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
