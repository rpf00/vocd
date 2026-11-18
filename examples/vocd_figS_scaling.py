"""
SI figure (Nature-Communications style) — runtime scaling with series length
(Supplementary Fig. 10).  Same style and palette as the other figures:

  a  total runtime (ms) vs series length T, log-log, every detector (Python
     circles, C++ squares).  A power law runtime = a * T^b is fitted per method
     on T >= --fit-min; the fit is drawn solid over the measured range and
     dashed beyond it up to --extrapolate (an extrapolation of the fitted law,
     not a measurement), and the legend gives the exponent b.
  b  cost per observation (us/obs) vs T: flat lines mean a constant per-step
     cost, i.e. linear scaling, the regime VOCD's O(S^2) per-observation
     recurrence predicts at a fixed regime bank.

Input: scaling_v3.csv from examples/scaling_v3.py (nothing is recomputed).

    python examples/vocd_figS_scaling.py
    python examples/vocd_figS_scaling.py --methods vocd,vocd (cpp),bocd,dm-bocd
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "results", "scaling")
OUTDIR = os.path.join(_REPO, "results", "scaling", "figures")

LAYOUT = {"fig_mm": (180.0, 92.0),
          "panels": {"a": dict(x=14.0, y=7.0, w=72.0, h=62.0),
                     "b": dict(x=104.0, y=7.0, w=72.0, h=62.0)},
          "legend_y": 86.0}
F = {}

METHOD_STYLE = {   # csv name : (label, colour, marker)
    "vocd":           ("VOCD (Py)",      "#009E73", "o"),
    "vocd (cpp)":     ("VOCD (C++)",     "#009E73", "s"),
    "bocd":           ("BOCD",           "#E69F00", "o"),
    "dm-bocd":        ("Dm-BOCD",        "#CC79A7", "o"),
    "focus":          ("FOCuS (Py)",     "#56B4E9", "o"),
    "focus (cpp)":    ("FOCuS (C++)",    "#56B4E9", "s"),
    "np-focus":       ("NP-FOCuS (Py)",  "#0072B2", "o"),
    "np-focus (cpp)": ("NP-FOCuS (C++)", "#0072B2", "s"),
    "nunc":           ("NUNC",           "#c9b400", "o"),
    "chad":           ("CHAD",           "#D55E00", "o"),
    "champ":          ("CHAMP",          "#000000", "o"),
}


def set_nature_style(base=7.0):
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    arial_family = ["Arial", "Arimo", "Liberation Sans", "Helvetica"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not (set(arial_family) & have):
        print("[figS] warning: no Arial-family font found; using DejaVu Sans.")
    F.update(panel=base + 1.0, small=base - 1.0, tiny=max(5.0, base - 1.5))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": arial_family + ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": base, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1, "legend.fontsize": base - 1,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 120, "savefig.dpi": 300,
    })


def ax_mm(fig, x, y, w, h):
    W, H = LAYOUT["fig_mm"]
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def plabel(fig, panel, dx_mm=-10.0, dy_mm=-1.0):
    p = LAYOUT["panels"][panel]
    W, H = LAYOUT["fig_mm"]
    fig.text((p["x"] + dx_mm) / W, 1.0 - (p["y"] + dy_mm) / H, panel,
             fontsize=F["panel"], fontweight="bold", va="bottom")


def load(path):
    data = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            data.setdefault(r["method"], []).append(
                (int(r["T"]), float(r["runtime_ms"]), float(r["runtime_min_ms"]),
                 float(r["runtime_max_ms"])))
    return {m: sorted(v) for m, v in data.items()}


def fit_power(T, rt, t_min):
    m = T >= t_min
    if m.sum() < 2:
        return np.nan, np.nan
    b, loga = np.polyfit(np.log10(T[m]), np.log10(rt[m]), 1)
    return 10 ** loga, b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(RES, "scaling_v3.csv"))
    ap.add_argument("--methods", default=",".join(METHOD_STYLE))
    ap.add_argument("--fit-min", type=int, default=1000, help="fit on T >= this")
    ap.add_argument("--extrapolate", type=float, default=1e6, help="dash the fit up to this T")
    ap.add_argument("--base-font", type=float, default=7.0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "figS_scaling.pdf"))
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    data = load(args.csv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in data]
    W_mm, H_mm = LAYOUT["fig_mm"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    ax_a = ax_mm(fig, **LAYOUT["panels"]["a"]); plabel(fig, "a")
    ax_b = ax_mm(fig, **LAYOUT["panels"]["b"]); plabel(fig, "b")

    handles, fits = [], {}
    for m in methods:
        lab, col, mk = METHOD_STYLE[m]
        T = np.array([r[0] for r in data[m]], float)
        rt = np.array([r[1] for r in data[m]])
        lo = np.array([r[2] for r in data[m]])
        hi = np.array([r[3] for r in data[m]])
        a, b = fit_power(T, rt, args.fit_min)
        fits[m] = (a, b)
        ls_meas = "-" if mk == "o" else (0, (1, 1))
        ax_a.errorbar(T, rt, yerr=[rt - lo, hi - rt], fmt=mk, ms=3.5, color=col,
                      markeredgecolor="0.2", markeredgewidth=0.3, elinewidth=0.6, capsize=1.5,
                      zorder=3)
        if np.isfinite(b):
            tt = np.logspace(np.log10(T.min()), np.log10(T.max()), 50)
            ax_a.plot(tt, a * tt ** b, color=col, lw=0.9, ls=ls_meas, zorder=2)
            te = np.logspace(np.log10(T.max()), np.log10(args.extrapolate), 30)
            ax_a.plot(te, a * te ** b, color=col, lw=0.8, ls="--", alpha=0.55, zorder=1)
        ax_b.plot(T, 1000.0 * rt / T, marker=mk, ms=3.5, color=col, lw=0.9, ls=ls_meas,
                  markeredgecolor="0.2", markeredgewidth=0.3, zorder=3)
        handles.append(Line2D([0], [0], color=col, marker=mk, ms=3.5, lw=0.9, ls=ls_meas,
                              markeredgecolor="0.2", markeredgewidth=0.3,
                              label=f"{lab}, b = {b:.2f}" if np.isfinite(b) else lab))

    for ax in (ax_a, ax_b):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Series length T (observations)")
        ax.grid(alpha=0.25, lw=0.4, which="both")
    ax_a.set_ylabel("Runtime (ms)")
    ax_a.axvspan(max(max(r[0] for r in v) for v in data.values()), args.extrapolate,
                 color="0.95", lw=0, zorder=0)
    ax_a.text(0.98, 0.03, "fitted law extrapolated", transform=ax_a.transAxes, ha="right",
              va="bottom", fontsize=F["tiny"], color="0.45")
    ax_b.set_ylabel(r"Cost per observation ($\mu$s)")
    tg = sorted({r[0] for v in data.values() for r in v})
    ax_b.set_xticks(tg, [f"{t:,}" for t in tg], rotation=0)
    ax_b.minorticks_off()
    W, H = LAYOUT["fig_mm"]
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - LAYOUT["legend_y"] / H), handlelength=1.8,
               columnspacing=1.0)
    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"wrote {args.out} and {png}")
    for m in methods:
        a, b = fits[m]
        print(f"  {m:15s} runtime = {a:.3g} * T^{b:.2f}  (fit on T >= {args.fit_min})")


if __name__ == "__main__":
    main()
