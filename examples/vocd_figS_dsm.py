"""
SI figures (Nature-Communications style) — classical CD benchmarks of the
DSM-bocd paper, one figure each (Supplementary Figs. 7, 8 and 9).  Same style
and palette as the other figures of the paper.  Each figure is one signal
panel and, below it, one row per detector with the CPs it places:

  Fig. 7  Twitter flash crash (DJIA, minute 'open', 23 April 2013): the price with
     the 4-min crash-and-rebound shaded, and below it one row per detector
     with the CPs it places (count in parentheses).  A robust detector should
     place no CP on the crash.
  Fig. 8  UK 10-year bond yield (daily, 2017-12 to 2022-12): same layout, with the
     political / Covid-19 events used in the DSM-bocd paper as gray lines.
  Fig. 9  Well-log (4,050 nuclear-magnetic-response samples): same layout; lasting
     level changes mark new strata, the narrow spikes are outliers.

VOCD's CPs are also drawn as thin lines on each signal.  Optional extra VOCD
rows (``--sens-dirs``) show its CPs at other fwer budgets from separate runs
of the benchmark scripts (``--fwer X --methods vocd --out-dir DIR``), as a
sensitivity check; they never replace the declared operating point.

Inputs (from examples/{flashcrash,bond,welllog}_data_v3.py; nothing is
recomputed): <name>_v3_cps.csv and <name>_v3_series.csv in --res-dir.

writes figS_dsm_flashcrash / figS_dsm_bond / figS_dsm_welllog (.pdf + .png).

    python examples/vocd_figS_dsm.py
    python examples/vocd_figS_dsm.py --sens-dirs results/dsm/fwer_1e-3,results/dsm/fwer_1e-4
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "results", "dsm")
OUTDIR = os.path.join(_REPO, "results", "dsm", "figures")

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_w": 180.0,                     # figure height follows from the number of rows
    "x": 30.0, "w": 146.0,              # shared axes span (row labels sit left of x)
    "sig_h": 34.0, "rug_row": 4.0,      # signal height; rug height per detector row
    "gap": 2.0, "top": 6.0, "bottom": 11.0,
}
F = {}

# ── palette: identical to Figs. 2-5 (Okabe-Ito methods) ──────────────────────
METHOD_STYLE = {   # csv name : (label, colour)
    "vocd":     ("VOCD",     "#009E73"),
    "bocd":     ("BOCD",     "#E69F00"),
    "dm-bocd":  ("Dm-BOCD",  "#CC79A7"),
    "focus":    ("FOCuS",    "#56B4E9"),
    "np-focus": ("NP-FOCuS", "#0072B2"),
    "nunc":     ("NUNC",     "#c9b400"),     # darker yellow: readable as thin ticks
    "chad":     ("CHAD",     "#D55E00"),
    "champ":    ("CHAMP",    "#000000"),
}
C_SIG = "0.25"
DATASETS = [   # name, title, y label, x-axis kind
    ("flashcrash", "Twitter flash crash, DJIA on 23 April 2013", "DJIA", "time"),
    ("bond", "UK 10-year government bond yield", "Yield (%)", "date"),
    ("welllog", "Well-log", "Nuclear response", "index"),
]
CRASH = ("13:08", "13:12")                    # the minutes of the crash and rebound
BOND_EVENTS = ["2019-05-24", "2019-07-24", "2020-02-21", "2020-04-15",
               "2021-01-08", "2021-07-19", "2021-12-06", "2022-09-06"]


# ── io ────────────────────────────────────────────────────────────────────────
def load_series(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r["label"] for r in rows], np.array([float(r["value"]) for r in rows])


def load_cps(path):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            idx = [int(c) for c in r["cp_idx"].split(";") if c.strip()]
            out[r["method"]] = dict(cps=idx, budget=r["budget"], runtime_ms=float(r["runtime_ms"]))
    return out


# ── style helpers (shared with figs 1-5) ─────────────────────────────────────
def set_nature_style(base=7.0):
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    arial_family = ["Arial", "Arimo", "Liberation Sans", "Helvetica"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not (set(arial_family) & have):
        print("[figS] warning: no Arial-family font found; using DejaVu Sans.")
    F.update(panel=base + 1.0, title=base, body=base - 0.5, small=base - 1.0,
             tiny=max(5.0, base - 1.5))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": arial_family + ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 120, "savefig.dpi": 300,
    })


def ax_mm(fig, x, y, w, h):
    W, H = fig._mm
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def x_ticks(ax, labels, kind):
    n = len(labels)
    if kind == "time":                          # every hour on the hour
        idx = [i for i, l in enumerate(labels) if l.endswith(":00")]
        ax.set_xticks(idx, [labels[i] for i in idx])
    elif kind == "date":                        # every January and July
        idx = [i for i, l in enumerate(labels) if l[5:] in ("01-01", "07-01")]
        ax.set_xticks(idx, [f"{'Jan' if labels[i][5:7] == '01' else 'Jul'} {labels[i][:4]}"
                            for i in idx])
    else:
        ax.set_xticks(np.arange(0, n, 500))
    ax.set_xlim(-0.5, n - 0.5)


# ── one benchmark block ──────────────────────────────────────────────────────
def block(fig, y0, name, ylab, kind, labels, values, cps, rows):
    L = LAYOUT
    n = len(values)
    ax = ax_mm(fig, L["x"], y0, L["w"], L["sig_h"])
    ax.plot(np.arange(n), values, color=C_SIG, lw=0.6, zorder=3)
    if "vocd" in cps:
        for c in cps["vocd"]["cps"]:
            ax.axvline(c, color=METHOD_STYLE["vocd"][1], lw=0.5, alpha=0.7, zorder=2)
    if name == "flashcrash":
        i0, i1 = labels.index(CRASH[0]), labels.index(CRASH[1])
        ax.axvspan(i0 - 0.5, i1 + 0.5, color="0.85", lw=0, zorder=1)
        ax.text(i1 + 3, 0.06, "crash and rebound", transform=ax.get_xaxis_transform(),
                fontsize=F["tiny"], color="0.35", va="bottom")
    if name == "bond":
        for k, d in enumerate(BOND_EVENTS, 1):
            if d in labels:
                i = labels.index(d)
                ax.axvline(i, color="0.6", lw=0.8, ls="-", zorder=1)
                ax.text(i, 1.01, str(k), transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=F["tiny"], color="0.35")
    ax.set_ylabel(ylab)
    x_ticks(ax, labels, kind)
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.25, lw=0.4)

    rug_h = L["rug_row"] * len(rows)
    ar = ax_mm(fig, L["x"], y0 + L["sig_h"] + L["gap"], L["w"], rug_h)
    for j, (key, lab, col) in enumerate(rows):
        yj = len(rows) - 1 - j
        cs = cps[key]["cps"]
        ar.vlines(cs, yj - 0.32, yj + 0.32, color=col, lw=0.9)
        ar.text(-0.008, yj, f"{lab} ({len(cs)})", transform=ar.get_yaxis_transform(),
                ha="right", va="center", fontsize=F["small"], color=col)
    ar.set_ylim(-0.6, len(rows) - 0.4)
    ar.set_yticks([])
    ar.spines["left"].set_visible(False)
    x_ticks(ar, labels, kind)
    ar.grid(alpha=0.25, lw=0.4, axis="x")
    ar.set_xlabel({"time": "Time (23 April 2013)", "date": "Date", "index": "Sample"}[kind])
    return y0 + L["sig_h"] + L["gap"] + rug_h


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res-dir", default=RES)
    ap.add_argument("--sens-dirs", default="",
                    help="comma list of folders with VOCD-only runs at other fwer budgets")
    ap.add_argument("--base-font", type=float, default=7.0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "figS_dsm.pdf"),
                    help="only its folder is used: one figS_dsm_<name> file per benchmark")
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sens = [d.strip() for d in args.sens_dirs.split(",") if d.strip()]

    out_dir = os.path.dirname(args.out) or "."
    for name, title, ylab, kind in DATASETS:
        labels, values = load_series(os.path.join(args.res_dir, f"{name}_v3_series.csv"))
        cps = load_cps(os.path.join(args.res_dir, f"{name}_v3_cps.csv"))
        rows = []
        for m, (lab, col) in METHOD_STYLE.items():
            if m not in cps:
                continue
            rows.append((m, lab, col))
            if m == "vocd":                      # sensitivity rows right under VOCD
                for d in sens:
                    p = os.path.join(d, f"{name}_v3_cps.csv")
                    if os.path.exists(p):
                        r = load_cps(p).get("vocd")
                        if r:
                            key = f"vocd@{d}"
                            cps[key] = r
                            rows.append((key, f"VOCD fwer {float(r['budget']):g}", "#7fcdb3"))
        L = LAYOUT
        H_mm = L["top"] + L["sig_h"] + L["gap"] + L["rug_row"] * len(rows) + L["bottom"]
        fig = plt.figure(figsize=(L["fig_w"] / 25.4, H_mm / 25.4))
        fig._mm = (L["fig_w"], H_mm)
        block(fig, L["top"], name, ylab, kind, labels, values, cps, rows)
        out = os.path.join(out_dir, f"figS_dsm_{name}.pdf")
        fig.savefig(out)
        fig.savefig(os.path.splitext(out)[0] + ".png", dpi=300)
        plt.close(fig)
        print(f"{name}: wrote {out} (+ .png), {L['fig_w']:g} x {H_mm:.0f} mm; "
              + ", ".join(f"{lab} {len(cps[key]['cps'])}" for key, lab, _ in rows))

if __name__ == "__main__":
    main()
