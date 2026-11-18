"""
SI figure — environmental monitoring, shifted-
date null (Supplementary Fig. 6).  Same style, palette and layout scheme as
the other SI figures:

  a-h  one panel per detector: the distribution of its consistency when the
       four storm dates are moved together to every other position in the
       record (circular one-day shifts, offsets within 2 x tol of the true
       dates excluded), as the % of shifts at each consistency value (gray
       bars), with the consistency on the TRUE dates as a coloured dashed line
       across the full panel (read on x only); the panel title gives that value
       and the % of shifts it exceeds.  The CPs are never changed: only the dates
       they are scored against move.

The null is recomputed from the CP dates (fast; identical to the numbers in
env_v3_null.csv written by env_sensitivity_v3.py, which is checked at run time
when that file exists).

Inputs (from examples/env_data_v3.py; no detector is re-run):
  --pos-neg   env_v3_pos_neg.csv   per (group, constituent, method) CP dates
  --series    env_v3_series.csv    pooled daily series (record start / end)
  --null      env_v3_null.csv      optional cross-check

LAYOUT: sizes and positions in millimetres from the top-left corner, at final
print size (Nature Portfolio: 180 mm wide, text 7 pt max / 5 pt min, panel
letters 8 pt bold).  Every text size derives from --base-font.

    python examples/vocd_figS_env_null.py
"""
from __future__ import annotations

import argparse
import csv
import os
import string

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "results", "env")
OUTDIR = os.path.join(_REPO, "results", "env", "figures")

EVENTS = {"Isaias": "2020-08-04", "Zeta": "2020-10-29",
          "Elsa": "2021-07-09", "Ida": "2021-09-01"}

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 108.0),
    "grid": dict(x0=14.0, y0=11.0, w=33.5, h=36.0, dx=42.5, dy=52.0, ncol=4),
}
F = {}        # font tiers, filled by set_nature_style(base)

# ── palette: identical to Fig. 5 (Okabe-Ito methods) ─────────────────────────
METHOD_STYLE = {   # csv name : (label, colour)
    "vocd":     ("VOCD",     "#009E73"),
    "bocd":     ("BOCD",     "#E69F00"),
    "dm-bocd":  ("Dm-BOCD",  "#CC79A7"),
    "focus":    ("FOCuS",    "#56B4E9"),
    "np-focus": ("NP-FOCuS", "#0072B2"),
    "nunc":     ("NUNC",     "#F0E442"),
    "chad":     ("CHAD",     "#D55E00"),
    "champ":    ("CHAMP",    "#000000"),
}
C_BAR = "0.72"


# ── null (same rules as env_sensitivity_v3.py) ───────────────────────────────
def load_cps(path):
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        ds = [] if pd.isna(r["cp_dates"]) else \
            [pd.Timestamp(x) for x in str(r["cp_dates"]).split(";") if x.strip()]
        out.setdefault(r["method"], {})[(r["group"], r["constituent"])] = ds
    return out


def consistency(cps, evs, tol):
    near = lambda d: any(abs((d - e).days) <= tol for e in evs.values())
    caught, fa, n_neg = set(), 0, 0
    for (g, _), ds in cps.items():
        if g == "POS":
            caught.update(k for d in ds for k, e in evs.items() if abs((d - e).days) <= tol)
        else:
            n_neg += 1
            fa += int(any(near(d) for d in ds))
    return len(caught) / len(evs) - fa / n_neg


def null_distribution(cps, evs, t0, t1, tol):
    L = (t1 - t0).days + 1
    shifts = [D for D in range(1, L) if min(D, L - D) > 2 * tol]
    vals = np.array([consistency(cps, {k: t0 + pd.Timedelta(days=((e - t0).days + D) % L)
                                       for k, e in evs.items()}, tol) for D in shifts])
    return consistency(cps, evs, tol), vals


# ── style helpers (shared with fig1 - fig5) ──────────────────────────────────
def set_nature_style(base=7.0):
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    arial_family = ["Arial", "Arimo", "Liberation Sans", "Helvetica"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not (set(arial_family) & have):
        print("[figS] warning: no Arial-family font found; using DejaVu Sans.")
    F.update(panel=base + 1.0, title=base, body=base - 0.5, small=base - 1.0,
             tiny=max(5.0, base - 1.5), math=base)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": arial_family + ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": F["small"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 120, "savefig.dpi": 300,
    })


def ax_mm(fig, x, y, w, h):
    W, H = LAYOUT["fig_mm"]
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def plabel(fig, x, y, letter, dx_mm=-6.5, dy_mm=-5.0):
    W, H = LAYOUT["fig_mm"]
    fig.text((x + dx_mm) / W, 1.0 - (y + dy_mm) / H, letter,
             fontsize=F["panel"], fontweight="bold", va="bottom", ha="left")


# ── panel ────────────────────────────────────────────────────────────────────
def panel(ax, m, obs, vals, xlim, ymax, first_col, last_row):
    lab, col = METHOD_STYLE[m]
    u, c = np.unique(np.round(vals, 6), return_counts=True)
    pct = 100.0 * c / vals.size
    ax.bar(u, pct, width=1 / 12 * 0.8, color=C_BAR, edgecolor="0.45", lw=0.3, zorder=2)
    # the true-date consistency is a POSITION on x: a dashed line across the full
    # panel height (the permutation-test convention), so it has no height to read
    lc = col if m != "nunc" else "#c9b400"
    ax.axvline(obs, color=lc, lw=1.4, ls=(0, (4, 2)), zorder=3)
    exc = 100.0 * np.mean(obs > vals)
    sgn = f"{obs:+.2f}".replace("-", "\u2212")
    ax.set_title(f"{lab}\ntrue dates {sgn}, exceeds {exc:.0f}%", fontsize=F["small"],
                 linespacing=1.3, pad=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25, lw=0.4, axis="y", zorder=0)
    if first_col:
        ax.set_ylabel("Shifts (%)")
    if last_row:
        ax.set_xlabel("Consistency")
    return exc


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos-neg", default=os.path.join(RES, "env_v3_pos_neg.csv"))
    ap.add_argument("--series", default=os.path.join(RES, "env_v3_series.csv"))
    ap.add_argument("--null", default=os.path.join(RES, "env_v3_null.csv"))
    ap.add_argument("--tol", type=int, default=15, help="event matching tolerance (days)")
    ap.add_argument("--base-font", type=float, default=7.0,
                    help="base font size in pt at final print size (journal max 7)")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "figS_env_null.pdf"))
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cps_by = load_cps(args.pos_neg)
    s = pd.read_csv(args.series, parse_dates=["date"])
    t0, t1 = s["date"].min(), s["date"].max()
    evs = {k: pd.Timestamp(v) for k, v in EVENTS.items()}
    methods = [m for m in METHOD_STYLE if m in cps_by]
    res = {m: null_distribution(cps_by[m], evs, t0, t1, args.tol) for m in methods}

    allv = np.concatenate([v for _, v in res.values()] + [[o for o, _ in res.values()]])
    xlim = (np.floor(allv.min() * 4) / 4 - 0.08, np.ceil(allv.max() * 4) / 4 + 0.08)
    ymax = max(100.0 * np.unique(np.round(v, 6), return_counts=True)[1].max() / v.size
               for _, v in res.values()) * 1.1

    g = LAYOUT["grid"]
    W_mm, H_mm = LAYOUT["fig_mm"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    exceeds = {}
    for i, m in enumerate(methods):
        r, cidx = divmod(i, g["ncol"])
        x, y = g["x0"] + cidx * g["dx"], g["y0"] + r * g["dy"]
        ax = ax_mm(fig, x, y, g["w"], g["h"])
        obs, vals = res[m]
        exceeds[m] = panel(ax, m, obs, vals, xlim, ymax, cidx == 0,
                           r == (len(methods) - 1) // g["ncol"])
        plabel(fig, x, y, string.ascii_lowercase[i])

    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    n = len(next(iter(res.values()))[1])
    print(f"wrote {args.out} and {png}  ({W_mm:g} x {H_mm:g} mm; {n} shifts, tol +/-{args.tol} d)")

    ref = {}
    if os.path.exists(args.null):
        with open(args.null, newline="") as fh:
            ref = {r["method"]: float(r["pct_exceeds"]) for r in csv.DictReader(fh)}
    for m in methods:
        chk = ""
        if m in ref:
            chk = "  (matches env_v3_null.csv)" if abs(ref[m] - exceeds[m]) < 1e-6 else \
                  f"  ! env_v3_null.csv says {ref[m]:.1f}%"
        print(f"  {m:9s} true {res[m][0]:+.3f}  exceeds {exceeds[m]:5.1f}% of shifts{chk}")


if __name__ == "__main__":
    main()