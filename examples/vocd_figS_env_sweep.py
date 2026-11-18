"""
SI figure — environmental monitoring, operating-
point robustness (Supplementary Fig. 5).  Same style, palette and layout scheme
as ``vocd_figS_isfet_na_k.py`` / ``vocd_figS_gas_diacetyl.py``:

  a  event recall vs fraction of controls with a storm-adjacent CP, one line
     per detector through its budget grid (strict -> loose), with the declared
     operating point of Fig. 5 drawn large with a black edge.  The shaded
     region above the diagonal is where recall exceeds false alarms (as in
     Fig. 5b).  Points that coincide across detectors are spread slightly so
     none hides another.
  b  consistency (recall - false-alarm fraction) vs the total number of CPs
     over the 14 series (log), same lines and declared points: how much of a
     detector's consistency is bought by firing more.

Dm-BOCD is not swept (minutes per grid point); its declared point is drawn
alone from the summary CSV.

Inputs (all from examples/env_data_v3.py; NOTHING is recomputed):
  --sweep     env_v3_sweep.csv     recall / false alarms / CPs per budget
  --summary   env_v3_summary.csv   the declared operating points (tol = --tol)

LAYOUT: sizes and positions in millimetres from the top-left corner, at final
print size (Nature Portfolio: 180 mm wide, text 7 pt max / 5 pt min, panel
letters 8 pt bold).  Every text size derives from --base-font.

    python examples/vocd_figS_env_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "results", "env")
OUTDIR = os.path.join(_REPO, "results", "env", "figures")

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 112.0),
    "legend_y": 104.0,                        # centre of the shared legend row
    "panels": {
        #      x      y      w      h
        "a": dict(x=14.0,  y=7.0, w=72.0, h=84.0),
        "b": dict(x=104.0, y=7.0, w=72.0, h=84.0),
    },
}
F = {}        # font tiers, filled by set_nature_style(base)

# ── palette: identical to Fig. 5 (Okabe-Ito methods) ─────────────────────────
C_VOCD, C_BOCD, C_DMBOCD = "#009E73", "#E69F00", "#CC79A7"
C_FOCUS, C_NPFOCUS, C_NUNC, C_CHAD, C_CHAMP = "#56B4E9", "#0072B2", "#F0E442", "#D55E00", "#000000"
METHOD_STYLE = {   # csv name : (label, colour)
    "vocd":     ("VOCD",     C_VOCD),
    "bocd":     ("BOCD",     C_BOCD),
    "dm-bocd":  ("Dm-BOCD",  C_DMBOCD),
    "focus":    ("FOCuS",    C_FOCUS),
    "np-focus": ("NP-FOCuS", C_NPFOCUS),
    "nunc":     ("NUNC",     C_NUNC),
    "chad":     ("CHAD",     C_CHAD),
    "champ":    ("CHAMP",    C_CHAMP),
}
C_OK = "#009E73"          # shading where recall exceeds false alarms (Fig. 5b)
JITTER = 0.012            # spread of coincident points in panel a (axis units)


# ── io ────────────────────────────────────────────────────────────────────────
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_rows(path):
    with open(path, newline="") as fh:
        return [{k: (v if k in ("method", "budget_kind", "events_matched") else fnum(v))
                 for k, v in r.items()} for r in csv.DictReader(fh)]


def sweep_by_method(rows):
    """method -> rows ordered strict -> loose (alpha / fwer up, hazard down)."""
    out = {}
    for r in rows:
        out.setdefault(r["method"], []).append(r)
    for m, rs in out.items():
        rs.sort(key=lambda r: r["budget"], reverse=(rs[0]["budget_kind"] == "hazard"))
    return out


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


def plabel(fig, panel, dx_mm=-2.5, dy_mm=-1.0):
    p = LAYOUT["panels"][panel]
    W, H = LAYOUT["fig_mm"]
    fig.text((p["x"] + dx_mm) / W, 1.0 - (p["y"] + dy_mm) / H, panel,
             fontsize=F["panel"], fontweight="bold", va="bottom", ha="left")


def spread(points):
    """{(method, i): (x, y)} -> same with coincident points spread in x."""
    groups = {}
    for key, (x, y) in points.items():
        groups.setdefault((round(x, 4), round(y, 4)), []).append(key)
    out = {}
    for (x, y), keys in groups.items():
        keys = sorted(keys, key=lambda k: list(METHOD_STYLE).index(k[0]))
        n = len(keys)
        for j, k in enumerate(keys):
            out[k] = (x + (j - (n - 1) / 2) * JITTER, y)
    return out


# ── panels ───────────────────────────────────────────────────────────────────
def draw_declared(ax, x, y, col):
    ax.plot(x, y, "o", ms=6.0, color=col, markeredgecolor="black",
            markeredgewidth=1.0, zorder=5)


def panel_a(ax, sweep, declared):
    ax.fill_between([0, 1.05], [0, 1.05], [1.05, 1.05], color=C_OK, alpha=0.08, lw=0, zorder=0)
    ax.plot([0, 1.05], [0, 1.05], ls="--", color="0.45", lw=0.7, zorder=1)
    ax.text(0.03, 1.0, "recall > false alarms", color=C_OK,
            fontsize=F["small"], style="italic", ha="left", va="top")
    pts = {(m, i): (r["neg_false_frac"], r["pos_recall"])
           for m, rs in sweep.items() for i, r in enumerate(rs)}
    for m, r in declared.items():
        pts[(m, "decl")] = (r["neg_false_frac"], r["pos_recall"])
    pos = spread(pts)
    for m, rs in sweep.items():
        col = METHOD_STYLE[m][1]
        xy = np.array([pos[(m, i)] for i in range(len(rs))])
        ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=0.8, alpha=0.8, zorder=2)
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=2.8, color=col, markeredgecolor="0.25",
                markeredgewidth=0.3, zorder=3)
    for m in declared:
        draw_declared(ax, *pos[(m, "decl")], METHOD_STYLE[m][1])
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Fraction of false alarms")
    ax.set_ylabel("Recall")
    ax.grid(alpha=0.25, lw=0.4)


def panel_b(ax, sweep, declared):
    tot = lambda r: r["pos_cp"] + r["neg_cp"]
    allc = [r["consistency"] for rs in sweep.values() for r in rs] + \
           [r["consistency"] for r in declared.values()]
    lo, hi = min(allc + [0.0]), max(allc + [0.0])
    pad = 0.08 * (hi - lo)
    ax.axhspan(0, hi + pad, color=C_OK, alpha=0.08, lw=0, zorder=0)
    ax.axhline(0, color="0.45", ls="--", lw=0.7, zorder=1)
    ax.text(0.02, 0.98, "consistency > 0", transform=ax.transAxes, color=C_OK,
            fontsize=F["small"], style="italic", ha="left", va="top")
    xs = []
    for m, rs in sweep.items():
        col = METHOD_STYLE[m][1]
        x = np.array([max(tot(r), 1) for r in rs], float)
        y = np.array([r["consistency"] for r in rs])
        xs += list(x)
        ax.plot(x, y, "-", color=col, lw=0.8, alpha=0.8, zorder=2)
        ax.plot(x, y, "o", ms=2.8, color=col, markeredgecolor="0.25",
                markeredgewidth=0.3, zorder=3)
    for m, r in declared.items():
        xs.append(max(tot(r), 1))
        draw_declared(ax, max(tot(r), 1), r["consistency"], METHOD_STYLE[m][1])
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("CPs over the 14 series")
    ax.set_ylabel("Consistency")
    ax.grid(alpha=0.25, lw=0.4, which="both")


def legend_row(fig, methods):
    from matplotlib.lines import Line2D
    hh = [Line2D([0], [0], color=METHOD_STYLE[m][1], marker="o", ms=3.5, lw=0.9,
                 markeredgecolor="0.25", markeredgewidth=0.3, label=METHOD_STYLE[m][0])
          for m in methods]
    hh.append(Line2D([0], [0], color="0.6", marker="o", ms=6.0, lw=0,
                     markeredgecolor="black", markeredgewidth=1.0,
                     label="declared operating point (Fig. 5)"))
    W, H = LAYOUT["fig_mm"]
    fig.legend(handles=hh, loc="center", bbox_to_anchor=(0.5, 1.0 - LAYOUT["legend_y"] / H),
               ncol=len(hh), frameon=False, handlelength=1.6, handletextpad=0.4,
               columnspacing=1.0)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=os.path.join(RES, "env_v3_sweep.csv"))
    ap.add_argument("--summary", default=os.path.join(RES, "env_v3_summary.csv"))
    ap.add_argument("--tol", type=float, default=15.0, help="matching tolerance (days)")
    ap.add_argument("--base-font", type=float, default=7.0,
                    help="base font size in pt at final print size (journal max 7)")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "figS_env_sweep.pdf"))
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    sweep = sweep_by_method([r for r in load_rows(args.sweep)
                             if r["method"] in METHOD_STYLE and r["tol"] == args.tol])
    declared = {r["method"]: r for r in load_rows(args.summary)
                if r["method"] in METHOD_STYLE and r["tol"] == args.tol}
    methods = [m for m in METHOD_STYLE if m in sweep or m in declared]
    missing = [m for m in declared if m not in sweep]
    if missing:
        print(f"[figS] not swept (declared point only): {missing}")

    W_mm, H_mm = LAYOUT["fig_mm"]
    P = LAYOUT["panels"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    ax_a = ax_mm(fig, **P["a"]); panel_a(ax_a, sweep, declared); plabel(fig, "a", dx_mm=-10.0)
    ax_b = ax_mm(fig, **P["b"]); panel_b(ax_b, sweep, declared); plabel(fig, "b", dx_mm=-10.0)
    legend_row(fig, methods)

    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"wrote {args.out} and {png}  ({W_mm:g} x {H_mm:g} mm)")
    for m in methods:
        rs = sweep.get(m, [])
        if rs:
            best = max(rs, key=lambda r: r["consistency"])
            print(f"  {m:9s} consistency over grid {min(r['consistency'] for r in rs):+.3f} to "
                  f"{best['consistency']:+.3f} (max at {best['budget_kind']}={best['budget']:g}); "
                  f"declared {declared[m]['consistency']:+.3f}")
        else:
            print(f"  {m:9s} declared {declared[m]['consistency']:+.3f} (not swept)")


if __name__ == "__main__":
    main()
