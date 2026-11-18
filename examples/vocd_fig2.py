"""
Fig 2 (Nature-Communications style) — VOCD robustness. VOCD only.

Panels (no plot titles; bold lowercase panel letters; Arial):
  a  scheme: synthetic signal at 0/2/5/7/10/15/20% outliers (outliers in red)
  b  heatmap  verify_w x fwer   (mean F1, 2% outliers)
  c  heatmap  dedup_w  x fwer   (mean F1, 2% outliers)
  d  mean F1 vs outlier fraction (defaults)
  e  tornado: internal-defaults one-at-a-time F1 range (role-appropriate ranges),
     2% and stress (10%) levels

  f  F1 vs runtime for the full benchmark panel, read from the committed sweep
     CSV (results/synthetic/synthetic_v3_best.csv): the 8 methods plus the
     three compiled twins (same detections, only runtime differs).

Same synthetic protocol as the paper (length/CP locations/means/seed identical;
only outlier fraction varies).  Saved to results/heatmap_robust/ as PDF (vector,
what the journal wants) and PNG (300 dpi preview).

LAYOUT — everything about size and placement lives in the LAYOUT dict below,
in MILLIMETRES at final print size, so the figure is built exactly as it will
appear on the page (no scaling by the journal, no shrinking fonts):

  LAYOUT["fig_mm"]          overall figure (width, height); double column = 180 mm
                            wide, must fit the 210 x 276 mm page with its legend
  LAYOUT["panels"][letter]  x, y, w, h of each panel's AXES box in mm, measured
                            from the TOP-LEFT corner of the figure (y grows down,
                            like a page).  Axis labels and tick labels sit
                            OUTSIDE the box, so leave ~10 mm to the left / below.
  LAYOUT["cbar_w"], ["cbar_gap"]   colour-bar width and gap for the heatmaps
  --base-font (pt)          every text size derives from it: ticks = base-1,
                            axis labels = base, panel letters = base+1 bold,
                            in-cell / in-plot annotations = base-1.5.  Nature
                            Communications lettering is typically 7-8 pt at
                            final size; the default is 7.

To move panel e 5 mm right: LAYOUT["panels"]["e"]["x"] += 5.  To make panel f
taller: ["f"]["h"] = 60 (and push nothing: y is top-anchored).  To change the
aspect of the whole figure: LAYOUT["fig_mm"] = (180, 220) and re-place.

    python examples/vocd_fig2_natcomms.py
    python examples/vocd_fig2_natcomms.py --fast          # a-e via the C++ engine
    python examples/vocd_fig2_natcomms.py --base-font 12  # larger text

Panels a-e are VOCD only and are recomputed here (identical to the Python
engine with --fast, which just runs ~50x faster).  Panel f never recomputes
anything: it aggregates the per-trial rows of the sweep CSV, so re-running
the sweep and re-running this script keeps the figure and the table in sync.
"""
from __future__ import annotations
import argparse, csv, dataclasses, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vocd
from vocd.core import DEFAULT_PPV, DEFAULT_TPR
from synthetic_data import (make_paper_data, contaminate, accuracy_delay,
                            SEED_CONTAM, CONTAM_SPIKE)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "heatmap_robust")

FWER = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
VERIFY_W = [10, 20, 30, 40, 50]
DEDUP_W = [10, 20, 30, 40, 50]
#OUTLIER_FRACS = [0.0, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20]
OUTLIER_FRACS = [0.02, 0.05, 0.07, 0.10, 0.15, 0.20]
DEFAULT_FWER, DEFAULT_VERIFY_W, DEFAULT_DEDUP_W, DEFAULT_MIN_SIDE = 0.01, 25, 15, 5

PPV_GRIDS = {"S": [6, 9, 12, 16], "confirm_k": [3, 5, 8, 12],
             "huber_delta": [1.5, 2.0, 3.0, 5.0], "pen_base": [3.0, 5.0, 8.0, 12.0],
             "min_dwell": [3, 5, 8, 12], "merge_window": [5, 7, 10, 15],
             "init_window": [75, 100, 150, 200]}
TPR_GRIDS = {"S": [3, 5, 8], "confirm_k": [1, 2, 3],
             "huber_delta": [1.5, 2.0, 3.0, 5.0], "pen_base": [0.0, 0.5, 1.0, 2.0],
             "min_dwell": [0, 1, 2, 3], "merge_window": [5, 7, 10, 15],
             "init_window": [75, 100, 150, 200]}
TOP_FIELDS = {"verify_w": [15, 20, 25, 30, 40, 50], "dedup_w": [5, 10, 15, 20, 30],
              "min_side": [3, 5, 8, 12]}

SYNTH_CSV = os.path.join(_REPO, "results", "synthetic", "synthetic_v3_best.csv")

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────
# Nature Portfolio production limits (also used by Nature Communications):
#   width  180 mm (double column; 88 mm single column)
#   height 170 mm MAXIMUM, so the legend fits underneath on the same page
#   text   7 pt maximum, 5 pt minimum; panel letters 8 pt bold lower-case
LAYOUT = {
    "fig_mm": (180.0, 170.0),
    "cbar_w": 2.5, "cbar_gap": 1.5,
    "panels": {
        #      x      y      w      h
        "a": dict(x=12.0, y=5.0,   w=42.0, h=45.0),
        "b": dict(x=66.0, y=5.0,   w=38.0, h=45.0),
        "c": dict(x=127.0, y=5.0,  w=38.0, h=45.0),
        "d": dict(x=12.0, y=67.0,  w=55.0, h=40.0),
        "e": dict(x=92.0, y=67.0,  w=82.0, h=40.0),
        "f": dict(x=12.0, y=120.0, w=162.0, h=40.0),
    },
}

# Okabe-Ito colour-blind-safe palette (Nature asks to avoid green/red pairs)
C_VOCD, C_BOCD, C_DMBOCD = "#009E73", "#E69F00", "#CC79A7"
C_FOCUS, C_NPFOCUS, C_NUNC, C_CHAD, C_CHAMP = "#56B4E9", "#0072B2", "#F0E442", "#D55E00", "#000000"
C_OUTLIER, C_STRESS, C_GREY = "#D55E00", "#D55E00", "0.70"


def ax_mm(fig, x, y, w, h):
    """Axes at (x, y) mm from the top-left of the figure, w x h mm."""
    W, H = LAYOUT["fig_mm"]
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def cbar_mm(fig, panel, im, label):
    """Colour bar to the right of a panel, same height."""
    p = LAYOUT["panels"][panel]
    cax = ax_mm(fig, p["x"] + p["w"] + LAYOUT["cbar_gap"], p["y"], LAYOUT["cbar_w"], p["h"])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=F["tick"])
    cb.ax.tick_params(labelsize=F["tick"], width=0.6, length=2)
    return cb


# panel f: CSV method name -> (label, colour, marker).  Python/C++ twins share a
# colour; circle = Python, square = C++.
PANEL_F_STYLE = {
    "vocd":           ("VOCD",      C_VOCD,    "o"),
    "vocd (cpp)":     ("VOCD (C++)",     C_VOCD,    "s"),
    "bocd":           ("BOCD",           C_BOCD,    "o"),
    "dm-bocd":        ("Dm-BOCD",        C_DMBOCD,  "o"),
    "focus":          ("FOCuS",     C_FOCUS,   "o"),
    "focus (cpp)":    ("FOCuS (C++)",    C_FOCUS,   "s"),
    "np-focus":       ("NP-FOCuS",  C_NPFOCUS, "o"),
    "np-focus (cpp)": ("NP-FOCuS (C++)", C_NPFOCUS, "s"),
    "nunc":           ("NUNC",           C_NUNC,    "o"),
    "chad":           ("CHAD",           C_CHAD,    "o"),
    "champ":          ("CHAMP",          C_CHAMP,   "o"),
}
# label offsets (points) so names do not collide; default is above-right
PANEL_F_OFFSET = {       # (x, y) in points; default (5, 4) = above-right
    "vocd (cpp)": (5, -9), "vocd": (5, -9), "focus (cpp)": (5, -9),
    "np-focus (cpp)": (5, -9), "nunc": (5, -9),
}


def load_panel_f(path):
    """Aggregate the sweep's per-trial rows: method -> (mean F1, std F1,
    mean runtime ms).  Order follows PANEL_F_STYLE; missing methods are skipped."""
    import collections
    f1, rt = collections.defaultdict(list), collections.defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            f1[row["method"]].append(float(row["f1"]))
            rt[row["method"]].append(float(row["runtime_ms"]))
    out = []
    for m in PANEL_F_STYLE:
        if m in f1:
            out.append((m, float(np.mean(f1[m])), float(np.std(f1[m])),
                        float(np.mean(rt[m]))))
    missing = [m for m in PANEL_F_STYLE if m not in f1]
    if missing:
        print(f"panel f: methods absent from {os.path.basename(path)}: {missing}")
    return out


def build_sweeps():
    out = [(f"ppv.{f}", "ppv", f, v) for f, v in PPV_GRIDS.items()]
    out += [(f"tpr.{f}", "tpr", f, v) for f, v in TPR_GRIDS.items()]
    out += [(f, "top", f, v) for f, v in TOP_FIELDS.items()]
    return out


def trials_at(frac, n):
    data, _ = make_paper_data()
    np.random.seed(SEED_CONTAM)
    return [contaminate(data, frac=frac, spike=CONTAM_SPIKE) for _ in range(n)]


def one_contaminated(frac):
    data, _ = make_paper_data()
    np.random.seed(SEED_CONTAM)
    return data.ravel(), contaminate(data, frac=frac, spike=CONTAM_SPIKE).ravel()


def mean_f1(trials, *, ppv=DEFAULT_PPV, tpr=DEFAULT_TPR, verify_w=DEFAULT_VERIFY_W,
            dedup_w=DEFAULT_DEDUP_W, min_side=DEFAULT_MIN_SIDE, fwer=DEFAULT_FWER):
    f1 = []
    for dc in trials:
        r = vocd.detect(dc.ravel(), ppv=ppv, tpr=tpr, verify_w=verify_w,
                        dedup_w=dedup_w, min_side=min_side)
        f1.append(accuracy_delay([0] + sorted(int(c) for c in r.changepoints(fwer=fwer)))["f1"])
    return float(np.mean(f1))


def f1_with(where, field, val, trials):
    kw = dict(ppv=DEFAULT_PPV, tpr=DEFAULT_TPR, verify_w=DEFAULT_VERIFY_W,
              dedup_w=DEFAULT_DEDUP_W, min_side=DEFAULT_MIN_SIDE, fwer=DEFAULT_FWER)
    if where == "ppv":
        kw["ppv"] = dataclasses.replace(DEFAULT_PPV, **{field: val})
    elif where == "tpr":
        kw["tpr"] = dataclasses.replace(DEFAULT_TPR, **{field: val})
    else:
        kw[field] = val
    return mean_f1(trials, **kw)


def panel_C(trials, sweeps):
    rows = []
    for label, where, field, values in sweeps:
        f1s = [f1_with(where, field, v, trials) for v in values]
        rows.append((label, min(f1s), max(f1s), max(f1s) - min(f1s)))
    return rows


F = {}   # derived font sizes, filled by set_nature_style(base)


def set_nature_style(base=7.0):
    """All text sizes derive from one base size in pt AT FINAL PRINT SIZE.
    Nature Communications lettering is typically 7-8 pt; panel letters are
    bold lower-case at about the same size."""
    import matplotlib as mpl
    F.update(base=base, label=base, tick=base - 1, legend=base - 1,
             small=base - 1.5,        # in-cell / in-plot annotations
             panel=base + 1)          # bold panel letters
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base, "axes.labelsize": F["label"], "axes.titlesize": F["label"],
        "xtick.labelsize": F["tick"], "ytick.labelsize": F["tick"],
        "legend.fontsize": F["legend"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # editable text in the PDF
        "figure.dpi": 120, "savefig.dpi": 300,
    })


def plabel(fig, panel, dx_mm=-9.0, dy_mm=-1.5):
    """Bold lower-case panel letter, placed dx/dy mm from the panel's top-left
    corner (negative dx = to the left, negative dy = above)."""
    p = LAYOUT["panels"][panel]
    W, H = LAYOUT["fig_mm"]
    fig.text((p["x"] + dx_mm) / W, 1.0 - (p["y"] + dy_mm) / H, panel,
             fontsize=F["panel"], fontweight="bold", va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--stress-frac", type=float, default=0.10)
    ap.add_argument("--sens-threshold", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "fig2_natcomms.pdf"),
                    help="vector output; a .png preview is written alongside")
    ap.add_argument("--base-font", type=float, default=7.0,
                    help="base font size in pt AT FINAL PRINT SIZE (180 mm wide); "
                         "every text size derives from it. Journal norm is 7-8")
    ap.add_argument("--synthetic-csv", default=SYNTH_CSV,
                    help="per-trial best-point CSV from synthetic_data_v3.py --sweep")
    ap.add_argument("--fast", action="store_true",
                    help="run panels a-e through the full C++ VOCD engine "
                         "(identical results, ~50x faster)")
    args = ap.parse_args()
    if args.fast:
        import vocd._fastdetect as fastdetect
        fastdetect.enable()
    if not os.path.exists(args.synthetic_csv):
        sys.exit(f"panel f needs {args.synthetic_csv}; run synthetic_data_v3.py --sweep first")
    panel_f = load_panel_f(args.synthetic_csv)
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    os.makedirs(OUTDIR, exist_ok=True)

    base = trials_at(0.02, args.trials)
    stress = trials_at(args.stress_frac, args.trials)
    sweeps = build_sweeps()

    print("panel b: verify_w x fwer …")
    Hb = np.array([[mean_f1(base, verify_w=w, fwer=f) for f in FWER] for w in VERIFY_W])
    print("panel c: dedup_w x fwer …")
    Hc = np.array([[mean_f1(base, dedup_w=d, fwer=f) for f in FWER] for d in DEDUP_W])
    print("panel d: F1 vs outlier fraction …")
    Bd = [mean_f1(trials_at(fr, args.trials)) for fr in OUTLIER_FRACS]
    print(f"panel e: tornado at 2% and {args.stress_frac:.0%} …")
    C2 = panel_C(base, sweeps)
    Cs = panel_C(stress, sweeps)
    f1_def2, f1_defS = mean_f1(base), mean_f1(stress)

    # CSVs
    def dump(path, header, rows):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); [w.writerow(r) for r in rows]
    dump(os.path.join(OUTDIR, "fig2_panelBc_verifyw.csv"), ["verify_w", "fwer", "f1"],
         [[VERIFY_W[i], FWER[j], Hb[i, j]] for i in range(len(VERIFY_W)) for j in range(len(FWER))])
    dump(os.path.join(OUTDIR, "fig2_panelCc_dedupw.csv"), ["dedup_w", "fwer", "f1"],
         [[DEDUP_W[i], FWER[j], Hc[i, j]] for i in range(len(DEDUP_W)) for j in range(len(FWER))])
    dump(os.path.join(OUTDIR, "fig2_panelD_outliers.csv"), ["frac", "f1"],
         [[fr, v] for fr, v in zip(OUTLIER_FRACS, Bd)])
    dump(os.path.join(OUTDIR, "fig2_panelE_tornado.csv"),
         ["param", "swing_2pct", "swing_stress"],
         [[C2[k][0], C2[k][3], Cs[k][3]] for k in range(len(C2))])

    # ── figure, built at final print size from LAYOUT (mm) ──────────
    W_mm, H_mm = LAYOUT["fig_mm"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    P = LAYOUT["panels"]

    # (a) scheme: stacked contaminated traces
    ax_a = ax_mm(fig, **P["a"])
    _, _probe = one_contaminated(0.20)
    off = float(np.ptp(_probe)) * 1.15
    n = len(OUTLIER_FRACS)
    for i, fr in enumerate(OUTLIER_FRACS):
        clean, dc = one_contaminated(fr)
        base_off = (n - 1 - i) * off
        ax_a.plot(clean + base_off, color="0.55", lw=0.5)
        omask = np.abs(dc - clean) > 1e-9
        ax_a.plot(np.where(~omask, dc, np.nan) + base_off, color="0.15", lw=0.5)
        idx = np.where(omask)[0]
        ax_a.scatter(idx, dc[idx] + base_off, s=3, color=C_OUTLIER, zorder=3, linewidths=0)
        ax_a.text(-25, base_off + off * 0.30, f"{fr*100:g}%", va="center", ha="right",
                  fontsize=F["small"])
        if i < n - 1:
            ax_a.axhline(base_off - off * 0.42, color="0.75", lw=0.4, ls="--")
    ax_a.set_xlim(-75, len(clean)); ax_a.set_yticks([])
    ax_a.set_xlabel("Time index"); ax_a.spines["left"].set_visible(False)
    plabel(fig, "a")

    # (b) verify_w x fwer heatmap
    def heatmap(panel, H, yvals, ylabel):
        ax = ax_mm(fig, **P[panel])
        im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(FWER)))
        ax.set_xticklabels([f"{f:g}" for f in FWER], rotation=45, ha="right")
        ax.set_yticks(range(len(yvals))); ax.set_yticklabels(yvals)
        ax.set_xlabel("Error budget (fwer)"); ax.set_ylabel(ylabel)
        for i in range(len(yvals)):
            for j in range(len(FWER)):
                ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                        color="white" if H[i, j] < 0.6 else "0.1", fontsize=F["small"])
        cbar_mm(fig, panel, im, "Mean F1")
        return ax
    heatmap("b", Hb, VERIFY_W, "Gate window (verify_w)"); plabel(fig, "b")
    heatmap("c", Hc, DEDUP_W, "Dedup window (dedup_w)"); plabel(fig, "c")

    # (d) F1 vs outlier fraction
    ax_d = ax_mm(fig, **P["d"])
    ax_d.plot([f * 100 for f in OUTLIER_FRACS], Bd, "o-", color=C_VOCD, lw=1.0, ms=3)
    ax_d.set_xlabel("Outlier fraction (%)"); ax_d.set_ylabel("Mean F1")
    ax_d.set_ylim(0.6, 1.05); ax_d.grid(alpha=0.25, lw=0.4)
    for i, (x, yv) in enumerate(zip([f * 100 for f in OUTLIER_FRACS], Bd)):
        ax_d.annotate(f"{yv:.3f}", (x, yv), textcoords="offset points",
                      xytext=(0, 4 if i % 2 == 0 else 11), ha="center", fontsize=F["small"])
    plabel(fig, "d")

    # (e) tornado, widest swing on top; colour = contamination level
    ax_e = ax_mm(fig, **P["e"])
    order = sorted(range(len(C2)), key=lambda k: max(C2[k][3], Cs[k][3]))
    labels = [C2[k][0] for k in order]
    def _bar(lo, hi, y, col, z):
        if hi - lo < 1e-4:
            ax_e.plot([lo], [y], "o", color=col, ms=2.2, zorder=z)
        else:
            ax_e.plot([lo, hi], [y, y], color=col, lw=3.0, solid_capstyle="round", zorder=z)
    for row, k in enumerate(order):
        _bar(C2[k][1], C2[k][2], row + 0.18, C_GREY, 2)
        _bar(Cs[k][1], Cs[k][2], row - 0.18, C_STRESS, 3)
    ax_e.axvline(f1_defS, color="0.35", ls="--", lw=0.7)
    ax_e.set_yticks(range(len(labels))); ax_e.set_yticklabels(labels)
    ax_e.set_ylim(-0.7, len(labels) - 0.3)
    lo_all = min(min(r[1] for r in C2), min(r[1] for r in Cs))
    ax_e.set_xlim(min(0.7, lo_all - 0.02), 1.005)
    ax_e.set_xlabel("Mean F1 range over one-at-a-time sweep")
    ax_e.grid(axis="x", alpha=0.25, lw=0.4)
    from matplotlib.lines import Line2D
    ax_e.legend(handles=[Line2D([0], [0], color=C_GREY, lw=3, label="2% outliers"),
                         Line2D([0], [0], color=C_STRESS, lw=3,
                                label=f"{int(args.stress_frac*100)}% outliers"),
                         Line2D([0], [0], color="0.35", ls="--", lw=0.7,
                                label=f"Default F1 ({int(args.stress_frac*100)}%)")],
                loc="lower left", frameon=False, handlelength=1.6)
    plabel(fig, "e", dx_mm=-28.0)

    # (f) F1 vs runtime, one point per benchmark row; twins share a colour
    ax_f = ax_mm(fig, **P["f"])
    for m, f1, sd, rt in panel_f:
        label, col, mk = PANEL_F_STYLE[m]
        ax_f.errorbar(rt, f1, yerr=sd, fmt=mk, color=col, ms=5, capsize=2,
                      elinewidth=0.7, markeredgecolor="0.2", markeredgewidth=0.4, zorder=3)
        xoff, yoff = PANEL_F_OFFSET.get(m, (5, 4))
        ax_f.annotate(label, (rt, f1), textcoords="offset points", xytext=(xoff, yoff),
                      fontsize=F["small"], color="0.15")
    ax_f.set_xscale("log")
    ax_f.set_xlabel("Runtime (ms)"); ax_f.set_ylabel("Mean F1")
    lo_f = min(f1 - sd for _, f1, sd, _ in panel_f)
    ax_f.set_ylim(min(0.60, lo_f - 0.03), 1.03)
    ax_f.grid(alpha=0.25, lw=0.4, which="both")
    ax_f.legend(handles=[Line2D([0], [0], marker="o", color="0.4", ls="", ms=5, label="Python"),
                         Line2D([0], [0], marker="s", color="0.4", ls="", ms=5,
                                label="C++")],
                loc="lower right", frameon=False)
    plabel(fig, "f")

    # vector for submission + raster preview; NO bbox_inches="tight", so the
    # file is exactly LAYOUT["fig_mm"] and nothing is rescaled
    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    print(f"\nwrote {args.out} and {png}  ({W_mm:g} x {H_mm:g} mm, {args.base_font:g} pt base)")
    print(f"default F1: 2%={f1_def2:.3f}, {int(args.stress_frac*100)}%={f1_defS:.3f}")


if __name__ == "__main__":
    main()
