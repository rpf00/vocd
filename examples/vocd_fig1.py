"""
Fig 1 — VOCD pipeline overview. Schematic only.

Panels (no plot titles; bold lowercase panel letters; Arial):
  a  full pipeline, start to finish: input time series (with change-points and
     outliers) -> two Viterbi passes (high-recall TPR / high-precision PPV) ->
     two candidate CP lists -> preliminary CP list (union) -> MW & KS gate ->
     verified CP list.  Circled tags (b/c/d) mark the stages the detail panels
     zoom into.
  b  zoom on the Viterbi pass: a 3-state (A/B/C) trellis over t=1..6 with the
     best path (Viterbi solution) highlighted and two change-points (CP1, CP2).
  c  the two passes side by side: high-recall (TPR) catches every true CP plus
     false alarms; high-precision (PPV) is clean but misses weak CPs; their union
     is the candidate set handed to the gate.
  d  zoom on the MW & KS gate: two windows of half-width verify_w around a
     candidate CP feed a Mann-Whitney U (rank-sum) and a Kolmogorov-Smirnov test;
     p_max = max(p_MW, p_KS) is compared to a Bonferroni budget alpha = fwer / m.

This figure is a schematic: the signal in a/c/d is a fixed deterministic demo
(seeded) chosen for legibility, not the paper's benchmark.  If the repo's
synthetic_data module is importable it is used for the demo trace; otherwise a
self-contained fallback is generated, so the script runs standalone with only
numpy + matplotlib.  Saved to results/pipeline_overview/ at 300 dpi.

    python examples/vocd_fig1.py
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "pipeline_overview")

# ── palette (matches the draft's colour coding) ───────────────────────────────
C_IO_F,  C_IO_E  = "#cfe0f2", "#2c5f8a"   # blue  : input / output boxes
C_PS_F,  C_PS_E  = "#f6c873", "#cf8a1e"   # gold  : Viterbi passes
C_LS_F,  C_LS_E  = "#eaeaea", "#8a8a8a"   # grey  : candidate / preliminary lists
C_GT_F,  C_GT_E  = "#f2b3a3", "#c0392b"   # salmon: MW & KS gate
GOLD, GREY, RED, BLUE = "#e29a1a", "#c9c9c9", "#d62728", "#2c5f8a"
BLUEW, ORANGEW = "#4c78b0", "#e08a3c"     # left / right window shades


# ── demo signal (deterministic; illustrative, not the benchmark) ──────────────
def demo_signal(n=320, seed=7):
    """Piecewise-constant trace with 3 change-points and a few additive outliers."""
    try:                                     # use repo data if available
        from synthetic_data import make_paper_data
        data, cps = make_paper_data()
        y = np.asarray(data).ravel().astype(float)
        cps = [int(c) for c in cps] if np.ndim(cps) else None
        if cps:
            rng = np.random.default_rng(seed)
            out = np.sort(rng.choice(len(y), size=5, replace=False))
            return y, cps, out
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    cps = [80, 150, 240]
    means = [0.0, 1.9, 0.6, 2.7]
    bounds = [0] + cps + [n]
    y = np.empty(n)
    for k in range(len(means)):
        y[bounds[k]:bounds[k + 1]] = means[k]
    y = y + rng.normal(0, 0.24, n)
    out = np.sort(rng.choice(n, size=5, replace=False))
    y[out] += rng.choice([-1, 1], size=5) * rng.uniform(1.7, 2.5, size=5)
    return y, cps, out


# ── style / small drawing helpers ─────────────────────────────────────────────
def set_nature_style():
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    # Prefer real Arial (present on most local machines); fall back to the
    # metric-compatible Arial clones Arimo / Liberation Sans so the figure looks
    # identical where Arial is absent (e.g. CI, this sandbox), and only then to
    # DejaVu.  If none of the Arial-family fonts resolve, warn once.
    arial_family = ["Arial", "Arimo", "Liberation Sans", "Helvetica"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not (set(arial_family) & have):
        print("[fig1] warning: no Arial-family font found; using DejaVu Sans. "
              "Install Arial (or 'fonts-liberation') for the intended look.")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": arial_family + ["DejaVu Sans"],
        # maths in matplotlib's built-in sans face (always resolves; a hair
        # different from Arial but avoids missing-glyph fallbacks for α, ⇔, …)
        "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def plabel(ax, letter, dx=-0.02, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="right")


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def rbox(ax, cx, cy, w, h, title, *, fill, edge, sub=None,
         fs=9.5, sub_fs=8.6, tc="0.12", lw=1.3):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.14",
                 linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(cx, cy - h * 0.24, sub, ha="center", va="center",
                fontsize=sub_fs, color=tc, style="italic", zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="0.35", lw=1.5, rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=13, lw=lw, color=color, shrinkA=0, shrinkB=0,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def tag(ax, x, y, letter, color):
    ax.text(x, y, letter, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="white", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.28", fc=color, ec="none"))


def series_in_box(ax, x0, x1, y0, y1, y, cps=(), outliers=(), lw=0.9,
                  color="0.22", cp_ax=True):
    """Map a 1-D series into the rectangle [x0,x1]x[y0,y1] on a schematic axis."""
    n = len(y)
    xs = x0 + (x1 - x0) * np.arange(n) / (n - 1)
    ys = y0 + (y1 - y0) * (y - y.min()) / (np.ptp(y) + 1e-9)
    ax.plot(xs, ys, color=color, lw=lw, zorder=3)
    for cp in cps:
        xc = x0 + (x1 - x0) * cp / (n - 1)
        ax.plot([xc, xc], [y0, y1], color=RED, ls="--", lw=0.8, zorder=2)
    if len(outliers):
        oi = np.asarray(outliers)
        ax.scatter(xs[oi], ys[oi], s=7, color=RED, zorder=4, linewidths=0)
    return xs


def curve(ax, rx, ry, x0, x1, y0, y1, **kw):
    ax.plot(x0 + (x1 - x0) * np.asarray(rx),
            y0 + (y1 - y0) * np.asarray(ry), **kw)


# ── panel a : full pipeline ───────────────────────────────────────────────────
def panel_a(ax, y, cps, outliers):
    blank(ax); ax.set_xlim(0, 10); ax.set_ylim(0.4, 13.3)

    # input (with a mini trace inside)
    rbox(ax, 5, 11.85, 7.4, 2.25, "", fill=C_IO_F, edge=C_IO_E)
    ax.text(5, 12.75, "Input time series", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="0.12", zorder=3)
    series_in_box(ax, 2.1, 7.9, 11.05, 12.35, y, cps, outliers, lw=0.7)

    arrow(ax, 5, 10.72, 5, 10.42)                       # split point

    # two passes
    rbox(ax, 2.65, 9.4, 4.0, 1.55, "Viterbi pass 1",
         fill=C_PS_F, edge=C_PS_E, sub="high recall  ·  TPR")
    rbox(ax, 7.35, 9.4, 4.0, 1.55, "Viterbi pass 2",
         fill=C_PS_F, edge=C_PS_E, sub="high precision  ·  PPV")
    arrow(ax, 5, 10.42, 2.65, 10.175, rad=0.0)
    arrow(ax, 5, 10.42, 7.35, 10.175, rad=0.0)

    # two candidate lists
    rbox(ax, 2.65, 7.35, 3.7, 1.2, "CP list 1",
         fill=C_LS_F, edge=C_LS_E, sub="recall candidates", fs=9)
    rbox(ax, 7.35, 7.35, 3.7, 1.2, "CP list 2",
         fill=C_LS_F, edge=C_LS_E, sub="precision candidates", fs=9)
    arrow(ax, 2.65, 8.625, 2.65, 7.95)
    arrow(ax, 7.35, 8.625, 7.35, 7.95)

    # union
    rbox(ax, 5, 5.35, 5.4, 1.35, "Preliminary CP list",
         fill=C_LS_F, edge=C_LS_E, sub="union")
    arrow(ax, 2.65, 6.75, 4.15, 6.025, rad=0.0)
    arrow(ax, 7.35, 6.75, 5.85, 6.025, rad=0.0)

    # gate
    rbox(ax, 5, 3.3, 5.6, 1.5, "MW & KS tests",
         fill=C_GT_F, edge=C_GT_E, sub="per candidate CP")
    arrow(ax, 5, 4.675, 5, 4.05)

    # output
    rbox(ax, 5, 1.3, 5.0, 1.3, "Verified CP list",
         fill=C_IO_F, edge=C_IO_E, sub="output")
    arrow(ax, 5, 2.55, 5, 1.95)

    # zoom tags to the detail panels
    tag(ax, 9.35, 9.4, "b", C_PS_E)
    tag(ax, 9.35, 7.35, "c", "0.45")
    tag(ax, 8.05, 3.3, "d", C_GT_E)

    plabel(ax, "a", dx=-0.01, dy=1.005)


# ── panel b : Viterbi trellis ─────────────────────────────────────────────────
def panel_b(ax):
    from matplotlib.patches import Ellipse
    blank(ax); ax.set_xlim(0.05, 6.95); ax.set_ylim(0.25, 3.95)
    rows = {"A": 1.0, "B": 2.0, "C": 3.0}
    T = [1, 2, 3, 4, 5, 6]
    best = {1: "A", 2: "A", 3: "B", 4: "B", 5: "C", 6: "C"}
    ew, eh = 0.52, 0.46

    # faint trellis transitions
    for t in T[:-1]:
        for s0 in rows:
            for s1 in rows:
                ax.plot([t, t + 1], [rows[s0], rows[s1]],
                        color="0.86", lw=0.5, ls=(0, (3, 3)), zorder=1)

    # nodes
    for t in T:
        for s, yy in rows.items():
            on = best[t] == s
            ax.add_patch(Ellipse((t, yy), ew, eh, facecolor=GOLD if on else "#dedede",
                         edgecolor=C_PS_E if on else "#b8b8b8", lw=1.0 if on else 0.7,
                         zorder=3))
            ax.text(t, yy, s, ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white" if on else "0.4", zorder=4)

    # best-path arrows
    for t in T[:-1]:
        arrow(ax, t + ew * 0.36, rows[best[t]], t + 1 - ew * 0.36, rows[best[t + 1]],
              color=C_PS_E, lw=2.0)

    # change-points
    for xc, lab in [(2.5, "CP$_1$"), (4.5, "CP$_2$")]:
        ax.plot([xc, xc], [0.45, 3.6], color=RED, ls="--", lw=1.1, zorder=2)
        ax.text(xc, 3.72, lab, ha="center", va="bottom", fontsize=7.5, color=RED)

    # row labels (each state = one latent regime with its own mean/variance)
    for s, yy in rows.items():
        ax.text(0.62, yy, f"{s}", ha="right", va="center", fontsize=8.5,
                fontweight="bold", color="0.25")
    ax.text(0.24, 2.0, r"regime bank  ($S$ states; 3 shown)", rotation=90,
            ha="center", va="center", fontsize=7.4, color="0.5", style="italic")

    # time axis
    for t in T:
        ax.text(t, 0.33, f"t={t}", ha="center", va="top", fontsize=6.8, color="0.4")

    # legend
    ax.scatter([], [], s=40, c=GOLD, marker="o", label="best path (Viterbi)")
    ax.scatter([], [], s=40, c="#dedede", marker="o", label="other paths")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False,
              fontsize=6.8, handletextpad=0.3, ncol=2, columnspacing=1.0)
    plabel(ax, "b")


# ── panel c : two passes + union ──────────────────────────────────────────────
def panel_c(ax, y, cps):
    blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    xL, xR = 0.24, 0.985
    n = len(y)

    def xcp(c):
        return xL + (xR - xL) * c / (n - 1)

    # signal on top
    series_in_box(ax, xL, xR, 0.64, 0.955, y, lw=0.8)
    for c in cps:                                   # true-CP guide lines
        ax.plot([xcp(c), xcp(c)], [0.09, 0.63], color="0.8", ls="--", lw=0.7, zorder=1)

    fp = [42, 205]                                  # recall-pass false alarms
    miss = [cps[1]]                                 # precision-pass miss (weak CP)
    prec_hits = [c for c in cps if c not in miss]

    def tick(x, yc, color, h=0.032, lw=1.8):
        ax.plot([x, x], [yc - h, yc + h], color=color, lw=lw,
                solid_capstyle="round", zorder=4)

    rails = [(0.47, "high recall", "TPR"),
             (0.31, "high precision", "PPV"),
             (0.15, "union", "→ gate")]
    for yc, t1, t2 in rails:
        ax.plot([xL, xR], [yc, yc], color="0.85", lw=1.0, zorder=2)
        ax.text(0.205, yc + 0.03, t1, ha="right", va="center",
                fontsize=8.2, fontweight="bold", color="0.2")
        ax.text(0.205, yc - 0.032, t2, ha="right", va="center",
                fontsize=7.6, color="0.5", style="italic")

    # recall: all true CPs (correct) + false alarms
    for c in cps:
        tick(xcp(c), 0.47, "#2ca02c")
    for c in fp:
        tick(xcp(c), 0.47, RED)
    # precision: subset of true CPs (clean), misses marked as open circles
    for c in prec_hits:
        tick(xcp(c), 0.31, "#2ca02c")
    for c in miss:
        ax.scatter([xcp(c)], [0.31], s=22, facecolors="none",
                   edgecolors="0.55", linewidths=1.0, zorder=4)
    # union: every unique candidate handed to the gate
    for c in sorted(set(cps) | set(fp)):
        tick(xcp(c), 0.15, "0.4")

    # key
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#2ca02c", lw=1.8, label="true CP"),
        Line2D([0], [0], color=RED, lw=1.8, label="false alarm"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="0.55", label="missed CP", markersize=5)],
        loc="lower left", bbox_to_anchor=(0.205, -0.02), frameon=False,
        fontsize=7.4, ncol=3, handletextpad=0.3, columnspacing=1.1)
    plabel(ax, "c")


# ── panel d : MW & KS gate ────────────────────────────────────────────────────
def panel_d(ax):
    from matplotlib.patches import FancyBboxPatch, Rectangle
    blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rng = np.random.default_rng(3)

    # (i) windows around the candidate CP -------------------------------------
    ax0, ax1, ay0, ay1 = 0.045, 0.325, 0.46, 0.93
    m = 30
    snip = np.concatenate([rng.normal(0.0, 0.30, m), rng.normal(1.5, 0.30, m)])
    ax.add_patch(Rectangle((ax0, ay0), (ax1 - ax0) / 2, ay1 - ay0,
                 facecolor=BLUEW, alpha=0.16, zorder=1, lw=0))
    ax.add_patch(Rectangle((ax0 + (ax1 - ax0) / 2, ay0), (ax1 - ax0) / 2, ay1 - ay0,
                 facecolor=ORANGEW, alpha=0.18, zorder=1, lw=0))
    series_in_box(ax, ax0, ax1, ay0 + 0.02, ay1 - 0.02, snip, lw=0.8, color="0.2")
    xc = (ax0 + ax1) / 2
    ax.plot([xc, xc], [ay0, ay1], color=RED, ls="--", lw=1.1, zorder=4)
    ax.text(xc, ay1 + 0.015, "candidate CP", ha="center", va="bottom",
            fontsize=7.2, color=RED, style="italic")
    for xa, xb, lab, col in [(ax0, xc, "verify_w", BLUEW), (xc, ax1, "verify_w", ORANGEW)]:
        ax.annotate("", (xa + 0.008, ay0 - 0.03), (xb - 0.008, ay0 - 0.03),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=0.9))
        ax.text((xa + xb) / 2, ay0 - 0.08, lab, ha="center", va="top",
                fontsize=7.2, color=col, style="italic")
    ax.text((ax0 + ax1) / 2, ay1 + 0.06, "two windows",
            ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="0.2")

    # (ii) Mann–Whitney : rank-sum on the two samples --------------------------
    bx0, bx1, by0, by1 = 0.38, 0.655, 0.50, 0.90
    xx = np.linspace(-3.6, 5.2, 220)
    pL = np.exp(-0.5 * (xx - 0.0) ** 2); pR = np.exp(-0.5 * (xx - 1.6) ** 2)
    rx = (xx - xx[0]) / np.ptp(xx)
    curve(ax, rx, pL / pL.max(), bx0, bx1, by0, by1, color=BLUEW, lw=1.4)
    curve(ax, rx, pR / pR.max(), bx0, bx1, by0, by1, color=ORANGEW, lw=1.4)
    ax.fill_between(bx0 + (bx1 - bx0) * rx, by0, by0 + (by1 - by0) * pL / pL.max(),
                    color=BLUEW, alpha=0.12, lw=0)
    ax.fill_between(bx0 + (bx1 - bx0) * rx, by0, by0 + (by1 - by0) * pR / pR.max(),
                    color=ORANGEW, alpha=0.12, lw=0)
    ax.text((bx0 + bx1) / 2, by1 + 0.06, "Mann–Whitney U",
            ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="0.2")
    ax.text((bx0 + bx1) / 2, by0 - 0.055, r"$p_{\mathrm{MW}}$",
            ha="center", va="top", fontsize=10, color="0.15")
    ax.text((bx0 + bx1) / 2, by0 - 0.135, "rank-sum · location",
            ha="center", va="top", fontsize=7.0, color="0.5", style="italic")

    # (iii) KS : max CDF gap ---------------------------------------------------
    cx0, cx1, cy0, cy1 = 0.70, 0.975, 0.50, 0.90
    cL = np.cumsum(pL); cL /= cL[-1]; cR = np.cumsum(pR); cR /= cR[-1]
    curve(ax, rx, cL, cx0, cx1, cy0, cy1, color=BLUEW, lw=1.4)
    curve(ax, rx, cR, cx0, cx1, cy0, cy1, color=ORANGEW, lw=1.4)
    di = int(np.argmax(np.abs(cL - cR)))
    xd = cx0 + (cx1 - cx0) * rx[di]
    ax.annotate("", (xd, cy0 + (cy1 - cy0) * cR[di]), (xd, cy0 + (cy1 - cy0) * cL[di]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
    ax.text(xd + 0.012, cy0 + (cy1 - cy0) * (cL[di] + cR[di]) / 2, "D",
            ha="left", va="center", fontsize=7.5, color=RED, fontweight="bold")
    ax.text((cx0 + cx1) / 2, cy1 + 0.06, "Kolmogorov–Smirnov",
            ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="0.2")
    ax.text((cx0 + cx1) / 2, cy0 - 0.055, r"$p_{\mathrm{KS}}$",
            ha="center", va="top", fontsize=10, color="0.15")
    ax.text((cx0 + cx1) / 2, cy0 - 0.135, "max CDF gap · shape",
            ha="center", va="top", fontsize=7.0, color="0.5", style="italic")

    # decision strip : DECIDE (multiple-testing correction) + MERGE (dedup) -----
    ax.add_patch(FancyBboxPatch((0.05, 0.035), 0.90, 0.265,
                 boxstyle="round,pad=0,rounding_size=0.02",
                 facecolor="#fbeeea", edgecolor=C_GT_E, lw=1.2, zorder=2))
    ax.text(0.50, 0.245,
            r"$p_{\max}=\max(p_{\mathrm{MW}},\,p_{\mathrm{KS}})$"
            r"$\quad$vs$\quad \alpha = \mathrm{fwer}\,/\,m$  (Bonferroni)",
            ha="center", va="center", fontsize=8.6, color="0.15", zorder=3)
    ax.text(0.50, 0.16,
            r"keep candidate CP  $\Leftrightarrow\ p_{\max} < \alpha$",
            ha="center", va="center", fontsize=8.2, color=C_GT_E,
            fontweight="bold", zorder=3)
    ax.text(0.50, 0.08, "then merge survivors within  dedup_w",
            ha="center", va="center", fontsize=7.2, color="0.45",
            style="italic", zorder=3)
    plabel(ax, "d")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "fig1.png"))
    args = ap.parse_args()
    set_nature_style()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTDIR, exist_ok=True)

    y, cps, outliers = demo_signal(seed=args.seed)

    fig = plt.figure(figsize=(15, 12.5))
    gs = fig.add_gridspec(3, 5, hspace=0.34, wspace=0.48,
                          width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
                          height_ratios=[1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:5])
    ax_c = fig.add_subplot(gs[1, 2:5])
    ax_d = fig.add_subplot(gs[2, 2:5])

    panel_a(ax_a, y, cps, outliers)
    panel_b(ax_b)
    panel_c(ax_c, y, cps)
    panel_d(ax_d)

    # reproducibility CSV for the illustrative demo trace
    with open(os.path.join(OUTDIR, "fig1_demo_signal.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["idx", "value", "is_cp", "is_outlier"])
        cpset, oset = set(cps), set(int(o) for o in outliers)
        for i, v in enumerate(y):
            w.writerow([i, f"{v:.5f}", int(i in cpset), int(i in oset)])

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}  (300 dpi)")
    print(f"demo CPs at {cps};  outliers at {list(int(o) for o in outliers)}")


if __name__ == "__main__":
    main()
