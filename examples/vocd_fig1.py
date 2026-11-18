"""
Fig 1 (Nature-Communications style) — VOCD pipeline overview. Schematic only.

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
numpy + matplotlib.  Saved to results/pipeline_overview/ as PDF (vector) + PNG.

    python examples/vocd_fig1_natcomms.py
    python examples/vocd_fig1_natcomms.py --base-font 7

LAYOUT (same scheme as vocd_fig2_natcomms.py): the figure is built AT FINAL
PRINT SIZE from the LAYOUT dict below, in millimetres measured from the
top-left corner.  Nature Portfolio production limits: 180 mm wide (double
column), 170 mm tall maximum (legend fits underneath), text 7 pt maximum and
5 pt minimum, panel letters 8 pt bold lower-case.  Every text size derives
from --base-font (default 7 = the maximum): panel letters base+1, box
titles base, body text base-0.5, captions base-1, fine print base-1.5.
Because the panels are schematics drawn in data coordinates, keep each
panel's aspect ratio close to the current one when resizing (a is ~1:2.2
tall, b-d are ~2.2:1 wide) or the boxes and ellipses will stretch.
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
GOLD, GREY, RED, BLUE = "#e29a1a", "#c9c9c9", "#D55E00", "#2c5f8a"   # RED = vermillion
BLUEW, ORANGEW = "#0072B2", "#E69F00"     # left / right window shades (Okabe-Ito)
TRUE_CP = "#0072B2"                       # true-CP ticks (was green; green/red is
                                          # the pair Nature asks authors to avoid)


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


# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 160.0),
    "panels": {
        #      x      y      w      h
        "a": dict(x=4.0,  y=5.0,   w=68.0, h=155.0),   # ~1:2.2 tall
        "b": dict(x=80.0, y=5.0,   w=90.0, h=50.0),    # ~2.2:1 wide
        "c": dict(x=75.0, y=50.0,  w=90.0, h=50.0),
        "d": dict(x=80.0, y=109.0, w=90.0, h=50.0),
    },
}
F = {}      # font tiers (pt), filled by set_nature_style(base)
LW = 0.62   # line-weight scale vs the old 381 mm canvas (keeps the same look)


def ax_mm(fig, x, y, w, h):
    """Axes at (x, y) mm from the top-left of the figure, w x h mm."""
    W, H = LAYOUT["fig_mm"]
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


# ── style / small drawing helpers ─────────────────────────────────────────────
def set_nature_style(base=7.0):
    """All text sizes derive from one base size in pt at final print size.
    Nature: 7 pt maximum, 5 pt minimum, panel letters 8 pt bold."""
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
    F.update(panel=base + 1.0, title=base, body=base - 0.5, small=base - 1.0,
             tiny=max(5.0, base - 1.5), math=base)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": arial_family + ["DejaVu Sans"],
        # maths in matplotlib's built-in sans face (always resolves; a hair
        # different from Arial but avoids missing-glyph fallbacks for α, ⇔, …)
        "mathtext.fontset": "dejavusans",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": F["small"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 120, "savefig.dpi": 300,
    })


def plabel(fig, panel, dx_mm=-2.5, dy_mm=-1.0):
    """Bold lower-case panel letter dx/dy mm from the panel's top-left corner."""
    p = LAYOUT["panels"][panel]
    W, H = LAYOUT["fig_mm"]
    fig.text((p["x"] + dx_mm) / W, 1.0 - (p["y"] + dy_mm) / H, panel,
             fontsize=F["panel"], fontweight="bold", va="bottom", ha="left")


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def rbox(ax, cx, cy, w, h, title, *, fill, edge, sub=None,
         fs=None, sub_fs=None, tc="0.12", lw=1.3):
    fs = F["title"] if fs is None else fs
    sub_fs = F["small"] if sub_fs is None else sub_fs
    lw = lw * LW
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
                 mutation_scale=13 * LW, lw=lw * LW, color=color, shrinkA=0, shrinkB=0,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def tag(ax, x, y, letter, color):
    ax.text(x, y, letter, ha="center", va="center", fontsize=F["title"],
            fontweight="bold", color="white", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.28", fc=color, ec="none"))


def series_in_box(ax, x0, x1, y0, y1, y, cps=(), outliers=(), lw=0.9,
                  color="0.22", cp_ax=True):
    """Map a 1-D series into the rectangle [x0,x1]x[y0,y1] on a schematic axis."""
    n = len(y)
    xs = x0 + (x1 - x0) * np.arange(n) / (n - 1)
    ys = y0 + (y1 - y0) * (y - y.min()) / (np.ptp(y) + 1e-9)
    ax.plot(xs, ys, color=color, lw=lw * LW, zorder=3)
    for cp in cps:
        xc = x0 + (x1 - x0) * cp / (n - 1)
        ax.plot([xc, xc], [y0, y1], color=RED, ls="--", lw=0.8 * LW, zorder=2)
    if len(outliers):
        oi = np.asarray(outliers)
        ax.scatter(xs[oi], ys[oi], s=3.5, color=RED, zorder=4, linewidths=0)
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
            fontsize=F["title"], fontweight="bold", color="0.12", zorder=3)
    series_in_box(ax, 2.1, 7.9, 11.05, 12.35, y, cps, outliers, lw=0.7)

    arrow(ax, 5, 10.72, 5, 10.42)                       # split point

    # two passes
    rbox(ax, 2.65, 9.4, 4.0, 1.40, "Viterbi pass 1",
         fill=C_PS_F, edge=C_PS_E, sub="high recall (TPR)")
    rbox(ax, 7.35, 9.4, 4.0, 1.40, "Viterbi pass 2",
         fill=C_PS_F, edge=C_PS_E, sub="high precision (PPV)")
    arrow(ax, 5, 10.42, 2.65, 10.175, rad=0.0)
    arrow(ax, 5, 10.42, 7.35, 10.175, rad=0.0)

    # two candidate lists
    rbox(ax, 2.65, 7.35, 3.7, 1.2, "CP list 1",
         fill=C_LS_F, edge=C_LS_E, sub="recall candidates")
    rbox(ax, 7.35, 7.35, 3.7, 1.2, "CP list 2",
         fill=C_LS_F, edge=C_LS_E, sub="precision candidates")
    arrow(ax, 2.65, 8.625, 2.65, 7.95)
    arrow(ax, 7.35, 8.625, 7.35, 7.95)

    # union
    rbox(ax, 5, 5.35, 5.2, 1.20, "Preliminary CP list",
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
    tag(ax, 9.70, 9.4, "b", C_PS_E)
    tag(ax, 9.60, 7.35, "c", "0.45")
    tag(ax, 8.20, 3.3, "d", C_GT_E)




# ── panel b : Viterbi trellis ─────────────────────────────────────────────────
def panel_b(ax):
    from matplotlib.patches import Ellipse
    blank(ax); ax.set_xlim(0.05, 6.95); ax.set_ylim(0.25, 4.55)   # headroom: legend above CP labels
    rows = {"A": 1.0, "B": 2.0, "C": 3.0}
    T = [1, 2, 3, 4, 5, 6]
    best = {1: "A", 2: "A", 3: "B", 4: "B", 5: "C", 6: "C"}
    ew, eh = 0.52, 0.46

    # faint trellis transitions
    for t in T[:-1]:
        for s0 in rows:
            for s1 in rows:
                ax.plot([t, t + 1], [rows[s0], rows[s1]],
                        color="0.86", lw=0.5 * LW, ls=(0, (3, 3)), zorder=1)

    # nodes
    for t in T:
        for s, yy in rows.items():
            on = best[t] == s
            ax.add_patch(Ellipse((t, yy), ew, eh, facecolor=GOLD if on else "#dedede",
                         edgecolor=C_PS_E if on else "#b8b8b8",
                         lw=(1.0 if on else 0.7) * LW, zorder=3))
            ax.text(t, yy, s, ha="center", va="center", fontsize=F["body"],
                    fontweight="bold", color="white" if on else "0.4", zorder=4)

    # best-path arrows
    for t in T[:-1]:
        arrow(ax, t + ew * 0.36, rows[best[t]], t + 1 - ew * 0.36, rows[best[t + 1]],
              color=C_PS_E, lw=2.0)

    # change-points
    for xc, lab in [(2.5, "CP$_1$"), (4.5, "CP$_2$")]:
        ax.plot([xc, xc], [0.45, 3.6], color=RED, ls="--", lw=1.1 * LW, zorder=2)
        ax.text(xc, 3.72, lab, ha="center", va="bottom", fontsize=F["small"], color=RED)

    # row labels (each state = one latent regime with its own mean/variance)
    for s, yy in rows.items():
        ax.text(0.62, yy, f"{s}", ha="right", va="center", fontsize=F["body"],
                fontweight="bold", color="0.25")
    ax.text(0.24, 2.0, r"$S$ states, 3 shown (A, B, C)", rotation=90,
            ha="center", va="center", fontsize=F["tiny"], color="0.5", style="italic")

    # time axis
    for t in T:
        ax.text(t, 0.33, f"t={t}", ha="center", va="top", fontsize=F["tiny"], color="0.4")

    # legend
    ax.scatter([], [], s=18, c=GOLD, marker="o", label="best path (Viterbi)")
    ax.scatter([], [], s=18, c="#dedede", marker="o", label="other paths")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              fontsize=F["tiny"], handletextpad=0.3, ncol=2, columnspacing=1.0)


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
        ax.plot([xcp(c), xcp(c)], [0.09, 0.63], color="0.8", ls="--", lw=0.7 * LW, zorder=1)

    fp = [42, 205]                                  # recall-pass false alarms
    miss = [cps[1]]                                 # precision-pass miss (weak CP)
    prec_hits = [c for c in cps if c not in miss]

    def tick(x, yc, color, h=0.032, lw=1.8):
        ax.plot([x, x], [yc - h, yc + h], color=color, lw=lw * LW,
                solid_capstyle="round", zorder=4)

    rails = [(0.47, "high recall", "TPR"),
             (0.31, "high precision", "PPV"),
             (0.15, "union", "→ gate")]
    for yc, t1, t2 in rails:
        ax.plot([xL, xR], [yc, yc], color="0.85", lw=1.0 * LW, zorder=2)
        ax.text(0.205, yc + 0.03, t1, ha="right", va="center",
                fontsize=F["body"], fontweight="bold", color="0.2")
        ax.text(0.205, yc - 0.032, t2, ha="right", va="center",
                fontsize=F["small"], color="0.5", style="italic")

    # recall: all true CPs (correct) + false alarms
    for c in cps:
        tick(xcp(c), 0.47, TRUE_CP)
    for c in fp:
        tick(xcp(c), 0.47, RED)
    # precision: subset of true CPs (clean), misses marked as open circles
    for c in prec_hits:
        tick(xcp(c), 0.31, TRUE_CP)
    for c in miss:
        ax.scatter([xcp(c)], [0.31], s=12, facecolors="none",
                   edgecolors="0.55", linewidths=0.7, zorder=4)
    # union: every unique candidate handed to the gate
    for c in sorted(set(cps) | set(fp)):
        tick(xcp(c), 0.15, "0.4")

    # key
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=TRUE_CP, lw=1.3, label="true CP"),
        Line2D([0], [0], color=RED, lw=1.3, label="false alarm"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="0.55", label="missed CP", markersize=3.5)],
        loc="lower left", bbox_to_anchor=(0.205, -0.02), frameon=False,
        fontsize=F["small"], ncol=3, handletextpad=0.3, columnspacing=1.1)


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
    ax.plot([xc, xc], [ay0, ay1], color=RED, ls="--", lw=1.1 * LW, zorder=4)
    ax.text(xc, ay1 + 0.015, "candidate CP", ha="center", va="bottom",
            fontsize=F["small"], color=RED, style="italic")
    for xa, xb, lab, col in [(ax0, xc, "verify_w", BLUEW), (xc, ax1, "verify_w", ORANGEW)]:
        ax.annotate("", (xa + 0.008, ay0 - 0.03), (xb - 0.008, ay0 - 0.03),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=0.9 * LW))
        ax.text((xa + xb) / 2, ay0 - 0.08, lab, ha="center", va="top",
                fontsize=F["small"], color=col, style="italic")
    ax.text((ax0 + ax1) / 2, ay1 + 0.06, "two windows",
            ha="center", va="bottom", fontsize=F["body"], fontweight="bold", color="0.2")

    # (ii) Mann–Whitney : rank-sum on the two samples --------------------------
    bx0, bx1, by0, by1 = 0.38, 0.655, 0.50, 0.90
    xx = np.linspace(-3.6, 5.2, 220)
    pL = np.exp(-0.5 * (xx - 0.0) ** 2); pR = np.exp(-0.5 * (xx - 1.6) ** 2)
    rx = (xx - xx[0]) / np.ptp(xx)
    curve(ax, rx, pL / pL.max(), bx0, bx1, by0, by1, color=BLUEW, lw=1.4 * LW)
    curve(ax, rx, pR / pR.max(), bx0, bx1, by0, by1, color=ORANGEW, lw=1.4 * LW)
    ax.fill_between(bx0 + (bx1 - bx0) * rx, by0, by0 + (by1 - by0) * pL / pL.max(),
                    color=BLUEW, alpha=0.12, lw=0)
    ax.fill_between(bx0 + (bx1 - bx0) * rx, by0, by0 + (by1 - by0) * pR / pR.max(),
                    color=ORANGEW, alpha=0.12, lw=0)
    ax.text((bx0 + bx1) / 2, by1 + 0.06, "Mann–Whitney U",
            ha="center", va="bottom", fontsize=F["body"], fontweight="bold", color="0.2")
    ax.text((bx0 + bx1) / 2, by0 - 0.055, r"$p_{\mathrm{MW}}$",
            ha="center", va="top", fontsize=F["math"], color="0.15")
    #ax.text((bx0 + bx1) / 2, by0 - 0.135, "rank-sum · location",
    #        ha="center", va="top", fontsize=F["tiny"], color="0.5", style="italic")

    # (iii) KS : max CDF gap ---------------------------------------------------
    cx0, cx1, cy0, cy1 = 0.70, 0.975, 0.50, 0.90
    cL = np.cumsum(pL); cL /= cL[-1]; cR = np.cumsum(pR); cR /= cR[-1]
    curve(ax, rx, cL, cx0, cx1, cy0, cy1, color=BLUEW, lw=1.4 * LW)
    curve(ax, rx, cR, cx0, cx1, cy0, cy1, color=ORANGEW, lw=1.4 * LW)
    di = int(np.argmax(np.abs(cL - cR)))
    xd = cx0 + (cx1 - cx0) * rx[di]
    ax.annotate("", (xd, cy0 + (cy1 - cy0) * cR[di]), (xd, cy0 + (cy1 - cy0) * cL[di]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1 * LW))
    ax.text(xd + 0.012, cy0 + (cy1 - cy0) * (cL[di] + cR[di]) / 2, "D",
            ha="left", va="center", fontsize=F["small"], color=RED, fontweight="bold")
    ax.text((cx0 + cx1) / 2, cy1 + 0.06, "Kolmogorov–Smirnov",
            ha="center", va="bottom", fontsize=F["body"], fontweight="bold", color="0.2")
    ax.text((cx0 + cx1) / 2, cy0 - 0.055, r"$p_{\mathrm{KS}}$",
            ha="center", va="top", fontsize=F["math"], color="0.15")
    #ax.text((cx0 + cx1) / 2, cy0 - 0.135, "max CDF gap · shape",
    #        ha="center", va="top", fontsize=F["tiny"], color="0.5", style="italic")

    # decision strip : DECIDE (multiple-testing correction) + MERGE (dedup) -----
    bx, by, bw, bh = 0.05, 0.035, 0.90, 0.20          # height was 0.265
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
             boxstyle="round,pad=0,rounding_size=0.02",
             facecolor="#fbeeea", edgecolor=C_GT_E, lw=1.2 * LW, zorder=2))
    rows = [by + bh * f for f in (0.78, 0.20)]
    ax.text(0.50, rows[0],
        r"$p_{\max}=\max(p_{\mathrm{MW}},\,p_{\mathrm{KS}})$"
        r"$\quad$vs$\quad \alpha = \mathrm{fwer}\,/\,m$  (Bonferroni)",
        ha="center", va="center", fontsize=F["body"], color="0.15", zorder=3)
    ax.text(0.50, rows[1],
        r"keep candidate CP  $\Leftrightarrow\ p_{\max} < \alpha$",
        ha="center", va="center", fontsize=F["body"], color=C_GT_E,
        fontweight="bold", zorder=3)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "fig1_natcomms.pdf"),
                    help="vector output; a .png preview is written alongside")
    ap.add_argument("--base-font", type=float, default=7.0,
                    help="base font size in pt at final print size (journal max 7)")
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTDIR, exist_ok=True)

    y, cps, outliers = demo_signal(seed=args.seed)

    W_mm, H_mm = LAYOUT["fig_mm"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    P = LAYOUT["panels"]
    ax_a = ax_mm(fig, **P["a"]); panel_a(ax_a, y, cps, outliers); plabel(fig, "a")
    ax_b = ax_mm(fig, **P["b"]); panel_b(ax_b);                   plabel(fig, "b")
    ax_c = ax_mm(fig, **P["c"]); panel_c(ax_c, y, cps);           
    plabel(fig, "c", dx_mm=5.0, dy_mm=7.5)
    ax_d = ax_mm(fig, **P["d"]); panel_d(ax_d);                   plabel(fig, "d")

    # reproducibility CSV for the illustrative demo trace
    with open(os.path.join(OUTDIR, "fig1_demo_signal.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["idx", "value", "is_cp", "is_outlier"])
        cpset, oset = set(cps), set(int(o) for o in outliers)
        for i, v in enumerate(y):
            w.writerow([i, f"{v:.5f}", int(i in cpset), int(i in oset)])

    # vector for submission + raster preview; no bbox_inches="tight", so the
    # file is exactly LAYOUT["fig_mm"] and nothing gets rescaled
    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    print(f"wrote {args.out} and {png}  ({W_mm:g} x {H_mm:g} mm, {args.base_font:g} pt base)")
    print(f"demo CPs at {cps};  outliers at {list(int(o) for o in outliers)}")


if __name__ == "__main__":
    main()
