"""
Fig 3 — ISFET (H+) drift correction.

Panels (bold lower-case letters; Arial; built AT FINAL PRINT SIZE):
  a  experiment: river intake -> reference pH probe -> ISFET probe -> readout,
     with an inset of the phenomenon V_H = E0(t) - S.pH.
  b  correction pipeline (the detector is the only swappable box).
  c  H+ ISFET (mV) vs day: raw V_H drifting away, the reference (-S.pH), and
     VOCD's corrected signal hugging it; VOCD's change-points and the burn-in.
  d  drift RMSE (mV) vs runtime (ms, log) for the full panel, 95% block-
     bootstrap CIs, with the context baselines as horizontal lines.

Inputs (all produced by examples/isfet_data_v3.py --sweep; NOTHING is
recomputed here):
  --best     isfet_v3_best.csv    one row per method: rmse_mv, rmse_lo/hi,
                                  runtime_ms (panel d)
  --series   isfet_v3_series.csv  per-day V_H, reference, true E0 and every
                                  method's E0_hat (panel c)
  --cps      isfet_v3_cps.csv     every detector's final boundary set (panel c)
  --best-extra  optional second best CSV whose rows OVERRIDE --best for the
                methods it contains (e.g. a BOCD / Dm-BOCD / CHAMP re-run on a
                corrected hazard grid); its series/cps are not needed.

LAYOUT (same scheme as vocd_fig2.py): sizes and positions live in
the LAYOUT dict below, in millimetres from the top-left corner, at final
print size (Nature Portfolio: 180 mm wide, 170 mm tall maximum, text 7 pt
maximum and 5 pt minimum, panel letters 8 pt bold).  Every text size derives
from --base-font.  Panels a and b are schematics drawn in data coordinates:
keep their aspect ratio (~1.3:1) when resizing or the boxes will stretch.

    python examples/vocd_fig3_isfet.py
    python examples/vocd_fig3_isfet.py --best-extra results/isfet/isfet_v3_best_hazard.csv
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "fig3_isfet")
RES = os.path.join(_REPO, "results", "isfet")
S_NERNST = 59.16          # mV / pH
MIN_FIT = 4               # days used for the single day-0 anchor (isfet_data)

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 166.0),
    "panels": {
        #      x      y      w      h
        "a": dict(x=4.0,   y=5.0,  w=84.0, h=70.0),   # schematic, keep ~1.3:1
        "b": dict(x=96.0,  y=5.0,  w=80.0, h=70.0),   # schematic, keep ~1.3:1
        "c": dict(x=14.0,  y=85.0, w=74.0, h=70.0),
        "d": dict(x=104.0, y=85.0, w=72.0, h=70.0),
    },
}
F = {}        # font tiers, filled by set_nature_style(base)
LW = 0.62     # line-weight scale for the schematics vs the old 381 mm canvas

# ── palette: methods = fig2 panel f (Okabe-Ito); boxes = fig1 ───────────────
C_VOCD, C_BOCD, C_DMBOCD = "#009E73", "#E69F00", "#CC79A7"
C_FOCUS, C_NPFOCUS, C_NUNC, C_CHAD, C_CHAMP = "#56B4E9", "#0072B2", "#F0E442", "#D55E00", "#000000"
METHOD_STYLE = {   # csv name : (label, colour, marker); twins share a colour
    "vocd":           ("VOCD (Py)",      C_VOCD,    "o"),
    "vocd (cpp)":     ("VOCD (C++)",     C_VOCD,    "s"),
    "bocd":           ("BOCD",           C_BOCD,    "o"),
    "dm-bocd":        ("Dm-BOCD",        C_DMBOCD,  "o"),
    "focus":          ("FOCuS (Py)",     C_FOCUS,   "o"),
    "focus (cpp)":    ("FOCuS (C++)",    C_FOCUS,   "s"),
    "np-focus":       ("NP-FOCuS (Py)",  C_NPFOCUS, "o"),
    "np-focus (cpp)": ("NP-FOCuS (C++)", C_NPFOCUS, "s"),
    "nunc":           ("NUNC",           C_NUNC,    "o"),
    "chad":           ("CHAD",           C_CHAD,    "o"),
    "champ":          ("CHAMP",          C_CHAMP,   "o"),
}
BASELINE_STYLE = {  # context rows drawn as horizontal lines in panel d
    "two-point": ("two-point", "0.45", "--"),
    "M0":        ("M0 online", "0.45", ":"),
    "exp":       ("exp + linear",   "0.45", "-."),
}
C_IO_F, C_IO_E = "#cfe0f2", "#2c5f8a"
C_PS_F, C_PS_E = "#f6c873", "#cf8a1e"
C_LS_F, C_LS_E = "#eaeaea", "#8a8a8a"
C_GT_F, C_GT_E = "#f2b3a3", "#c0392b"
RED, BLUE, GOLD = "#D55E00", "#2c5f8a", "#e29a1a"      # RED = vermillion (colour-safe)
C_RAW, C_REF = C_PS_E, "0.15"                            # panel c: raw V_H, reference


# ── io ────────────────────────────────────────────────────────────────────────
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_best(path):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["method"]] = {k: fnum(v) for k, v in r.items() if k != "method"}
    return out


def load_series(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    col = lambda k: np.array([fnum(r[k]) for r in rows])
    return {k: col(k) for k in rows[0].keys()}


def load_cps(path, method):
    days = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["method"] == method:
                days.append(float(r["cp_day"]))
    return sorted(days)


def corrected_mv(series, method, burn_in):
    """The harness's corrected signal on the reference scale, for ALL days:
    pH_hat = (E0_hat - V_H)/S, one bias from the first MIN_FIT scored days,
    then -S.(pH_hat - bias)  (isfet_data.score, verbatim anchoring)."""
    e0h, vh, ref = series[f"e0_hat:{method}"], series["v_h_mv"], series["ref_mv"]
    dd = series["day"]
    ph_hat, ph_ref = (e0h - vh) / S_NERNST, -ref / S_NERNST
    m = dd >= burn_in
    bias = float(np.mean(ph_hat[m][:MIN_FIT] - ph_ref[m][:MIN_FIT]))
    return -S_NERNST * (ph_hat - bias)


# ── style / drawing helpers (shared with fig1 / fig2) ────────────────────────
def set_nature_style(base=7.0):
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    arial_family = ["Arial", "Arimo", "Liberation Sans", "Helvetica"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not (set(arial_family) & have):
        print("[fig3] warning: no Arial-family font found; using DejaVu Sans.")
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


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def rbox(ax, cx, cy, w, h, title, *, fill, edge, sub=None,
         fs=None, sub_fs=None, tc="0.12", lw=1.3, ls="-"):
    from matplotlib.patches import FancyBboxPatch
    fs = F["title"] if fs is None else fs
    sub_fs = F["small"] if sub_fs is None else sub_fs
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.14", linestyle=ls,
                 linewidth=lw * LW, edgecolor=edge, facecolor=fill, zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(cx, cy - h * 0.23, sub, ha="center", va="center",
                fontsize=sub_fs, color=tc, style="italic", zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="0.35", lw=1.5, rad=0.0, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=13 * LW, lw=lw * LW, color=color, shrinkA=0,
                 shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=1))


def series_in_box(ax, x0, x1, y0, y1, y, *, lw=0.9, color="0.22", ylims=None):
    n = len(y)
    xs = x0 + (x1 - x0) * np.arange(n) / (n - 1)
    lo, hi = (np.nanmin(y), np.nanmax(y)) if ylims is None else ylims
    ys = y0 + (y1 - y0) * (y - lo) / (hi - lo + 1e-9)
    ax.plot(xs, ys, color=color, lw=lw * LW, zorder=3)
    return xs, ys


# ── panel a : experiment ─────────────────────────────────────────────────────
# Panel-a geometry, in the panel's data coordinates (x 0-10, y 0-6.1).  The
# boxes size themselves from their measured text, so only spacing lives here.
A_BOXES = [   # (title, subtitle, fill, edge) -- left to right
    ("River intake",    "Llobregat (40 L/s)",       C_IO_F, C_IO_E),
    ("Reference probe", "Orbisint pH",              C_LS_F, C_LS_E),
    ("ISFET probe",     r"H$^+$, Na$^+$, K$^+$",    C_PS_F, C_PS_E),
    ("Readout → PC",    "1 Hz, 90 days",            C_LS_F, C_LS_E),
]
A_BOX_Y, A_BOX_H = 5.25, 1.35      # box row centre and height (data units)
A_BOX_PAD_MM = 1.2                 # padding between text and box border, each side
A_MIN_GAP_MM = 3.0                 # smallest gap between boxes (holds the arrow)
A_INSET = dict(x0=0.9, x1=7.3, y0=0.55, y1=3.75)   # illustrative plot box


def panel_a(ax):
    blank(ax); ax.set_xlim(0, 13); ax.set_ylim(0.3, 6.1)
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    u_per_mm = 10.0 / LAYOUT["panels"]["a"]["w"]          # data units per mm
    pad, min_gap = A_BOX_PAD_MM * u_per_mm, A_MIN_GAP_MM * u_per_mm

    def width(t):                  # text width in data units
        bb = t.get_window_extent(r)
        return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]

    # 1) measure each box's text, width = widest line + padding
    texts, widths = [], []
    for title, sub, fill, edge in A_BOXES:
        t1 = ax.text(0, A_BOX_Y + A_BOX_H * 0.17, title, ha="center", va="center",
                     fontsize=F["body"], fontweight="bold", color="0.12", zorder=3)
        t2 = ax.text(0, A_BOX_Y - A_BOX_H * 0.23, sub, ha="center", va="center",
                     fontsize=F["tiny"], color="0.12", style="italic", zorder=3)
        texts.append((t1, t2))
        widths.append(max(width(t1), width(t2)) + 2 * pad)

    # 2) spread them across the panel with equal gaps
    span = 10.0 - 0.1
    gap = (span - sum(widths)) / (len(widths) - 1)
    if gap < min_gap:
        need = (sum(widths) + (len(widths) - 1) * min_gap) / u_per_mm
        print(f"[fig3a] boxes need {need:.1f} mm but panel a is "
              f"{LAYOUT['panels']['a']['w']:g} mm wide: widen it in LAYOUT or shorten a label")
        gap = min_gap
    x, edges = 0.05, []
    for (title, sub, fill, edge), w, (t1, t2) in zip(A_BOXES, widths, texts):
        cx = x + w / 2
        rbox(ax, cx, A_BOX_Y, w, A_BOX_H, "", fill=fill, edge=edge)
        t1.set_x(cx); t2.set_x(cx)
        edges.append((x, x + w))
        x += w + gap
    for (_, right), (left, _) in zip(edges[:-1], edges[1:]):
        arrow(ax, right, A_BOX_Y, left, A_BOX_Y)          # same arrows as panel b

    # 3) the phenomenon:  V_H = E0(t) - S*pH(t) + eps
    dx = 2.08                                   # shift the illustrative plot right
    ix0, ix1, iy0, iy1 = A_INSET["x0"] + dx, A_INSET["x1"] + dx, A_INSET["y0"], A_INSET["y1"]
    t = np.linspace(0, 90, 300)
    ref = 0.12 * np.sin(t / 5.0)
    e0 = -1.2 * t / 90 * 1.8 + 0.55 * np.exp(-t / 12.0)
    vh = e0 + ref
    lo, hi = min(vh.min(), ref.min()) - 0.1, max(vh.max(), ref.max()) + 0.35
    xs, ys_ref = series_in_box(ax, ix0, ix1, iy0, iy1, ref, color="0.45", lw=1.0, ylims=(lo, hi))
    _, ys_vh = series_in_box(ax, ix0, ix1, iy0, iy1, vh, color=C_PS_E, lw=1.3, ylims=(lo, hi))
    ax.plot([ix0, ix0, ix1], [iy1, iy0, iy0], color="0.3", lw=0.9 * LW, zorder=2)
    k = 230
    arrow(ax, xs[k], ys_ref[k], xs[k], ys_vh[k], color=RED, lw=1.0, style="<->")
    ax.text(xs[k] + 0.12, (ys_ref[k] + ys_vh[k]) / 2, r"$E_0(t)$", color=RED,
            fontsize=F["body"], va="center", ha="left")
    ax.text(ix1 + 0.08, ys_ref[-1], r"reference ($E_0$(0)−$S_N$·pH)", fontsize=F["tiny"], color="0.45", va="center")
    ax.text(ix1 + 0.08, ys_vh[-1], r"$V_H$ (ISFET)", fontsize=F["tiny"], color=C_PS_E, va="center")
    ax.text((ix0 + ix1) / 2, iy1 + 0.2, r"$V_H(t) = E_0(t) - S_N\cdot\mathrm{pH}(t) + \varepsilon$",
            ha="center", va="bottom", fontsize=F["math"], color="0.15")
    ax.text((ix0 + ix1) / 2, iy0 - 0.12, "Time (days)", ha="center", va="top",
            fontsize=F["small"], color="0.3")
    ax.text(ix0 - 0.2, (iy0 + iy1) / 2, r"H$^+$ ISFET (mV)", rotation=90, ha="center",
            va="center", fontsize=F["small"], color="0.3")


# ── panel b : correction pipeline ────────────────────────────────────────────
def panel_b(ax):
    blank(ax); ax.set_xlim(0, 10); ax.set_ylim(0.9, 13.9)
    cx, w, xn = 4.0, 6.6, 7.5
    note = dict(ha="left", va="center", fontsize=F["tiny"], color="0.35", style="italic")

    rbox(ax, cx, 12.85, w, 1.5, r"$V_H(d)$ — daily means, mV",
         fill=C_IO_F, edge=C_IO_E, sub="ISFET only (reference never seen)")
    arrow(ax, cx, 12.1, cx, 11.55)
    rbox(ax, cx, 10.65, w, 1.8, "change-point detector",
         fill=C_PS_F, edge=C_PS_E, sub="VOCD  |  BOCD  |  Dm-BOCD  |  …")
    ax.text(xn, 11.15, "swappable (only\nthis box changes)", **note)
    ax.text(xn, 10.05, r"causal: day $d$ sees" "\n" r"only $V_H[\,{:}\,d\,]$", **note)
    arrow(ax, cx, 9.75, cx, 9.2)
    rbox(ax, cx, 8.45, w, 1.5, r"segment boundaries $t_1 < \dots < t_K$",
         fill=C_LS_F, edge=C_LS_E, sub="held on day d (prefix replay, no look-ahead)")
    arrow(ax, cx, 7.7, cx, 7.15)
    rbox(ax, cx, 6.2, w, 1.9, "per-segment drift model", fill=C_GT_F, edge=C_GT_E,
         sub=r"linear:  $\hat E_0(t) = a_k + b_k\,t$  on segment $k$")
    ax.text(xn, 6.2, "re-fit as days arrive;\nM0 = the same line\nwith no boundaries", **note)
    arrow(ax, cx, 5.25, cx, 4.7)
    rbox(ax, cx, 3.95, w, 1.5, r"$V_{\rm corr}(d) = V_H(d) - \hat E_0(d)$",
         fill=C_IO_F, edge=C_IO_E, sub="corrected signal (output)")
    arrow(ax, cx, 3.2, cx, 2.65)
    rbox(ax, cx, 1.9, w, 1.5, "score vs reference", fill="white", edge="0.5", ls="--",
         sub=r"day-0 anchor, then RMSE of $\hat E_0 - E_0$")
    ax.text(xn, 1.9, "reference used here only\n" r"$E_0 = V_H + S_N\cdot\mathrm{pH}_{\rm ref}$", **note)


# ── panel c : signal vs time (raw, reference, VOCD corrected) ────────────────
def panel_c(ax, series, cps, burn_in, arrow_day=80.0):
    """Same view as panel a's inset.  The reference (-S.pH_ref) and the VOCD-
    corrected signal are shown on the ISFET's day-0 level by adding E0(0), the
    true offset on day 0 -- i.e. the trace a drift-free ISFET would produce --
    so raw V_H starts on top of them and the growing gap is the accumulated
    drift E0(t) - E0(0).  Set ``off = 0.0`` below to plot on the pH-electrode
    scale instead."""
    dd = series["day"]
    vh, ref = series["v_h_mv"], series["ref_mv"]
    corr = corrected_mv(series, "vocd", burn_in)
    off = float(series["e0_true_mv"][0])             # E0(0): day-0 ISFET level
    ref_s, corr_s = ref + off, corr + off

    ax.axvspan(dd[0] - 0.5, burn_in, color="0.92", lw=0, zorder=0)
    ax.text(burn_in / 2, 0.985, "burn-in", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=F["tiny"], color="0.45")
    for i, c in enumerate(cps, 1):
        ax.axvline(c, color=RED, ls="--", lw=0.7, zorder=1)
        ax.text(c, 1.0, rf"CP$_{i}$", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=F["tiny"], color=RED)
    ax.plot(dd, vh, color=C_RAW, lw=1.0, label=r"raw $V_H$ (ISFET)", zorder=3)
    ax.plot(dd, ref_s, color=C_REF, lw=1.0,
            label=r"reference, $E_0(0) - S_N\cdot\mathrm{pH}_{\rm ref}$", zorder=3)
    ax.plot(dd, corr_s, color=C_VOCD, lw=1.0, label="VOCD corrected", zorder=4)

    # the drift gap, drawn like the inset in panel a
    k = int(np.argmin(np.abs(dd - arrow_day)))
    #ax.annotate("", (dd[k], vh[k]), (dd[k], ref_s[k]),
    #            arrowprops=dict(arrowstyle="<->", color=RED, lw=0.7, shrinkA=0, shrinkB=0),
    #            zorder=5)
    #ax.text(dd[k] - 1.5, (vh[k] + ref_s[k]) / 2, r"$E_0(t) - E_0(0)$", color=RED,
    #        fontsize=F["small"], ha="right", va="center", zorder=5)

    ax.set_xlabel("Time (days)"); ax.set_ylabel(r"H$^+$ ISFET (mV)")
    ax.set_xlim(dd[0] - 0.5, dd[-1] + 0.5)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)


# ── panel d : RMSE vs runtime ────────────────────────────────────────────────
def panel_d(ax, best, ylim, floor_mv):
    for m, (lab, col, mk) in METHOD_STYLE.items():
        if m not in best:
            continue
        x, y = best[m]["runtime_ms"], best[m]["rmse_mv"]
        lo, hi = best[m].get("rmse_lo", np.nan), best[m].get("rmse_hi", np.nan)
        yerr = [[max(0.0, y - lo)], [max(0.0, hi - y)]] if np.isfinite(lo) and np.isfinite(hi) else None
        ax.errorbar(x, y, yerr=yerr, fmt=mk, color=col, ms=4.5, capsize=2, elinewidth=0.7,
                    markeredgecolor="0.2", markeredgewidth=0.4, zorder=3, label=lab)
    ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.3,
              columnspacing=0.8, borderaxespad=0.2)

    groups = {}   # merge baselines whose RMSE coincide to 0.1 mV
    for m, (lab, col, ls) in BASELINE_STYLE.items():
        if m in best and np.isfinite(best[m]["rmse_mv"]):
            groups.setdefault(round(best[m]["rmse_mv"], 1), []).append((lab, col, ls, best[m]["rmse_mv"]))
    for _, items in groups.items():
        lab = " = ".join(i[0] for i in items)
        col, ls, val = items[0][1], items[0][2], items[0][3]
        ax.axhline(val, color=col, ls=ls, lw=0.7, zorder=1)
        ax.text(0.99, val, f"{lab} ({val:.1f})", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=F["tiny"], color=col)
    if "none" in best:
        ax.text(0.02, 0.97, f"no correction: {best['none']['rmse_mv']:.1f} mV",
                transform=ax.transAxes, ha="left", va="top", fontsize=F["tiny"], color="0.4")
    if floor_mv > 0:
        ax.text(0.02, 0.91, f"instrument floor ≈ {floor_mv:.1f} mV",
                transform=ax.transAxes, ha="left", va="top", fontsize=F["tiny"], color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (ms)"); ax.set_ylabel("RMSE (mV) relative to the reference")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.4, which="both")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best", default=os.path.join(RES, "isfet_v3_best.csv"))
    ap.add_argument("--best-extra", default=None,
                    help="second best CSV; its rows override --best for the methods it has")
    ap.add_argument("--series", default=os.path.join(RES, "isfet_v3_series.csv"))
    ap.add_argument("--cps", default=os.path.join(RES, "isfet_v3_cps.csv"))
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--ref-sigma-ph", type=float, default=0.02)
    ap.add_argument("--digit-sigma-mv", type=float, default=0.0)
    ap.add_argument("--ylim", type=float, nargs=2, default=(5.0, 17.0),
                    help="RMSE axis; the top ~5 mV hold the legend (nothing plots above 12)")
    ap.add_argument("--base-font", type=float, default=7.0,
                    help="base font size in pt at final print size (journal max 7)")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "fig3_isfet.pdf"),
                    help="vector output; a .png preview is written alongside")
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTDIR, exist_ok=True)

    best = load_best(args.best)
    if args.best_extra:
        extra = load_best(args.best_extra)
        best.update({m: r for m, r in extra.items() if m in METHOD_STYLE})
        print(f"overrode {sorted(m for m in extra if m in METHOD_STYLE)} from {args.best_extra}")
    series = load_series(args.series)
    cps = load_cps(args.cps, "vocd")
    floor_mv = float(np.hypot(S_NERNST * args.ref_sigma_ph, args.digit_sigma_mv))

    with open(os.path.join(OUTDIR, "fig3_panelD.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "rmse_mv", "rmse_lo", "rmse_hi", "runtime_ms"])
        for m in METHOD_STYLE:
            if m in best:
                w.writerow([m] + [best[m][k] for k in ("rmse_mv", "rmse_lo", "rmse_hi", "runtime_ms")])

    W_mm, H_mm = LAYOUT["fig_mm"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    P = LAYOUT["panels"]
    ax_a = ax_mm(fig, **P["a"]); panel_a(ax_a); plabel(fig, "a")
    ax_b = ax_mm(fig, **P["b"]); panel_b(ax_b); plabel(fig, "b")
    ax_c = ax_mm(fig, **P["c"]); panel_c(ax_c, series, cps, args.burn_in); plabel(fig, "c", dx_mm=-10.0)
    ax_d = ax_mm(fig, **P["d"]); panel_d(ax_d, best, tuple(args.ylim), floor_mv); plabel(fig, "d", dx_mm=-10.0)

    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    print(f"wrote {args.out} and {png}  ({W_mm:g} x {H_mm:g} mm, {args.base_font:g} pt base)")
    print(f"VOCD change-points at days {cps}; instrument floor {floor_mv:.2f} mV")
    for m in METHOD_STYLE:
        if m in best:
            r = best[m]
            print(f"  {m:15s} RMSE {r['rmse_mv']:.2f} mV  [{r['rmse_lo']:.2f}, {r['rmse_hi']:.2f}]"
                  f"  {r['runtime_ms']:.0f} ms")


if __name__ == "__main__":
    main()