"""
Fig 4 — MOX gas-sensor (e-nose) drift compensation

Panels (bold lower-case letters; Arial; built AT FINAL PRINT SIZE):
  a  experiment: compressed air -> mass-flow controller -> humidifier -> sample
     bottle -> e-nose (62 MOX) -> PC, with an inset of one 20-min cycle whose
     baseline wanders day to day:  R_t(tau) = R_ref(tau) + dR_t + eps.
  b  correction pipeline (the detector is the only swappable box).
  c  how VOCD's compensation error evolves over the record: one curve per
     test day of the relative error (R_ref - R_VOCD) / R_ref in %, vs time in
     cycle, coloured from dark blue (day 2) to orange (day 39), lightly
     smoothed for display, over the three stage bands.
  d  MAE (kOhm) vs runtime (ms, log) for the full panel with 95% block-
     bootstrap CIs over days, the oracle as a horizontal line, and Dm-BOCD
     real if it was run, otherwise an ESTIMATE (hollow marker, no CI): runtime
     = runtime(VOCD on gas) x (Dm-BOCD / VOCD) from the ISFET table, MAE at
     BOCD's gas value (--dm-bocd-mae).

Inputs (all from examples/gas_data_v3.py; NOTHING is recomputed here):
  --best        gas_v3_best.csv      one row per method (panel d)
  --best-extra  optional second best CSV; its rows OVERRIDE --best for the
                methods it contains (e.g. the CHAMP / Dm-BOCD single-point run)
  --resid       gas_v3_resid_all.csv reference - corrected on every test day
                (columns resid:vocd:dayN; written by --resid-all-out)
  --isfet-best  isfet_v3_best.csv    only for the Dm-BOCD runtime estimate

LAYOUT (same scheme as Figs. 2-3): sizes and positions in the LAYOUT dict, in
millimetres from the top-left corner, at final print size (180 mm wide,
<= 170 mm tall, 7 pt text max, panel letters 8 pt bold).  Panels a and b are
schematics drawn in data coordinates: keep their aspect (~1.3:1) when resizing.

    python examples/gas_data_v3.py --sweep --analyte EtOH --methods vocd \\
        --out results/gas/gas_v3_best_c.csv --sweep-out results/gas/gas_v3_sweep_c.csv \\
        --series-out results/gas/gas_v3_series_c.csv --onsets-out results/gas/gas_v3_onsets_c.csv \\
        --resid-all-out results/gas/gas_v3_resid_all.csv
    python examples/vocd_fig4_gas.py --best-extra results/gas/gas_v3_single_champ.csv
"""
from __future__ import annotations
import argparse, csv, os, re, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "fig4_gas")
RES = os.path.join(_REPO, "results", "gas")

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 166.0),
    "cbar_w": 2.5, "cbar_gap": 1.5,          # panel c day colour bar
    "panels": {
        #      x      y      w      h
        "a": dict(x=1.5,   y=5.0,  w=99.0, h=69.0),   # schematic (5 boxes need the width)
        "b": dict(x=101.0, y=5.0,  w=75.0, h=69.0),   # schematic, keep ~1.2:1
        "c": dict(x=14.0,  y=80.0, w=70.0, h=77.0),
        "d": dict(x=106.0, y=80.0, w=70.0, h=77.0),
    },
}
F = {}
LW = 0.62

# ── palette: methods = fig2 panel f (Okabe-Ito); boxes = figs 1/3 ───────────
C_VOCD, C_BOCD, C_DMBOCD = "#009E73", "#E69F00", "#CC79A7"
C_FOCUS, C_NPFOCUS, C_NUNC, C_CHAD, C_CHAMP = "#56B4E9", "#0072B2", "#F0E442", "#D55E00", "#000000"
METHOD_STYLE = {
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
C_IO_F, C_IO_E = "#cfe0f2", "#2c5f8a"
C_PS_F, C_PS_E = "#f6c873", "#cf8a1e"
C_LS_F, C_LS_E = "#eaeaea", "#8a8a8a"
C_GT_F, C_GT_E = "#f2b3a3", "#c0392b"
RED = "#D55E00"
STAGE_COL = {1: "#7fb3d5", 2: "#f0a860", 3: "#9fd39f"}
STAGE_NAME = {1: "baseline", 2: "gas injection", 3: "recovery"}
DAY_COL = ["0.15", C_NPFOCUS, C_BOCD, C_DMBOCD, C_CHAD]     # panel c: day 1 = near-black


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
    return {k: np.array([fnum(r[k]) for r in rows]) for k in rows[0].keys()}


def load_offsets(path, method):
    """day number -> (onset, offset) for one method, from the onsets CSV."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["method"] != method:
                continue
            n = re.findall(r"\d+", r["day_folder"])
            if n:
                out[int(n[0])] = (int(r["onset_idx"]), float(r["offset_ohm"]))
    return out


# ── style / drawing helpers (shared with figs 1-3) ──────────────────────────
def set_nature_style(base=7.0):
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    arial_family = ["Arial", "Arimo", "Liberation Sans", "Helvetica"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not (set(arial_family) & have):
        print("[fig4] warning: no Arial-family font found; using DejaVu Sans.")
    F.update(panel=base + 1.0, title=base, body=base - 0.5, small=base - 1.0,
             tiny=max(5.0, base - 1.5), math=base)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": arial_family + ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1, "legend.fontsize": F["small"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 120, "savefig.dpi": 300,
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


def rbox(ax, cx, cy, w, h, title, *, fill, edge, sub=None, fs=None, sub_fs=None,
         tc="0.12", lw=1.3, ls="-"):
    from matplotlib.patches import FancyBboxPatch
    fs = F["title"] if fs is None else fs
    sub_fs = F["small"] if sub_fs is None else sub_fs
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.14", linestyle=ls,
                 linewidth=lw * LW, edgecolor=edge, facecolor=fill, zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tc, zorder=3)
        ax.text(cx, cy - h * 0.23, sub, ha="center", va="center", fontsize=sub_fs,
                color=tc, style="italic", zorder=3)
    elif title:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=tc, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="0.35", lw=1.5, rad=0.0, style="-|>", ms=13):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms * LW,
                 lw=lw * LW, color=color, shrinkA=0, shrinkB=0,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def line_in_box(ax, x0, x1, y0, y1, y, *, lw=0.9, color="0.22", ylims=None, **kw):
    n = len(y)
    xs = x0 + (x1 - x0) * np.arange(n) / (n - 1)
    lo, hi = (np.nanmin(y), np.nanmax(y)) if ylims is None else ylims
    ys = y0 + (y1 - y0) * (y - lo) / (hi - lo + 1e-9)
    ax.plot(xs, ys, color=color, lw=lw * LW, zorder=3, **kw)
    return xs, ys


def cycle_shape(n=1000):
    """Baseline flat -> sharp rise in gas -> decay in recovery."""
    stage = np.where(np.arange(n) < 0.30 * n, 1, np.where(np.arange(n) < 0.62 * n, 2, 3))
    t = np.arange(n, dtype=float)
    g0, r0 = 0.30 * n, 0.62 * n
    peak, floor, tr, tf = 1.0, 0.18, 0.06 * n, 0.14 * n
    gas = peak * (1 - np.exp(-(t - g0) / tr))
    gas_end = peak * (1 - np.exp(-(r0 - g0) / tr))
    rec = floor + (gas_end - floor) * np.exp(-(t - r0) / tf)
    return stage, np.where(stage == 1, 0.0, np.where(stage == 2, gas, rec))


# ── panel a : experiment ─────────────────────────────────────────────────────
A_BOXES = [   # (title, subtitle, fill, edge), left to right; boxes size themselves
    ("Compressed air",  "carrier gas",           C_LS_F, C_LS_E),
    ("Mass-flow ctrl.", "100 sccm",              C_LS_F, C_LS_E),
    ("Humidifier",      "gas washing",           C_IO_F, C_IO_E),
    ("Sample bottle",   "50 mL liquid",          C_IO_F, C_IO_E),
    ("E-nose → PC",     "62 MOX, 1 Hz",         C_PS_F, C_PS_E),
]
A_BOX_Y, A_BOX_H = 5.0, 1.3
A_BOX_PAD_MM, A_MIN_GAP_MM = 0.8, 2.2
A_INSET = dict(w=8.6, y0=0.6, y1=3.45)    # plot width (data units); centred under the boxes
#A_INSET = dict(x0=1.3, x1=8.4, y0=0.6, y1=3.45)


def panel_a(ax):
    from matplotlib.patches import Rectangle
    blank(ax); ax.set_xlim(0, 11.7); ax.set_ylim(0.3, 6.0)
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    u_per_mm = 11.7 / LAYOUT["panels"]["a"]["w"]
    pad, min_gap = A_BOX_PAD_MM * u_per_mm, A_MIN_GAP_MM * u_per_mm

    def width(t):
        bb = t.get_window_extent(r)
        return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]

    texts, widths = [], []
    for title, sub, fill, edge in A_BOXES:
        t1 = ax.text(0, A_BOX_Y + A_BOX_H * 0.17, title, ha="center", va="center",
                     fontsize=F["body"], fontweight="bold", color="0.12", zorder=3)
        t2 = ax.text(0, A_BOX_Y - A_BOX_H * 0.23, sub, ha="center", va="center",
                     fontsize=F["tiny"], color="0.12", style="italic", zorder=3)
        texts.append((t1, t2)); widths.append(max(width(t1), width(t2)) + 2 * pad)
    gap = (11.7 - 0.1 - sum(widths)) / (len(widths) - 1)
    if gap < min_gap:
        need = (sum(widths) + (len(widths) - 1) * min_gap) / u_per_mm
        print(f"[fig4a] boxes need {need:.1f} mm but panel a is "
              f"{LAYOUT['panels']['a']['w']:g} mm wide: widen it in LAYOUT or shorten a label")
        gap = min_gap
    x, edges = 0.05, []
    for (title, sub, fill, edge), w, (t1, t2) in zip(A_BOXES, widths, texts):
        cx = x + w / 2
        rbox(ax, cx, A_BOX_Y, w, A_BOX_H, "", fill=fill, edge=edge)
        t1.set_x(cx); t2.set_x(cx); edges.append((x, x + w)); x += w + gap
    for (_, right), (left, _) in zip(edges[:-1], edges[1:]):
        arrow(ax, right, A_BOX_Y, left, A_BOX_Y)

    cx_row = (edges[0][0] + edges[-1][1]) / 2          # centre of the box sequence
    ix0, ix1 = cx_row - A_INSET["w"] / 2, cx_row + A_INSET["w"] / 2
    iy0, iy1 = A_INSET["y0"], A_INSET["y1"]
    #ix0, ix1, iy0, iy1 = A_INSET["x0"], A_INSET["x1"], A_INSET["y0"], A_INSET["y1"]
    stage, base = cycle_shape(1000)
    curves = [(base + 0.0, C_PS_E, 1.4, "day 1"),
              (base + 0.16, "0.55", 1.0, "another day"),
              (base - 0.14, "0.72", 1.0, "another day")]
    lo = min(c[0].min() for c in curves) - 0.05
    hi = max(c[0].max() for c in curves) + 0.30
    for s in (1, 2, 3):
        m = np.where(stage == s)[0]
        xa, xb = ix0 + (ix1 - ix0) * m[0] / 999, ix0 + (ix1 - ix0) * m[-1] / 999
        ax.add_patch(Rectangle((xa, iy0), xb - xa, iy1 - iy0, facecolor=STAGE_COL[s],
                               alpha=0.22, lw=0, zorder=0))
        ax.text((xa + xb) / 2, iy1 - 0.02, STAGE_NAME[s], ha="center", va="top",
                fontsize=F["tiny"], color="0.35")
    ys_by = {}
    for y, col, lw, lab in curves:
        xs, ys = line_in_box(ax, ix0, ix1, iy0, iy1, y, color=col, lw=lw, ylims=(lo, hi))
        ys_by.setdefault(lab, (xs, ys, col))
    ax.text(ix1 + 0.08, ys_by["day 1"][1][-1], "day 1", fontsize=F["tiny"], color=C_PS_E, va="center")
    kb = 150
    y1b, y2b = ys_by["day 1"][1][kb], ys_by["another day"][1][kb]
    arrow(ax, ys_by["day 1"][0][kb], y1b, ys_by["day 1"][0][kb], y2b, color=RED, lw=1.0,
          style="<->", ms=9)
    ax.text(ys_by["day 1"][0][kb] + 0.12, (y1b + y2b) / 2, r"$\Delta R_t$", color=RED,
            fontsize=F["body"], va="center", ha="left")
    ax.plot([ix0, ix0, ix1], [iy1, iy0, iy0], color="0.3", lw=0.9 * LW, zorder=2)
    ax.text((ix0 + ix1) / 2, iy1 + 0.2, r"$R_t(\tau) = R_{\rm ref}(\tau) + \Delta R_t + \varepsilon$",
            ha="center", va="bottom", fontsize=F["math"], color="0.15")
    ax.text((ix0 + ix1) / 2, iy0 - 0.12, "time in cycle (min)", ha="center", va="top",
            fontsize=F["small"], color="0.3")
    ax.text(ix0 - 0.2, (iy0 + iy1) / 2, "R1 (kΩ)", rotation=90, ha="center", va="center",
            fontsize=F["small"], color="0.3")


# ── panel b : correction pipeline ────────────────────────────────────────────
def panel_b(ax):
    blank(ax); ax.set_xlim(0, 10); ax.set_ylim(0.9, 13.9)
    cx, w, xn = 4.0, 6.6, 7.5
    note = dict(ha="left", va="center", fontsize=F["tiny"], color="0.35", style="italic")
    rbox(ax, cx, 12.85, w, 1.5, r"$R_t(\tau)$: one 20-min cycle, day $t$",
         fill=C_IO_F, edge=C_IO_E, sub="sensor R1 (stage labels never seen)")
    arrow(ax, cx, 12.1, cx, 11.55)
    rbox(ax, cx, 10.65, w, 1.8, "changepoint detector", fill=C_PS_F, edge=C_PS_E,
         sub="VOCD  |  BOCD  |  Dm-BOCD  |  …")
    ax.text(xn, 10.65, "causal within the cycle\n(no future days used)", **note)
    arrow(ax, cx, 9.75, cx, 9.2)
    rbox(ax, cx, 8.45, w, 1.5, r"first changepoint $\tau^*_t$", fill=C_LS_F, edge=C_LS_E,
         sub="baseline → gas onset")
    arrow(ax, cx, 7.7, cx, 7.15)
    rbox(ax, cx, 6.2, w, 1.9, "baseline offset (no curve fit)", fill=C_GT_F, edge=C_GT_E,
         sub=r"$\Delta R_t = \overline{R_t[\tau<\tau^*_t]} - \overline{R_1[\mathrm{baseline}]}$")
    ax.text(xn, 6.2, "one constant per day\n(independent of the\nresponse shape)", **note)
    arrow(ax, cx, 5.25, cx, 4.7)
    rbox(ax, cx, 3.95, w, 1.5, r"$R_{{\rm comp},t}(\tau) = R_t(\tau) - \Delta R_t$",
         fill=C_IO_F, edge=C_IO_E, sub="compensated cycle (output)")
    arrow(ax, cx, 3.2, cx, 2.65)
    rbox(ax, cx, 1.9, w, 1.5, "score vs day-1 reference", fill="white", edge="0.5", ls="--",
         sub=r"MAE $=\overline{|R_{{\rm comp},t} - R_1|}$ over $\tau$, days $t>1$")


# ── panel c : residual vs time in cycle, one curve per day ─────────────────
C_DAY0, C_DAY1 = "#0b2a6f", "#E69F00"     # colour ramp: dark blue (early) -> orange (late)


def day_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("days", [C_DAY0, "#0072B2", "#8fb3cf", C_DAY1])


def smooth(y, n):
    if n <= 1:
        return y
    k = np.ones(n) / n
    pad = n // 2
    yp = np.pad(y, (pad, n - 1 - pad), mode="edge")
    return np.convolve(yp, k, mode="valid")


def panel_c(fig, ax, resid, method="vocd", smooth_s=30.0, sample_hz=1.0, ref=None):
    """relative error (R_ref - R_comp) / R_ref in %, every test day,
    coloured by day.  ``ref`` = the day-1 cycle (the 'ref' column of the
    residual CSV, or raw:day1 from the series CSV)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    t = resid["t_min"]
    stage = resid.get("stage")
    if stage is not None:
        for s_ in (1, 2, 3):
            m = np.where(stage == s_)[0]
            if m.size:
                ax.axvspan(t[m[0]], t[m[-1]], color=STAGE_COL[s_], alpha=0.18, lw=0, zorder=0)
                ax.text((t[m[0]] + t[m[-1]]) / 2, 0.985, STAGE_NAME[s_],
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=F["tiny"], color="0.35")
    cols = sorted((k for k in resid if k.startswith(f"resid:{method}:day")),
                  key=lambda k: int(re.findall(r"\d+", k.split(":")[-1])[0]))
    if not cols:
        print(f"[fig4c] no resid:{method}:dayN columns: re-run gas_data_v3.py "
              f"(writes --resid-all-out) with --resid-methods {method}")
        return
    day_of = lambda k: int(re.findall(r"\d+", k.split(":")[-1])[0])
    d_lo, d_hi = day_of(cols[0]), day_of(cols[-1])
    cmap, norm = day_cmap(), Normalize(d_lo, d_hi)
    n = int(round(smooth_s * sample_hz))
    if ref is None:
        print("[fig4c] no reference cycle: pass --series with raw:day1 or re-run the "
              "driver (its residual CSV now carries a 'ref' column)")
        return
    for k in cols:
        y = smooth(100.0 * resid[k] / ref, n)
        ax.plot(t, y, color=cmap(norm(day_of(k))), lw=0.55, alpha=0.85, zorder=3)
    ax.axhline(0, color="0.2", lw=0.6, zorder=2)
    ax.set_xlabel("Time in cycle (min)")
    ax.set_ylabel("VOCD relative error (%) to the day-1 reference")
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.25, lw=0.4)
    p = LAYOUT["panels"]["c"]
    cax = ax_mm(fig, p["x"] + p["w"] + LAYOUT["cbar_gap"], p["y"], LAYOUT["cbar_w"], p["h"])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Day", fontsize=F["tick"] if "tick" in F else F["small"])
    cb.ax.tick_params(labelsize=F["small"], width=0.6, length=2)
    #if smooth_s > 0:
    #    ax.text(0.01, 0.02, f"{smooth_s:g} s moving average", transform=ax.transAxes,
    #            ha="left", va="bottom", fontsize=F["tiny"], color="0.45")


# ── panel d : MAE vs runtime ─────────────────────────────────────────────────
DODGE_LOG10 = 0.07     # markers closer than this in log10(runtime) and 0.5 kOhm are spread


def dodge_x(points, ylim_span):
    """Spread near-coincident markers horizontally (log axis) so none hides
    another; returns {method: x_plot}."""
    pts = sorted(points.items(), key=lambda kv: kv[1][0])
    out, i = {}, 0
    while i < len(pts):
        j = i + 1
        while (j < len(pts) and abs(np.log10(pts[j][1][0]) - np.log10(pts[i][1][0])) < DODGE_LOG10
               and abs(pts[j][1][1] - pts[i][1][1]) < 0.5):
            j += 1
        group = pts[i:j]
        c = np.mean([np.log10(x) for _, (x, _) in group])
        for g, (m, (x, _)) in enumerate(group):
            off = (g - (len(group) - 1) / 2) * DODGE_LOG10 * 1.2
            out[m] = 10 ** (c + off) if len(group) > 1 else x
        if len(group) > 1:
            print(f"[fig4d] markers dodged in x for visibility: {[m for m, _ in group]}")
        i = j
    return out


def panel_d(ax, best, dm_est, metric, ylim, dm_marker="hollow"):
    k = 1e3
    pts = {m: (best[m]["runtime_ms"], best[m][metric] / k)
           for m in METHOD_STYLE if m in best and m != "dm-bocd"}
    if dm_est is not None:
        pts["dm-bocd"] = (dm_est["runtime_ms"], dm_est[metric] / k)
    xplot = dodge_x(pts, ylim[1] - ylim[0])
    for m, (lab, col, mk) in METHOD_STYLE.items():
        if m not in best or m == "dm-bocd":
            continue
        y = best[m][metric] / k
        lo, hi = best[m].get(f"{metric}_lo", np.nan), best[m].get(f"{metric}_hi", np.nan)
        yerr = [[max(0.0, y - lo / k)], [max(0.0, hi / k - y)]] if np.isfinite(lo) and np.isfinite(hi) else None
        ax.errorbar(xplot[m], y, yerr=yerr, fmt=mk, color=col, ms=4.5, capsize=2, elinewidth=0.7,
                    markeredgecolor="0.2", markeredgewidth=0.4, label=lab,
                    zorder=4 if m.startswith("vocd") else 3)
    if dm_est is not None:
        lab, col, mk = METHOD_STYLE["dm-bocd"]
        est = dm_est["label"] == "estimated"
        y = dm_est[metric] / k
        lo, hi = dm_est.get("err", (np.nan, np.nan))
        yerr = [[max(0.0, y - lo / k)], [max(0.0, hi / k - y)]] if np.isfinite(lo) and np.isfinite(hi) else None
        hollow = est and dm_marker == "hollow"
        ax.errorbar(xplot["dm-bocd"], y, yerr=yerr, fmt=mk, color=col, ms=4.5, capsize=2,
                    elinewidth=0.7, markerfacecolor="white" if hollow else col,
                    markeredgecolor=col if hollow else "0.2", markeredgewidth=0.9 if hollow else 0.4,
                    zorder=3, label="Dm-BOCD" if est else lab)
    if "oracle" in best and np.isfinite(best["oracle"][metric]):
        v = best["oracle"][metric] / k
        ax.axhline(v, color="0.45", ls="--", lw=0.7, zorder=1,
                   label=f"oracle, stage labels ({v:.1f})")
    ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8,
              borderaxespad=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel(("MAE" if metric == "mae" else "RMSE") + " (kΩ) relative to the day-1 reference")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.4, which="both")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best", default=os.path.join(RES, "gas_v3_best.csv"))
    ap.add_argument("--best-extra", default=None,
                    help="second best CSV; its rows override --best for the methods it has")
    ap.add_argument("--resid", default=os.path.join(RES, "gas_v3_resid_all.csv"),
                    help="reference - compensated on every test day (gas_data_v3 --resid-all-out)")
    ap.add_argument("--smooth-s", type=float, default=30.0,
                    help="moving-average window (s) for the panel-c curves; 0 = raw")
    ap.add_argument("--sample-hz", type=float, default=1.0)
    ap.add_argument("--isfet-best", default=os.path.join(_REPO, "results", "isfet", "isfet_v3_best.csv"),
                    help="for the Dm-BOCD runtime estimate when it was not run on gas")
    ap.add_argument("--dm-bocd-mae", default="bocd",
                    help="'bocd' | 'oracle' | a number (Ω): the estimate's MAE; the CI "
                         "is copied from the same source when it is a method")
    ap.add_argument("--dm-bocd-marker", choices=["hollow", "filled"], default="hollow",
                    help="how the Dm-BOCD estimate is drawn (hollow = visibly estimated)")
    ap.add_argument("--series", default=os.path.join(RES, "gas_v3_series_c.csv"),
                    help="only used for the day-1 cycle if the residual CSV has no 'ref' column")
    ap.add_argument("--metric", choices=["mae", "rmse"], default="mae")
    ap.add_argument("--ylim", type=float, nargs=2, default=(10.0, 38.0),
                    help="kOhm; the top ~8 hold the legend")
    ap.add_argument("--base-font", type=float, default=7.0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "fig4_gas.pdf"))
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTDIR, exist_ok=True)

    best = load_best(args.best)
    if args.best_extra and os.path.exists(args.best_extra):
        extra = load_best(args.best_extra)
        best.update({m: r for m, r in extra.items() if m in METHOD_STYLE})
        print(f"overrode {sorted(m for m in extra if m in METHOD_STYLE)} from {args.best_extra}")
    resid = load_series(args.resid)
    ref = resid.get("ref")
    if ref is None and os.path.exists(args.series):
        ref = load_series(args.series).get("raw:day1")

    dm_est = None
    if "dm-bocd" in best:
        r = best["dm-bocd"]
        dm_est = {"runtime_ms": r["runtime_ms"], args.metric: r[args.metric], "label": "real",
                  "err": (r.get(f"{args.metric}_lo", np.nan), r.get(f"{args.metric}_hi", np.nan))}
    elif os.path.exists(args.isfet_best):
        isf = load_best(args.isfet_best)
        if "vocd" in isf and "dm-bocd" in isf and "vocd" in best:
            ratio = isf["dm-bocd"]["runtime_ms"] / isf["vocd"]["runtime_ms"]
            rt = best["vocd"]["runtime_ms"] * ratio
            if args.dm_bocd_mae == "bocd":
                y = best.get("bocd", best["vocd"])[args.metric]
            elif args.dm_bocd_mae == "oracle":
                y = best["oracle"][args.metric]
            else:
                y = float(args.dm_bocd_mae)
            src = best.get("bocd", best["vocd"]) if args.dm_bocd_mae == "bocd" else \
                  best["oracle"] if args.dm_bocd_mae == "oracle" else {}
            err = (src.get(f"{args.metric}_lo", np.nan), src.get(f"{args.metric}_hi", np.nan))
            dm_est = {"runtime_ms": rt, args.metric: y, "label": "estimated", "err": err}
            print(f"[fig4] Dm-BOCD ESTIMATE: runtime {rt:,.0f} ms (VOCD x {ratio:.0f}, "
                  f"the ISFET ratio); {args.metric.upper()} and CI copied from "
                  f"{args.dm_bocd_mae} ({y:,.0f} Ohm)")

    with open(os.path.join(OUTDIR, "fig4_panelD.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", args.metric, f"{args.metric}_lo", f"{args.metric}_hi", "runtime_ms", "kind"])
        for m in METHOD_STYLE:
            if m in best and m != "dm-bocd":
                r = best[m]
                w.writerow([m, r[args.metric], r.get(f"{args.metric}_lo", ""),
                            r.get(f"{args.metric}_hi", ""), r["runtime_ms"], "real"])
        if dm_est is not None:
            lo, hi = dm_est.get("err", (np.nan, np.nan))
            w.writerow(["dm-bocd", dm_est[args.metric], lo, hi, dm_est["runtime_ms"], dm_est["label"]])

    W_mm, H_mm = LAYOUT["fig_mm"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    P = LAYOUT["panels"]
    ax_a = ax_mm(fig, **P["a"]); panel_a(ax_a); plabel(fig, "a", dx_mm=3.0)
    ax_b = ax_mm(fig, **P["b"]); panel_b(ax_b); plabel(fig, "b", dx_mm=3.0)
    ax_c = ax_mm(fig, **P["c"]); panel_c(fig, ax_c, resid, smooth_s=args.smooth_s, sample_hz=args.sample_hz, ref=ref); plabel(fig, "c", dx_mm=-10.0)
    ax_d = ax_mm(fig, **P["d"]); panel_d(ax_d, best, dm_est, args.metric, tuple(args.ylim), args.dm_bocd_marker); plabel(fig, "d", dx_mm=-7.0)

    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    print(f"wrote {args.out} and {png}  ({W_mm:g} x {H_mm:g} mm, {args.base_font:g} pt base)")
    for m in METHOD_STYLE:
        if m in best and m != "dm-bocd":
            r = best[m]
            print(f"  {m:15s} {args.metric.upper()} {r[args.metric]/1e3:6.2f} kOhm  "
                  f"[{r.get(args.metric+'_lo', np.nan)/1e3:.2f}, {r.get(args.metric+'_hi', np.nan)/1e3:.2f}]"
                  f"  {r['runtime_ms']:,.0f} ms")


if __name__ == "__main__":
    main()
