"""
Fig 5 — environmental monitoring: change-points vs
extreme-weather events in NoVA stream/reservoir water quality (2020-21).

Panels (bold lower-case letters; Arial; built AT FINAL PRINT SIZE):
  a  the setting: Virginia with the study area, the four tropical systems on a
     2020-21 timeline (+/- tol windows), and the constituent design (POSITIVE
     TP/TN vs dilution / flush NEGATIVE controls).
  b  left: event recall vs fraction of controls with a storm-adjacent CP, one
     point per method (the shaded half is where recall exceeds false alarms).
     Right: consistency (recall - false-alarm fraction) vs runtime over all 14
     series for all 11 rows (circles Python, squares the compiled twins, same
     colour).  One shared legend.  Both at the headline tolerance (--tol).
  c  TP daily medians (markers only; every --thin-th point in the series
     panel, every --thin-zoom-th in the zooms; the detectors ran on ALL points)
     with storm windows and VOCD's change-points (green), an
     event raster underneath with every method's change-points (one row per
     method, CP count in brackets), and one zoom per storm with every method's
     change-points.
  d  the same for a negative control (default FIELDPH).

Inputs (all from examples/env_data_v3.py; NOTHING is recomputed here):
  --summary  env_v3_summary.csv   per method x tolerance (panel b)
  --posneg   env_v3_pos_neg.csv   CP dates per constituent and method (c, d)
  --series   env_v3_series.csv    pooled daily series (c, d)
  --geojson  virginia.geojson     state outline (panel a; optional)

LAYOUT (same scheme as Figs. 2-4): sizes and positions live in the LAYOUT dict,
in millimetres from the top-left corner, at final print size (180 mm wide,
<= 170 mm tall, 7 pt text max, panel letters 8 pt bold).  Panels c and d are a
stack of three rows (series, raster, zooms) whose heights and gaps are the
"c_stack" entries; panel b is "b" (scatter), "b2" (consistency vs runtime)
and "bl" (their shared legend).

    python examples/vocd_fig5_env.py
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, os
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "fig5_env")
RES = os.path.join(_REPO, "results", "env")

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 170.0),
    "panels": {
        #      x      y      w      h
        "a":  dict(x=1.0,   y=4.0,  w=89.0, h=70.0),   # schematic (panel mm coords)
        "b":  dict(x=104.0, y=6.0,  w=30.0, h=43.0),   # recall vs false alarms
        "b2": dict(x=147.0, y=6.0,  w=30.0, h=43.0),   # consistency vs runtime
        "bl": dict(x=98.0,  y=60.0, w=80.0, h=9.0),    # shared legend strip
        "c":  dict(x=14.0,  y=76.0, w=73.0, h=36.0),   # series (raster + zooms below)
        "d":  dict(x=105.0, y=76.0, w=73.0, h=36.0),
    },
    "c_stack": dict(raster_gap=1.2, raster_h=18.0,     # event raster under the series
                    zoom_gap=12.0, zoom_h=18.0,        # storm zooms under the raster
                    zoom_hgap=5.6, zoom_left=1.5),
}
F = {}
LW = 0.62
SERIES_MS = 1.6     # marker size of the (unconnected) daily medians in panels c, d

# ── palette: methods = figs 2-4 (Okabe-Ito); boxes = figs 1/3/4 ─────────────
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
BASE_METHODS = [m for m in METHOD_STYLE if "(cpp)" not in m]      # 8 distinct detectors
short = lambda m: METHOD_STYLE[m][0].replace(" (Py)", "")

C_IO_F, C_IO_E = "#cfe0f2", "#2c5f8a"
C_PS_F, C_PS_E = "#f6c873", "#cf8a1e"
STORM_COL = "#7f8c8d"

RAIN_EVENTS = {"Isaias": dt.date(2020, 8, 4), "Zeta": dt.date(2020, 10, 29),
               "Elsa": dt.date(2021, 7, 9), "Ida": dt.date(2021, 9, 1)}
POSITIVE = {"TP": "Total phosphorus", "TN": "Total nitrogen"}
NEGATIVE = {"COND25": "Specific conductance", "TDS": "Total dissolved solids",
            "CL": "Chloride", "SO4": "Sulfate", "THARD": "Total hardness",
            "TALK": "Total alkalinity", "TURB": "Turbidity", "TSS": "Total suspended solids",
            "COD": "Chemical oxygen demand", "TOC": "Total organic carbon",
            "DOC": "Dissolved organic carbon", "FIELDPH": "Field pH"}
UNITS = {"TP": r"mg L$^{-1}$", "TN": r"mg L$^{-1}$", "FIELDPH": "pH",
         "THARD": r"mg L$^{-1}$ CaCO$_3$", "COD": r"mg L$^{-1}$", "COND25": r"µS cm$^{-1}$",
         "CL": r"mg L$^{-1}$", "TALK": r"mg L$^{-1}$ CaCO$_3$"}
T0, T1 = dt.date(2020, 1, 1), dt.date(2021, 12, 31)


# ── io ────────────────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_summary(path, tol):
    out = {}
    for r in read_csv(path):
        if "tol" in r and r["tol"] and int(float(r["tol"])) != tol:
            continue
        out[r["method"]] = {k: (fnum(v) if k not in ("method", "events_matched", "budget_kind")
                                else v) for k, v in r.items()}
    return out


def load_cps(path):
    out = {}
    for r in read_csv(path):
        ds = [dt.date.fromisoformat(x) for x in r["cp_dates"].split(";") if x.strip()]
        out.setdefault(r["constituent"], {})[r["method"]] = ds
    return out


def load_series(path):
    out = {}
    for r in read_csv(path):
        out.setdefault(r["constituent"], ([], []))
        out[r["constituent"]][0].append(dt.date.fromisoformat(r["date"]))
        out[r["constituent"]][1].append(float(r["value"]))
    return {k: (d, np.asarray(v, float)) for k, (d, v) in out.items()}


# ── style / helpers (shared with figs 1-4) ──────────────────────────────────
def set_nature_style(base=7.0):
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    fam = ["Arial", "Arimo", "Liberation Sans", "Helvetica"]
    if not ({f.name for f in fm.fontManager.ttflist} & set(fam)):
        print("[fig5] warning: no Arial-family font found; using DejaVu Sans.")
    F.update(panel=base + 1.0, title=base, body=base - 0.5, small=base - 1.0,
             tiny=max(5.0, base - 1.5), math=base)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fam + ["DejaVu Sans"],
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


def plabel(fig, panel, dx_mm=-2.5, dy_mm=-1.0, letter=None):
    p = LAYOUT["panels"][panel]
    W, H = LAYOUT["fig_mm"]
    fig.text((p["x"] + dx_mm) / W, 1.0 - (p["y"] + dy_mm) / H, letter or panel,
             fontsize=F["panel"], fontweight="bold", va="bottom", ha="left")


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ── panel a : setting (drawn in the panel's own mm, y measured from the top) ─
def draw_virginia(ax, geojson, x0, x1, y0, y1, study=(-77.4, 38.75)):
    from matplotlib.patches import Polygon
    g = json.load(open(geojson))
    geom = g["geometry"] if "geometry" in g else g["features"][0]["geometry"]
    rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    allp = np.array([p for r in rings for p in r], float)
    lon0, lon1 = allp[:, 0].min(), allp[:, 0].max()
    lat0, lat1 = allp[:, 1].min(), allp[:, 1].max()
    kx = np.cos(np.deg2rad((lat0 + lat1) / 2))
    s = min((x1 - x0) / ((lon1 - lon0) * kx), (y1 - y0) / (lat1 - lat0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    f = lambda lon, lat: (cx + (lon - (lon0 + lon1) / 2) * kx * s, cy + (lat - (lat0 + lat1) / 2) * s)
    for r in rings:
        ax.add_patch(Polygon(np.array([f(lo, la) for lo, la in r]), closed=True,
                             facecolor="#e8eef5", edgecolor="0.45", lw=0.8 * LW, zorder=1))
    sx, sy = f(*study)
    ax.plot([sx], [sy], "o", ms=4.5, mfc=C_PS_F, mec=C_PS_E, mew=0.9, zorder=3)
    return sx, sy


def rbox(ax, cx, cy, w, h, title, *, fill, edge, sub=None, fs=None, sub_fs=None,
         tc="0.12", lw=1.3, ls="-", rounding=0.9):
    from matplotlib.patches import FancyBboxPatch
    fs = F["body"] if fs is None else fs
    sub_fs = F["tiny"] if sub_fs is None else sub_fs
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rounding}", linestyle=ls,
                 linewidth=lw * LW, edgecolor=edge, facecolor=fill, zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tc, zorder=3)
        ax.text(cx, cy - h * 0.23, sub, ha="center", va="center", fontsize=sub_fs,
                color=tc, style="italic", zorder=3)
    elif title:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=tc, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="0.35", lw=1.5, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13 * LW,
                 lw=lw * LW, color=color, shrinkA=0, shrinkB=0, zorder=1))


# Panel-a geometry, in the panel's own mm (y measured from the panel top).
C_LS_F, C_LS_E = "#eaeaea", "#8a8a8a"
A_BOXES = [   # (title, subtitle, fill, edge), left to right; boxes size themselves
    ("Stations",       "22 sites (~weekly)",    C_IO_F, C_IO_E),
    ("Lab analysis",   "6,815 samples (SM)",   C_LS_F, C_LS_E),
    ("Daily medians",  "pooled (14 series)",    C_LS_F, C_LS_E),
    ("Detector → PC",  "CP index → date",      C_PS_F, C_PS_E),
]
A_ROW_Y, A_ROW_H = 6.0, 8.5          # workflow row: centre and height (mm from top)
A_BOX_PAD_MM, A_MIN_GAP_MM = 1.1, 3.0
A_MAP = dict(x0=0.5, x1=47.0, top=16.0, bottom=46.0)    # was x1=38.0, bottom=42.0
A_TEXT_X = 49.0                                         # was 41.0
#A_MAP = dict(x0=0.5, x1=38.0, top=16.0, bottom=42.0)    # map box (mm)
#A_TEXT_X = 41.0                      # left edge of the station note / constituents
A_TIMELINE_Y = 55.0                  # timeline baseline (mm from top)


def panel_a(ax, geojson, neg, tol):
    from matplotlib.patches import Rectangle
    p = LAYOUT["panels"]["a"]
    W, H = p["w"], p["h"]
    blank(ax); ax.set_xlim(0, W); ax.set_ylim(0, H)
    Y = lambda y_top: H - y_top
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(t):
        bb = t.get_window_extent(r)
        return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]

    # 1) workflow row: boxes sized from their text, spread evenly, plain arrows
    yb = Y(A_ROW_Y)
    texts, widths = [], []
    for title, sub, fill, edge in A_BOXES:
        t1 = ax.text(0, yb + A_ROW_H * 0.17, title, ha="center", va="center",
                     fontsize=F["body"], fontweight="bold", color="0.12", zorder=3)
        t2 = ax.text(0, yb - A_ROW_H * 0.23, sub, ha="center", va="center",
                     fontsize=F["tiny"], color="0.12", style="italic", zorder=3)
        texts.append((t1, t2)); widths.append(max(width(t1), width(t2)) + 2 * A_BOX_PAD_MM)
    gap = (W - 1.0 - sum(widths)) / (len(widths) - 1)
    if gap < A_MIN_GAP_MM:
        need = sum(widths) + (len(widths) - 1) * A_MIN_GAP_MM
        print(f"[fig5a] boxes need {need:.1f} mm but panel a is {W:g} mm wide: "
              "widen it in LAYOUT or shorten a label")
        gap = A_MIN_GAP_MM
    x, edges = 0.5, []
    for (title, sub, fill, edge), w, (t1, t2) in zip(A_BOXES, widths, texts):
        cx = x + w / 2
        rbox(ax, cx, yb, w, A_ROW_H, "", fill=fill, edge=edge)
        t1.set_x(cx); t2.set_x(cx); edges.append((x, x + w)); x += w + gap
    for (_, right), (left, _) in zip(edges[:-1], edges[1:]):
        arrow(ax, right, yb, left, yb)

    # 2) Virginia with the study area, and the station note
    note = "Northern Virginia\n22 stream and reservoir stations\n~weekly sampling, 2020–21"
    if geojson and os.path.exists(geojson):
        sx, sy = draw_virginia(ax, geojson, A_MAP["x0"], A_MAP["x1"], Y(A_MAP["bottom"]), Y(A_MAP["top"]))
        ax.annotate(note, (sx, sy), (A_TEXT_X, Y(A_MAP["top"] + 1.0)), ha="left", va="top",
                    fontsize=F["small"], color="0.25", linespacing=1.2,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6 * LW))
    else:
        ax.text((A_MAP["x0"] + A_MAP["x1"]) / 2, Y((A_MAP["top"] + A_MAP["bottom"]) / 2),
                "Virginia outline\n(--geojson not found)", ha="center", va="center",
                fontsize=F["tiny"], color="0.5", style="italic")
        ax.text(A_TEXT_X, Y(A_MAP["top"] + 1.0), note, ha="left", va="top", fontsize=F["small"],
                color="0.25", linespacing=1.2)

    # 3) constituents: two colour-keyed lines (positive vs negative control)
    #yc = A_MAP["top"] + 12.5
    yc = A_MAP["top"] + 14.0                     # was + 12.5
    for k, (fill, edge, head, body) in enumerate((
            (C_PS_F, C_PS_E, "positive", "TP, TN: storm response expected"),
            (C_IO_F, C_IO_E, "negative controls", f"{neg} + 11 dilution / flush"))):
        yy = Y(yc + 7.0 * k) 
        #yy = Y(yc + 5.6 * k)
        ax.add_patch(Rectangle((A_TEXT_X, yy - 1.3), 2.6, 2.6, facecolor=fill, edgecolor=edge,
                               lw=0.8 * LW, zorder=2))
        ax.text(A_TEXT_X + 3.6, yy + 0.3, head, ha="left", va="bottom", fontsize=F["small"],
                fontweight="bold", color=edge)
        ax.text(A_TEXT_X + 3.6, yy - 0.2, body, ha="left", va="top", fontsize=F["small"], color="0.25")

    # 4) one storm timeline: names AND dates, labels staggered on two levels
    xa, xb = 3.0, W - 3.0
    xt = lambda d: xa + (xb - xa) * (d - T0).days / (T1 - T0).days
    yl = Y(A_TIMELINE_Y)
    ax.plot([xa, xb], [yl, yl], color="0.3", lw=1.0 * LW, zorder=1)
    for yr in (2020, 2021):
        for mo in (1, 4, 7, 10):
            d = dt.date(yr, mo, 1)
            ax.plot([xt(d), xt(d)], [yl - 0.6, yl + 0.6], color="0.3", lw=0.7 * LW)
            ax.text(xt(d), yl - 1.2, str(yr) if mo == 1 else d.strftime("%b"), ha="center",
                    va="top", fontsize=F["tiny"], color="0.3" if mo == 1 else "0.45")
    # labels below the line (Zeta, Ida) end where panel b's legend text ends: lay out a
    # throw-away copy of that legend in its strip, read its bottom edge, then remove it
    from matplotlib.lines import Line2D
    fig = ax.figure
    rr = fig.canvas.get_renderer()
    inv_ = ax.transData.inverted()
    tmp = ax_mm(fig, **LAYOUT["panels"]["bl"]); blank(tmp)
    hh = [Line2D([0], [0], marker="o", ls="", ms=4.2, label=short(m)) for m in BASE_METHODS]
    hh += [Line2D([0], [0], marker="o", ls="", ms=4.2, label="Python"),
           Line2D([0], [0], marker="s", ls="", ms=4.2, label="C++")]
    leg = tmp.legend(handles=hh, loc="center", ncol=5, frameon=False, fontsize=F["tiny"],
                     handletextpad=0.2, columnspacing=0.9, borderaxespad=0.0)
    leg_bottom_px = leg.get_window_extent(rr).y0
    tmp.remove()
    y_below = inv_.transform((0, leg_bottom_px))[1]
    for k, (name, d) in enumerate(RAIN_EVENTS.items()):
        w0, w1 = xt(d - dt.timedelta(days=tol)), xt(d + dt.timedelta(days=tol))
        ax.add_patch(Rectangle((w0, yl - 1.0), w1 - w0, 2.0, facecolor=C_VOCD, alpha=0.25, lw=0))
        if k % 2 == 0:                                          # Isaias, Elsa: above the line
            top = yl + 2.4
            ax.plot([xt(d), xt(d)], [yl - 1.4, top], color=STORM_COL, lw=1.0 * LW, zorder=2)
            ax.text(xt(d), top + 0.3, f"{name}\n{d:%Y-%m-%d}", ha="center", va="bottom",
                    fontsize=F["small"], color="0.25", linespacing=1.1, fontweight="normal")
        else:                                                   # Zeta, Ida: below the line
            t = ax.text(xt(d), y_below, f"{name}\n{d:%Y-%m-%d}", ha="center", va="bottom",
                        fontsize=F["tiny"], color="0.25", linespacing=1.1, fontweight="normal")
            bb = t.get_window_extent(rr)
            label_top = inv_.transform((0, bb.y1))[1]
            ax.plot([xt(d), xt(d)], [yl + 1.4, label_top + 0.3], color=STORM_COL, lw=1.0 * LW, zorder=2)
    ax.text(xb, yl + 1.2, f"±{tol} d windows", ha="right", va="bottom", fontsize=F["small"] - 0.3,
            color=C_VOCD, style="italic")


# ── panel b : recall vs false alarms  +  consistency vs runtime ───────────────
def panel_b(ax_s, ax_c, ax_l, summ):
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    # (left) recall vs false-alarm fraction; coincident points are dodged
    ax_s.add_patch(Polygon([(-0.05, -0.05), (-0.05, 1.08), (1.08, 1.08)], closed=True,
                           facecolor=C_VOCD, alpha=0.08, lw=0, zorder=0))
    ax_s.plot([-0.05, 1.08], [-0.05, 1.08], color="0.6", ls="--", lw=0.6, zorder=1)
    ax_s.text(0.0, 1.03, "recall > false alarms", ha="left", va="top",
              fontsize=F["tiny"], color=C_VOCD, style="italic")
    fa = lambda m: (summ[m]["neg_false_frac"] if np.isfinite(summ[m].get("neg_false_frac", np.nan))
                    else summ[m]["neg_false"] / summ[m]["n_neg"])
    pts = {m: (fa(m), summ[m]["pos_recall"]) for m in BASE_METHODS if m in summ}
    used = set()
    for m in pts:
        if m in used:
            continue
        g = [n for n in pts if n not in used and np.hypot(pts[n][0] - pts[m][0], pts[n][1] - pts[m][1]) < 0.02]
        used.update(g)
        for i, n in enumerate(g):
            x, y = pts[n]
            ax_s.plot(x + (i - (len(g) - 1) / 2) * 0.035, y, "o", color=METHOD_STYLE[n][1], ms=4.2,
                      mec="0.2", mew=0.4, zorder=4 if n == "vocd" else 3, clip_on=False)
    ax_s.set_xlim(-0.05, 1.08); ax_s.set_ylim(-0.05, 1.08)
    ax_s.set_xticks([0, 0.5, 1.0]); ax_s.set_yticks([0, 0.5, 1.0])
    ax_s.set_xlabel("Fraction of false alarms"); ax_s.set_ylabel("Recall")
    ax_s.grid(alpha=0.25, lw=0.4)

    # (right) consistency vs runtime, all 11 rows
    rows = [m for m in METHOD_STYLE if m in summ]
    cons = [summ[m]["consistency"] for m in rows]
    lo, hi = min(cons + [0.0]), max(cons + [0.0])
    ylo, yhi = lo - 0.1, max(hi + 0.15, 0.25)
    ax_c.axhspan(0, yhi, color=C_VOCD, alpha=0.08, lw=0, zorder=0)
    ax_c.axhline(0, color="0.6", ls="--", lw=0.6, zorder=1)
    ax_c.text(0.03, 0.97, "consistency > 0", transform=ax_c.transAxes, ha="left", va="top",
              fontsize=F["tiny"], color=C_VOCD, style="italic")
    for m in rows:
        lab, col, mk = METHOD_STYLE[m]
        ax_c.plot(summ[m]["runtime_ms"], summ[m]["consistency"], mk, color=col, ms=4.2,
                  mec="0.2", mew=0.4, zorder=4 if m.startswith("vocd") else 3, clip_on=False)
    rts = [summ[m]["runtime_ms"] for m in rows]
    ax_c.set_xscale("log")
    ax_c.set_xlim(min(rts) / 2.5, max(rts) * 2.5)
    ax_c.set_ylim(ylo, yhi)
    import matplotlib.ticker as mt
    ax_c.xaxis.set_major_locator(mt.LogLocator(base=10, numticks=4))
    ax_c.yaxis.set_major_locator(mt.MaxNLocator(4))
    ax_c.set_xlabel("Runtime (ms)"); ax_c.set_ylabel("Consistency")
    ax_c.grid(alpha=0.25, lw=0.4, which="major")

    # shared legend: one colour per method + the marker key
    blank(ax_l)
    h = [Line2D([0], [0], marker="o", ls="", color=METHOD_STYLE[m][1], mec="0.2", mew=0.4,
                ms=4.2, label=short(m)) for m in BASE_METHODS if m in summ]
    h += [Line2D([0], [0], marker="o", ls="", color="0.55", mec="0.2", mew=0.4, ms=4.2, label="Python"),
          Line2D([0], [0], marker="s", ls="", color="0.55", mec="0.2", mew=0.4, ms=4.2, label="C++")]
    ax_l.legend(handles=h, loc="center", ncol=5, frameon=False, fontsize=F["tiny"],
                handletextpad=0.2, columnspacing=0.9, borderaxespad=0.0)


# ── panels c / d : series + event raster + storm zooms ───────────────────────
def panel_series(fig, panel, col, days, y, cps, tol, thin=3, thin_zoom=1):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mt
    from matplotlib.patches import ConnectionPatch
    p, st = LAYOUT["panels"][panel], LAYOUT["c_stack"]
    ax_t = ax_mm(fig, **p)
    ax_r = ax_mm(fig, p["x"], p["y"] + p["h"] + st["raster_gap"], p["w"], st["raster_h"])
    zy = p["y"] + p["h"] + st["raster_gap"] + st["raster_h"] + st["zoom_gap"]
    zw = (p["w"] - st["zoom_left"] - 3 * st["zoom_hgap"]) / 4
    ax_z = [ax_mm(fig, p["x"] + st["zoom_left"] + k * (zw + st["zoom_hgap"]), zy, zw, st["zoom_h"])
            for k in range(4)]
    days = list(days); y = np.asarray(y, float)
    td = dt.timedelta

    # series, storm windows, VOCD's change-points
    for name, d in RAIN_EVENTS.items():
        for a in (ax_t, ax_r):
            a.axvspan(d - td(days=tol), d + td(days=tol), color=STORM_COL, alpha=0.14, lw=0, zorder=0)
            a.axvline(d, color=STORM_COL, lw=0.7, zorder=1)
        ax_t.text(d, 1.01, name, transform=ax_t.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=F["tiny"], color="0.3")
    k = max(1, int(thin))                    # display only: every k-th daily median, no line
    ax_t.plot(days, y, ls="none", marker="none")          # invisible: keeps the full-series y-range
    ax_t.plot(days[::k], y[::k], "o", color="0.25", ms=SERIES_MS, zorder=3)
    for d in cps.get("vocd", []):
        ax_t.axvline(d, color=C_VOCD, lw=1.0, zorder=4)
    ax_t.set_ylabel(f"{col} ({UNITS.get(col, '')})".replace(" ()", ""))
    ax_t.set_xlim(T0, T1); ax_t.tick_params(axis="x", labelbottom=False)
    ax_t.yaxis.set_major_locator(mt.MaxNLocator(4))
    ax_t.grid(alpha=0.25, lw=0.4)

    # event raster: one row per method, every change-point as a tick
    rows = [m for m in BASE_METHODS]
    for i, m in enumerate(rows):
        ds = cps.get(m, [])
        if ds:
            ax_r.broken_barh([(mdates.date2num(d) - 2.5, 5.0) for d in ds], (i - 0.36, 0.72),
                             facecolors=METHOD_STYLE[m][1], edgecolors="0.25", linewidth=0.25,
                             zorder=3)
        ax_r.text(-0.008, i, f"{short(m)} ({len(ds)})", transform=ax_r.get_yaxis_transform(),
                  ha="right", va="center", fontsize=F["tiny"] - 0.3, color="0.2")
    ax_r.set_ylim(len(rows) - 0.5, -0.5); ax_r.set_yticks([])
    ax_r.set_xlim(T0, T1)
    ax_r.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_r.tick_params(axis="x", labelsize=F["tiny"])
    for s in ("left",):
        ax_r.spines[s].set_visible(False)
    ax_r.grid(axis="x", alpha=0.25, lw=0.4)

    # zooms: one per storm, +/- (tol + 7) days, every method's change-points
    pad = tol + 7
    for (name, e), az in zip(RAIN_EVENTS.items(), ax_z):
        a, b = e - td(days=pad), e + td(days=pad)
        sel = [(d, v) for d, v in zip(days, y) if a <= d <= b]
        az.axvspan(e - td(days=tol), e + td(days=tol), color=STORM_COL, alpha=0.18, lw=0, zorder=0)
        az.axvline(e, color=STORM_COL, lw=0.9, zorder=1)
        if sel:
            kz = max(1, int(thin_zoom))
            az.plot([d for d, _ in sel], [v for _, v in sel], ls="none", marker="none")   # full y-range
            az.plot([d for d, _ in sel][::kz], [v for _, v in sel][::kz], "o", color="0.25",
                    ms=SERIES_MS, zorder=3)
        for m in reversed(rows):                                 # VOCD drawn last, on top
            for dd in cps.get(m, []):
                if a <= dd <= b:
                    az.axvline(dd, color=METHOD_STYLE[m][1], lw=1.3 if m == "vocd" else 0.8,
                               alpha=0.95, zorder=5 if m == "vocd" else 4)
        az.set_xlim(a, b)
        az.set_xticks([e - td(days=tol), e + td(days=tol)])
        az.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        az.tick_params(axis="x", labelsize=F["tiny"] - 0.5, rotation=30, pad=1)
        az.tick_params(axis="y", labelsize=F["tiny"] - 0.5, pad=1)
        az.yaxis.set_major_locator(mt.MaxNLocator(3))
        for lbl in az.get_xticklabels():
            lbl.set_ha("right")
        az.set_title(name, fontsize=F["tiny"], color="0.3", pad=2)
        az.grid(alpha=0.25, lw=0.4)
        for xa, fx in ((a, 0.0), (b, 1.0)):                     # connectors from the raster
            fig.add_artist(ConnectionPatch(xyA=(mdates.date2num(xa), 0.0), coordsA=ax_r.get_xaxis_transform(),
                                           xyB=(fx, 1.0), coordsB=az.transAxes,
                                           color="0.78", lw=0.4, zorder=0))
    return ax_t


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=os.path.join(RES, "env_v3_summary.csv"))
    ap.add_argument("--posneg", default=os.path.join(RES, "env_v3_pos_neg.csv"))
    ap.add_argument("--series", default=os.path.join(RES, "env_v3_series.csv"))
    ap.add_argument("--geojson", default=os.path.join(_REPO, "data", "env_monitoring", "virginia.geojson"))
    ap.add_argument("--pos", default="TP", help="positive constituent for panel c")
    ap.add_argument("--neg", default="FIELDPH", help="negative control for panel d")
    ap.add_argument("--tol", type=int, default=15)
    ap.add_argument("--thin", type=int, default=3,
                    help="panels c/d series: plot every N-th daily median (1 = all); display only")
    ap.add_argument("--thin-zoom", type=int, default=1,
                    help="panels c/d storm zooms: plot every N-th daily median (1 = all)")
    ap.add_argument("--base-font", type=float, default=7.0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "fig5_env.pdf"))
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTDIR, exist_ok=True)

    summ = load_summary(args.summary, args.tol)
    cps = load_cps(args.posneg)
    series = load_series(args.series)
    for c in (args.pos, args.neg):
        if c not in series:
            raise SystemExit(f"no series for {c!r} in {args.series}")

    W_mm, H_mm = LAYOUT["fig_mm"]
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    P = LAYOUT["panels"]
    ax_a = ax_mm(fig, **P["a"]); panel_a(ax_a, args.geojson, args.neg, args.tol); plabel(fig, "a", dx_mm=2.0)
    ax_bs = ax_mm(fig, **P["b"]); ax_bc = ax_mm(fig, **P["b2"]); ax_bl = ax_mm(fig, **P["bl"])
    panel_b(ax_bs, ax_bc, ax_bl, summ); plabel(fig, "b", dx_mm=-9.0)
    panel_series(fig, "c", args.pos, *series[args.pos], cps.get(args.pos, {}), args.tol,
                 thin=args.thin, thin_zoom=args.thin_zoom)
    plabel(fig, "c", dx_mm=-12.0, dy_mm=-2.5)
    panel_series(fig, "d", args.neg, *series[args.neg], cps.get(args.neg, {}), args.tol,
                 thin=args.thin, thin_zoom=args.thin_zoom)
    plabel(fig, "d", dx_mm=-12.0, dy_mm=-2.5)

    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300)
    print(f"wrote {args.out} and {png}  ({W_mm:g} x {H_mm:g} mm, {args.base_font:g} pt base)")
    for m in METHOD_STYLE:
        if m in summ:
            s = summ[m]
            print(f"  {m:15s} recall {s['pos_recall']:.2f}  FA {int(s['neg_false'])}/{int(s['n_neg'])}"
                  f"  consistency {s['consistency']:+.2f}  {s['runtime_ms']:,.0f} ms")


if __name__ == "__main__":
    main()