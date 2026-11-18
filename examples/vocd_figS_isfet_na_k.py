"""
SI figures (Nature-Communications style) — ISFET Na+ and K+ drift compensation,
reference-free.  One figure per ion, two panels each, same style, palette and
layout scheme as ``vocd_fig3_isfet.py`` (panels c and d there):

  a  <ion> ISFET (mV) vs day: raw V, the M0 (online straight line) compensated
     signal and the VOCD compensated signal, both drawn on the channel's day-0
     level (V_comp + V(0)), so raw starts on top of them and the growing gap is
     the accumulated drift.  VOCD's change-points and the burn-in are marked.
  b  residual drift beta_comp (mV/day, slope of the compensated signal over the
     post-burn-in window) vs runtime (ms, log) for the full panel, with 95%
     moving-block-bootstrap CIs.  beta_comp = 0 (no residual drift) is the
     target; M0 and exp + linear are drawn as horizontal context lines and the
     uncompensated slope is quoted in the corner.  Each detector's change-point
     count is in the legend: beta_comp alone rewards over-segmentation (a
     detector that fires constantly flattens the signal), so the count is the
     context that keeps the panel honest.

Inputs (all produced by examples/si_isfet_na_k.py; NOTHING is recomputed):
  --summary  si_isfet_summary.csv        beta_corr, beta_lo/hi, runtime_ms, n_cp
  --series   si_isfet_{ion}_series.csv   per-day V and every method's E0_hat
  --cps      si_isfet_cps.csv            every detector's final boundary set

LAYOUT: sizes and positions in millimetres from the top-left corner, at final
print size (Nature Portfolio: 180 mm wide, text 7 pt max / 5 pt min, panel
letters 8 pt bold).  Every text size derives from --base-font.

    python examples/vocd_figS_isfet_na_k.py
    python examples/vocd_figS_isfet_na_k.py --ions k --ylim-b k -0.30 0.25
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "si_isfet")
OUTDIR = os.path.join(_REPO, "si_isfet", "figures")

ION_TEX = {"na": r"Na$^+$", "k": r"K$^+$", "h": r"H$^+$"}

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 120.0),
    "panels": {
        #      x      y      w      h
        "a": dict(x=14.0,  y=7.0, w=74.0, h=100.0),
        "b": dict(x=104.0, y=7.0, w=72.0, h=100.0),
    },
}
F = {}        # font tiers, filled by set_nature_style(base)

# ── palette: identical to Fig. 3 (Okabe-Ito methods) ─────────────────────────
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
BASELINE_STYLE = {  # context rows drawn as horizontal lines in panel b
    "M0":  ("M0 online",    "0.45", ":"),
    "exp": ("exp + linear", "0.45", "-."),
}
C_RAW = "#cf8a1e"        # Fig. 3 raw-V_H colour (ISFET box edge)
C_M0 = "0.55"
RED = "#D55E00"


# ── io ────────────────────────────────────────────────────────────────────────
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_summary(path, ion):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["ion"] == ion:
                out[r["method"]] = {k: fnum(v) for k, v in r.items()
                                    if k not in ("ion", "method", "budget_kind", "cp_days")}
    if not out:
        raise SystemExit(f"no rows for ion {ion!r} in {path}")
    return out


def load_series(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([fnum(r[k]) for r in rows]) for k in rows[0].keys()}


def load_cps(path, ion, method):
    days = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["ion"] == ion and r["method"] == method:
                days.append(float(r["cp_day"]))
    return sorted(days)


def compensated_mv(series, method):
    """V - E0_hat on the channel's day-0 level: V_comp + V(0).  E0_hat(0) =
    V(0) for every causal estimator in the harness, so this starts on raw V."""
    v = series["v_mv"]
    return v - series[f"e0_hat:{method}"] + v[0]


# ── style helpers (shared with fig1 / fig2 / fig3) ───────────────────────────
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


# ── panel a : raw / M0 / VOCD compensated ────────────────────────────────────
def panel_a(ax, ion, series, cps, burn_in, legend_loc):
    dd, v = series["day"], series["v_mv"]
    ax.axvspan(dd[0] - 0.5, burn_in, color="0.92", lw=0, zorder=0)
    ax.text(burn_in / 2, 0.985, "burn-in", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=F["tiny"], color="0.45")
    for i, c in enumerate(cps, 1):
        ax.axvline(c, color=RED, ls="--", lw=0.7, zorder=1)
        ax.text(c, 1.0, rf"CP$_{i}$", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=F["tiny"], color=RED)
    ax.plot(dd, v, color=C_RAW, lw=1.0, label=rf"raw $V$ ({ION_TEX[ion]} ISFET)", zorder=3)
    ax.plot(dd, compensated_mv(series, "M0"), color=C_M0, lw=0.9,
            label="M0 compensated", zorder=3)
    ax.plot(dd, compensated_mv(series, "vocd"), color=C_VOCD, lw=1.0,
            label="VOCD compensated", zorder=4)
    ax.set_xlabel("Time (days)"); ax.set_ylabel(rf"{ION_TEX[ion]} ISFET (mV)")
    x0, x1 = dd[0] - 0.5, dd[-1] + 0.5
    ax.set_xlim(x0, x1)
    ax.grid(alpha=0.25, lw=0.4)
    if legend_loc == "upper left":      # start the legend just right of the burn-in band
        ax.legend(loc="upper left", bbox_to_anchor=((burn_in - x0) / (x1 - x0) + 0.01, 1.0),
                  frameon=False, handlelength=1.6)
    else:
        ax.legend(loc=legend_loc, frameon=False, handlelength=1.6)


# ── panel b : beta_comp vs runtime ───────────────────────────────────────────
def auto_ylim(best, headroom=0.75):
    """Span every plotted CI and baseline, then leave the top for the legend."""
    vals = [0.0]
    for m in list(METHOD_STYLE) + list(BASELINE_STYLE):
        if m in best:
            vals += [best[m][k] for k in ("beta_corr", "beta_lo", "beta_hi")
                     if np.isfinite(best[m].get(k, np.nan))]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    return lo - 0.08 * span, hi + headroom * span


def panel_b(ax, best, ylim, cp_labels=True):
    for m, (lab, col, mk) in METHOD_STYLE.items():
        if m not in best:
            continue
        r = best[m]
        x, y = r["runtime_ms"], r["beta_corr"]
        lo, hi = r.get("beta_lo", np.nan), r.get("beta_hi", np.nan)
        yerr = ([[max(0.0, y - lo)], [max(0.0, hi - y)]]
                if np.isfinite(lo) and np.isfinite(hi) else None)
        if cp_labels and "(cpp)" not in m and np.isfinite(r.get("n_cp", np.nan)):
            lab = f"{lab}, {int(r['n_cp'])} CP"
        ax.errorbar(x, y, yerr=yerr, fmt=mk, color=col, ms=4.5, capsize=2, elinewidth=0.7,
                    markeredgecolor="0.2", markeredgewidth=0.4, zorder=3, label=lab)
    ax.axhline(0.0, color="0.2", lw=0.6, zorder=1, label="no residual drift")
    ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.3,
              columnspacing=0.8, borderaxespad=0.2)
    groups = {}   # merge baselines whose slopes coincide to 0.001 mV/day
    for m, (lab, col, ls) in BASELINE_STYLE.items():
        if m in best and np.isfinite(best[m]["beta_corr"]):
            groups.setdefault(round(best[m]["beta_corr"], 3), []).append(
                (lab, col, ls, best[m]["beta_corr"]))
    for _, items in groups.items():
        lab = " = ".join(i[0] for i in items)
        col, ls, val = items[0][1], items[0][2], items[0][3]
        ax.axhline(val, color=col, ls=ls, lw=0.7, zorder=1)
        ax.text(0.99, val, f"{lab} ({val:+.3f})", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=F["tiny"], color=col)
    if "none" in best:
        ax.text(0.98, 0.03, f"no compensation: {best['none']['beta_corr']:+.2f} mV day$^{{-1}}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=F["tiny"], color="0.4")
    xs = [best[m]["runtime_ms"] for m in METHOD_STYLE
          if m in best and np.isfinite(best[m]["runtime_ms"])]
    if xs:
        ax.set_xlim(min(xs) / 2.0, max(xs) * 2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel(r"Residual drift $\beta_{\rm comp}$ (mV day$^{-1}$)")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.4, which="both")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res-dir", default=RES, help="folder written by si_isfet_na_k.py")
    ap.add_argument("--ions", default="k,na")
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--ylim-b", nargs=3, action="append", metavar=("ION", "LO", "HI"),
                    default=[], help="fix panel b's y-range for one ion (repeatable); "
                                     "default spans every CI and leaves the top for the legend")
    ap.add_argument("--legend-a", default="upper left",
                    help="panel a legend location (matplotlib loc string)")
    ap.add_argument("--no-cp-labels", action="store_true",
                    help="drop the change-point counts from panel b's legend")
    ap.add_argument("--base-font", type=float, default=7.0,
                    help="base font size in pt at final print size (journal max 7)")
    ap.add_argument("--out-dir", default=OUTDIR,
                    help="writes figS_isfet_<ion>.pdf (+ .png preview, + panel-b CSV)")
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)
    ylim_override = {i.lower(): (float(lo), float(hi)) for i, lo, hi in args.ylim_b}

    summary = os.path.join(args.res_dir, "si_isfet_summary.csv")
    cps_path = os.path.join(args.res_dir, "si_isfet_cps.csv")
    W_mm, H_mm = LAYOUT["fig_mm"]
    P = LAYOUT["panels"]

    for ion in [s.strip().lower() for s in args.ions.split(",") if s.strip()]:
        best = load_summary(summary, ion)
        series = load_series(os.path.join(args.res_dir, f"si_isfet_{ion}_series.csv"))
        cps = load_cps(cps_path, ion, "vocd")
        ylim = ylim_override.get(ion, auto_ylim(best))

        with open(os.path.join(args.out_dir, f"figS_isfet_{ion}_panelB.csv"), "w",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "beta_comp", "beta_lo", "beta_hi", "n_cp", "runtime_ms"])
            for m in list(BASELINE_STYLE) + ["none"] + list(METHOD_STYLE):
                if m in best:
                    r = best[m]
                    w.writerow([m] + [r[k] for k in ("beta_corr", "beta_lo", "beta_hi",
                                                     "n_cp", "runtime_ms")])

        fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
        ax_a = ax_mm(fig, **P["a"])
        panel_a(ax_a, ion, series, cps, args.burn_in, args.legend_a)
        plabel(fig, "a", dx_mm=-10.0)
        ax_b = ax_mm(fig, **P["b"])
        panel_b(ax_b, best, ylim, cp_labels=not args.no_cp_labels)
        plabel(fig, "b", dx_mm=-10.0)

        out = os.path.join(args.out_dir, f"figS_isfet_{ion}.pdf")
        fig.savefig(out)
        png = os.path.splitext(out)[0] + ".png"
        fig.savefig(png, dpi=300)
        plt.close(fig)

        print(f"\n{ion.upper()}: wrote {out} and {png}  ({W_mm:g} x {H_mm:g} mm)")
        print(f"  VOCD change-points at days {cps}; "
              f"uncompensated slope {best['none']['beta_corr']:+.3f} mV/day")
        for m in METHOD_STYLE:
            if m in best:
                r = best[m]
                print(f"  {m:15s} beta_comp {r['beta_corr']:+.4f} "
                      f"[{r['beta_lo']:+.4f}, {r['beta_hi']:+.4f}] mV/day  "
                      f"{int(r['n_cp']):>2d} CP  {r['runtime_ms']:>9.1f} ms")


if __name__ == "__main__":
    main()
