"""
SI figures (Nature-Communications style) — MOX gas-sensor baseline re-anchoring
on diacetyl (0.1 ppm and 1 ppm).  One figure per analyte, two panels each,
same style, palette and panels as ``vocd_fig4_gas.py`` (panels c and d there)
and same layout scheme as ``vocd_figS_isfet_na_k.py``:

  a  how VOCD's compensation error evolves over the record: one curve per
     test day of the relative error (R_1 - R_comp,t) / R_1 in %, vs time in
     cycle, coloured from dark blue (day 2) to orange (day 39), lightly
     smoothed for display, over the three stage bands of the day-1 cycle.
  b  MAE (kOhm) vs runtime (ms, log) for the full panel with 95% block-
     bootstrap CIs over days and the oracle as a horizontal line.  Near-
     coincident markers are spread in x for visibility (say so in the
     caption).  Dm-BOCD is drawn as run if it is in the best CSV; otherwise
     it is an ESTIMATE: MAE and CI copied from BOCD, runtime = CHAMP's runtime
     here x (Dm-BOCD / CHAMP) as plotted in Fig. 4d (read from
     fig4_panelD.csv), so the two slow rows keep the Fig. 4 proportion.  Say
     so in the caption.

Inputs (all produced by examples/gas_data_v3_si.py; NOTHING is recomputed):
  gas_v3_<tag>_best.csv        one row per method (panel b)
  gas_v3_<tag>_resid_all.csv   reference - compensated on every test day,
                               with the day-1 'ref' and 'stage' columns (panel a)
with <tag> = diacetyl_0_1ppm, diacetyl_1ppm in --res-dir, plus
  fig4_panelD.csv              (from vocd_fig4_gas.py) the plotted Fig. 4d
                               runtimes of Dm-BOCD and CHAMP, for the estimate

LAYOUT: sizes and positions in millimetres from the top-left corner, at final
print size (Nature Portfolio: 180 mm wide, text 7 pt max / 5 pt min, panel
letters 8 pt bold).  Every text size derives from --base-font.

    python examples/vocd_figS_gas_diacetyl.py
    python examples/vocd_figS_gas_diacetyl.py --tags diacetyl_1ppm --ylim-b diacetyl_1ppm 15 45
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO, "results", "gas_si")
OUTDIR = os.path.join(_REPO, "results", "gas_si", "figures")

# ── LAYOUT (mm, final print size; y from the top edge) ───────────────────────
LAYOUT = {
    "fig_mm": (180.0, 120.0),
    "cbar_w": 2.5, "cbar_gap": 1.5,          # panel a day colour bar
    "panels": {
        #      x      y      w      h
        "a": dict(x=14.0,  y=7.0, w=68.0, h=100.0),
        "b": dict(x=106.0, y=7.0, w=70.0, h=100.0),
    },
}
F = {}        # font tiers, filled by set_nature_style(base)

# ── palette: identical to Fig. 4 (Okabe-Ito methods) ─────────────────────────
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
STAGE_COL = {1: "#7fb3d5", 2: "#f0a860", 3: "#9fd39f"}
STAGE_NAME = {1: "baseline", 2: "gas injection", 3: "recovery"}
C_DAY0, C_DAY1 = "#0b2a6f", "#E69F00"     # colour ramp: dark blue (early) -> orange (late)
DODGE_LOG10 = 0.07     # markers closer than this in log10(runtime) and 0.5 kOhm are spread


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
    if not out:
        raise SystemExit(f"no rows in {path}")
    return out


def load_series(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([fnum(r[k]) for r in rows]) for k in rows[0].keys()}


def load_runtimes(path):
    """method -> plotted runtime (ms) from fig4_panelD.csv."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["method"]] = fnum(r["runtime_ms"])
    return out


def dm_bocd_estimate(best, fig4_rt, metric):
    """Dm-BOCD not run here: MAE and CI copied from BOCD; runtime = CHAMP's
    runtime x (Dm-BOCD / CHAMP) from the plotted Fig. 4d."""
    need = [m for m in ("dm-bocd", "champ") if not np.isfinite(fig4_rt.get(m, np.nan))]
    if need or "champ" not in best or "bocd" not in best:
        print(f"  ! no Dm-BOCD estimate: missing {need or ['champ/bocd in this best CSV']}")
        return None
    ratio = fig4_rt["dm-bocd"] / fig4_rt["champ"]
    est = {k: best["bocd"][k] for k in (metric, f"{metric}_lo", f"{metric}_hi")
           if k in best["bocd"]}
    est.update(runtime_ms=best["champ"]["runtime_ms"] * ratio,
               n_missed=best["bocd"].get("n_missed", np.nan), _est=True)
    print(f"  Dm-BOCD ESTIMATE: runtime {est['runtime_ms']:,.0f} ms (CHAMP x {ratio:.2f}, "
          f"the Fig. 4d ratio); {metric.upper()} and CI copied from BOCD")
    return est


# ── style helpers (shared with fig1 / fig2 / fig3 / fig4) ────────────────────
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


# ── panel a : VOCD relative error, every test day (Fig. 4c) ──────────────────
def panel_a(fig, ax, resid, method="vocd", smooth_s=30.0, sample_hz=1.0):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    t, ref = resid["t_min"], resid.get("ref")
    stage = resid.get("stage")
    if stage is not None:
        for s_ in (1, 2, 3):
            m = np.where(stage == s_)[0]
            if m.size:
                ax.axvspan(t[m[0]], t[m[-1]], color=STAGE_COL[s_], alpha=0.18, lw=0, zorder=0)
                ax.text((t[m[0]] + t[m[-1]]) / 2, 0.985, STAGE_NAME[s_],
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=F["tiny"], color="0.35")
    day_of = lambda k: int(re.findall(r"\d+", k.split(":")[-1])[0])
    cols = sorted((k for k in resid if k.startswith(f"resid:{method}:day")), key=day_of)
    if not cols or ref is None:
        raise SystemExit(f"[figS] no resid:{method}:dayN or 'ref' columns: re-run "
                         f"gas_data_v3_si.py with --resid-methods {method}")
    d_lo, d_hi = day_of(cols[0]), day_of(cols[-1])
    cmap, norm = day_cmap(), Normalize(d_lo, d_hi)
    n = int(round(smooth_s * sample_hz))
    for k in cols:
        ax.plot(t, smooth(100.0 * resid[k] / ref, n), color=cmap(norm(day_of(k))),
                lw=0.55, alpha=0.85, zorder=3)
    ax.axhline(0, color="0.2", lw=0.6, zorder=2)
    ax.set_xlabel("Time in cycle (min)")
    ax.set_ylabel("VOCD relative error (%) to the day-1 reference")
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.25, lw=0.4)
    p = LAYOUT["panels"]["a"]
    cax = ax_mm(fig, p["x"] + p["w"] + LAYOUT["cbar_gap"], p["y"], LAYOUT["cbar_w"], p["h"])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Day", fontsize=F["small"])
    cb.ax.tick_params(labelsize=F["small"], width=0.6, length=2)


# ── panel b : MAE vs runtime (Fig. 4d) ───────────────────────────────────────
def dodge_x(points):
    """Spread near-coincident markers horizontally (log axis) so none hides
    another; returns {method: x_plot}.  Say so in the caption."""
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
            print(f"  [figS b] markers dodged in x for visibility: {[m for m, _ in group]}")
        i = j
    return out


def auto_ylim(best, metric, headroom=0.75):
    """kOhm; span every plotted CI and the oracle, then leave the top for the legend."""
    k, vals = 1e3, []
    for m in list(METHOD_STYLE) + ["oracle"]:
        if m in best:
            vals += [best[m][c] / k for c in (metric, f"{metric}_lo", f"{metric}_hi")
                     if np.isfinite(best[m].get(c, np.nan))]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    return lo - 0.08 * span, hi + headroom * span


def panel_b(ax, best, metric, ylim, dm_marker="filled"):
    k = 1e3
    pts = {m: (best[m]["runtime_ms"], best[m][metric] / k) for m in METHOD_STYLE if m in best}
    xplot = dodge_x(pts)
    for m, (lab, col, mk) in METHOD_STYLE.items():
        if m not in best:
            continue
        y = best[m][metric] / k
        lo, hi = best[m].get(f"{metric}_lo", np.nan), best[m].get(f"{metric}_hi", np.nan)
        yerr = ([[max(0.0, y - lo / k)], [max(0.0, hi / k - y)]]
                if np.isfinite(lo) and np.isfinite(hi) else None)
        hollow = best[m].get("_est", False) and dm_marker == "hollow"
        ax.errorbar(xplot[m], y, yerr=yerr, fmt=mk, color=col, ms=4.5, capsize=2, elinewidth=0.7,
                    markerfacecolor="white" if hollow else col,
                    markeredgecolor=col if hollow else "0.2",
                    markeredgewidth=0.9 if hollow else 0.4, label=lab,
                    zorder=4 if m.startswith("vocd") else 3)
    if "oracle" in best and np.isfinite(best["oracle"][metric]):
        v = best["oracle"][metric] / k
        ax.axhline(v, color="0.45", ls="--", lw=0.7, zorder=1,
                   label=f"oracle, stage labels ({v:.1f})")
    ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8,
              borderaxespad=0.2)
    xs = list(xplot.values())
    if xs:
        ax.set_xlim(min(xs) / 2.0, max(xs) * 2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel(("MAE" if metric == "mae" else "RMSE") + " (kΩ) relative to the day-1 reference")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.4, which="both")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res-dir", default=RES, help="folder written by gas_data_v3_si.py")
    ap.add_argument("--tags", default="diacetyl_0_1ppm,diacetyl_1ppm",
                    help="analyte tags of the gas_v3_<tag>_*.csv files")
    ap.add_argument("--metric", choices=["mae", "rmse"], default="mae")
    ap.add_argument("--smooth-s", type=float, default=30.0,
                    help="moving-average window (s) for the panel-a curves; 0 = raw")
    ap.add_argument("--sample-hz", type=float, default=1.0)
    ap.add_argument("--ylim-b", nargs=3, action="append", metavar=("TAG", "LO", "HI"),
                    default=[], help="fix panel b's y-range (kOhm) for one analyte "
                                     "(repeatable); default spans every CI and the legend")
    ap.add_argument("--fig4-paneld", default=os.path.join(_REPO, "results", "fig4_gas",
                                                          "fig4_panelD.csv"),
                    help="Fig. 4d data (vocd_fig4_gas.py): Dm-BOCD / CHAMP runtime ratio")
    ap.add_argument("--dm-bocd-marker", choices=["hollow", "filled"], default="filled",
                    help="how the Dm-BOCD estimate is drawn (match Fig. 4d)")
    ap.add_argument("--base-font", type=float, default=7.0,
                    help="base font size in pt at final print size (journal max 7)")
    ap.add_argument("--out-dir", default=OUTDIR,
                    help="writes figS_gas_<tag>.pdf (+ .png preview, + panel-b CSV)")
    args = ap.parse_args()
    set_nature_style(args.base_font)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)
    ylim_override = {t: (float(lo), float(hi)) for t, lo, hi in args.ylim_b}
    W_mm, H_mm = LAYOUT["fig_mm"]
    P = LAYOUT["panels"]
    fig4_rt = load_runtimes(args.fig4_paneld) if os.path.exists(args.fig4_paneld) else {}
    if not fig4_rt:
        print(f"! {args.fig4_paneld} not found: run vocd_fig4_gas.py first for the "
              f"Dm-BOCD estimate")

    for tag in [s.strip() for s in args.tags.split(",") if s.strip()]:
        best = load_best(os.path.join(args.res_dir, f"gas_v3_{tag}_best.csv"))
        resid = load_series(os.path.join(args.res_dir, f"gas_v3_{tag}_resid_all.csv"))
        print(f"\n{tag}:")
        if "dm-bocd" not in best:
            est = dm_bocd_estimate(best, fig4_rt, args.metric)
            if est is not None:
                best["dm-bocd"] = est
        ylim = ylim_override.get(tag, auto_ylim(best, args.metric))

        with open(os.path.join(args.out_dir, f"figS_gas_{tag}_panelB.csv"), "w",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", args.metric, f"{args.metric}_lo", f"{args.metric}_hi",
                        "n_missed", "runtime_ms", "kind"])
            for m in list(METHOD_STYLE) + ["oracle"]:
                if m in best:
                    r = best[m]
                    w.writerow([m] + [r.get(c, "") for c in (args.metric, f"{args.metric}_lo",
                                                             f"{args.metric}_hi", "n_missed",
                                                             "runtime_ms")]
                               + ["estimated" if r.get("_est") else "real"])

        fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
        ax_a = ax_mm(fig, **P["a"])
        panel_a(fig, ax_a, resid, smooth_s=args.smooth_s, sample_hz=args.sample_hz)
        plabel(fig, "a", dx_mm=-10.0)
        ax_b = ax_mm(fig, **P["b"])
        panel_b(ax_b, best, args.metric, ylim, args.dm_bocd_marker)
        plabel(fig, "b", dx_mm=-10.0)

        out = os.path.join(args.out_dir, f"figS_gas_{tag}.pdf")
        fig.savefig(out)
        png = os.path.splitext(out)[0] + ".png"
        fig.savefig(png, dpi=300)
        plt.close(fig)

        print(f"  wrote {out} and {png}  ({W_mm:g} x {H_mm:g} mm)")
        ora = best.get("oracle", {}).get(args.metric, np.nan)
        for m in list(METHOD_STYLE) + ["oracle"]:
            if m in best:
                r = best[m]
                marg = (r[args.metric] / ora - 1) * 100 if np.isfinite(ora) and ora > 0 else np.nan
                print(f"  {m:15s} {args.metric.upper()} {r[args.metric] / 1e3:6.2f} kOhm "
                      f"[{r.get(args.metric + '_lo', np.nan) / 1e3:.2f}, "
                      f"{r.get(args.metric + '_hi', np.nan) / 1e3:.2f}]  "
                      f"{r['runtime_ms']:>10,.1f} ms  missed {int(r.get('n_missed', 0) or 0):>2d}  "
                      f"{marg:+5.1f}% vs oracle" + ("  (ESTIMATE)" if r.get("_est") else ""))


if __name__ == "__main__":
    main()