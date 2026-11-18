"""
ISFET Na+/K+ drift correction — VOCD only, reference-free self-consistency.

The H+ ISFET has a reference pH probe, so its drift can be scored directly
(see ``isfet_data.py``).  The Na+ and K+ channels have NO reference, so
RMSE-vs-reference is unavailable.  This script tests whether VOCD is still
useful on them with two reference-free ideas:

1. Detection (qualitative).  Run VOCD causally on each channel and report the
   drift events it flags.  We do NOT assume the channels share change-points —
   fouling/leaching need not hit every ion identically — so cross-channel
   timing is reported descriptively, not scored.

2. beta self-consistency (quantitative, reference-free).  A good drift
   correction should remove the *persistent* trend, leaving a stationary,
   mean-reverting analyte signal.  So we compare the residual drift rate
       beta_raw   = slope of the raw daily V_ion
       beta_corr  = slope of V_ion - E0_hat, VOCD's causal piecewise correction
   over the post-burn-in window.  |beta_corr| << |beta_raw| (ideally ~0) is
   evidence the drift was removed, and needs no reference.

Validation.  Include H+ (``--ions h,na,k``): there we independently know the
correction is good (low RMSE vs the probe in isfet_data.py), so beta_corr ~ 0 on
H+ vouches for reading beta_corr ~ 0 on Na+/K+ as genuine correction rather than
an artefact.

Caveat to state in the paper: a real long-term analyte trend (seasonal shift,
sustained discharge) would also register as residual drift and could be mistaken
for correction failure; beta self-consistency assumes the analyte is stationary
on the record's timescale.

The correction machinery, causal prefix replay and daily-mean loading match
``isfet_data.py`` / ``isfet_benchmark.py``.

Usage
-----
    python examples/isfet_na_k.py --data-dir data/isfet
    python examples/isfet_na_k.py --data-dir data/isfet --ions h,na,k
    python examples/isfet_na_k.py --data-dir data/isfet --fig results/isfet/na_k.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas:  pip install pandas")

import vocd

SAMPLE_MIN = 5.0
BURN_IN = 10
MIN_FIT = 4


# ── data: daily means of one ISFET channel (no reference needed) ──────

def load_daily_ion(directory: str, ion: str):
    f_ion = os.path.join(directory, f"{ion}_mv.csv")
    if not os.path.exists(f_ion):
        sys.exit(f"not found: {f_ion}")
    a = (pd.read_csv(f_ion, header=None, names=["timeslot", "mv"])
         .sort_values("timeslot").reset_index(drop=True).dropna())
    ts = a["timeslot"].to_numpy(float)
    days = (ts - ts[0]) * SAMPLE_MIN / (60.0 * 24.0)
    v = a["mv"].to_numpy(float)
    idx = np.floor(days).astype(int)
    dd, vd = [], []
    for d in range(int(idx.min()), int(idx.max()) + 1):
        m = idx == d
        if m.any():
            dd.append(d + 0.5)
            vd.append(float(np.mean(v[m])))
    return np.array(dd), np.array(vd)


# ── correction machinery (verbatim from isfet_benchmark.py) ───────────

def _line_at(t, y, at):
    if t.size < 2:
        return float(y[-1]) if y.size else 0.0
    if t.size < MIN_FIT:
        return float(np.mean(y))
    c = np.polyfit(t, y, 1)
    return float(np.polyval(c, at))


def est_from_cp_sets(dd, vd, cp_sets):
    out = np.empty(dd.size)
    for i in range(dd.size):
        start = 0
        for c in sorted(cp_sets[i]):
            if c <= i:
                start = c
            else:
                break
        if i - start + 1 < MIN_FIT:
            start = max(0, i - MIN_FIT + 1)
        out[i] = _line_at(dd[start:i + 1], vd[start:i + 1], dd[i])
    return out


# ── VOCD causal boundary sets by prefix replay ────────────────────────

def vocd_cp_sets(vd, fwer, backend="python"):
    sets = []
    for i in range(vd.size):
        try:
            r = vocd.detect(vd[:i + 1], backend=backend)
            cps = [int(c) for c in r.changepoints(fwer=fwer)]
        except Exception:
            cps = []
        sets.append(sorted(int(c) for c in cps))
    return sets


def slope(dd, y, mask):
    return float(np.polyfit(dd[mask], y[mask], 1)[0]) if mask.sum() > 2 else np.nan


def within_seg_rms(dd, vd, final_cps):
    """RMS of (V - per-segment line): the internal scatter VOCD leaves — a
    reference-free noise-floor / stationarity indicator."""
    bounds = [0] + sorted(int(c) for c in final_cps if 0 < c < vd.size) + [vd.size]
    res = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t, y = dd[a:b], vd[a:b]
        if t.size >= MIN_FIT:
            fit = np.polyval(np.polyfit(t, y, 1), t)
            res.extend((y - fit).tolist())
        elif t.size:
            res.extend((y - np.mean(y)).tolist())
    return float(np.sqrt(np.mean(np.square(res)))) if res else np.nan


# ── figure ────────────────────────────────────────────────────────────

def make_figure(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("! matplotlib not available; skipping figure")
        return
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, R in zip(axes, results):
        dd, vd, e0_hat = R["dd"], R["vd"], R["e0_hat"]
        ax.plot(dd, vd, color="0.4", lw=1.0, label="raw $V_{ion}$")
        ax.plot(dd, vd - e0_hat + np.mean(e0_hat), color="C0", lw=1.2,
                label="corrected (offset)")
        for c in R["cp_days"]:
            ax.axvline(c, color="C3", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylabel(f"{R['ion'].upper()}  (mV)")
        ax.legend(loc="best", fontsize=8, framealpha=0.6)
        ax.set_title(f"{R['ion'].upper()}: "
                     f"beta_raw={R['beta_raw']:+.3f} -> beta_corr={R['beta_corr']:+.3f} "
                     f"mV/day, {R['n_cp']} CPs", fontsize=9)
    axes[-1].set_xlabel("day")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote figure -> {path}")


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/isfet")
    ap.add_argument("--ions", default="na,k",
                    help="comma list; add 'h' to include the validation channel")
    ap.add_argument("--fwer", type=float, default=0.01, help="VOCD budget")
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--backend", choices=["python", "cpp", "auto"], default="auto")
    ap.add_argument("--fig", default="results/isfet/isfet_na_k.png")
    ap.add_argument("--out", default="results/isfet/isfet_na_k.csv")
    args = ap.parse_args()

    ions = [s.strip() for s in args.ions.split(",") if s.strip()]
    print(f"VOCD reference-free self-consistency — ions={ions}, "
          f"fwer={args.fwer:g}, burn-in {args.burn_in}d\n")

    results = []
    for ion in ions:
        dd, vd = load_daily_ion(args.data_dir, ion)
        sets = vocd_cp_sets(vd, args.fwer, backend=args.backend)
        e0_hat = est_from_cp_sets(dd, vd, sets)
        corrected = vd - e0_hat
        m = dd >= args.burn_in
        b_raw = slope(dd, vd, m)
        b_corr = slope(dd, corrected, m)
        final = sets[-1]
        cp_days = [float(dd[c]) for c in sorted(final) if 0 < c < dd.size]
        wrms = within_seg_rms(dd, vd, final)
        red = (1 - abs(b_corr) / abs(b_raw)) * 100 if b_raw not in (0.0, np.nan) else np.nan
        results.append({"ion": ion, "dd": dd, "vd": vd, "e0_hat": e0_hat,
                        "n_cp": len(final), "cp_days": cp_days,
                        "beta_raw": b_raw, "beta_corr": b_corr,
                        "drift_reduction_pct": red, "within_seg_rms": wrms,
                        "n_days": dd.size})

    # table
    print(f"{'ion':<5}{'days':>6}{'#CP':>5}{'beta_raw':>11}{'beta_corr':>11}"
          f"{'drift_red%':>12}{'wseg_RMS':>10}")
    print("-" * 60)
    for R in results:
        print(f"{R['ion'].upper():<5}{R['n_days']:>6}{R['n_cp']:>5}"
              f"{R['beta_raw']:>+11.3f}{R['beta_corr']:>+11.3f}"
              f"{R['drift_reduction_pct']:>11.1f}%{R['within_seg_rms']:>10.3f}")
    print("-" * 60)
    print("beta in mV/day over the post-burn-in window. |beta_corr| << |beta_raw|")
    print("(ideally ~0) = VOCD removed the persistent drift, reference-free.")
    print("CP days per channel:")
    for R in results:
        days = ", ".join(f"{d:.0f}" for d in R["cp_days"]) or "(none)"
        print(f"  {R['ion'].upper()}: {days}")
    print("Channels are NOT assumed to share CPs; timing is descriptive.")

    # csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["ion", "n_days", "n_cp", "beta_raw", "beta_corr",
            "drift_reduction_pct", "within_seg_rms", "cp_days"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for R in results:
            w.writerow({"ion": R["ion"], "n_days": R["n_days"], "n_cp": R["n_cp"],
                        "beta_raw": R["beta_raw"], "beta_corr": R["beta_corr"],
                        "drift_reduction_pct": R["drift_reduction_pct"],
                        "within_seg_rms": R["within_seg_rms"],
                        "cp_days": ";".join(f"{d:.1f}" for d in R["cp_days"])})
    print(f"\nwrote {len(results)} rows -> {args.out}")
    make_figure(results, args.fig)


if __name__ == "__main__":
    main()
