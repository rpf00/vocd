"""
SI — ISFET Na+ / K+ drift compensation with the full v3 panel, reference-free.

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)
    + the reference-free baselines none, M0 (online line), exp (exp+linear)

The H+ channel is scored against a co-located pH probe (``isfet_data_v3.py``).
Na+ and K+ have NO reference, so drift RMSE cannot be computed.  Instead every
detector supplies the boundaries for the SAME causal piecewise-line correction
(prefix replay, verbatim from v2/v3) and is scored reference-free:

  beta_corr     slope (mV/day) of the corrected signal over the post-burn-in
                window, with a 95% moving-block-bootstrap CI (7-day blocks).
                Persistent drift removed  =>  beta_corr ~ 0.
  drift_red_pct 1 - |beta_corr| / |beta_raw|.
  resid_var_ratio  var(corrected) / var(M0-corrected).  GUARD AGAINST
                OVER-FIRING: a detector that segments constantly makes the
                causal line track everything, so beta_corr -> 0 trivially while
                the analyte itself is compensated away.  A ratio well below 1
                together with many change-points flags that failure mode.
  wseg_rms      RMS of V minus the per-segment line on the final boundary set
                (descriptive: internal scatter left after segmentation).

Budgets are TRANSFERRED, not tuned.  Each detector runs at its H+ best budget,
read from ``isfet_v3_best.csv`` (optionally overridden by a corrected-hazard
re-run, as in ``vocd_fig3_isfet.py --best-extra``).  Same sensor, same
defaults, three ions.  ``--sweep`` additionally writes every budget's metrics
as a sensitivity DIAGNOSTIC; it never selects the reported row (minimising
|beta_corr| would reward over-firing).

Validation.  H+ is processed alongside Na+/K+ (``--ions h,na,k``, the
default).  On H+ the reference exists, so each detector's RMSE is computed in
the same run and the Spearman rank correlation between RMSE and the
reference-free metrics is reported.  Agreement licenses reading the Na+/K+
table quantitatively; disagreement (plausible: all H+ CIs overlap) means the
Na+/K+ rows should be reported descriptively.  The H+ daily loader (v2) and the
Na+/K+ loader below are also checked for identical binning.

Caveat for the Methods: beta self-consistency assumes the analyte is
stationary over the 90-day record.  The pH reference establishes this for H+;
nothing establishes it for Na+/K+, so a real seasonal ionic trend would read as
residual drift.

Outputs (all in --out-dir, default ``si_isfet/``):
  si_isfet_summary.csv           one row per (ion, method): budget, metrics,
                                 runtime, CP days; RMSE columns filled for H+
  si_isfet_{ion}_series.csv      per-day V, every method's E0_hat and corrected V
                                 (+ ref and true E0 for H+)
  si_isfet_cps.csv               every detector's final boundary set, per ion
  si_isfet_validation.csv        H+ per-method RMSE vs reference-free metrics
  si_isfet_validation_stats.csv  Spearman rho / p for each proxy
  si_isfet_sweep.csv             (--sweep only) metrics at every budget
  si_isfet_quicklook.png         raw / M0 / VOCD-corrected per ion, VOCD CPs

Usage::

    python examples/si_isfet_na_k.py --data-dir data/isfet --dsm-path ../DSM-bocd \\
        --h-best results/isfet/isfet_v3_best.csv
    python examples/si_isfet_na_k.py ... --h-best-extra results/isfet/isfet_v3_best_hazard.csv
    python examples/si_isfet_na_k.py ... --sweep          # + sensitivity diagnostic

Without ``--dsm-path`` / ``VOCD_DSM_PATH`` the BOCD / Dm-BOCD rows are skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas:  pip install pandas")

import vocd                                   # noqa: E402,F401
from alternative_models import REGISTRY       # noqa: E402
import isfet_data as v2                       # noqa: E402  (verbatim problem)
import isfet_data_v3 as v3                    # noqa: E402  (verbatim panel + routing)

BURN_IN, MIN_FIT = v3.BURN_IN, v3.MIN_FIT
SAMPLE_MIN = getattr(v2, "SAMPLE_MIN", 5.0)
est_from_cp_sets = v3.est_from_cp_sets
PANEL, TWINS, TWIN_METHODS = v3.PANEL, v3.TWINS, v3.TWIN_METHODS
HAZARD_METHODS = v3.HAZARD_METHODS
BASELINES = (("none", v3.est_none), ("M0", v3.est_M0), ("exp", v3.est_exp))

SUMMARY_COLS = ["ion", "method", "budget_kind", "budget", "n_cp", "cp_days",
                "beta_raw", "beta_corr", "beta_lo", "beta_hi", "drift_red_pct",
                "resid_var_ratio", "corr_std", "wseg_rms", "runtime_ms",
                "rmse_mv", "rmse_lo", "rmse_hi"]
SWEEP_COLS = ["ion", "method", "budget_kind", "budget", "n_cp", "beta_corr",
              "beta_lo", "beta_hi", "drift_red_pct", "resid_var_ratio",
              "wseg_rms", "runtime_ms", "rmse_mv"]


# ── data ──────────────────────────────────────────────────────────────

def load_daily_ion(directory, ion):
    """Daily means of one ISFET channel, no reference needed.  Same binning as
    v2.load_daily (checked against it on H+ at run time)."""
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


def load_channel(directory, ion):
    """(dd, vd, ph_ref, e0).  H+ goes through v2's loader so its rows match
    Fig. 3 exactly; Na+/K+ have ph_ref = e0 = None."""
    if ion != "h":
        dd, vd = load_daily_ion(directory, ion)
        return dd, vd, None, None
    dd, vd, ph_ref, e0 = v2.load_daily(directory, "h")
    dd2, vd2 = load_daily_ion(directory, "h")
    common, i1, i2 = np.intersect1d(np.round(dd, 3), np.round(dd2, 3),
                                    return_indices=True)
    if common.size == 0 or not np.allclose(vd[i1], vd2[i2], atol=1e-6):
        print("! WARNING: the Na+/K+ loader does NOT reproduce v2's H+ daily means; "
              "the channels are not binned identically")
    elif common.size != dd.size or dd.size != dd2.size:
        print(f"  loader check: H+ daily means identical on {common.size} shared days "
              f"(v2 {dd.size} d, ion loader {dd2.size} d; v2 keeps reference-covered days)")
    else:
        print("  loader check: Na+/K+ loader reproduces v2's H+ daily means exactly")
    return dd, vd, ph_ref, e0


# ── budgets transferred from the H+ run ───────────────────────────────

def load_budgets(path, extra=None):
    out = {}
    for p in (path, extra):
        if not p:
            continue
        if not os.path.exists(p):
            sys.exit(f"budget file not found: {p}  (run isfet_data_v3.py --sweep first)")
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                b = (r.get("budget") or "").strip()
                if b:
                    out[r["method"]] = float(b)       # later files override
    return out


def budget_kind(name):
    if name == "vocd":
        return "fwer"
    return "hazard" if name in HAZARD_METHODS else "alpha"


# ── reference-free metrics ────────────────────────────────────────────

def _slope(t, y):
    t = np.asarray(t, float) - np.mean(t)
    den = float(np.dot(t, t))
    return float(np.dot(t, y - np.mean(y)) / den) if den > 0 else np.nan


def slope_ci(t, y, block=7, n_boot=2000, seed=0):
    """95% percentile CI of the slope from a moving-block bootstrap over days
    (same block length and resample count as the H+ RMSE CIs)."""
    n = t.size
    if n < block + 1:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, size=k)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = _slope(t[idx], y[idx])
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def within_seg_rms(dd, vd, final_cps):
    bounds = [0] + sorted(int(c) for c in final_cps if 0 < c < vd.size) + [vd.size]
    res = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t, y = dd[a:b], vd[a:b]
        if t.size >= MIN_FIT:
            res.extend((y - np.polyval(np.polyfit(t, y, 1), t)).tolist())
        elif t.size:
            res.extend((y - np.mean(y)).tolist())
    return float(np.sqrt(np.mean(np.square(res)))) if res else np.nan


def ref_free_metrics(dd, vd, e0_hat, final_cps, m0_var, args, ci=True):
    m = dd >= args.burn_in
    corr = vd - e0_hat
    t, yc = dd[m], corr[m]
    b_raw, b_corr = _slope(t, vd[m]), _slope(t, yc)
    lo, hi = slope_ci(t, yc, args.block, args.n_boot) if ci else (np.nan, np.nan)
    red = ((1.0 - abs(b_corr) / abs(b_raw)) * 100.0
           if np.isfinite(b_raw) and b_raw != 0 else np.nan)
    var = float(np.var(yc))
    return {"beta_raw": b_raw, "beta_corr": b_corr, "beta_lo": lo, "beta_hi": hi,
            "drift_red_pct": red,
            "resid_var_ratio": var / m0_var if m0_var > 0 else np.nan,
            "corr_std": float(np.sqrt(var)),
            "wseg_rms": within_seg_rms(dd, vd, final_cps) if final_cps is not None else np.nan}


def ref_rmse(e0_hat, dd, vd, ph_ref, n_cp, args):
    """H+ only: the v3 scoring (reference-anchored drift RMSE + block CI)."""
    r = v3.score(e0_hat, vd, ph_ref, dd, args.burn_in, n_cp=n_cp)
    return {"rmse_mv": r["rmse_mv"], "rmse_lo": r["rmse_lo"], "rmse_hi": r["rmse_hi"]}


# ── evaluation ────────────────────────────────────────────────────────

def run_detector(cp_sets, name, dd, vd, budget, vocd_backend=None):
    vocd_backend = vocd_backend or v3.VOCD_FAIR_BACKEND
    t0 = time.perf_counter()
    sets = cp_sets(name, vd, budget, vocd_backend=vocd_backend)
    e0_hat = est_from_cp_sets(dd, vd, sets)
    ms = (time.perf_counter() - t0) * 1000.0
    return e0_hat, sets, ms


def make_row(ion, method, budget, kind, ch, e0_hat, sets, ms, m0_var, args,
             final=None, ci=True):
    dd, vd, ph_ref, _ = ch
    if sets is not None:
        final = sets[-1]
    n_cp = len(final) if (sets is not None) else np.nan
    row = {"ion": ion, "method": method, "budget_kind": kind,
           "budget": budget if budget is not None else "",
           "n_cp": n_cp, "runtime_ms": ms,
           "cp_days": ";".join(f"{dd[c]:.1f}" for c in (final or []) if 0 < c < dd.size),
           "rmse_mv": np.nan, "rmse_lo": np.nan, "rmse_hi": np.nan,
           "_e0_hat": np.asarray(e0_hat, float), "_sets": sets}
    row.update(ref_free_metrics(dd, vd, e0_hat, final, m0_var, args, ci=ci))
    if ph_ref is not None:
        row.update(ref_rmse(e0_hat, dd, vd, ph_ref, n_cp, args))
    return row


def eval_ion(ion, ch, wanted, budgets, cp_sets, args):
    dd, vd, _, _ = ch
    m = dd >= args.burn_in
    m0_hat = v3.est_M0(dd, vd)
    m0_var = float(np.var((vd - m0_hat)[m]))
    rows = {}

    if not args.no_baselines:
        for label, fn in BASELINES:
            t0 = time.perf_counter()
            e0_hat = fn(dd, vd)
            ms = (time.perf_counter() - t0) * 1000.0
            final = [] if label == "M0" else None    # M0 = one line, no boundaries
            rows[label] = make_row(ion, label, None, "—", ch, e0_hat, None, ms,
                                   m0_var, args, final=final)

    for name in wanted:
        b, kind = budgets[name], budget_kind(name)
        print(f"  [{name}] {kind}={b:g} ...", end="", flush=True)
        e0_hat, sets, ms = run_detector(cp_sets, name, dd, vd, b)
        rows[name] = make_row(ion, name, b, kind, ch, e0_hat, sets, ms, m0_var, args)
        print(f" {len(sets[-1])} CPs, beta_corr={rows[name]['beta_corr']:+.3f} mV/d,"
              f" {ms:.0f} ms")

        if name in TWINS:                         # compiled timing twin, same budget
            label, method = TWINS[name]
            try:
                if method is None:
                    e0t, st, mst = run_detector(cp_sets, "vocd", dd, vd, b,
                                                vocd_backend=v3.VOCD_FAST_BACKEND)
                else:
                    e0t, st, mst = run_detector(cp_sets, method, dd, vd, b)
            except Exception as e:                # noqa: BLE001
                print(f"  ! {label}: compiled twin unavailable ({e}); row skipped")
                continue
            if st == sets:
                print(f"    {label}: boundary sets identical (only runtime differs)")
            else:
                nd = sum(1 for x, y in zip(st, sets) if x != y)
                print(f"  ! WARNING: {label} differs from {name} on {nd}/{len(sets)} "
                      "prefixes — NOT a pure timing twin")
            rows[label] = make_row(ion, label, b, kind, ch, e0t, st, mst, m0_var,
                                   args, ci=False)
            for k in ("beta_lo", "beta_hi"):      # identical boundaries => same CI
                rows[label][k] = rows[name][k]
    return rows, m0_var


def sweep_ion(ion, ch, wanted, cp_sets, m0_var, args):
    """Sensitivity diagnostic only: metrics at every budget, never selection."""
    alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
    hazard_grid = [float(v) for v in args.hazard_grid.split(",") if v.strip()]
    dm_grid = [float(v) for v in args.dm_hazard_grid.split(",") if v.strip()]
    dd, vd, _, _ = ch
    out = []
    for name in wanted:
        grid = (dm_grid if name == "dm-bocd" else
                hazard_grid if name in HAZARD_METHODS else alpha_grid)
        kind = budget_kind(name)
        print(f"  sweep [{name}] ({kind}) ", end="", flush=True)
        for b in grid:
            e0_hat, sets, ms = run_detector(cp_sets, name, dd, vd, b)
            r = make_row(ion, name, b, kind, ch, e0_hat, sets, ms, m0_var, args,
                         ci=args.sweep_ci)
            out.append({k: r[k] for k in SWEEP_COLS})
            print(".", end="", flush=True)
        print()
    return out


# ── validation on H+ ──────────────────────────────────────────────────

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x[ok], y[ok])
        return float(r[0]), float(r[1])
    except ImportError:
        rx = pd.Series(x[ok]).rank().to_numpy()
        ry = pd.Series(y[ok]).rank().to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1]), np.nan


def validate_h(rows, wanted, out_dir):
    dets = [m for m in wanted if m in rows]
    rmse = [rows[m]["rmse_mv"] for m in dets]
    proxies = {
        "|beta_corr|": [abs(rows[m]["beta_corr"]) for m in dets],
        "wseg_rms": [rows[m]["wseg_rms"] for m in dets],
        "resid_var_ratio": [rows[m]["resid_var_ratio"] for m in dets],
    }
    with open(os.path.join(out_dir, "si_isfet_validation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "rmse_mv", "rmse_lo", "rmse_hi", "n_cp"] + list(proxies))
        for i, m in enumerate(dets):
            r = rows[m]
            w.writerow([m, r["rmse_mv"], r["rmse_lo"], r["rmse_hi"], r["n_cp"]]
                       + [proxies[k][i] for k in proxies])
    stats = []
    print(f"\nH+ validation — Spearman(RMSE, proxy) over {len(dets)} detectors "
          "(Python rows only; twins and baselines excluded):")
    for k, v in proxies.items():
        rho, p = spearman(rmse, v)
        stats.append({"proxy": k, "n": len(dets), "spearman_rho": rho, "p_value": p})
        print(f"  {k:<16} rho = {rho:+.2f}   p = {p:.3g}")
    print("  Positive rho with a small p: the proxy ranks methods as the reference "
          "does,\n  so the Na+/K+ table can be read quantitatively.  Otherwise report "
          "Na+/K+\n  descriptively (CP days, drift reduction, analyte retained).  With "
          f"n = {len(dets)},\n  p < 0.05 needs |rho| of roughly 0.7 or more.")
    with open(os.path.join(out_dir, "si_isfet_validation_stats.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["proxy", "n", "spearman_rho", "p_value"])
        w.writeheader()
        w.writerows(stats)


# ── IO ────────────────────────────────────────────────────────────────

def fmt(x, spec):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        w = "".join(ch for ch in spec.split(".")[0] if ch.isdigit())
        return format("nan", f">{w}") if w else "nan"
    return format(x, spec)


def print_table(ion, order, rows, has_ref):
    hdr = (f"{'method':<16}{'budget':>9}{'#CP':>5}{'beta_corr':>11}{'95% CI':>18}"
           f"{'drift_red%':>11}{'var_ratio':>10}{'wseg_RMS':>10}{'ms':>10}")
    if has_ref:
        hdr += f"{'RMSE_mV':>9}"
    print(f"\n{ion.upper()}  (beta_raw = {rows[order[0]]['beta_raw']:+.3f} mV/day)")
    print(hdr)
    print("-" * len(hdr))
    for name in order:
        r = rows[name]
        b = r["budget"]
        bcol = f"{b:>9.4g}" if b != "" else f"{'—':>9}"
        ncp = "—" if not np.isfinite(r["n_cp"]) else f"{int(r['n_cp'])}"
        ci = (f"[{fmt(r['beta_lo'], '+.3f')}, {fmt(r['beta_hi'], '+.3f')}]"
              if np.isfinite(r["beta_lo"]) else "—")
        line = (f"{name:<16}{bcol}{ncp:>5}{fmt(r['beta_corr'], '+11.3f')}{ci:>18}"
                f"{fmt(r['drift_red_pct'], '10.1f')}%{fmt(r['resid_var_ratio'], '10.3f')}"
                f"{fmt(r['wseg_rms'], '10.3f')}{fmt(r['runtime_ms'], '10.1f')}")
        if has_ref:
            line += fmt(r["rmse_mv"], "9.3f")
        print(line)
    print("-" * len(hdr))


def write_series(path, order, rows, ch):
    dd, vd, ph_ref, e0 = ch
    cols = ["day", "v_mv"]
    if ph_ref is not None:
        cols += ["ref_mv", "e0_true_mv"]
    for name in order:
        cols += [f"e0_hat:{name}", f"vcorr:{name}"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(dd.size):
            row = [f"{dd[i]:.1f}", f"{vd[i]:.4f}"]
            if ph_ref is not None:
                row += [f"{-v3.S_NERNST * ph_ref[i]:.4f}", f"{e0[i]:.4f}"]
            for name in order:
                eh = rows[name]["_e0_hat"][i]
                row += [f"{eh:.4f}", f"{vd[i] - eh:.4f}"]
            w.writerow(row)


def make_quicklook(per_ion, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("! matplotlib not available; skipping quick-look figure")
        return
    n = len(per_ion)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), sharex=True, squeeze=False)
    for ax, (ion, ch, rows) in zip(axes[:, 0], per_ion):
        dd, vd = ch[0], ch[1]
        ax.plot(dd, vd - vd[0], color="0.55", lw=1.0, label="raw")
        for name, col in (("M0", "C1"), ("vocd", "C0")):
            if name in rows:
                c = vd - rows[name]["_e0_hat"]
                ax.plot(dd, c - c[0], color=col, lw=1.1, label=f"{name} corrected")
        if "vocd" in rows and rows["vocd"]["_sets"]:
            for c in rows["vocd"]["_sets"][-1]:
                if 0 < c < dd.size:
                    ax.axvline(dd[c], color="C3", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylabel(f"{ion.upper()} (mV, rel.)")
        ax.legend(loc="best", fontsize=7, framealpha=0.6)
    axes[-1, 0].set_xlabel("day")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote quick-look figure -> {path}")


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/isfet")
    ap.add_argument("--ions", default="h,na,k",
                    help="comma list; keep 'h' for the RMSE-vs-proxy validation")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--out-dir", default="si_isfet")
    # transferred budgets
    ap.add_argument("--h-best", default="results/isfet/isfet_v3_best.csv",
                    help="H+ best-operating-point CSV from isfet_data_v3.py --sweep")
    ap.add_argument("--h-best-extra", default=None,
                    help="optional CSV whose rows OVERRIDE --h-best (e.g. corrected "
                         "hazard-grid re-run), as in vocd_fig3_isfet.py --best-extra")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="fallback fwer/alpha for a method missing from --h-best")
    ap.add_argument("--hazard", type=float, default=30.0,
                    help="fallback hazard timescale (days) for a missing method")
    # problem settings (identical to the H+ run)
    ap.add_argument("--min-seg", type=int, default=4)
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    ap.add_argument("--block", type=int, default=7, help="bootstrap block (days)")
    ap.add_argument("--n-boot", type=int, default=2000)
    # CHAMP / NUNC a-priori settings (identical to isfet_data_v3.py)
    ap.add_argument("--champ-gamma", type=float, default=0.05)
    ap.add_argument("--champ-em-iters", type=int, default=2)
    ap.add_argument("--champ-max-particles", type=int, default=None)
    ap.add_argument("--champ-lag", type=int, default=20)
    ap.add_argument("--champ-confirm", type=int, default=None)
    ap.add_argument("--nunc-window", type=int, default=50)
    ap.add_argument("--nunc-sims", type=int, default=20000)
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full")
    # sensitivity diagnostic
    ap.add_argument("--sweep", action="store_true",
                    help="also write metrics at every budget (diagnostic; slow with "
                         "dm-bocd: ~70 s per point per ion)")
    ap.add_argument("--sweep-ci", action="store_true",
                    help="bootstrap beta CIs inside the sweep too (slower)")
    ap.add_argument("--alpha-grid",
                    default="1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1")
    ap.add_argument("--hazard-grid", default="2,5,10,20,30,50,80,150")
    ap.add_argument("--dm-hazard-grid", default="2,5,10,20,30,50,80,150")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    os.makedirs(args.out_dir, exist_ok=True)

    # panel (same validation as v3)
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"vocd"} | set(REGISTRY)
    for m in wanted:
        if m not in known:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(known)}")
    wanted = [m for m in wanted if m not in TWIN_METHODS]     # twins are paired
    needs = [m for m in wanted if m in REGISTRY and REGISTRY[m][1]]
    if needs and not dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}")
        wanted = [m for m in wanted if m not in needs]

    # transferred budgets
    h_budgets = load_budgets(args.h_best, args.h_best_extra)
    budgets = {}
    for m in wanted:
        if m in h_budgets:
            budgets[m] = h_budgets[m]
        else:
            budgets[m] = args.hazard if m in HAZARD_METHODS else args.alpha
            print(f"! {m}: not in the H+ best CSV; fallback {budget_kind(m)}={budgets[m]:g}")
        if m in HAZARD_METHODS and budgets[m] < 2:
            print(f"! WARNING: {m} hazard timescale {budgets[m]:g} < 2 is invalid "
                  "(per-step change probability > 1/2); re-run the H+ sweep on the "
                  "corrected grid and pass it via --h-best-extra")

    # compiled engines (one-off, outside the timers)
    if args.vocd_cpp == "full":
        import vocd._fastdetect as fastdetect
        fastdetect.enable()
    elif args.vocd_cpp == "gate":
        import vocd._fastgate as fastgate
        fastgate.enable()
    if any(m in TWINS for m in wanted):
        from alternative_models import _cppalarms
        _cppalarms._load()

    cp_sets = v3.make_cp_sets_fn(args, dsm_path)
    ions = [s.strip().lower() for s in args.ions.split(",") if s.strip()]

    print(f"\nSI ISFET drift compensation, reference-free — ions={ions}")
    print(f"methods: {', '.join(wanted)}  (+ twins "
          f"{', '.join(TWINS[m][0] for m in wanted if m in TWINS)})")
    print("budgets transferred from the H+ run, no re-tuning: "
          + ", ".join(f"{m}={budgets[m]:g}" for m in wanted))
    print(f"burn-in {args.burn_in} d, min_seg {args.min_seg} d, "
          f"beta CI = {args.block}-day block bootstrap x {args.n_boot}\n")

    summary, sweep_rows, cps_rows, per_ion = [], [], [], []
    h_rows = None
    for ion in ions:
        print(f"── {ion.upper()} " + "─" * 60)
        ch = load_channel(args.data_dir, ion)
        print(f"  {ch[0].size} daily means"
              + ("  (reference available: RMSE computed for validation)"
                 if ch[2] is not None else "  (no reference)"))
        rows, m0_var = eval_ion(ion, ch, wanted, budgets, cp_sets, args)

        det_order = [m for m in v3.insert_twins(wanted) if m in rows]
        order = [m for m in ("none", "M0", "exp") if m in rows] + det_order
        print_table(ion, order, rows, ch[2] is not None)

        summary += [{k: rows[n][k] for k in SUMMARY_COLS} for n in order]
        for n in det_order:
            for c in rows[n]["_sets"][-1]:
                cps_rows.append([ion, n, c, f"{ch[0][c]:.1f}"])
        write_series(os.path.join(args.out_dir, f"si_isfet_{ion}_series.csv"),
                     order, rows, ch)
        per_ion.append((ion, ch, rows))
        if ion == "h":
            h_rows = rows
        if args.sweep:
            sweep_rows += sweep_ion(ion, ch, wanted, cp_sets, m0_var, args)

    # outputs
    p = os.path.join(args.out_dir, "si_isfet_summary.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {len(summary)} rows -> {p}")
    p = os.path.join(args.out_dir, "si_isfet_cps.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ion", "method", "cp_index", "cp_day"])
        w.writerows(cps_rows)
    print(f"wrote final boundary sets -> {p}")
    print(f"wrote per-day series -> {args.out_dir}/si_isfet_{{{','.join(ions)}}}_series.csv")
    if sweep_rows:
        p = os.path.join(args.out_dir, "si_isfet_sweep.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SWEEP_COLS)
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"wrote {len(sweep_rows)} sweep rows (diagnostic) -> {p}")

    if h_rows is not None:
        validate_h(h_rows, wanted, args.out_dir)
    else:
        print("\n! 'h' not in --ions: no RMSE-vs-proxy validation; read the Na+/K+ "
              "rows descriptively")

    print("\nCPs are NOT assumed shared across ions; timing is descriptive.  "
          "beta self-consistency\nassumes a stationary analyte over the record "
          "(established for H+ by the pH probe only).")
    if not args.no_fig:
        make_quicklook(per_ion, os.path.join(args.out_dir, "si_isfet_quicklook.png"))


if __name__ == "__main__":
    main()
