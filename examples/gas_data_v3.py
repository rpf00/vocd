"""
MOX gas-sensor drift-correction benchmark, v3 — the final paper panel (Fig. 4c/d).

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)
    + the oracle (true cycle_stage baseline; the non-causal ceiling)

Same task, data, correction and scoring as ``gas_data.py`` (imported verbatim
from it): on each day t > 1 a detector's FIRST change-point on the raw cycle
R_t(tau) marks the baseline -> gas onset; the baseline before it is re-anchored
to day 1 by one constant; the corrected cycle is scored against the day-1
cycle by MAE (and RMSE) over tau, averaged over the test days; the sweep
MINIMISES mean MAE.  What this file changes is the panel and the plumbing,
identical to ``synthetic_data_v3.py`` / ``isfet_data_v3.py``:

* R-FOCuS dropped; FOCuS, NUNC and CHAMP added through the same
  ``alternative_models.REGISTRY`` entries and budget kinds (alarm methods on
  alpha, BOCD / Dm-BOCD / CHAMP on the hazard grid, VOCD on fwer).
* Three compiled timing twins paired at their Python row's best budget, with
  the per-day onsets asserted identical (never swept on their own).
* Dm-BOCD swept on its own coarse grid (``--dm-hazard-grid``; 'inherit' = the
  old protocol).  It is the slow row (~minutes per grid point here).
* 95% moving-block-bootstrap CIs over the per-day MAE (5-day blocks, 2000
  resamples; the paper's Methods) added to every row -> Fig. 4d error bars.
* ``--series-out`` dumps the per-sample series behind Fig. 4c: the raw cycles
  of the days in ``--show-days`` with their stage labels, and the residual
  (day-1 reference minus corrected) on ``--resid-day`` for every method at its
  best point.  ``--onsets-out`` dumps every method's onset and the offset it
  subtracted per day, so raw - offset reproduces any corrected cycle.
  ``--resid-all-out`` dumps reference - corrected on EVERY test day for
  ``--resid-methods`` (default vocd): the family of residual curves in Fig. 4c.

A-priori settings for THIS record (declare them with ``min_seg = 30``): NUNC
window W = max(2 x VOCD's gate window, 2 x min_seg) = 60 -- the 50 of the
other runs cannot hold two 30-sample blocks, so no split would be tested;
CHAMP gamma = 0.05, confirmation = min_seg, lag = 20 (= BOCD's causal
extraction spacing), and a particle cap of 100 (the published algorithm; the
exact recursion is O(n^3) per cycle and these cycles are ~1200 points -- on
the synthetic task the cap gave identical detections).

Why MAE and not RMSE (both are written out).  Within the gas-injection stage
the residual between a corrected cycle and the day-1 cycle is dominated by
day-to-day changes in the sensor's RESPONSE SHAPE, which an additive baseline
offset cannot remove by design and which are the same for every detector.
RMSE squares those systematic excursions and so measures response-shape
mismatch; MAE weights every sample linearly, so the baseline and recovery
stages -- where the correction actually acts and where detectors differ --
carry their proportionate share.  MAE is also the metric of the original
benchmark and of the oracle comparison.

Usage::

    python examples/gas_data_v3.py --sweep --dsm-path ../DSM-bocd --analyte Ethanol \\
        --out results/gas/gas_v3_best.csv --sweep-out results/gas/gas_v3_sweep.csv \\
        --series-out results/gas/gas_v3_series.csv --onsets-out results/gas/gas_v3_onsets.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import vocd                                   # noqa: E402
from alternative_models import REGISTRY       # noqa: E402
import gas_data as v2                         # noqa: E402  (verbatim task)

MIN_BASE = v2.MIN_BASE
load_all_days = v2.load_all_days
base_mask_from_cps, correct_measurement, curve_metrics = (
    v2.base_mask_from_cps, v2.correct_measurement, v2.curve_metrics)
VOCD_FAIR_BACKEND, VOCD_FAST_BACKEND = v2.VOCD_FAIR_BACKEND, v2.VOCD_FAST_BACKEND
ANALYTE_LABELS = v2.ANALYTE_LABELS
_DEFAULT_GAS_DIR = v2._DEFAULT_GAS_DIR

PANEL = "vocd,bocd,dm-bocd,focus,np-focus,nunc,chad,champ"
HAZARD_METHODS = ("bocd", "dm-bocd", "champ")
TWINS = {
    "vocd":     ("vocd (cpp)",     None),
    "focus":    ("focus (cpp)",    "focus-cpp"),
    "np-focus": ("np-focus (cpp)", "np-focus-cpp"),
}
TWIN_METHODS = {m for _, m in TWINS.values() if m}
METRIC_KEYS = ["mae", "mae_lo", "mae_hi", "rmse", "n_days", "n_missed", "runtime_ms"]


# ── statistics ────────────────────────────────────────────────────────

def block_bootstrap_mean(x, block=5, n_boot=2000, seed=0):
    """95% percentile interval of the mean from a moving-block bootstrap over
    days (the paper's Methods: 5-day blocks, 2,000 resamples)."""
    x = np.asarray(x, float)
    n = x.size
    if n < block + 1:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, size=k)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = np.mean(x[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ── detector -> change-points on one cycle, unified routing ───────────

def make_cps_fn(args, dsm_path):
    champ_kw = dict(gamma=args.champ_gamma, em_iters=args.champ_em_iters,
                    max_particles=args.champ_max_particles, extract="causal",
                    lag=args.champ_lag, confirm=args.champ_confirm)
    nunc_kw = dict(window=args.nunc_window, n_sims=args.nunc_sims)

    def cps(name, r, budget, vocd_backend=VOCD_FAIR_BACKEND, nunc_cache=None):
        if name == "vocd":
            try:
                return [int(c) for c in vocd.detect(r, backend=vocd_backend)
                        .changepoints(fwer=budget)]
            except Exception:
                return []
        fn, needs_dsm = REGISTRY[name]
        if needs_dsm:
            return fn(r, mean0=None, dsm_path=dsm_path, extract=args.bocd_extract,
                      hazard_timescale=budget)
        if name == "champ":
            return fn(r, hazard_timescale=budget, min_seg=args.min_seg, **champ_kw)
        if name == "nunc":
            return fn(r, alpha=budget, min_seg=args.min_seg, cache=nunc_cache, **nunc_kw)
        return fn(r, alpha=budget, min_seg=args.min_seg)
    return cps


def eval_method(cps_fn, name, days, budget, args, vocd_backend=VOCD_FAIR_BACKEND):
    """Per-day correction of every test day; mean MAE / RMSE, CI, onsets."""
    d0, r0, cs0 = days[0]
    nunc_cache = {} if name == "nunc" else None    # one calibration per record
    if name == "oracle":
        bm0 = cs0 == 1
    else:
        bm0 = base_mask_from_cps(cps_fn(name, r0, budget, vocd_backend, nunc_cache), len(r0))
    if bm0 is None or bm0.sum() < 5:               # day-1 fallback: true baseline
        bm0 = cs0 == 1
    ref_base_mean = float(r0[bm0].mean())
    ref_r = r0
    offsets = {d0: (int(bm0.sum()), 0.0)}      # day 1: onset used for the reference, no correction

    t0 = time.perf_counter()
    maes, rmses, onsets, corrected = [], [], [], {}
    for d, r, cs in days[1:]:
        if name == "oracle":
            bm = cs == 1
            onset = int(np.argmax(~bm)) if (~bm).any() else -1
        else:
            c = cps_fn(name, r, budget, vocd_backend, nunc_cache)
            bm = base_mask_from_cps(c, len(r))
            onset = int(bm.sum()) if bm is not None else -1
        rc = correct_measurement(r, bm, ref_base_mean)
        mae, rmse = curve_metrics(rc, ref_r)
        maes.append(mae); rmses.append(rmse); onsets.append((d, onset))
        corrected[d] = rc
        offsets[d] = (onset, float(r[0] - rc[0]))      # the constant actually subtracted
    ms = (time.perf_counter() - t0) * 1000.0
    lo, hi = block_bootstrap_mean(maes)
    return {"mae": float(np.mean(maes)), "mae_lo": lo, "mae_hi": hi,
            "rmse": float(np.mean(rmses)), "n_days": len(maes),
            "n_missed": int(sum(1 for _, o in onsets if o < 0)),
            "runtime_ms": ms, "_maes": tuple(maes), "_onsets": onsets,
            "_corrected": corrected, "_offsets": offsets,
            "_ref_base_mean": ref_base_mean}


def paired_row(cps_fn, base, days, b_star, args, base_row):
    label, method = TWINS[base]
    try:
        if method is None:
            r = eval_method(cps_fn, "vocd", days, b_star, args, vocd_backend=VOCD_FAST_BACKEND)
        else:
            r = eval_method(cps_fn, method, days, b_star, args)
    except Exception as e:
        print(f"! {label}: compiled twin unavailable ({e}); row skipped")
        return None
    if r["_onsets"] == base_row["_onsets"]:
        print(f"  {label}: onsets identical to {base} on every day (only runtime differs)")
    else:
        nd = sum(1 for a, b in zip(r["_onsets"], base_row["_onsets"]) if a != b)
        print(f"! WARNING: {base} python vs cpp onsets differ on {nd}/{len(r['_onsets'])} "
              f"days — '{label}' is NOT a pure timing twin")
    return r


# ── IO ────────────────────────────────────────────────────────────────

def print_table(order, rows, budgets):
    print(f"\n{'method':<16}{'budget':>10}{'MAE':>11}{'95% CI':>22}{'RMSE':>11}{'days':>6}"
          f"{'miss':>6}{'Runtime(ms)':>13}")
    print("-" * 95)
    for name in order:
        r = rows[name]
        b = budgets.get(name)
        bcol = f"{b:>10.4g}" if b is not None else f"{'—':>10}"
        ci = f"[{r['mae_lo']:.1f}, {r['mae_hi']:.1f}]"
        print(f"{name:<16}{bcol}{r['mae']:>11.1f}{ci:>22}{r['rmse']:>11.1f}{r['n_days']:>6}"
              f"{r['n_missed']:>6}{r['runtime_ms']:>13.1f}")
    print("-" * 95)


def write_rows(path, order, rows, budgets):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget"] + METRIC_KEYS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name in order:
            w.writerow({"method": name, "budget": budgets.get(name, ""),
                        **{k: rows[name][k] for k in METRIC_KEYS}})


def write_curve(path, curve):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget_kind", "budget"] + METRIC_KEYS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in curve:
            w.writerow(r)


def write_series(path, order, rows, days, show_days, resid_day, sample_hz):
    """Per-sample series behind Fig. 4c: raw cycles of ``show_days`` with stage
    labels, and reference - corrected on ``resid_day`` for every method."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    by_day = {d: (r, cs) for d, r, cs in days}
    folders = [d for d, _, _ in days]
    pick = lambda i: folders[min(max(i - 1, 0), len(folders) - 1)]   # day number -> folder
    shown = [pick(i) for i in show_days]
    rday = pick(resid_day)
    ref = days[0][1]
    T = min(len(by_day[d][0]) for d in set(shown) | {rday} | {folders[0]})
    cols = ["idx", "t_min"]
    for i, d in zip(show_days, shown):
        cols += [f"raw:day{i}", f"stage:day{i}"]
    cols += [f"resid:{m}" for m in order if m in rows and rday in rows[m].get("_corrected", {})]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for k in range(T):
            row = [k, f"{k / sample_hz / 60.0:.4f}"]
            for d in shown:
                r, cs = by_day[d]
                row += [f"{r[k]:.3f}", int(cs[k])]
            for m in order:
                if m in rows and rday in rows[m].get("_corrected", {}):
                    row.append(f"{ref[k] - rows[m]['_corrected'][rday][k]:.3f}")
            w.writerow(row)
    print(f"wrote per-sample series (days {show_days}, residual day {resid_day} = "
          f"'{rday}') -> {path}")


def write_resid_all(path, methods, rows, days, sample_hz):
    """reference - corrected on EVERY test day for the given methods (Fig. 4c):
    columns resid:<method>:day<N>, with the day-1 stage labels."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ref, cs0 = days[0][1], days[0][2]
    folders = [d for d, _, _ in days]
    T = min(len(r) for _, r, _ in days)
    cols = ["idx", "t_min", "stage", "ref"]
    plan = []
    for m in methods:
        if m not in rows:
            continue
        for i, d in enumerate(folders[1:], start=2):
            if d in rows[m].get("_corrected", {}):
                cols.append(f"resid:{m}:day{i}")
                plan.append(rows[m]["_corrected"][d])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for k in range(T):
            w.writerow([k, f"{k / sample_hz / 60.0:.4f}", int(cs0[k]), f"{ref[k]:.2f}"]
                       + [f"{ref[k] - rc[k]:.2f}" for rc in plan])
    print(f"wrote per-day residuals for {methods} ({len(plan)} day-columns) -> {path}")


def write_onsets(path, order, rows):
    """Every method's onset AND the offset subtracted, per day (day 1 included
    with offset 0), plus the day-1 baseline mean it was referenced to; so a
    corrected cycle on any dumped day is simply raw - offset."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "day_folder", "onset_idx", "offset_ohm", "ref_base_mean_ohm"])
        for m in order:
            rb = rows[m].get("_ref_base_mean", np.nan)
            for d, (o, off) in rows[m].get("_offsets", {}).items():
                w.writerow([m, d, o, f"{off:.4f}", f"{rb:.4f}"])


def best_point(evals, tol=1e-9):
    """Centre of the MIN-MAE plateau (the synthetic / ISFET rule)."""
    mn = min(e[1]["mae"] for e in evals)
    cand = [e for e in evals if e[1]["mae"] <= mn + tol]
    cand.sort(key=lambda e: e[0])
    b_star, r_star = cand[len(cand) // 2]
    return b_star, r_star, mn, len(cand)


def insert_twins(order):
    out = []
    for m in order:
        out.append(m)
        if m in TWINS:
            out.append(TWINS[m][0])
    return out


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_GAS_DIR)
    ap.add_argument("--sensor", type=int, default=0, help="sensor id (0-based -> R1)")
    ap.add_argument("--analyte", default="Ethanol", choices=list(ANALYTE_LABELS))
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--sample-hz", type=float, default=1.0, help="for the t_min column")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-seg", type=int, default=30,
                    help="min samples between change-points (per cycle)")
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    # sweep
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--alpha-grid",
                    default="1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1")
    ap.add_argument("--hazard-grid", default="100,300,600,1000,2000,5000,10000",
                    help="length-prior timescales (samples) swept for BOCD and CHAMP")
    ap.add_argument("--dm-hazard-grid", default="100,300,1000,3000,10000",
                    help="coarse grid swept for Dm-BOCD on its own (slow); "
                         "'inherit' = BOCD's best (the old protocol)")
    ap.add_argument("--dm-hazard", type=float, default=None,
                    help="single hazard for Dm-BOCD, overriding --dm-hazard-grid")
    # CHAMP / NUNC a-priori settings
    ap.add_argument("--champ-gamma", type=float, default=0.05)
    ap.add_argument("--champ-em-iters", type=int, default=2)
    ap.add_argument("--champ-max-particles", type=int, default=100,
                    help="particle cap (the published algorithm); exact = 0 is O(n^3)")
    ap.add_argument("--champ-lag", type=int, default=20)
    ap.add_argument("--champ-confirm", type=int, default=None, help="default = --min-seg")
    ap.add_argument("--nunc-window", type=int, default=None,
                    help="NUNC rolling window; default max(50, 2*min_seg) so two "
                         "min_seg blocks fit (= 60 here)")
    ap.add_argument("--nunc-sims", type=int, default=20000)
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full")
    # outputs
    ap.add_argument("--out", default="results/gas/gas_v3_best.csv")
    ap.add_argument("--sweep-out", default="results/gas/gas_v3_sweep.csv")
    ap.add_argument("--series-out", default="results/gas/gas_v3_series.csv")
    ap.add_argument("--onsets-out", default="results/gas/gas_v3_onsets.csv")
    ap.add_argument("--resid-all-out", default="results/gas/gas_v3_resid_all.csv",
                    help="reference - corrected on every test day (Fig. 4c)")
    ap.add_argument("--resid-methods", default="vocd",
                    help="methods written to --resid-all-out (comma-separated)")
    ap.add_argument("--show-days", default="1,15,30,39",
                    help="day numbers whose raw cycles go to --series-out (Fig. 4c top)")
    ap.add_argument("--resid-day", type=int, default=39,
                    help="day whose reference - corrected residual goes to --series-out")
    args = ap.parse_args()
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    if args.champ_max_particles is not None and args.champ_max_particles <= 0:
        args.champ_max_particles = None
    if args.nunc_window is None:
        args.nunc_window = max(50, 2 * args.min_seg)
    show_days = [int(v) for v in args.show_days.split(",") if v.strip()]

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"vocd"} | set(REGISTRY)
    for m in wanted:
        if m not in known:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(known)}")
    if any(m in TWIN_METHODS for m in wanted):
        print("note: compiled twins are paired automatically; list 'focus' / 'np-focus' instead")
        wanted = [m for m in wanted if m not in TWIN_METHODS]
    needs = [m for m in wanted if m in REGISTRY and REGISTRY[m][1]]
    if needs and not dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}\n")
        wanted = [m for m in wanted if m not in needs]

    if args.vocd_cpp == "full":
        import vocd._fastdetect as fastdetect
        fastdetect.enable()
    elif args.vocd_cpp == "gate":
        import vocd._fastgate as fastgate
        fastgate.enable()
    if any(m in TWINS for m in wanted):
        from alternative_models import _cppalarms
        _cppalarms._load()                   # one-off compile, outside timers

    print(f"loading gas data: sensor R{args.sensor + 1}, analyte {args.analyte}, "
          f"replicate {args.replicate}")
    days = load_all_days(args.data_dir, args.analyte, args.replicate, args.sensor)
    print(f"  {len(days)} days ({len(days) - 1} test days), ~{len(days[0][1])} pts/cycle")
    cps_fn = make_cps_fn(args, dsm_path)

    print(f"methods: {', '.join(wanted)}; twins: "
          f"{', '.join(TWINS[m][0] for m in wanted if m in TWINS)}; vocd C++ engine = {args.vocd_cpp}")
    if "nunc" in wanted:
        print(f"nunc : window={args.nunc_window} (= max(2 x VOCD gate window, 2 x min_seg)); "
              f"calibration once per record evaluation, inside the timer")
    if "champ" in wanted:
        print(f"champ: gamma={args.champ_gamma}, lag={args.champ_lag}, "
              f"confirm={args.champ_confirm or args.min_seg}, "
              f"{'exact' if not args.champ_max_particles else f'max_particles={args.champ_max_particles}'}")
    print()

    rows, budgets, curve = {}, {}, []
    if not args.no_oracle:
        rows["oracle"] = eval_method(cps_fn, "oracle", days, None, args)

    if not args.sweep:
        print(f"single operating point: alpha={args.alpha:g} (hazard methods at 300)\n")
        for name in wanted:
            b = 300.0 if name in HAZARD_METHODS else args.alpha
            print(f"[{name}] ...", end="", flush=True)
            r = eval_method(cps_fn, name, days, b, args)
            rows[name], budgets[name] = r, b
            print(f" MAE={r['mae']:.2f}")
    else:
        alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
        hazard_grid = [float(v) for v in args.hazard_grid.split(",") if v.strip()]
        print("SWEEP mode — reporting each detector's MIN-MAE operating point.")
        print(f"  alpha grid : {alpha_grid}")
        print(f"  hazard grid: {hazard_grid}\n")
        best_by = {}
        proc = list(wanted)
        if "bocd" in proc and "dm-bocd" in proc:
            proc.remove("dm-bocd")
            proc.insert(proc.index("bocd") + 1, "dm-bocd")
        for name in proc:
            if name == "dm-bocd":
                kind = "hazard"
                if args.dm_hazard is not None:
                    grid, src = [args.dm_hazard], "--dm-hazard"
                elif args.dm_hazard_grid.strip().lower() == "inherit":
                    grid, src = [best_by.get("bocd", 300.0)], "inherited from bocd best"
                else:
                    grid = [float(v) for v in args.dm_hazard_grid.split(",") if v.strip()]
                    src = "own grid"
                print(f"[{name}] (hazard, {src}) ", end="", flush=True)
            elif name in HAZARD_METHODS:
                grid, kind = hazard_grid, "hazard"
                print(f"[{name}] (hazard) ", end="", flush=True)
            else:
                grid, kind = alpha_grid, ("fwer" if name == "vocd" else "alpha")
                print(f"[{name}] ({kind}) ", end="", flush=True)
            evals = []
            for b in grid:
                r = eval_method(cps_fn, name, days, b, args)
                curve.append({"method": name, "budget_kind": kind, "budget": b,
                              **{k: r[k] for k in METRIC_KEYS}})
                evals.append((b, r))
                print(".", end="", flush=True)
            b_star, r_star, mn, n_plat = best_point(evals)
            rows[name], budgets[name], best_by[name] = r_star, b_star, b_star
            plateau = "" if n_plat == 1 else f" ({n_plat}-pt plateau)"
            edge = " [EDGE of grid]" if len(grid) > 1 and b_star in (grid[0], grid[-1]) else ""
            print(f" min MAE={mn:.2f} @ {kind}={b_star:g}{plateau}{edge}")

    for base in wanted:
        if base in TWINS and base in rows:
            tw = paired_row(cps_fn, base, days, budgets[base], args, rows[base])
            if tw:
                rows[TWINS[base][0]] = tw
                budgets[TWINS[base][0]] = budgets[base]

    det_order = [m for m in insert_twins(wanted) if m in rows]
    order = det_order + (["oracle"] if "oracle" in rows else [])
    print_table(order, rows, budgets)
    print("mean MAE / RMSE (Ohm) of the corrected cycle vs the day-1 cycle over the test "
          "days; CI = 95%\nmoving-block bootstrap over days (5-day blocks, 2000 resamples). "
          "'miss' = days with no usable\nonset (no correction applied). '(cpp)' rows reuse "
          "their Python row's budget and onsets.\noracle uses the true cycle_stage "
          "(non-causal ceiling).")

    write_rows(args.out, order, rows, budgets)
    print(f"\nwrote {len(order)} rows -> {args.out}")
    if curve:
        write_curve(args.sweep_out, curve)
        print(f"wrote {len(curve)} curve rows -> {args.sweep_out}")
    write_series(args.series_out, order, rows, days, show_days, args.resid_day, args.sample_hz)
    write_onsets(args.onsets_out, det_order, rows)
    print(f"wrote onsets -> {args.onsets_out}")
    write_resid_all(args.resid_all_out, [m.strip() for m in args.resid_methods.split(",")],
                    rows, days, args.sample_hz)


if __name__ == "__main__":
    main()
