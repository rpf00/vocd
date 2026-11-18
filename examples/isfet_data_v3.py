"""
ISFET drift-correction benchmark, v3 — the final paper panel (Fig. 3c/d).

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)

Same problem, data, correction machinery and scoring as ``isfet_data.py``
(imported verbatim from it): no change-point ground truth, so each detector
supplies the segment boundaries for the SAME causal piecewise-line correction
by prefix replay, and is scored by drift RMSE against the reference; the
sweep MINIMISES RMSE.  What this file changes is the PANEL and the plumbing,
so every experiment in the paper calls the detectors the same way as
``examples/synthetic_data_v3.py``:

* R-FOCuS is dropped (a Huber-clipped FOCuS we built ourselves, unpublished).
* FOCuS (plain), NUNC and CHAMP are added, called through the same
  ``alternative_models.REGISTRY`` entries and with the same budget kinds:
  alarm methods on the alpha grid, BOCD / Dm-BOCD / CHAMP on the hazard
  (length-prior timescale) grid, VOCD on fwer.
* Three compiled timing twins are paired automatically at their Python row's
  best budget with the boundary sets asserted identical: 'vocd (cpp)' (whole
  detect in C++ via vocd/_fastdetect), 'focus (cpp)' and 'np-focus (cpp)'
  (functional pruning).  Twins are never swept on their own.
* Dm-BOCD is swept on its own coarse grid rather than inheriting BOCD's
  hazard (on the synthetic task the two optima differ).
* 95% moving-block-bootstrap confidence intervals of the RMSE over days
  (7-day blocks, 2000 resamples) are added to every row (Fig. 3d error bars).
* ``--series-out`` dumps the per-day series behind Fig. 3c: V_H, reference,
  true E0, and every method's E0_hat and corrected V_H at its best operating
  point, plus every method's final boundary set (``--cps-out``).

A-priori method settings for THIS record (declare them in the Methods, like
``min_seg``): NUNC window W = 50 = 2 x VOCD's gate window (the same rule as
the synthetic benchmark; its window grows until it reaches W); CHAMP outlier
ratio gamma = 0.05, confirmation = min_seg, commit spacing lag = 20 (the same
spacing BOCD's causal extraction uses), exact recursion (no particle cap;
T <= 90 makes it trivial).  NUNC's Monte-Carlo calibration is paid once per
record evaluation, inside the timer: a deployed NUNC calibrates once per
sensor, so rebuilding it on each of the 90 replay prefixes would charge a
harness artefact, not the method.

Usage::

    python examples/isfet_data_v3.py --sweep --data-dir data/isfet --dsm-path ../DSM-bocd \\
        --out results/isfet/isfet_v3_best.csv --sweep-out results/isfet/isfet_v3_sweep.csv \\
        --series-out results/isfet/isfet_v3_series.csv --cps-out results/isfet/isfet_v3_cps.csv

Without ``--dsm-path`` / ``VOCD_DSM_PATH`` the two BOCD rows are skipped.
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

import vocd                                   # noqa: E402
from alternative_models import REGISTRY       # noqa: E402
import isfet_data as v2                       # noqa: E402  (verbatim problem)

S_NERNST, BURN_IN, MIN_FIT = v2.S_NERNST, v2.BURN_IN, v2.MIN_FIT
load_daily = v2.load_daily
est_none, est_twopoint, est_M0, est_exp = v2.est_none, v2.est_twopoint, v2.est_M0, v2.est_exp
est_from_cp_sets, est_oracle = v2.est_from_cp_sets, v2.est_oracle
VOCD_FAIR_BACKEND, VOCD_FAST_BACKEND = v2.VOCD_FAIR_BACKEND, v2.VOCD_FAST_BACKEND

PANEL = "vocd,bocd,dm-bocd,focus,np-focus,nunc,chad,champ"
HAZARD_METHODS = ("bocd", "dm-bocd", "champ")
TWINS = {
    "vocd":     ("vocd (cpp)",     None),
    "focus":    ("focus (cpp)",    "focus-cpp"),
    "np-focus": ("np-focus (cpp)", "np-focus-cpp"),
}
TWIN_METHODS = {m for _, m in TWINS.values() if m}
METRIC_KEYS = ["rmse_mv", "rmse_lo", "rmse_hi", "rmse_ph", "mae_mv", "beta",
               "n_cp", "runtime_ms"]
BASELINES = ("none", "two-point", "M0", "exp", "oracle")


# ── scoring: v2's score + a moving-block bootstrap CI over days ───────

def block_bootstrap_rmse(err, block=7, n_boot=2000, seed=0):
    """95% percentile interval of the RMSE from a moving-block bootstrap over
    days (the paper's Methods: 7-day blocks, 2,000 resamples)."""
    err = np.asarray(err, float)
    n = err.size
    if n < block + 1:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts_max = n - block
    k = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, starts_max + 1, size=k)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = np.sqrt(np.mean(err[idx] ** 2))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def score(e0_hat, vd, ph_ref, dd, burn_in=BURN_IN, n_cp=np.nan):
    """Verbatim v2 scoring plus the CI; also returns the anchored error series."""
    r = v2.score(e0_hat, vd, ph_ref, dd, burn_in, n_cp)
    ph_hat = (e0_hat - vd) / S_NERNST
    m = dd >= burn_in
    bias = float(np.mean(ph_hat[m][:MIN_FIT] - ph_ref[m][:MIN_FIT]))
    err_mv = (ph_hat[m] - bias - ph_ref[m]) * S_NERNST
    r["rmse_lo"], r["rmse_hi"] = block_bootstrap_rmse(err_mv)
    full = np.full(dd.size, np.nan)
    full[m] = err_mv
    r["_err_mv"] = full
    r["_e0_hat"] = np.asarray(e0_hat, float)
    return r


# ── detectors by prefix replay (strictly causal), unified routing ─────

def make_cp_sets_fn(args, dsm_path):
    champ_kw = dict(gamma=args.champ_gamma, em_iters=args.champ_em_iters,
                    max_particles=args.champ_max_particles, extract="causal",
                    lag=args.champ_lag, confirm=args.champ_confirm)

    def cp_sets(name, vd, budget, vocd_backend=VOCD_FAIR_BACKEND):
        """cp_sets[d] = the detector's boundaries computed on V_H[:d+1]."""
        sets = []
        nunc_cache = {} if name == "nunc" else None    # one calibration per record
        for i in range(vd.size):
            xi = vd[:i + 1]
            if name == "vocd":
                try:
                    cps = [int(c) for c in vocd.detect(xi, backend=vocd_backend)
                           .changepoints(fwer=budget)]
                except Exception:
                    cps = []
            else:
                fn, needs_dsm = REGISTRY[name]
                if needs_dsm:
                    cps = fn(xi, mean0=None, dsm_path=dsm_path,
                             extract=args.bocd_extract, hazard_timescale=budget)
                elif name == "champ":
                    cps = fn(xi, hazard_timescale=budget, min_seg=args.min_seg, **champ_kw)
                elif name == "nunc":
                    cps = fn(xi, alpha=budget, min_seg=args.min_seg,
                             window=args.nunc_window, n_sims=args.nunc_sims,
                             cache=nunc_cache)
                else:
                    cps = fn(xi, alpha=budget, min_seg=args.min_seg)
            sets.append(sorted(int(c) for c in cps))
        return sets
    return cp_sets


def eval_detector(cp_sets, name, dd, vd, ph_ref, budget, args,
                  vocd_backend=VOCD_FAIR_BACKEND):
    t0 = time.perf_counter()
    sets = cp_sets(name, vd, budget, vocd_backend=vocd_backend)
    e0_hat = est_from_cp_sets(dd, vd, sets)
    ms = (time.perf_counter() - t0) * 1000.0
    r = score(e0_hat, vd, ph_ref, dd, args.burn_in, n_cp=len(sets[-1]))
    r["runtime_ms"] = ms
    r["_sets"] = sets
    return r


def paired_row(cp_sets, base, dd, vd, ph_ref, b_star, args, base_row):
    label, method = TWINS[base]
    try:
        if method is None:
            r = eval_detector(cp_sets, "vocd", dd, vd, ph_ref, b_star, args,
                              vocd_backend=VOCD_FAST_BACKEND)
        else:
            r = eval_detector(cp_sets, method, dd, vd, ph_ref, b_star, args)
    except Exception as e:
        print(f"! {label}: compiled twin unavailable ({e}); row skipped")
        return None
    if r["_sets"] == base_row["_sets"]:
        print(f"  {label}: boundary sets identical to {base} (only runtime differs)")
    else:
        nd = sum(1 for a, b in zip(r["_sets"], base_row["_sets"]) if a != b)
        print(f"! WARNING: {base} python vs cpp boundary sets differ on {nd}/"
              f"{len(r['_sets'])} prefixes — '{label}' is NOT a pure timing twin")
    return r


def eval_baselines(dd, vd, ph_ref, e0, args):
    rows = {}
    specs = [("none", est_none, {}),
             ("two-point", est_twopoint, {"e0": e0, "period": args.twopoint_period}),
             ("M0", est_M0, {}), ("exp", est_exp, {}),
             ("oracle", est_oracle, {"e0": e0})]
    for label, fn, kw in specs:
        t0 = time.perf_counter()
        e0_hat = fn(dd, vd, **kw)
        ms = (time.perf_counter() - t0) * 1000.0
        r = score(e0_hat, vd, ph_ref, dd, args.burn_in, n_cp=np.nan)
        r["runtime_ms"] = ms
        rows[label] = r
    return rows


# ── IO ────────────────────────────────────────────────────────────────

def print_table(order, rows, budgets):
    print(f"\n{'method':<16}{'budget':>10}{'RMSE_mV':>9}{'95% CI':>16}{'RMSE_pH':>9}"
          f"{'MAE_mV':>9}{'beta':>8}{'#CP':>5}{'Runtime(ms)':>13}")
    print("-" * 95)
    for name in order:
        r = rows[name]
        b = budgets.get(name)
        bcol = f"{b:>10.4g}" if b is not None else f"{'—':>10}"
        ncp = "—" if np.isnan(r["n_cp"]) else f"{int(r['n_cp'])}"
        beta = "nan" if np.isnan(r["beta"]) else f"{r['beta']:+.3f}"
        ci = f"[{r['rmse_lo']:.2f}, {r['rmse_hi']:.2f}]"
        print(f"{name:<16}{bcol}{r['rmse_mv']:>9.3f}{ci:>16}{r['rmse_ph']:>9.4f}"
              f"{r['mae_mv']:>9.3f}{beta:>8}{ncp:>5}{r['runtime_ms']:>13.1f}")
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


def write_curve(path, curve_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget_kind", "budget"] + METRIC_KEYS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in curve_rows:
            w.writerow(r)


def write_series(path, order, rows, dd, vd, ph_ref, e0):
    """Per-day series behind Fig. 3c, every method at its reported point."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ref_mv = -S_NERNST * ph_ref
    cols = ["day", "v_h_mv", "ref_mv", "e0_true_mv", "e0_true_rel_mv"]
    for name in order:
        cols += [f"e0_hat:{name}", f"vcorr:{name}", f"err_mv:{name}"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(dd.size):
            row = [f"{dd[i]:.1f}", f"{vd[i]:.4f}", f"{ref_mv[i]:.4f}",
                   f"{e0[i]:.4f}", f"{e0[i] - e0[0]:.4f}"]
            for name in order:
                r = rows[name]
                eh, er = r["_e0_hat"][i], r["_err_mv"][i]
                row += [f"{eh:.4f}", f"{vd[i] - eh:.4f}",
                        "" if np.isnan(er) else f"{er:.4f}"]
            w.writerow(row)


def write_cps(path, order, rows, dd):
    """Every detector's FINAL boundary set (the one held on the last day)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "cp_index", "cp_day"])
        for name in order:
            sets = rows[name].get("_sets")
            if sets:
                for c in sets[-1]:
                    w.writerow([name, c, f"{dd[c]:.1f}"])


def best_point(evals, tol=1e-9):
    """Centre of the MIN-RMSE plateau (mirrors the synthetic driver's rule)."""
    min_rmse = min(e[1]["rmse_mv"] for e in evals)
    cand = [e for e in evals if e[1]["rmse_mv"] <= min_rmse + tol]
    cand.sort(key=lambda e: e[0])
    b_star, r_star = cand[len(cand) // 2]
    return b_star, r_star, min_rmse, len(cand)


def insert_twins(order):
    out = []
    for m in order:
        out.append(m)
        if m in TWINS:
            out.append(TWINS[m][0])
    return out


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/isfet")
    ap.add_argument("--ion", default="h")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="single-point mode: fwer(vocd)/alpha(alarm) budget")
    ap.add_argument("--min-seg", type=int, default=4,
                    help="min days between change-points / per-segment burn-in")
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--twopoint-period", type=float, default=7.0)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    # sweep
    ap.add_argument("--sweep", action="store_true",
                    help="sweep each detector's budget; report the min-RMSE point")
    ap.add_argument("--alpha-grid",
                    default="1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1")
    ap.add_argument("--hazard-grid", default="2,5,10,20,30,50,80,150",
                    help="length-prior timescales (days) swept for BOCD and CHAMP")
    ap.add_argument("--dm-hazard-grid", default="2,5,10,20,30,50,80,150",
                    help="hazard grid swept for Dm-BOCD on its own (slow: ~70 s "
                         "per point on this record); 'inherit' = BOCD's best")
    ap.add_argument("--dm-hazard", type=float, default=None,
                    help="single hazard for Dm-BOCD, overriding --dm-hazard-grid")
    # CHAMP / NUNC a-priori settings (declare in the Methods)
    ap.add_argument("--champ-gamma", type=float, default=0.05)
    ap.add_argument("--champ-em-iters", type=int, default=2)
    ap.add_argument("--champ-max-particles", type=int, default=None,
                    help="default None = exact (T <= 90 is trivial)")
    ap.add_argument("--champ-lag", type=int, default=20,
                    help="= BOCD's causal-extraction spacing")
    ap.add_argument("--champ-confirm", type=int, default=None,
                    help="default None = --min-seg")
    ap.add_argument("--nunc-window", type=int, default=50,
                    help="= 2 x VOCD's gate window (same rule as the synthetic run)")
    ap.add_argument("--nunc-sims", type=int, default=20000)
    # VOCD compiled engine for the twin row
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full")
    # outputs
    ap.add_argument("--out", default="results/isfet/isfet_v3_best.csv")
    ap.add_argument("--sweep-out", default="results/isfet/isfet_v3_sweep.csv")
    ap.add_argument("--series-out", default="results/isfet/isfet_v3_series.csv",
                    help="per-day series for Fig. 3c (every method at its best point)")
    ap.add_argument("--cps-out", default="results/isfet/isfet_v3_cps.csv",
                    help="every detector's final boundary set")
    args = ap.parse_args()
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"vocd"} | set(REGISTRY)
    for m in wanted:
        if m not in known:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(known)}")
    if any(m in TWIN_METHODS for m in wanted):
        print("note: compiled twins are paired automatically; list 'focus' / "
              "'np-focus' instead")
        wanted = [m for m in wanted if m not in TWIN_METHODS]
    needs = [m for m in wanted if m in REGISTRY and REGISTRY[m][1]]
    if needs and not dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}\n"
              "  git clone https://github.com/maltamiranomontero/DSM-bocd.git\n")
        wanted = [m for m in wanted if m not in needs]

    if args.vocd_cpp == "full":
        import vocd._fastdetect as fastdetect
        fastdetect.enable()                 # routes backend="cpp" only
    elif args.vocd_cpp == "gate":
        import vocd._fastgate as fastgate
        fastgate.enable()
    if any(m in TWINS for m in wanted):      # one-off compile, outside timers
        from alternative_models import _cppalarms
        _cppalarms._load()

    dd, vd, ph_ref, e0 = load_daily(args.data_dir, args.ion)
    cp_sets = make_cp_sets_fn(args, dsm_path)

    print(f"ISFET drift correction (reference-free) — ion={args.ion}, {dd.size} days, "
          f"burn-in {args.burn_in} d, min_seg {args.min_seg} d")
    print(f"methods: {', '.join(wanted)}")
    print(f"twins paired at their Python row's budget: "
          f"{', '.join(TWINS[m][0] for m in wanted if m in TWINS)}; "
          f"vocd C++ engine = {args.vocd_cpp}")
    if "nunc" in wanted:
        print(f"nunc : window={args.nunc_window} (= 2 x VOCD gate window), calibration "
              f"paid once per record evaluation, inside the timer")
    if "champ" in wanted:
        print(f"champ: gamma={args.champ_gamma}, lag={args.champ_lag}, "
              f"confirm={args.champ_confirm or args.min_seg}, "
              f"{'exact' if not args.champ_max_particles else f'max_particles={args.champ_max_particles}'}")
    print()

    rows, budgets, curve_rows = {}, {}, []
    if not args.no_baselines:
        rows.update(eval_baselines(dd, vd, ph_ref, e0, args))

    if not args.sweep:
        print(f"single operating point: alpha={args.alpha:g} "
              f"(hazard methods at timescale 30 d)\n")
        for name in wanted:
            b = 30.0 if name in HAZARD_METHODS else args.alpha
            print(f"[{name}] ...", end="", flush=True)
            r = eval_detector(cp_sets, name, dd, vd, ph_ref, b, args)
            rows[name], budgets[name] = r, b
            print(f" RMSE={r['rmse_mv']:.3f} mV")
    else:
        alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
        hazard_grid = [float(v) for v in args.hazard_grid.split(",") if v.strip()]
        print("SWEEP mode — reporting each detector's MIN-RMSE operating point.")
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
                    grid, src = [best_by.get("bocd", 30.0)], "inherited from bocd best"
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
                r = eval_detector(cp_sets, name, dd, vd, ph_ref, b, args)
                curve_rows.append({"method": name, "budget_kind": kind, "budget": b,
                                   **{k: r[k] for k in METRIC_KEYS}})
                evals.append((b, r))
                print(".", end="", flush=True)
            b_star, r_star, min_rmse, n_plat = best_point(evals)
            rows[name], budgets[name], best_by[name] = r_star, b_star, b_star
            plateau = "" if n_plat == 1 else f" ({n_plat}-pt plateau)"
            edge = " [EDGE of grid]" if len(grid) > 1 and b_star in (grid[0], grid[-1]) else ""
            print(f" min RMSE={min_rmse:.3f} mV @ {kind}={b_star:g}{plateau}{edge}")

    for base in wanted:
        if base in TWINS and base in rows:
            tw = paired_row(cp_sets, base, dd, vd, ph_ref, budgets[base], args, rows[base])
            if tw:
                rows[TWINS[base][0]] = tw
                budgets[TWINS[base][0]] = budgets[base]

    det_order = [m for m in insert_twins(wanted) if m in rows]
    ctx_top = [m for m in ("none", "two-point", "M0", "exp") if m in rows]
    ctx_bot = [m for m in ("oracle",) if m in rows]
    order = ctx_top + det_order + ctx_bot

    print_table(order, rows, budgets)
    print("scored on drift RMSE vs the reference (mV) over the post-burn-in window; "
          "CI = 95% moving-block bootstrap\nover days (7-day blocks, 2000 resamples). "
          "'(cpp)' rows reuse their Python row's budget and boundaries;\nonly runtime "
          "differs. runtime = total prefix-replay build cost over the record.")

    write_rows(args.out, order, rows, budgets)
    print(f"\nwrote {len(order)} rows -> {args.out}")
    if curve_rows:
        write_curve(args.sweep_out, curve_rows)
        print(f"wrote {len(curve_rows)} curve rows -> {args.sweep_out}")
    write_series(args.series_out, order, rows, dd, vd, ph_ref, e0)
    print(f"wrote per-day series ({len(order)} methods) -> {args.series_out}")
    write_cps(args.cps_out, det_order, rows, dd)
    print(f"wrote final boundary sets -> {args.cps_out}")


if __name__ == "__main__":
    main()
