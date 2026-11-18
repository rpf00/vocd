"""
ISFET drift-correction benchmark: VOCD vs BOCD, Dm-BOCD, NP-FOCuS, R-FOCuS, CHAD.

Structurally the twin of ``examples/synthetic_data.py`` — same six-method panel,
same paired ``vocd`` / ``vocd (cpp)`` rows, same ``--sweep`` with central-plateau
selection, same BOCD-swept / Dm-BOCD-inherits-hazard logic, same fair-runtime
framing.  What differs is the PROBLEM, and therefore the objective:

    there is no change-point ground truth here.  The reference sensor gives the
    true drift curve E0(t), not a labelled list of changes.  So methods are
    scored by DRIFT RMSE against the reference, and the sweep MINIMISES RMSE
    instead of maximising F1.

Signal model (Margarit-Taule et al. 2022, S&A B 353, 131123):

    V_H(t) = E0(t) - S . pH(t)                 S = 59.16 mV/pH (Nernst)
    truth  E0(t) = V_H(t) + S . pH_ref(t)      (ref_mv.csv stores -S.pH)
    method E0_hat(t) = f(V_H up to t)          reference NOT used by the method
    score  RMSE(E0_hat - E0) in mV (= pH error x S)

This is the reference-free ("Condition B") comparison: every method estimates
the slow, persistent component of V_H from the sensor alone and calls it drift;
the reference is used only to score.  All estimators, the correction machinery
and the scoring are reused verbatim from ``isfet_benchmark.py``.

Detectors and causality
-----------------------
The six change-point detectors do NOT get scored directly.  Each supplies the
segment boundaries for the SAME causal piecewise-line correction
(``est_from_cp_sets``); holding the correction fixed and swapping the detector
is what isolates detection quality.  To stay strictly causal with no leakage,
each detector's boundary set on day d is recomputed on the prefix V_H[:d+1]
(prefix replay) — a change at day c cannot influence the correction before day c.
The reported runtime is therefore the total cost of building the causal boundary
sets over the record, not the (cheaper, incremental) deployment cost.

Context rows (not swept): ``none`` (calibrate once — the drift ceiling),
``two-point`` (reset to true E0 every W days — the optimistic industry
baseline), ``M0`` (online line), ``exp`` (Jamasb exponential+linear online fit),
and ``oracle`` (perfect E0 — the floor).  ``none`` and ``oracle`` bracket the
table so the gap is visible.

Honest ceiling (state in the paper): the drift in this record is a near-linear
ramp, so the online line ``M0`` is near-optimal by construction and VOCD can at
best TIE it.  Beating two-point calibration while assuming nothing about the
drift shape is the ISFET result; model-agnosticism is demonstrated on the gas
sensors, not here.  Expect a TIGHT spread across the detector rows.

Usage
-----
    python examples/isfet_data.py --data-dir data/isfet
    python examples/isfet_data.py --data-dir data/isfet --dsm-path ./DSM-bocd
    python examples/isfet_data.py --sweep --data-dir data/isfet --dsm-path ./DSM-bocd \
        --out results/isfet/isfet_best.csv --sweep-out results/isfet/isfet_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")   # DSM-BOCD log(0) and curve_fit covariance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    from scipy.interpolate import interp1d
except ImportError:
    sys.exit("needs pandas and scipy:  pip install pandas scipy")

import vocd
from alternative_models import REGISTRY

SAMPLE_MIN = 5.0
S_NERNST = 59.16
BURN_IN = 10        # days before scoring starts
MIN_FIT = 4         # days; fewer and a line is meaningless

DETECTORS = ["vocd", "bocd", "dm-bocd", "np-focus", "r-focus", "chad"]
VOCD_FAIR_BACKEND = "python"
VOCD_FAST_BACKEND = "cpp"
METRIC_KEYS = ["rmse_mv", "rmse_ph", "mae_mv", "beta", "n_cp", "runtime_ms"]


# ── data (verbatim logic from isfet_benchmark.py) ─────────────────────

def load_daily(directory: str, ion: str):
    f_ion = os.path.join(directory, f"{ion}_mv.csv")
    f_ref = os.path.join(directory, "ref_mv.csv")
    for f in (f_ion, f_ref):
        if not os.path.exists(f):
            sys.exit(f"not found: {f}")
    a = (pd.read_csv(f_ion, header=None, names=["timeslot", "mv"])
         .sort_values("timeslot").reset_index(drop=True).dropna())
    b = (pd.read_csv(f_ref, header=None, names=["timeslot", "ref"])
         .sort_values("timeslot").reset_index(drop=True).dropna())
    ts = a["timeslot"].to_numpy(float)
    days = (ts - ts[0]) * SAMPLE_MIN / (60.0 * 24.0)
    ref_mv = interp1d(b["timeslot"].to_numpy(float), b["ref"].to_numpy(float),
                      kind="linear", bounds_error=False,
                      fill_value=(float(b["ref"].iloc[0]),
                                  float(b["ref"].iloc[-1])))(ts)
    v = a["mv"].to_numpy(float)
    idx = np.floor(days).astype(int)
    dd, vd, rd = [], [], []
    for d in range(int(idx.min()), int(idx.max()) + 1):
        m = idx == d
        if m.any():
            dd.append(d + 0.5)
            vd.append(float(np.mean(v[m])))
            rd.append(float(np.mean(ref_mv[m])))
    dd, vd, rd = np.array(dd), np.array(vd), np.array(rd)
    ph_ref = -rd / S_NERNST
    e0 = vd - rd
    print(f"loaded {len(a)} samples -> {dd.size} daily means; "
          f"true drift {e0[-1]-e0[0]:+.1f} mV "
          f"({(e0[-1]-e0[0])/(dd[-1]-dd[0]):+.2f} mV/day)")
    return dd, vd, ph_ref, e0


# ── correction machinery (verbatim from isfet_benchmark.py) ───────────

def _line_at(t, y, at):
    if t.size < 2:
        return float(y[-1]) if y.size else 0.0
    if t.size < MIN_FIT:
        return float(np.mean(y))
    c = np.polyfit(t, y, 1)
    return float(np.polyval(c, at))


def est_none(dd, vd, **kw):
    return np.full(dd.size, float(np.mean(vd[:MIN_FIT])))


def est_twopoint(dd, vd, e0=None, period=7.0, **kw):
    out = np.empty(dd.size)
    last, held = -1e9, e0[0]
    for i, d in enumerate(dd):
        if d - last >= period:
            held, last = e0[i], d
        out[i] = held
    return out


def est_M0(dd, vd, **kw):
    out = np.empty(dd.size)
    for i in range(dd.size):
        out[i] = _line_at(dd[:i + 1], vd[:i + 1], dd[i])
    return out


def est_exp(dd, vd, **kw):
    from scipy.optimize import curve_fit

    def f(x, a, tau, b, c):
        return a * np.exp(-(x - dd[0]) / max(tau, 1e-6)) + b * x + c

    out = np.empty(dd.size)
    for i in range(dd.size):
        t, y = dd[:i + 1], vd[:i + 1]
        if t.size < 6:
            out[i] = _line_at(t, y, dd[i])
            continue
        try:
            span = max(t[-1] - t[0], 1.0)
            p, _ = curve_fit(f, t, y, p0=[y[0] - y[-1], span / 4, 0.0, y[-1]],
                             maxfev=6000)
            out[i] = float(f(dd[i], *p))
        except Exception:
            out[i] = _line_at(t, y, dd[i])
    return out


def est_from_cp_sets(dd, vd, cp_sets, **kw):
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


def est_oracle(dd, vd, e0=None, **kw):
    return e0.copy()


# ── scoring ───────────────────────────────────────────────────────────

def score(e0_hat, vd, ph_ref, dd, burn_in=BURN_IN, n_cp=np.nan):
    """Drift-estimation error against the reference, with one day-0 anchor.

    Verbatim logic from isfet_benchmark.py:
        pH_hat = (E0_hat - V_H) / S ;  a single calibration removes the mean
        bias over the first MIN_FIT scored days (one deployment calibration);
        err_mv = (E0_hat - E0) up to that anchor.
    """
    ph_hat = (e0_hat - vd) / S_NERNST
    m = dd >= burn_in
    bias = float(np.mean(ph_hat[m][:MIN_FIT] - ph_ref[m][:MIN_FIT]))
    err_ph = ph_hat[m] - bias - ph_ref[m]
    err_mv = err_ph * S_NERNST
    rmse_mv = float(np.sqrt(np.mean(err_mv ** 2)))
    mae_mv = float(np.mean(np.abs(err_mv)))
    beta = float(np.polyfit(dd[m], err_mv, 1)[0]) if m.sum() > 2 else np.nan
    return {"rmse_mv": rmse_mv, "rmse_ph": rmse_mv / S_NERNST, "mae_mv": mae_mv,
            "beta": beta, "n_cp": float(n_cp), "runtime_ms": np.nan}


# ── detector boundary sets by prefix replay (strictly causal) ─────────

def detector_cp_sets(name, vd, budget, args, mean0_none=True,
                     vocd_backend=VOCD_FAIR_BACKEND):
    """cp_sets[d] = the detector's boundaries computed on V_H[:d+1]."""
    T = vd.size
    sets = []
    for i in range(T):
        xi = vd[:i + 1]
        if name == "vocd":
            try:
                r = vocd.detect(xi, backend=vocd_backend)
                cps = [int(c) for c in r.changepoints(fwer=budget)]
            except Exception:
                cps = []
        else:
            fn, needs_dsm = REGISTRY[name]
            if needs_dsm:
                cps = fn(xi, mean0=None, dsm_path=args._dsm_path,
                         extract=args.bocd_extract, hazard_timescale=budget)
            else:
                cps = fn(xi, alpha=budget, min_seg=args.min_seg)
        sets.append(sorted(int(c) for c in cps))
    return sets


def eval_detector(name, dd, vd, ph_ref, e0, budget, args,
                  vocd_backend=VOCD_FAIR_BACKEND):
    t0 = time.perf_counter()
    sets = detector_cp_sets(name, vd, budget, args, vocd_backend=vocd_backend)
    e0_hat = est_from_cp_sets(dd, vd, sets)
    ms = (time.perf_counter() - t0) * 1000.0
    r = score(e0_hat, vd, ph_ref, dd, args.burn_in, n_cp=len(sets[-1]))
    r["runtime_ms"] = ms
    r["_sets"] = sets
    return r


# ── fixed context baselines (no budget) ───────────────────────────────

def eval_baselines(dd, vd, ph_ref, e0, args):
    rows = {}
    specs = [
        ("none", est_none, {}),
        ("two-point", est_twopoint, {"e0": e0, "period": args.twopoint_period}),
        ("M0", est_M0, {}),
        ("exp", est_exp, {}),
        ("oracle", est_oracle, {"e0": e0}),
    ]
    for label, fn, kw in specs:
        t0 = time.perf_counter()
        e0_hat = fn(dd, vd, **kw)
        ms = (time.perf_counter() - t0) * 1000.0
        r = score(e0_hat, vd, ph_ref, dd, args.burn_in, n_cp=np.nan)
        r["runtime_ms"] = ms
        rows[label] = r
    return rows


# ── printing / IO ──────────────────────────────────────────────────────

def agg1(r, key):  # baselines/detectors here are single evaluations, not trials
    return r[key]


def print_table(order, rows, budgets, show_budget=True):
    bh = f"{'budget':>10}" if show_budget else ""
    print(f"\n{'method':<12}{bh}{'RMSE_mV':>10}{'RMSE_pH':>10}{'MAE_mV':>10}"
          f"{'beta':>10}{'#CP':>7}{'Runtime(ms)':>14}")
    print("-" * (73 + (10 if show_budget else 0)))
    for name in order:
        r = rows[name]
        b = budgets.get(name)
        bcol = (f"{b:>10.4g}" if (show_budget and b is not None) else
                (f"{'—':>10}" if show_budget else ""))
        ncp = "—" if np.isnan(r["n_cp"]) else f"{int(r['n_cp'])}"
        beta = "nan" if np.isnan(r["beta"]) else f"{r['beta']:+.3f}"
        print(f"{name:<12}{bcol}{r['rmse_mv']:>10.3f}{r['rmse_ph']:>10.4f}"
              f"{r['mae_mv']:>10.3f}{beta:>10}{ncp:>7}{r['runtime_ms']:>14.1f}")
    print("-" * (73 + (10 if show_budget else 0)))


def write_rows(path, order, rows, budgets):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget", "rmse_mv", "rmse_ph", "mae_mv", "beta",
            "n_cp", "runtime_ms"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name in order:
            r = rows[name]
            w.writerow({"method": name, "budget": budgets.get(name, ""),
                        **{k: r[k] for k in cols if k not in ("method", "budget")}})


def write_curve(path, curve_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget_kind", "budget", "rmse_mv", "rmse_ph", "mae_mv",
            "beta", "n_cp", "runtime_ms"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in curve_rows:
            w.writerow(r)


def vocd_cpp_paired(dd, vd, ph_ref, e0, b_star, args, python_row):
    try:
        r = eval_detector("vocd", dd, vd, ph_ref, e0, b_star, args,
                          vocd_backend=VOCD_FAST_BACKEND)
    except Exception as e:
        print(f"! cpp backend unavailable ({e}); skipping 'vocd (cpp)' row")
        return None
    same = r["_sets"] == python_row["_sets"]
    if same:
        print("  vocd (cpp): boundary sets identical to python (only runtime differs)")
    else:
        print("! WARNING: vocd python vs cpp boundary sets differ — 'vocd (cpp)' "
              "is NOT a pure timing twin; investigate.")
    return r


def budget_grid(name, alpha_grid, hazard_grid):
    if name in ("bocd", "dm-bocd"):
        return hazard_grid, "hazard"
    if name == "vocd":
        return alpha_grid, "fwer"
    return alpha_grid, "alpha"


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/isfet",
                    help="directory holding {ion}_mv.csv and ref_mv.csv")
    ap.add_argument("--ion", default="h", help="ISFET ion (h, na, k)")
    ap.add_argument("--dsm-path", default=None,
                    help="clone of maltamiranomontero/DSM-bocd (for BOCD / Dm-BOCD)")
    ap.add_argument("--methods", default=",".join(DETECTORS))
    ap.add_argument("--no-baselines", action="store_true",
                    help="omit the none/two-point/M0/exp/oracle context rows")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="single-point mode: fwer(vocd)/alpha(alarm) budget")
    ap.add_argument("--min-seg", type=int, default=4,
                    help="min days between change-points / per-segment burn-in")
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--twopoint-period", type=float, default=7.0)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    # sweep
    ap.add_argument("--sweep", action="store_true",
                    help="sweep each detector's budget; report min-RMSE point")
    ap.add_argument("--alpha-grid",
                    default="1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1")
    ap.add_argument("--hazard-grid", default="5,10,20,30,50,80",
                    help="hazard timescales swept for BOCD (days-scale for this "
                         "record); Dm-BOCD inherits BOCD's best")
    ap.add_argument("--dm-hazard", type=float, default=None,
                    help="fix Dm-BOCD's hazard (default: inherit BOCD's best)")
    ap.add_argument("--sweep-out", default="results/isfet/isfet_sweep.csv")
    ap.add_argument("--out", default="results/isfet/isfet_best.csv")
    args = ap.parse_args()
    args._dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in wanted:
        if m not in DETECTORS:
            raise SystemExit(f"unknown detector {m!r}; choose from {DETECTORS}")
    needs = [m for m in wanted if REGISTRY.get(m, (None, False))[1]]
    if needs and not args._dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}\n"
              "  git clone https://github.com/maltamiranomontero/DSM-bocd.git\n")
        wanted = [m for m in wanted if m not in needs]

    dd, vd, ph_ref, e0 = load_daily(args.data_dir, args.ion)
    has_vocd = "vocd" in wanted

    print(f"ISFET drift correction (reference-free / Condition B) — ion={args.ion}, "
          f"{dd.size} days, burn-in {args.burn_in}d")
    if has_vocd:
        print(f"vocd row = {VOCD_FAIR_BACKEND} backend; 'vocd (cpp)' = same "
              f"boundaries, {VOCD_FAST_BACKEND} runtime")
    print(f"detectors: {', '.join(wanted)}\n")

    rows, budgets = {}, {}
    curve_rows = []

    # context baselines (once)
    if not args.no_baselines:
        rows.update(eval_baselines(dd, vd, ph_ref, e0, args))

    if not args.sweep:
        print(f"single operating point: alpha={args.alpha:g}\n")
        for name in wanted:
            print(f"[{name}] ...", end="", flush=True)
            r = eval_detector(name, dd, vd, ph_ref, e0, args.alpha, args)
            rows[name] = r
            budgets[name] = args.alpha
            print(f" RMSE={r['rmse_mv']:.3f} mV")
        det_order = list(wanted)
        if has_vocd:
            cpp = vocd_cpp_paired(dd, vd, ph_ref, e0, args.alpha, args, rows["vocd"])
            if cpp:
                rows["vocd (cpp)"] = cpp
                budgets["vocd (cpp)"] = args.alpha
                det_order = _insert_cpp(wanted)
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
                if args.dm_hazard is not None:
                    h, src = args.dm_hazard, "--dm-hazard"
                elif "bocd" in best_by:
                    h, src = best_by["bocd"], "inherited from bocd best"
                else:
                    h, src = 30.0, "fallback (no bocd swept)"
                grid, kind = [h], "hazard"
                print(f"[{name}] (hazard={h:g}, {src}) ", end="", flush=True)
            else:
                grid, kind = budget_grid(name, alpha_grid, hazard_grid)
                print(f"[{name}] ({kind}) ", end="", flush=True)

            evals = []
            for b in grid:
                r = eval_detector(name, dd, vd, ph_ref, e0, b, args)
                curve_rows.append({"method": name, "budget_kind": kind,
                                   "budget": b,
                                   **{k: r[k] for k in
                                      ["rmse_mv", "rmse_ph", "mae_mv", "beta",
                                       "n_cp", "runtime_ms"]}})
                evals.append((b, r))
                print(".", end="", flush=True)
            # central point of the MIN-RMSE plateau (mirrors synthetic's logic,
            # minimising RMSE instead of maximising F1)
            tol = 1e-9
            min_rmse = min(e[1]["rmse_mv"] for e in evals)
            cand = [e for e in evals if e[1]["rmse_mv"] <= min_rmse + tol]
            cand.sort(key=lambda e: e[0])
            b_star, r_star = cand[len(cand) // 2]
            rows[name] = r_star
            budgets[name] = b_star
            best_by[name] = b_star
            plateau = "" if len(cand) == 1 else f" ({len(cand)}-pt plateau)"
            print(f" min RMSE={min_rmse:.3f} mV @ {kind}={b_star:g}{plateau}")

        det_order = list(wanted)
        if has_vocd:
            cpp = vocd_cpp_paired(dd, vd, ph_ref, e0, budgets["vocd"], args, rows["vocd"])
            if cpp:
                rows["vocd (cpp)"] = cpp
                budgets["vocd (cpp)"] = budgets["vocd"]
                det_order = _insert_cpp(wanted)

    # assemble display order: context rows bracket the detectors
    ctx_top = [m for m in ("none", "two-point", "M0", "exp") if m in rows]
    ctx_bot = [m for m in ("oracle",) if m in rows]
    order = ctx_top + det_order + ctx_bot

    print_table(order, rows, budgets)
    print("scored on drift RMSE vs the reference (mV) over the post-burn-in "
          "window.\ndetectors supply boundaries for the shared causal piecewise "
          "correction;\n'vocd (cpp)' reuses vocd's budget+boundaries, only runtime "
          "differs.\nnear-linear drift here => tight spread expected; model-"
          "agnosticism is a\ngas-sensor claim, not an ISFET one.")

    write_rows(args.out, order, rows, budgets)
    print(f"\nwrote {len(order)} rows -> {args.out}")
    if curve_rows:
        write_curve(args.sweep_out, curve_rows)
        print(f"wrote {len(curve_rows)} curve rows -> {args.sweep_out}")


def _insert_cpp(order):
    out = []
    for m in order:
        out.append(m)
        if m == "vocd":
            out.append("vocd (cpp)")
    return out


if __name__ == "__main__":
    main()
