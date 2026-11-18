"""
MOX gas-sensor drift-correction benchmark: VOCD vs BOCD, Dm-BOCD, NP-FOCuS,
R-FOCuS, CHAD (Worner et al. 2025).

The structural twin of ``examples/isfet_data.py``, using the SAME vocd package
and the SAME alternative_models panel — so gas is apples-to-apples with the
synthetic and ISFET experiments.  The viterbi_mwks / .so path of the original
gas_sensor_stats script is NOT used; every detector here is the pip ``vocd``
(MW/KS rank gate) or an ``alternative_models`` reference detector.

Task (verbatim from gas_sensor_stats_v2.py)
-------------------------------------------
Each measurement is a ~1200-pt raw resistance curve R(t) with three stages:
baseline (clean air) -> sample (gas) -> recovery (purge).  Ground truth is the
``cycle_stage`` column.  Drift correction is additive baseline offset:

    day 1  : reference.  ref_R = R, ref_base_mean = mean(R over baseline).
    day t>1: a detector finds the baseline->gas transition (first change-point)
             from R(t) ALONE; base_mask = samples before it;
             offset = mean(R[base_mask]) - ref_base_mean ;  R_corr = R - offset.
    score  : MAE, RMSE of (R_corr - ref_R) in Ohm, averaged over days t>1.
             LOWER is better.

Only the baseline transition (first CP) is needed for the correction, so every
detector reduces to "where does the sensor leave baseline", which is exactly the
kind of sharp mean shift the whole panel detects — and is why this record can
finally exercise VOCD's model-agnosticism (the drift here is NOT a clean ramp).

Panel & fairness (identical to ISFET)
-------------------------------------
vocd (+ paired ``vocd (cpp)`` = same detections, cpp runtime), bocd, dm-bocd,
np-focus, r-focus, chad, plus the ``oracle`` row (true cycle_stage baseline — the
ceiling).  ``--sweep`` gives each detector its own best budget (vocd->fwer,
alarm->alpha, bocd->hazard swept, dm-bocd inherits bocd's best), minimising mean
MAE, with central-plateau selection.  Runtime reports vocd in the PYTHON backend
for a like-for-like comparison; ``vocd (cpp)`` is the compiled bonus.

Causality: each measurement is processed as a whole (you get the full 20-min
curve, then correct it before the next); no stage labels and no future days are
used.  This matches the original benchmark's online-across-days framing.

Depends on ``gas_sensor_data.py`` (loaders) and the Worner dataset on disk
(see --data-dir).  NOT executable without them.

Usage
-----
    python examples/gas_data.py --sensor 0 --analyte Diacetyl_1ppm
    python examples/gas_data.py --sweep --dsm-path ./DSM-bocd \
        --out results/gas/gas_best.csv --sweep-out results/gas/gas_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vocd
from alternative_models import REGISTRY
from gas_sensor_data import (ANALYTE_LABELS,
                             parse_measurement, _canonical_analyte)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_GAS_DIR = os.path.join(_REPO, "data", "gas_sensor_worner2025")

DETECTORS = ["vocd", "bocd", "dm-bocd", "np-focus", "r-focus", "chad"]
VOCD_FAIR_BACKEND = "python"
VOCD_FAST_BACKEND = "cpp"
MIN_BASE = 10   # min baseline samples for a usable correction


# ── data loaders (reimplemented from gas_sensor_stats_v2.py) ──────────

def get_day_folders(data_dir):
    def key(d):
        n = re.findall(r"\d+", d)
        return int(n[0]) if n else 0
    return sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and re.search(r"\d", d)],
                  key=key)


def load_raw_measurement(data_dir, day_folder, analyte, replicate, sensor_id):
    folder = os.path.join(data_dir, day_folder)
    cands = sorted(f for f in os.listdir(folder)
                   if f.endswith(".csv") and _canonical_analyte(f) == analyte)
    if not cands:
        raise FileNotFoundError(f"no CSV for {analyte} in {folder}")
    path = os.path.join(folder, cands[min(replicate - 1, len(cands) - 1)])
    df = parse_measurement(path)
    scol = f"R{sensor_id + 1}"
    if scol not in df.columns:
        raise ValueError(f"{scol} not in {path}")
    r = df[scol].values.astype(np.float64)
    cs = (df["cycle_stage"].values.astype(int) if "cycle_stage" in df.columns
          else np.ones(len(r), dtype=int))
    return r, cs


def load_all_days(data_dir, analyte, replicate, sensor_id):
    days = get_day_folders(data_dir)
    out = []
    for d in days:
        try:
            r, cs = load_raw_measurement(data_dir, d, analyte, replicate, sensor_id)
            out.append((d, r, cs))
        except Exception as e:
            print(f"  ! skip {d}: {e}")
    if len(out) < 2:
        sys.exit("need >=2 days (day-1 reference + >=1 test day)")
    return out


# ── correction + scoring (verbatim from gas_sensor_stats_v2.py) ───────

def base_mask_from_cps(cps, T):
    cps = sorted(int(c) for c in cps if 0 < c < T)
    if not cps:
        return None
    m = np.zeros(T, dtype=bool)
    m[:cps[0]] = True
    return m if m.sum() >= MIN_BASE else None


def correct_measurement(r, base_mask, ref_base_mean):
    if base_mask is None or base_mask.sum() < 5:
        return r.copy()
    offset = float(r[base_mask].mean()) - ref_base_mean
    return r - offset


def curve_metrics(rc, rr):
    n = min(len(rc), len(rr))
    d = rc[:n] - rr[:n]
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


# ── detector -> change-points on one measurement ──────────────────────

def measure_cps(name, r, budget, args, vocd_backend=VOCD_FAIR_BACKEND):
    if name == "vocd":
        try:
            res = vocd.detect(r, backend=vocd_backend)
            return [int(c) for c in res.changepoints(fwer=budget)]
        except Exception:
            return []
    fn, needs_dsm = REGISTRY[name]
    if needs_dsm:
        return fn(r, mean0=None, dsm_path=args._dsm_path,
                  extract=args.bocd_extract, hazard_timescale=budget)
    return fn(r, alpha=budget, min_seg=args.min_seg)


def eval_method(name, days, budget, args, vocd_backend=VOCD_FAIR_BACKEND):
    """Mean MAE/RMSE over days t>1 for one (method, budget). Oracle uses labels."""
    d0, r0, cs0 = days[0]
    if name == "oracle":
        bm0 = cs0 == 1
    else:
        bm0 = base_mask_from_cps(measure_cps(name, r0, budget, args, vocd_backend),
                                 len(r0))
    if bm0 is None or bm0.sum() < 5:      # day-1 fallback: true baseline
        bm0 = cs0 == 1
    ref_base_mean = float(r0[bm0].mean())
    ref_r = r0

    t0 = time.perf_counter()
    maes, rmses = [], []
    for _, r, cs in days[1:]:
        if name == "oracle":
            bm = cs == 1
        else:
            bm = base_mask_from_cps(measure_cps(name, r, budget, args, vocd_backend),
                                    len(r))
        rc = correct_measurement(r, bm, ref_base_mean)
        mae, rmse = curve_metrics(rc, ref_r)
        maes.append(mae)
        rmses.append(rmse)
    ms = (time.perf_counter() - t0) * 1000.0
    return {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
            "n_days": len(maes), "runtime_ms": ms, "_maes": tuple(maes)}


# ── sweep helpers (identical logic to isfet_data.py) ──────────────────

def budget_grid(name, alpha_grid, hazard_grid):
    if name in ("bocd", "dm-bocd"):
        return hazard_grid, "hazard"
    if name == "vocd":
        return alpha_grid, "fwer"
    return alpha_grid, "alpha"


def vocd_cpp_paired(days, b_star, args, python_row):
    try:
        r = eval_method("vocd", days, b_star, args, vocd_backend=VOCD_FAST_BACKEND)
    except Exception as e:
        print(f"! cpp backend unavailable ({e}); skipping 'vocd (cpp)'")
        return None
    if r["_maes"] == python_row["_maes"]:
        print("  vocd (cpp): corrections identical to python (only runtime differs)")
    else:
        print("! WARNING: vocd python vs cpp corrections differ — not a timing twin")
    return r


def print_table(order, rows, budgets):
    print(f"\n{'method':<12}{'budget':>10}{'mean MAE':>12}{'mean RMSE':>12}"
          f"{'days':>7}{'Runtime(ms)':>14}")
    print("-" * 67)
    for name in order:
        r = rows[name]
        b = budgets.get(name)
        bcol = f"{b:>10.4g}" if b is not None else f"{'—':>10}"
        print(f"{name:<12}{bcol}{r['mae']:>12.2f}{r['rmse']:>12.2f}"
              f"{r['n_days']:>7}{r['runtime_ms']:>14.1f}")
    print("-" * 67)


def write_rows(path, order, rows, budgets):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget", "mae", "rmse", "n_days", "runtime_ms"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name in order:
            r = rows[name]
            w.writerow({"method": name, "budget": budgets.get(name, ""),
                        "mae": r["mae"], "rmse": r["rmse"],
                        "n_days": r["n_days"], "runtime_ms": r["runtime_ms"]})


def write_curve(path, curve):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget_kind", "budget", "mae", "rmse", "n_days", "runtime_ms"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in curve:
            w.writerow(r)


def _insert_cpp(order):
    out = []
    for m in order:
        out.append(m)
        if m == "vocd":
            out.append("vocd (cpp)")
    return out


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_GAS_DIR)
    ap.add_argument("--sensor", type=int, default=0, help="sensor id (0-based -> R1)")
    ap.add_argument("--analyte", default="Diacetyl_1ppm", choices=list(ANALYTE_LABELS))
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=",".join(DETECTORS))
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-seg", type=int, default=30,
                    help="min samples between change-points (per-measurement)")
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--alpha-grid",
                    default="1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1")
    ap.add_argument("--hazard-grid", default="100,300,600,1000,2000",
                    help="hazard timescales swept for BOCD (~1200-pt measurements)")
    ap.add_argument("--dm-hazard", type=float, default=None)
    ap.add_argument("--sweep-out", default="results/gas/gas_sweep.csv")
    ap.add_argument("--out", default="results/gas/gas_best.csv")
    args = ap.parse_args()
    args._dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in wanted:
        if m not in DETECTORS:
            raise SystemExit(f"unknown detector {m!r}; choose from {DETECTORS}")
    needs = [m for m in wanted if REGISTRY.get(m, (None, False))[1]]
    if needs and not args._dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}\n")
        wanted = [m for m in wanted if m not in needs]

    print(f"loading gas data: sensor R{args.sensor+1}, analyte {args.analyte}, "
          f"replicate {args.replicate}")
    days = load_all_days(args.data_dir, args.analyte, args.replicate, args.sensor)
    print(f"  {len(days)} days ({len(days)-1} test days), "
          f"~{len(days[0][1])} pts/measurement")
    has_vocd = "vocd" in wanted
    if has_vocd:
        print(f"vocd row = {VOCD_FAIR_BACKEND} backend; 'vocd (cpp)' = same "
              f"corrections, {VOCD_FAST_BACKEND} runtime")
    print(f"detectors: {', '.join(wanted)}\n")

    rows, budgets, curve = {}, {}, []
    if not args.no_oracle:
        rows["oracle"] = eval_method("oracle", days, None, args)

    if not args.sweep:
        print(f"single operating point: alpha={args.alpha:g}\n")
        for name in wanted:
            print(f"[{name}] ...", end="", flush=True)
            rows[name] = eval_method(name, days, args.alpha, args)
            budgets[name] = args.alpha
            print(f" MAE={rows[name]['mae']:.2f}")
        det_order = list(wanted)
        if has_vocd:
            cpp = vocd_cpp_paired(days, args.alpha, args, rows["vocd"])
            if cpp:
                rows["vocd (cpp)"] = cpp
                budgets["vocd (cpp)"] = args.alpha
                det_order = _insert_cpp(wanted)
    else:
        alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
        hazard_grid = [float(v) for v in args.hazard_grid.split(",") if v.strip()]
        print("SWEEP mode — each detector's MIN-MAE operating point.")
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
                    h, src = 600.0, "fallback"
                grid, kind = [h], "hazard"
                print(f"[{name}] (hazard={h:g}, {src}) ", end="", flush=True)
            else:
                grid, kind = budget_grid(name, alpha_grid, hazard_grid)
                print(f"[{name}] ({kind}) ", end="", flush=True)
            evals = []
            for b in grid:
                r = eval_method(name, days, b, args)
                curve.append({"method": name, "budget_kind": kind, "budget": b,
                              "mae": r["mae"], "rmse": r["rmse"],
                              "n_days": r["n_days"], "runtime_ms": r["runtime_ms"]})
                evals.append((b, r))
                print(".", end="", flush=True)
            tol = 1e-9
            best_mae = min(e[1]["mae"] for e in evals)
            cand = sorted([e for e in evals if e[1]["mae"] <= best_mae + tol],
                          key=lambda e: e[0])
            b_star, r_star = cand[len(cand) // 2]
            rows[name] = r_star
            budgets[name] = b_star
            best_by[name] = b_star
            plateau = "" if len(cand) == 1 else f" ({len(cand)}-pt plateau)"
            print(f" min MAE={best_mae:.2f} @ {kind}={b_star:g}{plateau}")
        det_order = list(wanted)
        if has_vocd:
            cpp = vocd_cpp_paired(days, budgets["vocd"], args, rows["vocd"])
            if cpp:
                rows["vocd (cpp)"] = cpp
                budgets["vocd (cpp)"] = budgets["vocd"]
                det_order = _insert_cpp(wanted)

    order = det_order + (["oracle"] if "oracle" in rows else [])
    print_table(order, rows, budgets)
    print("mean MAE/RMSE of corrected curve vs day-1 reference (Ohm), over test "
          "days.\ndetectors supply the baseline transition for the shared additive "
          "correction;\n'vocd (cpp)' reuses vocd's budget+corrections, only runtime "
          "differs.\noracle uses ground-truth cycle_stage (non-causal ceiling). "
          "unlike ISFET, gas\ndrift is non-linear — this is where model-agnosticism "
          "can actually show.")
    write_rows(args.out, order, rows, budgets)
    print(f"\nwrote {len(order)} rows -> {args.out}")
    if curve:
        write_curve(args.sweep_out, curve)
        print(f"wrote {len(curve)} curve rows -> {args.sweep_out}")


if __name__ == "__main__":
    main()
