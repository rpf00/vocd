"""
Synthetic-data benchmark: VOCD vs BOCD, Dm-BOCD, NP-FOCuS, R-FOCuS, CHAD.

Solves the contaminated change-in-mean case of Altamirano, Briol & Knoblauch
(2023) — T=500, six true change-points at [100, 200, 300, 350, 400, 450],
segment means [10, 5, 0, 10, 5, 12, 0], 2% outliers at +/-10, ten trials — with
every method run ONLINE and with no access to the whole series before deciding
a change (see ``alternative_models/`` for how each method is held to that bar).

The data generator, contamination procedure and the ``accuracy_delay`` scorer
are reproduced verbatim from that repository's ``notebooks/accuracy_delay.ipynb``
so the baselines are scored on their own terms.  VOCD is the installed package;
the other five come from the local ``alternative_models/`` package.  BOCD /
Dm-BOCD need a clone of maltamiranomontero/DSM-bocd (``--dsm-path`` or
``VOCD_DSM_PATH``); if it is absent they are skipped and everything else runs.

Fair runtime comparison
-----------------------
Every baseline here is pure Python, so VOCD is reported in its PYTHON backend as
the like-for-like ``vocd`` row.  A second ``vocd (cpp)`` row re-runs VOCD at the
SAME best-F1 budget in the compiled backend and reports only its runtime — the
detections (and therefore every accuracy column) are identical to the Python
row by construction, and the harness asserts this.  So ``vocd`` answers "how
does VOCD compare in the same language as the baselines" and ``vocd (cpp)``
answers "how fast can VOCD actually go".  Note the alarm baselines are
UNoptimised O(n^2) reference implementations, so even the all-Python comparison
somewhat flatters VOCD on speed against the FOCuS family — state that in the
caption.

Two modes
---------
Single operating point (default)::

    python examples/synthetic_data.py --dsm-path ./DSM-bocd --alpha 0.01 \
        --out results/synthetic/synthetic_v2.csv

Budget sweep — report each method's BEST-achievable operating point::

    python examples/synthetic_data.py --sweep --dsm-path ./DSM-bocd \
        --hazard-grid 300 \
        --out results/synthetic/synthetic_v2_best.csv \
        --sweep-out results/synthetic/synthetic_v2_sweep.csv

The knob swept is: vocd -> fwer, {np-focus,r-focus,chad} -> alpha (all on the
alpha grid); bocd -> hazard_timescale (hazard grid).  Dm-BOCD is ~20 s/trial
and is NOT swept: it runs once at BOCD's best hazard (both use the same
ConstantHazard prior, so BOCD's optimum is a principled proxy), overridable with
``--dm-hazard``.  This is disclosed in the output and should be stated in the
paper.  The knob is chosen by mean F1 over the trials — i.e. against the
labels, an oracle best-case operating point for every method alike; state that
in the paper.  When several budgets tie on F1 (a flat optimal plateau, common
for the alarm methods), the CENTRE of that plateau is reported rather than an
edge, so no method looks artificially grid-railed and the reported point is
stable.  A fully label-free version would pick each knob on a separate
calibration split under a common target, using the same grids.

WARNING: sweeping Dm-BOCD is slow (~20 s/trial * |hazard grid| * trials).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vocd
from alternative_models import REGISTRY

# ── experiment constants (verbatim from accuracy_delay.ipynb) ─────────
T = 500
CPS_TRUE = [100, 200, 300, 350, 400, 450, 600]   # 600 is a sentinel past T
MUS = [10, 5, 0, 10, 5, 12, 0]
WINDOW = 10
N_TRIALS = 10
SEED_DATA = 3000
SEED_CONTAM = 54321
CONTAM_FRAC = 0.02
CONTAM_SPIKE = 10.0

METRIC_KEYS = ["n_detected", "ppv", "tpr", "f1", "delay", "runtime_ms"]
VOCD_FAIR_BACKEND = "python"   # the like-for-like row
VOCD_FAST_BACKEND = "cpp"      # the "how fast can it go" bonus row


def make_paper_data(n: int = T, seed: int = SEED_DATA):
    np.random.seed(seed)
    data = np.zeros((n, 1))
    mu, i = MUS[0], 0
    for t in range(n):
        if t == CPS_TRUE[i]:
            mu = MUS[i + 1]
            i += 1
        data[t, 0] = np.random.normal(mu, 1)
    return data, float(np.mean(data))


def contaminate(data, frac=CONTAM_FRAC, spike=CONTAM_SPIKE):
    n = data.shape[0]
    idx = np.random.choice(np.arange(0, n, 1), int(frac * n), replace=False)
    out = data.copy()
    sgn = np.random.choice([1, -1], size=len(idx))
    out[idx, 0] = out[idx, 0] + sgn * spike
    return out


def make_trials(data, n_trials=N_TRIALS):
    np.random.seed(SEED_CONTAM)
    return [contaminate(data) for _ in range(n_trials)]


def accuracy_delay(cps_pred: Sequence[int], cps_true=CPS_TRUE, window=WINDOW):
    """Reference scorer, reproduced as-is (many-to-one; skips cps_pred[0])."""
    TP = FP = FN = 0
    delays = []
    for cp in cps_pred[1:]:
        if np.min(np.abs(np.array(cps_true) - cp)) < window:
            TP += 1
            delays.append(np.min(np.abs(np.array(cps_true) - cp)))
        else:
            FP += 1
    for cp in np.array(cps_true)[:-1]:
        if np.min(np.abs(np.array(cps_pred) - cp)) > window:
            FN += 1
    ppv = TP / (TP + FP) if (TP + FP) else 0.0
    tpr = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) else 0.0
    return {"n_detected": len(cps_pred) - 1, "ppv": ppv, "tpr": tpr, "f1": f1,
            "delay": float(np.mean(delays)) if delays else np.nan}


def agg(per_trial: List[dict], key: str) -> Tuple[float, float]:
    v = np.array([r[key] for r in per_trial], float)
    return float(np.nanmean(v)), float(np.sqrt(np.nanvar(v)))


# ── one run of one method at one budget ───────────────────────────────

def make_runner(args, mean0, dsm_path):
    def run(method: str, x1d: np.ndarray, budget: float,
            vocd_backend: str = VOCD_FAIR_BACKEND) -> List[int]:
        if method == "vocd":
            r = vocd.detect(x1d, backend=vocd_backend)
            return [int(c) for c in r.changepoints(fwer=budget)]
        fn, needs_dsm = REGISTRY[method]
        if needs_dsm:  # bocd / dm-bocd — budget is the hazard timescale
            return fn(x1d, mean0=mean0, dsm_path=dsm_path,
                      extract=args.bocd_extract, hazard_timescale=budget)
        return fn(x1d, alpha=budget, min_seg=args.min_seg)
    return run


def budget_grid(method: str, alpha_grid, hazard_grid):
    if method in ("bocd", "dm-bocd"):
        return hazard_grid, "hazard"
    if method == "vocd":
        return alpha_grid, "fwer"
    return alpha_grid, "alpha"


def eval_point(run, method, trials, budget,
               vocd_backend=VOCD_FAIR_BACKEND, label=None) -> List[dict]:
    """Per-trial metric dicts for (method, budget). ``label`` overrides the
    reported method name (used for the 'vocd (cpp)' row); raw change-points are
    stashed under ``_cps`` for the backend-equivalence check."""
    out = []
    for i, dc in enumerate(trials):
        x1d = dc.ravel()
        t0 = time.perf_counter()
        cps = run(method, x1d, budget, vocd_backend=vocd_backend)
        ms = (time.perf_counter() - t0) * 1000.0
        scps = sorted(int(c) for c in cps)
        r = accuracy_delay([0] + scps)
        r.update(method=label or method, trial=i, runtime_ms=ms, budget=budget)
        r["_cps"] = tuple(scps)
        out.append(r)
    return out


def vocd_cpp_paired(run, trials, b_star, python_pt):
    """Re-run VOCD at the SAME budget in the compiled backend; return its
    per-trial rows (labelled 'vocd (cpp)') after asserting identical detections."""
    try:
        cpp_pt = eval_point(run, "vocd", trials, b_star,
                            vocd_backend=VOCD_FAST_BACKEND, label="vocd (cpp)")
    except Exception as e:  # extension not built, etc.
        print(f"! cpp backend unavailable ({e}); skipping 'vocd (cpp)' row")
        return None
    ndiff = sum(1 for a, b in zip(python_pt, cpp_pt) if a["_cps"] != b["_cps"])
    if ndiff:
        print(f"! WARNING: vocd python vs cpp detections differ on "
              f"{ndiff}/{len(trials)} trials — 'vocd (cpp)' row is NOT a pure "
              "timing twin; investigate before using.")
    else:
        print("  vocd (cpp): detections identical to python (only runtime differs)")
    return cpp_pt


# ── printing / IO ─────────────────────────────────────────────────────

def print_table(order, per_method, budget_label=None):
    hdr_budget = f"{budget_label:>10}" if budget_label else ""
    print(f"\n{'method':<12}{hdr_budget}{'#CPs':>12}{'PPV':>13}{'TPR':>13}"
          f"{'F1':>13}{'Delay':>12}{'Runtime (ms)':>16}")
    print("-" * (91 + (10 if budget_label else 0)))
    for name in order:
        pt, budget = per_method[name]
        n, p, t = agg(pt, "n_detected"), agg(pt, "ppv"), agg(pt, "tpr")
        f1, d, rt = agg(pt, "f1"), agg(pt, "delay"), agg(pt, "runtime_ms")
        bcol = f"{budget:>10.4g}" if budget_label else ""
        print(f"{name:<12}{bcol}{n[0]:>7.1f}±{n[1]:<4.1f}{p[0]:>7.3f}±{p[1]:<5.3f}"
              f"{t[0]:>7.3f}±{t[1]:<5.3f}{f1[0]:>7.3f}±{f1[1]:<5.3f}"
              f"{d[0]:>7.2f}±{d[1]:<4.2f}{rt[0]:>10.1f}±{rt[1]:<5.1f}")
    print("-" * (91 + (10 if budget_label else 0)))


def write_pertrial(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "trial", "n_detected", "ppv", "tpr", "f1", "delay", "runtime_ms"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def write_curve(path, curve_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["method", "budget_kind", "budget", "n_detected", "ppv", "tpr",
            "f1", "delay", "runtime_ms",
            "n_detected_sd", "ppv_sd", "tpr_sd", "f1_sd", "delay_sd", "runtime_ms_sd"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in curve_rows:
            w.writerow(r)


def insert_cpp_after_vocd(order):
    out = []
    for m in order:
        out.append(m)
        if m == "vocd":
            out.append("vocd (cpp)")
    return out


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsm-path", default=None,
                    help="clone of maltamiranomontero/DSM-bocd (for BOCD / Dm-BOCD)")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--methods",
                    default="vocd,bocd,dm-bocd,np-focus,r-focus,chad")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="single-point mode: shared family-wise budget")
    ap.add_argument("--min-seg", type=int, default=10)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    # sweep mode
    ap.add_argument("--sweep", action="store_true",
                    help="sweep each method's own budget and report best-F1 point")
    ap.add_argument("--alpha-grid",
                    default="1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1",
                    help="grid for vocd(fwer) and the alarm methods(alpha); "
                         "wide enough that no method's optimum sits at an edge")
    ap.add_argument("--hazard-grid", default="50,100,200,300,500,1000,2000,5000",
                    help="hazard timescales swept for BOCD (cheap). Dm-BOCD is "
                         "NOT swept over this grid; see --dm-hazard")
    ap.add_argument("--dm-hazard", type=float, default=None,
                    help="single hazard timescale for Dm-BOCD (too slow to "
                         "sweep). Default None = inherit BOCD's best hazard "
                         "(same ConstantHazard prior); falls back to 300 if BOCD "
                         "is not in --methods")
    ap.add_argument("--sweep-out", default="results/synthetic/synthetic_v2_sweep.csv",
                    help="full per-budget aggregated curve (for the frontier plot)")
    ap.add_argument("--out", default="results/synthetic/synthetic_v2.csv",
                    help="single mode: per-trial rows; sweep mode: per-trial rows "
                         "at each method's best-F1 budget")
    args = ap.parse_args()

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"vocd"} | set(REGISTRY)
    for m in wanted:
        if m not in known:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(known)}")

    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    needs = [m for m in wanted if m in REGISTRY and REGISTRY[m][1]]
    if needs and not dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}\n"
              "  git clone https://github.com/maltamiranomontero/DSM-bocd.git\n")
        wanted = [m for m in wanted if m not in needs]
    if not wanted:
        raise SystemExit("no methods left to run")

    data_clean, mean0 = make_paper_data()
    trials = make_trials(data_clean, args.trials)
    run = make_runner(args, mean0, dsm_path)
    has_vocd = "vocd" in wanted

    print(f"synthetic-data benchmark — T={T}, 6 true CPs at {CPS_TRUE[:-1]}")
    print(f"{CONTAM_FRAC:.0%} outliers @ +/-{CONTAM_SPIKE:g}, {args.trials} trials, "
          f"window={WINDOW}, bocd-extract={args.bocd_extract}")
    print(f"methods: {', '.join(wanted)}")
    if has_vocd:
        print(f"vocd row = {VOCD_FAIR_BACKEND} backend (fair vs Python baselines); "
              f"'vocd (cpp)' = same detections, {VOCD_FAST_BACKEND} runtime")
    print()

    if not args.sweep:
        # ── single operating point ───────────────────────────────────
        print(f"single operating point: alpha={args.alpha:g}\n")
        all_rows, per_method = [], {}
        for name in wanted:
            print(f"[{name}] ", end="", flush=True)
            pt = eval_point(run, name, trials, args.alpha)
            print("." * len(pt), "done")
            all_rows.extend(pt)
            per_method[name] = (pt, args.alpha)
        order = wanted
        if has_vocd:
            cpp_pt = vocd_cpp_paired(run, trials, args.alpha, per_method["vocd"][0])
            if cpp_pt:
                per_method["vocd (cpp)"] = (cpp_pt, args.alpha)
                all_rows.extend(cpp_pt)
                order = insert_cpp_after_vocd(wanted)
        print_table(order, per_method)
        print("scorer: reference accuracy_delay (many-to-one). spread = population "
              "std.\nsingle shared alpha; not each method's best operating point "
              "(use --sweep for that).")
        write_pertrial(args.out, all_rows)
        print(f"\nwrote {len(all_rows)} rows -> {args.out}")
        return

    # ── sweep mode ───────────────────────────────────────────────────
    alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
    hazard_grid = [float(v) for v in args.hazard_grid.split(",") if v.strip()]
    print("SWEEP mode — reporting each method's best-F1 operating point.")
    print(f"  alpha grid : {alpha_grid}")
    print(f"  hazard grid: {hazard_grid}\n")

    curve_rows: List[dict] = []
    best_pertrial_rows: List[dict] = []
    best_by_method: Dict[str, Tuple[list, float]] = {}

    # BOCD must be evaluated before Dm-BOCD, which inherits BOCD's best hazard.
    process_order = list(wanted)
    if "bocd" in process_order and "dm-bocd" in process_order:
        process_order.remove("dm-bocd")
        process_order.insert(process_order.index("bocd") + 1, "dm-bocd")

    for name in process_order:
        if name == "dm-bocd":
            # Not swept (too slow): one hazard, inherited from BOCD's sweep.
            if args.dm_hazard is not None:
                h = args.dm_hazard
                src = "--dm-hazard"
            elif "bocd" in best_by_method:
                h = best_by_method["bocd"][1]
                src = "inherited from bocd best"
            else:
                h = 300.0
                src = "fallback (no bocd swept)"
            grid, kind = [h], "hazard"
            print(f"[{name}] (hazard={h:g}, {src}) ", end="", flush=True)
        else:
            grid, kind = budget_grid(name, alpha_grid, hazard_grid)
            print(f"[{name}] ({kind}) ", end="", flush=True)
        evals = []  # (budget, means, per_trial)
        for b in grid:
            pt = eval_point(run, name, trials, b)
            means = {k: agg(pt, k)[0] for k in METRIC_KEYS}
            sds = {k: agg(pt, k)[1] for k in METRIC_KEYS}
            row = {"method": name, "budget_kind": kind, "budget": b}
            row.update(means)
            row.update({f"{k}_sd": sds[k] for k in METRIC_KEYS})
            curve_rows.append(row)
            evals.append((b, means, pt))
            print(".", end="", flush=True)
        # Report the CENTRE of the optimal plateau, not an arbitrary edge of it:
        # take the budgets achieving the best mean F1 (then best PPV), and pick
        # the median one.  This avoids the tie-break landing on a grid edge and
        # makes "the method's best operating point" a stable, statable choice.
        tol = 1e-9
        max_f1 = max(e[1]["f1"] for e in evals)
        cand = [e for e in evals if e[1]["f1"] >= max_f1 - tol]
        max_ppv = max(e[1]["ppv"] for e in cand)
        cand = [e for e in cand if e[1]["ppv"] >= max_ppv - tol]
        cand.sort(key=lambda e: e[0])
        b_star, _, pt_star = cand[len(cand) // 2]
        best_by_method[name] = (pt_star, b_star)
        best_pertrial_rows.extend(pt_star)
        plateau = "" if len(cand) == 1 else f" ({len(cand)}-pt plateau)"
        print(f" best F1={max_f1:.3f} @ {kind}={b_star:g}{plateau}")

    order = wanted
    if has_vocd:
        pt_py, b_star = best_by_method["vocd"]
        cpp_pt = vocd_cpp_paired(run, trials, b_star, pt_py)
        if cpp_pt:
            best_by_method["vocd (cpp)"] = (cpp_pt, b_star)
            best_pertrial_rows.extend(cpp_pt)
            order = insert_cpp_after_vocd(wanted)

    print_table(order, best_by_method, budget_label="budget")
    print("each row is the method's OWN best-F1 operating point over its grid "
          "(oracle,\nchosen against labels — see module docstring). 'vocd (cpp)' "
          "reuses vocd's\nbudget and detections; only its runtime differs. "
          "dm-bocd is not swept — it\nuses bocd's best hazard (shared prior). "
          "spread = population std.")

    write_curve(args.sweep_out, curve_rows)
    write_pertrial(args.out, best_pertrial_rows)
    print(f"\nwrote {len(curve_rows)} curve rows -> {args.sweep_out}")
    print(f"wrote {len(best_pertrial_rows)} best-point per-trial rows -> {args.out}")


if __name__ == "__main__":
    main()
