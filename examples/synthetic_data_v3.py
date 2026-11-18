"""
Synthetic-data benchmark, v3 — the final paper panel.

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP.

Same contaminated change-in-mean task as ``synthetic_data.py`` (Altamirano,
Briol & Knoblauch 2023: T=500, six changes, 2% outliers at +/-10, ten
trials), same verbatim data generator / contamination / ``accuracy_delay``
scorer, same every-method-online bar.  This file only changes the PANEL and
the routing of the two methods that do not fit the old driver:

* R-FOCuS is dropped (it was a Huber-clipped FOCuS we built ourselves, not a
  published method).
* Three compiled timing twins are paired automatically with their Python row:
  'vocd (cpp)' (whole detect in C++, vocd/_vocd_full.cpp), 'focus (cpp)' and
  'np-focus (cpp)' (functional pruning, alternative_models/_cpp_alarms.cpp).
  Each twin is run at its Python row's best budget, its detections are
  asserted identical, and only its runtime differs.  Twins are never swept
  on their own.  The table therefore has 11 rows for the default panel.
* FOCuS (plain) is now reported: it is Gaussian and NOT outlier-robust, so
  on this contaminated task it acts as the non-robust CUSUM control.
* NUNC (Austin et al. 2023) is new: a rolling-window non-parametric eCDF
  test, the closest published relative of VOCD's gate.  Alarm-style, so it
  runs through ``run_restart`` on the shared alpha grid.  Its threshold is
  calibrated by Monte Carlo under H0 as in the paper (exact for any
  continuous null, since the statistic is rank-based).  The calibration is
  part of the method and is rebuilt inside every timed detect call, so it
  is charged to NUNC's runtime like every other detector's work; it is NOT
  a one-off that other methods get for free.
* CHAMP (Niekum et al. 2015 / Fearnhead & Liu 2007) is new: a genuine online
  MAP (Viterbi) segmentation with BIC-penalised per-segment ML fits and a
  model-based outlier mixture — the closest published relative of VOCD's
  proposal stage.  It is NOT alarm-style and does not restart; its sweep
  knob is the length-prior timescale, swept on the SAME hazard grid as BOCD
  (its default geometric length prior is BOCD's constant hazard with a
  minimum run length).  Change-points are read off causally with BOCD's
  commit rule plus a stabilisation step (see ``alternative_models/champ``).
  By default every candidate start is kept, so the recursion is exact and
  CHAMP's accuracy is its ceiling; ``--champ-max-particles`` enables the
  paper's particle cap for speed (identical detections on this task).

Fairness notes to carry into the paper
--------------------------------------
* On THIS task the parametric methods (BOCD, Dm-BOCD, CHAMP) have a
  correctly specified model: the data are Gaussian with a constant mean per
  segment.  The sensor data sets, not this benchmark, test misspecification.
* CHAMP's outlier ratio ``gamma`` and NUNC's window ``W`` are fixed a priori
  (not tuned on the test series) and must be reported alongside VOCD's gate
  window.  Defaults: gamma=0.05 (the contamination is 2%, so gamma is not
  matched to it); W=50 = the 2 x 25 points VOCD's gate inspects.
* All alarm baselines and CHAMP are reference reimplementations, unoptimised
  pure Python; BOCD / Dm-BOCD run the authors' code.  Runtime columns are
  comparable within that caveat (state it in the caption).  CHAMP's exact
  recursion is O(n^3) per series (~5 s/trial at n=500).

Usage::

    # single shared operating point
    python examples/synthetic_data_v3.py --dsm-path ./DSM-bocd --alpha 0.01

    # budget sweep — each method's best-F1 point (what the paper reports)
    python examples/synthetic_data_v3.py --sweep --dsm-path ./DSM-bocd \\
        --out results/synthetic/synthetic_v3_best.csv \\
        --sweep-out results/synthetic/synthetic_v3_sweep.csv

    # faster CHAMP sweep (particle cap; same detections on this task)
    python examples/synthetic_data_v3.py --sweep --champ-max-particles 100

Without ``--dsm-path`` / ``VOCD_DSM_PATH`` the two BOCD rows are skipped and
everything else runs.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root: vocd, alternative_models
sys.path.insert(0, _HERE)                     # sibling: synthetic_data (v2)

import vocd                                   # noqa: E402
from alternative_models import REGISTRY       # noqa: E402
import synthetic_data as v2                   # noqa: E402  (verbatim protocol)

# reuse the verbatim protocol and the reporting helpers — one source of truth
T, CPS_TRUE, WINDOW, N_TRIALS = v2.T, v2.CPS_TRUE, v2.WINDOW, v2.N_TRIALS
CONTAM_FRAC, CONTAM_SPIKE = v2.CONTAM_FRAC, v2.CONTAM_SPIKE
METRIC_KEYS = v2.METRIC_KEYS
VOCD_FAIR_BACKEND, VOCD_FAST_BACKEND = v2.VOCD_FAIR_BACKEND, v2.VOCD_FAST_BACKEND
make_paper_data, make_trials = v2.make_paper_data, v2.make_trials
agg, eval_point = v2.agg, v2.eval_point
print_table, write_pertrial, write_curve = v2.print_table, v2.write_pertrial, v2.write_curve

PANEL = "vocd,bocd,dm-bocd,focus,np-focus,nunc,chad,champ"
HAZARD_METHODS = ("bocd", "dm-bocd", "champ")   # swept on the hazard grid

# base row -> (twin label, registry method or None for the VOCD backend switch)
TWINS = {
    "vocd":     ("vocd (cpp)",     None),
    "focus":    ("focus (cpp)",    "focus-cpp"),
    "np-focus": ("np-focus (cpp)", "np-focus-cpp"),
}
TWIN_METHODS = {m for _, m in TWINS.values() if m}


def paired_rows(run, base, trials, b_star, pt_base):
    """Run ``base``'s compiled twin at the SAME budget; return its per-trial
    rows after asserting identical detections (None if unavailable)."""
    label, method = TWINS[base]
    try:
        if method is None:
            pt = eval_point(run, "vocd", trials, b_star,
                            vocd_backend=VOCD_FAST_BACKEND, label=label)
        else:
            pt = eval_point(run, method, trials, b_star, label=label)
    except Exception as e:
        print(f"! {label}: compiled twin unavailable ({e}); row skipped")
        return None
    ndiff = sum(1 for a, b in zip(pt_base, pt) if a["_cps"] != b["_cps"])
    if ndiff:
        print(f"! WARNING: {base} python vs cpp detections differ on {ndiff}/"
              f"{len(trials)} trials — '{label}' is NOT a pure timing twin")
    else:
        print(f"  {label}: detections identical to {base} (only runtime differs)")
    return pt


def insert_twins(order):
    out = []
    for m in order:
        out.append(m)
        if m in TWINS:
            out.append(TWINS[m][0])
    return out


# ── one run of one method at one budget ───────────────────────────────

def make_runner(args, mean0, dsm_path):
    champ_kw = dict(gamma=args.champ_gamma, em_iters=args.champ_em_iters,
                    max_particles=args.champ_max_particles,
                    extract=args.champ_extract, lag=args.champ_lag,
                    confirm=args.champ_confirm, length_prior=args.champ_prior)
    nunc_kw = dict(window=args.nunc_window, n_sims=args.nunc_sims,
                   shared_cache=args.nunc_cache)

    def run(method: str, x1d: np.ndarray, budget: float,
            vocd_backend: str = VOCD_FAIR_BACKEND) -> List[int]:
        if method == "vocd":
            r = vocd.detect(x1d, backend=vocd_backend)
            return [int(c) for c in r.changepoints(fwer=budget)]
        fn, needs_dsm = REGISTRY[method]
        if needs_dsm:                      # bocd / dm-bocd: budget = hazard
            return fn(x1d, mean0=mean0, dsm_path=dsm_path,
                      extract=args.bocd_extract, hazard_timescale=budget)
        if method == "champ":              # MAP segmentation: budget = hazard
            return fn(x1d, hazard_timescale=budget, min_seg=args.min_seg,
                      **champ_kw)
        if method == "nunc":               # alarm: budget = alpha
            return fn(x1d, alpha=budget, min_seg=args.min_seg, **nunc_kw)
        return fn(x1d, alpha=budget, min_seg=args.min_seg)   # other alarms
    return run


def budget_grid(method: str, alpha_grid, hazard_grid):
    if method in HAZARD_METHODS:
        return hazard_grid, "hazard"
    if method == "vocd":
        return alpha_grid, "fwer"
    return alpha_grid, "alpha"


def best_point(evals, tol=1e-9):
    """Centre of the best-F1 (then best-PPV) plateau, as in v2."""
    max_f1 = max(e[1]["f1"] for e in evals)
    cand = [e for e in evals if e[1]["f1"] >= max_f1 - tol]
    max_ppv = max(e[1]["ppv"] for e in cand)
    cand = [e for e in cand if e[1]["ppv"] >= max_ppv - tol]
    cand.sort(key=lambda e: e[0])
    b_star, _, pt_star = cand[len(cand) // 2]
    return b_star, pt_star, max_f1, len(cand)


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsm-path", default=None,
                    help="clone of maltamiranomontero/DSM-bocd (for BOCD / Dm-BOCD)")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="single-point mode: shared family-wise budget")
    ap.add_argument("--min-seg", type=int, default=10)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    # sweep
    ap.add_argument("--sweep", action="store_true",
                    help="sweep each method's own budget and report best-F1 point")
    ap.add_argument("--alpha-grid",
                    default="1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1",
                    help="grid for vocd(fwer) and the alarm methods(alpha); "
                         "extended below v2 so no method's optimum sits on an edge")
    ap.add_argument("--hazard-grid", default="50,100,200,300,500,1000,2000,5000,10000,20000,50000",
                    help="hazard / length-prior timescales swept for BOCD and "
                         "CHAMP. Dm-BOCD is NOT swept; see --dm-hazard")
    ap.add_argument("--dm-hazard", type=float, default=None,
                    help="single hazard for Dm-BOCD (too slow to sweep). Default "
                         "None = inherit BOCD's best hazard; falls back to 300")
    # CHAMP
    ap.add_argument("--champ-gamma", type=float, default=0.05,
                    help="CHAMP outlier ratio (fixed a priori; model-based robustness)")
    ap.add_argument("--champ-em-iters", type=int, default=2,
                    help="EM steps of the inlier/outlier mixture per segment fit")
    ap.add_argument("--champ-max-particles", type=int, default=None,
                    help="particle cap (default None = exact recursion, every "
                         "candidate start kept). 100 gives identical detections "
                         "here at ~4x the speed")
    ap.add_argument("--champ-extract", choices=["causal", "final"], default="causal",
                    help="'final' = retrospective MAP path at t=n (NOT online; "
                         "reference only, like --bocd-extract upstream)")
    ap.add_argument("--champ-lag", type=int, default=20,
                    help="min spacing between committed CPs (= BOCD's find_cp_causal lag)")
    ap.add_argument("--champ-confirm", type=int, default=None,
                    help="steps the MAP segment start must be stable before it is "
                         "committed (default None = --min-seg)")
    ap.add_argument("--champ-prior", choices=["geometric", "gaussian"], default="geometric",
                    help="length prior: geometric (= BOCD constant hazard, default) "
                         "or CHAMP's truncated Gaussian centred on the timescale")
    # NUNC
    ap.add_argument("--nunc-window", type=int, default=50,
                    help="NUNC rolling window W (fixed a priori; = 2 x VOCD gate window)")
    ap.add_argument("--nunc-sims", type=int, default=20000,
                    help="Monte Carlo draws for NUNC's null calibration. The "
                         "calibration is rebuilt inside every timed detect "
                         "call, so it is part of NUNC's reported runtime")
    ap.add_argument("--nunc-cache", action="store_true",
                    help="DEV ONLY: share NUNC's null tables across calls. "
                         "Removes the calibration from the timed region and "
                         "is therefore NOT a fair setting for reported runtimes")
    # VOCD compiled engine for the 'vocd (cpp)' row
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full",
                    help="which C++ engine the 'vocd (cpp)' row uses: hybrid = "
                         "compiled passes + NumPy gate (original); gate = + compiled "
                         "gate; full = whole detect in C++ (default). Detections "
                         "are identical for all three; only runtime differs. The "
                         "'vocd' row is always the pure-Python engine")
    # IO
    ap.add_argument("--sweep-out", default="results/synthetic/synthetic_v3_sweep.csv")
    ap.add_argument("--out", default="results/synthetic/synthetic_v3.csv")
    args = ap.parse_args()

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"vocd"} | set(REGISTRY)
    for m in wanted:
        if m not in known:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(known)}")
    if any(m in TWIN_METHODS for m in wanted):
        print("note: compiled twins (focus-cpp, np-focus-cpp) are paired with "
              "their Python row automatically and are not swept on their own; "
              "list 'focus' / 'np-focus' instead")
        wanted = [m for m in wanted if m not in TWIN_METHODS]

    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    needs = [m for m in wanted if m in REGISTRY and REGISTRY[m][1]]
    if needs and not dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}\n"
              "  git clone https://github.com/maltamiranomontero/DSM-bocd.git\n")
        wanted = [m for m in wanted if m not in needs]
    if not wanted:
        raise SystemExit("no methods left to run")

    if args.vocd_cpp == "full":
        import vocd._fastdetect as fastdetect
        fastdetect.enable()           # routes backend="cpp" only; "python" untouched
    elif args.vocd_cpp == "gate":
        import vocd._fastgate as fastgate
        fastgate.enable()

    # Build the compiled twins' libraries now (a one-off compile, like the
    # VOCD extension) so a first-use build is never charged to a trial's
    # runtime.  Method-level setup that IS part of an algorithm (NUNC's
    # Monte Carlo calibration) stays inside the timed call.
    if any(m in TWINS for m in wanted):
        from alternative_models import _cppalarms
        _cppalarms._load()

    data_clean, mean0 = make_paper_data()
    trials = make_trials(data_clean, args.trials)
    run = make_runner(args, mean0, dsm_path)
    has_vocd = "vocd" in wanted


    print(f"synthetic-data benchmark v3 — T={T}, 6 true CPs at {CPS_TRUE[:-1]}")
    print(f"{CONTAM_FRAC:.0%} outliers @ +/-{CONTAM_SPIKE:g}, {args.trials} trials, "
          f"window={WINDOW}, min_seg={args.min_seg}, bocd-extract={args.bocd_extract}")
    print(f"methods: {', '.join(wanted)}")
    if "champ" in wanted:
        mp = args.champ_max_particles
        print(f"champ: gamma={args.champ_gamma}, em_iters={args.champ_em_iters}, "
              f"{'exact (no particle cap)' if not mp else f'max_particles={mp}'}, "
              f"extract={args.champ_extract}, lag={args.champ_lag}, "
              f"confirm={args.champ_confirm or args.min_seg}, prior={args.champ_prior}")
    if "nunc" in wanted:
        print(f"nunc : window={args.nunc_window} (fixed a priori), MC null "
              f"calibration with {args.nunc_sims} draws "
              f"{'SHARED across calls (dev only, runtime NOT fair)' if args.nunc_cache else 'rebuilt per call (charged to runtime)'}")
    if has_vocd:
        print(f"vocd row = {VOCD_FAIR_BACKEND} backend (fair vs Python baselines); "
              f"'vocd (cpp)' = same detections, C++ engine = {args.vocd_cpp}")
    twins = [TWINS[m][0] for m in wanted if m in TWINS]
    if twins:
        print(f"compiled twins paired at their Python row's budget: {', '.join(twins)}")
    print()

    if not args.sweep:
        # ── single operating point ───────────────────────────────────
        print(f"single operating point: alpha={args.alpha:g} "
              f"(hazard methods use their default timescale 300)\n")
        all_rows, per_method = [], {}
        for name in wanted:
            budget = 300.0 if name in HAZARD_METHODS else args.alpha
            print(f"[{name}] ", end="", flush=True)
            pt = eval_point(run, name, trials, budget)
            print("." * len(pt), "done")
            all_rows.extend(pt)
            per_method[name] = (pt, budget)
        order = list(wanted)
        for base in wanted:
            if base in TWINS:
                pt_base, b = per_method[base]
                tw = paired_rows(run, base, trials, b, pt_base)
                if tw:
                    per_method[TWINS[base][0]] = (tw, b)
                    all_rows.extend(tw)
        order = [m for m in insert_twins(wanted) if m in per_method]
        print_table(order, per_method, budget_label="budget")
        print("scorer: reference accuracy_delay (many-to-one). spread = population "
              "std.\nsingle shared alpha (hazard methods at timescale 300); not "
              "each method's best operating point (use --sweep for that).")
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

    # BOCD before Dm-BOCD (Dm-BOCD inherits BOCD's best hazard).
    process_order = list(wanted)
    if "bocd" in process_order and "dm-bocd" in process_order:
        process_order.remove("dm-bocd")
        process_order.insert(process_order.index("bocd") + 1, "dm-bocd")

    for name in process_order:
        if name == "dm-bocd":
            if args.dm_hazard is not None:
                h, src = args.dm_hazard, "--dm-hazard"
            elif "bocd" in best_by_method:
                h, src = best_by_method["bocd"][1], "inherited from bocd best"
            else:
                h, src = 300.0, "fallback (no bocd swept)"
            grid, kind = [h], "hazard"
            print(f"[{name}] (hazard={h:g}, {src}) ", end="", flush=True)
        else:
            grid, kind = budget_grid(name, alpha_grid, hazard_grid)
            print(f"[{name}] ({kind}) ", end="", flush=True)
        evals = []
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
        b_star, pt_star, max_f1, n_plat = best_point(evals)
        best_by_method[name] = (pt_star, b_star)
        best_pertrial_rows.extend(pt_star)
        plateau = "" if n_plat == 1 else f" ({n_plat}-pt plateau)"
        print(f" best F1={max_f1:.3f} @ {kind}={b_star:g}{plateau}")

    for base in wanted:
        if base in TWINS and base in best_by_method:
            pt_base, b_star = best_by_method[base]
            tw = paired_rows(run, base, trials, b_star, pt_base)
            if tw:
                best_by_method[TWINS[base][0]] = (tw, b_star)
                best_pertrial_rows.extend(tw)
    order = [m for m in insert_twins(wanted) if m in best_by_method]

    print_table(order, best_by_method, budget_label="budget")
    print("each row is the method's OWN best-F1 operating point over its grid "
          "(oracle,\nchosen against labels). '(cpp)' rows reuse their Python row's "
          "budget and\ndetections; only runtime differs. dm-bocd is not swept "
          "(bocd's best hazard).\nchamp and bocd share the hazard grid. spread = "
          "population std.")

    write_curve(args.sweep_out, curve_rows)
    write_pertrial(args.out, best_pertrial_rows)
    print(f"\nwrote {len(curve_rows)} curve rows -> {args.sweep_out}")
    print(f"wrote {len(best_pertrial_rows)} best-point per-trial rows -> {args.out}")


if __name__ == "__main__":
    main()
