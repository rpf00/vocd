"""
Runtime scaling with series length, v3 — Supplementary Fig. 10.

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)

Every detector runs ONCE on the first T points of one fixed synthetic series
(piecewise-constant Gaussian mean, a new segment every --seg points, unit
noise, seed 0), for T on --t-grid, through the same ``alternative_models``
REGISTRY routing as every other experiment, at one declared operating point
(VOCD fwer = 0.01; alpha = 0.01; hazard 300 for BOCD, Dm-BOCD and CHAMP;
min_seg = 10).  Runtime is the median wall-clock of --reps runs.  The figure
script fits runtime = a * T^b per method, so the measured exponent b can be
compared with the theory: VOCD O(S^2) per observation at a fixed regime bank,
i.e. O(S^2 T) per record (b = 1); BOCD and Dm-BOCD with the DSM-bocd pruning
to K = 50 run lengths, O(K T) (b = 1); unpruned BOCD would be O(T^2).

Slow methods can be capped with --max-t (e.g. dm-bocd=8000): rows above the
cap are skipped, never extrapolated here (the figure extrapolates the FIT).

Output: --out CSV with method, T, runtime_ms (median), runtime_min/max_ms,
us_per_obs, n_cp.

    python examples/scaling_v3.py --dsm-path ../DSM-bocd
    python examples/scaling_v3.py --dsm-path ../DSM-bocd --t-grid 500,1000,2000,4000 --reps 1
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

PANEL = "vocd,bocd,dm-bocd,focus,np-focus,nunc,chad,champ"
HAZARD_METHODS = ("bocd", "dm-bocd", "champ")
TWINS = {"vocd": ("vocd (cpp)", None), "focus": ("focus (cpp)", "focus-cpp"),
         "np-focus": ("np-focus (cpp)", "np-focus-cpp")}
TWIN_METHODS = {m for _, m in TWINS.values() if m}


def make_series(T, seg, seed=0):
    rng = np.random.default_rng(seed)
    n_seg = int(np.ceil(T / seg))
    means = np.cumsum(rng.choice([-1.0, 1.0], n_seg) * rng.uniform(1.0, 2.5, n_seg))
    return np.repeat(means, seg)[:T] + rng.normal(0.0, 1.0, T)


def make_cps_fn(args, dsm_path):
    champ_kw = dict(gamma=args.champ_gamma, em_iters=args.champ_em_iters,
                    max_particles=args.champ_max_particles, extract="causal",
                    lag=args.champ_lag, confirm=args.champ_confirm)

    def cps(name, y, vocd_backend="python"):
        if name == "vocd":
            return list(vocd.detect(y, backend=vocd_backend).changepoints(fwer=args.fwer))
        fn, needs_dsm = REGISTRY[name]
        if needs_dsm:
            return fn(y, mean0=None, dsm_path=dsm_path, extract="causal",
                      hazard_timescale=args.hazard)
        if name == "champ":
            return fn(y, hazard_timescale=args.hazard, min_seg=args.min_seg, **champ_kw)
        if name == "nunc":
            return fn(y, alpha=args.alpha, min_seg=args.min_seg, window=args.nunc_window,
                      n_sims=args.nunc_sims, cache={})
        return fn(y, alpha=args.alpha, min_seg=args.min_seg)
    return cps


def time_runs(fn, reps):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(min(ts)), float(max(ts)), out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--t-grid", default="250,500,1000,2000,4000,8000,16000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seg", type=int, default=250, help="segment length of the series")
    ap.add_argument("--max-t", default="dm-bocd=8000,champ=8000",
                    help="per-method T cap, e.g. dm-bocd=8000,champ=4000")
    ap.add_argument("--fwer", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--hazard", type=float, default=300.0)
    ap.add_argument("--min-seg", type=int, default=10)
    ap.add_argument("--champ-gamma", type=float, default=0.05)
    ap.add_argument("--champ-em-iters", type=int, default=2)
    ap.add_argument("--champ-max-particles", type=int, default=100)
    ap.add_argument("--champ-lag", type=int, default=20)
    ap.add_argument("--champ-confirm", type=int, default=None)
    ap.add_argument("--nunc-window", type=int, default=50)
    ap.add_argument("--nunc-sims", type=int, default=20000)
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full")
    ap.add_argument("--out", default="results/scaling/scaling_v3.csv")
    args = ap.parse_args()
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    grid = sorted(int(v) for v in args.t_grid.split(",") if v.strip())
    caps = {k.strip(): int(v) for k, v in (p.split("=") for p in args.max_t.split(",") if "=" in p)}

    wanted = [m.strip() for m in args.methods.split(",") if m.strip() and m.strip() not in TWIN_METHODS]
    if not dsm_path:
        skip = [m for m in wanted if m in REGISTRY and REGISTRY[m][1]]
        if skip:
            print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {skip}")
        wanted = [m for m in wanted if m not in skip]
    if args.vocd_cpp == "full":
        import vocd._fastdetect as fastdetect
        fastdetect.enable()
    elif args.vocd_cpp == "gate":
        import vocd._fastgate as fastgate
        fastgate.enable()
    if any(m in TWINS for m in wanted):
        from alternative_models import _cppalarms
        _cppalarms._load()

    y_all = make_series(max(grid), args.seg)
    cps = make_cps_fn(args, dsm_path)
    rows = []
    print(f"scaling: T grid {grid}, {args.reps} reps (median), segment {args.seg}; "
          f"methods {', '.join(wanted)}; caps {caps}")
    jobs = []
    for m in wanted:
        jobs.append((m, m, "python"))
        if m in TWINS:
            label, method = TWINS[m]
            jobs.append((label, method or "vocd", "cpp" if method is None else "python"))
    for label, method, backend in jobs:
        base = label.split(" ")[0]
        for T in grid:
            if T > caps.get(base, 10**12):
                continue
            y = y_all[:T]
            med, lo, hi, out = time_runs(lambda: cps(method, y, vocd_backend=backend), args.reps)
            rows.append(dict(method=label, T=T, runtime_ms=med, runtime_min_ms=lo,
                             runtime_max_ms=hi, us_per_obs=1000.0 * med / T, n_cp=len(out)))
            print(f"  {label:<15} T={T:>6}  {med:>12,.1f} ms  ({1000 * med / T:8.1f} us/obs)  "
                  f"{len(out)} CPs", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
