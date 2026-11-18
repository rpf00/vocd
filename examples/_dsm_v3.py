"""
Shared plumbing for the three classical CD benchmarks, v3 (Supplementary
Information): Twitter flash crash, UK 10-year bond yield, well-log.

Replaces ``_dsm_problems.our_panel_1d`` with the SAME panel and routing as
``synthetic_data_v3.py`` / ``isfet_data_v3.py`` / ``gas_data_v3.py``:

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)

* every method except Dm-BOCD is called through ``alternative_models.REGISTRY``
  with the v3 budget kinds (VOCD on fwer, alarm methods on alpha, BOCD / CHAMP
  on the hazard timescale) at ONE declared operating point: these series have
  no ground truth, so nothing is tuned;
* Dm-BOCD is the AUTHORS' per-series implementation (their model, priors,
  b = 20, K = 50, hazard and ``np.random.seed(100)`` before ``bocpd``, as in
  the DSM-bocd notebooks), read causally (no retrospective revision) like
  BOCD, with the robustness weight fixed at omega = 0.1 (the synthetic
  notebook's value, as in every other experiment of the paper);
  ``--estimate-omega`` restores the notebooks' per-series estimation.  Each
  script builds it and hands it to ``run_benchmark``;
* the twins reuse their Python row's budget and the CP sets are asserted
  identical, so only runtime differs;
* every method runs ONCE on the whole series (offline benchmark, causal CP
  extraction); runtime is that call's wall-clock.  Dm-BOCD's omega estimation
  (only with ``--estimate-omega``) is timed separately (``omega_ms``) and not
  added to its runtime.

The author's DSM objects and causal extractor come from ``_dsm_problems``
(``load_dsm``, ``load_estimators``, ``m1``, ``grad_m1``), as before.

Outputs (in --out-dir, default ``results/dsm``), consumed by the SI figure:
  <name>_v3_cps.csv     method, budget, n_cp, runtime, CP indices and labels
  <name>_v3_series.csv  index, label (date or sample), raw value, standardized value
"""
from __future__ import annotations

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

VOCD_FAIR_BACKEND, VOCD_FAST_BACKEND = "python", "cpp"
PANEL = "vocd,bocd,dm-bocd,focus,np-focus,nunc,chad,champ"
HAZARD_METHODS = ("bocd", "champ")            # Dm-BOCD: author's own hazard per series
TWINS = {
    "vocd":     ("vocd (cpp)",     None),
    "focus":    ("focus (cpp)",    "focus-cpp"),
    "np-focus": ("np-focus (cpp)", "np-focus-cpp"),
}
TWIN_METHODS = {m for _, m in TWINS.values() if m}
SEED = 100                                    # the DSM-bocd notebooks' seed


# ── arguments shared by the three scripts ────────────────────────────

def add_common_args(ap, default_out="results/dsm"):
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--fwer", type=float, default=0.01, help="VOCD family-wise budget")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="alarm level of FOCuS, NP-FOCuS, NUNC, CHAD")
    ap.add_argument("--hazard", type=float, default=300.0,
                    help="hazard timescale (samples) of BOCD and CHAMP")
    ap.add_argument("--min-seg", type=int, default=10)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    ap.add_argument("--omega", type=float, default=0.1,
                    help="Dm-BOCD robustness weight omega (default 0.1, the DSM-bocd synthetic "
                         "notebook's value and the one used in every other experiment)")
    ap.add_argument("--estimate-omega", action="store_true",
                    help="instead estimate omega per series as in the DSM-bocd notebooks")
    ap.add_argument("--champ-gamma", type=float, default=0.05)
    ap.add_argument("--champ-em-iters", type=int, default=2)
    ap.add_argument("--champ-max-particles", type=int, default=100,
                    help="particle cap (the published algorithm); 0 = exact")
    ap.add_argument("--champ-lag", type=int, default=20)
    ap.add_argument("--champ-confirm", type=int, default=None, help="default = --min-seg")
    ap.add_argument("--nunc-window", type=int, default=None,
                    help="default max(50, 2*min_seg)")
    ap.add_argument("--nunc-sims", type=int, default=20000)
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full")
    ap.add_argument("--out-dir", default=default_out)


def setup(args):
    """Resolve the panel, enable the compiled engines (outside the timers)."""
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    if args.champ_max_particles is not None and args.champ_max_particles <= 0:
        args.champ_max_particles = None
    if args.nunc_window is None:
        args.nunc_window = max(50, 2 * args.min_seg)
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"vocd"} | set(REGISTRY)
    for m in wanted:
        if m not in known:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(known)}")
    if any(m in TWIN_METHODS for m in wanted):
        print("note: compiled twins are paired automatically; list 'focus' / 'np-focus' instead")
        wanted = [m for m in wanted if m not in TWIN_METHODS]
    needs = [m for m in wanted if m in ("bocd", "dm-bocd")]
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
        _cppalarms._load()
    return dsm_path, wanted


# ── detector -> CPs on the whole series, unified routing ─────────────

def make_cps_fn(args, dsm_path):
    champ_kw = dict(gamma=args.champ_gamma, em_iters=args.champ_em_iters,
                    max_particles=args.champ_max_particles, extract="causal",
                    lag=args.champ_lag, confirm=args.champ_confirm)

    def cps(name, y, budget, vocd_backend=VOCD_FAIR_BACKEND):
        if name == "vocd":
            try:
                return [int(c) for c in vocd.detect(y, backend=vocd_backend)
                        .changepoints(fwer=budget)]
            except Exception:
                return []
        fn, needs_dsm = REGISTRY[name]
        if needs_dsm:
            return fn(y, mean0=None, dsm_path=dsm_path, extract=args.bocd_extract,
                      hazard_timescale=budget)
        if name == "champ":
            return fn(y, hazard_timescale=budget, min_seg=args.min_seg, **champ_kw)
        if name == "nunc":
            return fn(y, alpha=budget, min_seg=args.min_seg, window=args.nunc_window,
                      n_sims=args.nunc_sims, cache={})
        return fn(y, alpha=budget, min_seg=args.min_seg)
    return cps


def budget_for(name, args):
    if name == "vocd":
        return args.fwer, "fwer"
    if name in HAZARD_METHODS:
        return args.hazard, "hazard"
    return args.alpha, "alpha"


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t0) * 1000.0


def clean(cps, T):
    return sorted({int(c) for c in cps if 0 < int(c) < T})


# ── the benchmark ────────────────────────────────────────────────────

def run_benchmark(name, y, labels, raw, args, dsm_path, wanted, dm_bocd_fn=None):
    """y: the series the panel sees (standardized); labels: date / sample per
    index; raw: unscaled values (written out for the figure); dm_bocd_fn():
    the script's faithful author run -> (cps, runtime_ms, info dict)."""
    T = len(y)
    cps_fn = make_cps_fn(args, dsm_path)
    rows, order = {}, []
    print(f"{name}: {T} points; panel {', '.join(wanted)}")
    print(f"  VOCD fwer={args.fwer:g}; alpha={args.alpha:g} (FOCuS, NP-FOCuS, NUNC, CHAD); "
          f"hazard={args.hazard:g} (BOCD, CHAMP); min_seg={args.min_seg}; "
          f"NUNC window={args.nunc_window}; CHAMP gamma={args.champ_gamma}")

    for m in wanted:
        if m == "dm-bocd":
            if dm_bocd_fn is None:
                continue
            print("  [dm-bocd] author's configuration ...", end="", flush=True)
            cps, ms, info = dm_bocd_fn()
            rows[m] = dict(budget=info.get("hazard"), kind="hazard (author)",
                           cps=clean(cps, T), runtime_ms=ms, **info)
            print(f" {len(rows[m]['cps'])} CPs, {ms:,.0f} ms (omega={info.get('omega'):.5g})")
            order.append(m)
            continue
        b, kind = budget_for(m, args)
        print(f"  [{m}] {kind}={b:g} ...", end="", flush=True)
        cps, ms = timed(cps_fn, m, y, b)
        rows[m] = dict(budget=b, kind=kind, cps=clean(cps, T), runtime_ms=ms)
        print(f" {len(rows[m]['cps'])} CPs, {ms:,.1f} ms")
        order.append(m)
        if m in TWINS:                                  # compiled twin, same budget
            label, method = TWINS[m]
            try:
                if method is None:
                    tc, tms = timed(cps_fn, "vocd", y, b, vocd_backend=VOCD_FAST_BACKEND)
                else:
                    tc, tms = timed(cps_fn, method, y, b)
            except Exception as e:                       # noqa: BLE001
                print(f"  ! {label}: compiled twin unavailable ({e}); row skipped")
                continue
            tc = clean(tc, T)
            same = tc == rows[m]["cps"]
            print(f"    {label}: {tms:,.1f} ms, "
                  + ("CPs identical (only runtime differs)" if same else
                     "! WARNING: CPs differ from the Python row — NOT a pure timing twin"))
            rows[label] = dict(budget=b, kind=kind, cps=tc, runtime_ms=tms)
            order.append(label)

    print_table(name, order, rows, labels)
    write_outputs(name, order, rows, labels, raw, y, args.out_dir)
    return rows


# ── IO ────────────────────────────────────────────────────────────────

def print_table(name, order, rows, labels):
    print(f"\n{name}: detected change-points (no ground truth)")
    print(f"{'method':<16}{'budget':>14}{'#CP':>6}{'runtime (ms)':>15}  CP locations")
    print("-" * 100)
    for m in order:
        r = rows[m]
        b = "—" if r["budget"] is None else f"{r['budget']:g}"
        locs = ", ".join(str(labels[c]) for c in r["cps"][:8])
        more = f" (+{len(r['cps']) - 8})" if len(r["cps"]) > 8 else ""
        print(f"{m:<16}{b:>14}{len(r['cps']):>6}{r['runtime_ms']:>15,.1f}  {locs}{more}")
    print("-" * 100)


def write_outputs(name, order, rows, labels, raw, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}_v3_cps.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "budget_kind", "budget", "n_cp", "runtime_ms", "omega",
                    "omega_ms", "cp_idx", "cp_label"])
        for m in order:
            r = rows[m]
            w.writerow([m, r["kind"], "" if r["budget"] is None else r["budget"],
                        len(r["cps"]), f"{r['runtime_ms']:.3f}", r.get("omega", ""),
                        "" if r.get("omega_ms") is None else f"{r['omega_ms']:.1f}",
                        ";".join(str(c) for c in r["cps"]),
                        ";".join(str(labels[c]) for c in r["cps"])])
    print(f"wrote {len(order)} rows -> {p}")
    p = os.path.join(out_dir, f"{name}_v3_series.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "label", "value", "value_std"])
        for i in range(len(y)):
            w.writerow([i, labels[i], f"{raw[i]:.6g}", f"{y[i]:.6g}"])
    print(f"wrote series -> {p}")


def standardize(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()
