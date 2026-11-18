"""
Environmental-monitoring benchmark, v3 — the final paper panel (Fig. 5b-d).

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)

Same data, design and scoring as ``env_data.py`` (the loader and the event
matching are imported verbatim from it): pooled daily-median series of 2
POSITIVE constituents (TP, TN; a sustained post-storm shift is expected) and
12 NEGATIVE controls (dilution / flush constituents; no persistent shift), the
four tropical rain systems that hit Northern Virginia in 2020-21, and

    recall       = fraction of the 4 events matched by a CP on TP or TN
    false alarms = fraction of the 12 controls with a CP within +/-tol of an event
    consistency  = recall - false-alarm fraction

What this file changes is the panel and the plumbing, so every experiment in
the paper calls the detectors the same way (synthetic / ISFET / gas v3):
same ``alternative_models.REGISTRY`` entries and budget kinds (VOCD on fwer,
alarm methods on alpha, BOCD / Dm-BOCD / CHAMP on the hazard), R-FOCuS
dropped, FOCuS / NUNC / CHAMP added, and the three compiled twins paired with
their CP dates asserted identical on every series.

ONE PRE-DECLARED OPERATING POINT, NO SWEEP SELECTION.  Unlike the other three
experiments there is no ground truth to tune against legitimately: choosing
each method's budget to maximise consistency would tune it to the four storm
dates being scored.  So every method runs at one operating point declared in
advance and applied identically to positives and controls -- the paper's
Methods: fwer = 0.05 (VOCD), alpha = 0.05 (alarm methods), hazard = 30
(BOCD, Dm-BOCD, CHAMP), min_seg = 5.  ``--sweep`` additionally writes the
consistency of every method across its budget grid as a ROBUSTNESS curve
(Supplementary), without changing the reported point; Dm-BOCD is excluded
from it by default (``--sweep-dm`` to include it; minutes per grid point).

Tolerances: the headline is +/-15 days (the Fig. 5 windows); ``--tols``
recomputes the summary at further tolerances (default 20 and 30) from the same
CP dates, so the "ranking unchanged" sentence in the paper is checked here.

A-priori settings for THIS record: NUNC window W = 50 = 2 x VOCD's gate window
(same rule as the other runs), its Monte-Carlo calibration paid once for the
whole 14-series panel inside its timer (a deployed network calibrates once);
CHAMP gamma = 0.05, confirmation = min_seg, lag = 20 (= BOCD's causal-
extraction spacing), exact recursion (series <= ~280 points).

Outputs (all consumed by vocd_fig5_env.py; nothing is recomputed there):
  --out          per (group, constituent, method): n_cp, matched events,
                 n_cp near an event, runtime, CP dates             (panels c, d)
  --summary-out  per method x tolerance: recall, false alarms, consistency,
                 runtime summed over all 14 series                 (panel b)
  --series-out   the pooled daily series of every constituent      (panels c, d)
  --sweep-out    optional robustness curve                          (SI)

Usage::

    python examples/env_data_v3.py --data-dir data/env_monitoring --dsm-path ../DSM-bocd
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
import env_data as v2                         # noqa: E402  (verbatim data + events)

RAIN_EVENTS, POSITIVE, NEGATIVE = v2.RAIN_EVENTS, v2.POSITIVE, v2.NEGATIVE
load_pooled_daily, near_event, events_matched = (
    v2.load_pooled_daily, v2.near_event, v2.events_matched)
_DEFAULT_ENV_DIR, _DEFAULT_FILE, _DEFAULT_SHEET = (
    v2._DEFAULT_ENV_DIR, v2._DEFAULT_FILE, v2._DEFAULT_SHEET)
VOCD_FAIR_BACKEND, VOCD_FAST_BACKEND = "python", "cpp"

PANEL = "vocd,bocd,dm-bocd,focus,np-focus,nunc,chad,champ"
HAZARD_METHODS = ("bocd", "dm-bocd", "champ")
TWINS = {
    "vocd":     ("vocd (cpp)",     None),
    "focus":    ("focus (cpp)",    "focus-cpp"),
    "np-focus": ("np-focus (cpp)", "np-focus-cpp"),
}
TWIN_METHODS = {m for _, m in TWINS.values() if m}


# ── detector -> CP indices on one series, unified routing ─────────────

def make_cps_fn(args, dsm_path):
    champ_kw = dict(gamma=args.champ_gamma, em_iters=args.champ_em_iters,
                    max_particles=args.champ_max_particles, extract="causal",
                    lag=args.champ_lag, confirm=args.champ_confirm)

    def cps(name, y, budget, vocd_backend=VOCD_FAIR_BACKEND, nunc_cache=None):
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
                      n_sims=args.nunc_sims, cache=nunc_cache)
        return fn(y, alpha=budget, min_seg=args.min_seg)
    return cps


def budget_for(name, args):
    if name == "vocd":
        return args.fwer, "fwer"
    if name in HAZARD_METHODS:
        return args.hazard, "hazard"
    return args.alpha, "alpha"


def run_method(cps_fn, name, series, budget, vocd_backend=VOCD_FAIR_BACKEND):
    """CP dates on every series + total runtime over the panel."""
    out, total = {}, 0.0
    cache = {} if name == "nunc" else None          # one calibration per panel
    for (group, col), (days, y) in series.items():
        t0 = time.perf_counter()
        idx = cps_fn(name, y, budget, vocd_backend, cache)
        total += (time.perf_counter() - t0) * 1000.0
        idx = sorted(int(c) for c in idx if 0 < c < len(y))
        out[(group, col)] = [days[c] for c in idx]
    return out, total


def summarise(dates_by_series, tol):
    """recall / false-alarm fraction / consistency at one tolerance."""
    caught, pos_cp, neg_cp, neg_fa, n_neg = set(), 0, 0, 0, 0
    for (group, col), cdates in dates_by_series.items():
        if group == "POS":
            caught.update(events_matched(cdates, tol))
            pos_cp += len(cdates)
        else:
            n_neg += 1
            neg_cp += len(cdates)
            neg_fa += int(any(near_event(cd, tol) for cd in cdates))
    recall = len(caught) / len(RAIN_EVENTS)
    fa = neg_fa / n_neg if n_neg else 0.0
    return {"pos_recall": recall, "pos_cp": pos_cp, "neg_cp": neg_cp,
            "neg_false": neg_fa, "n_neg": n_neg, "neg_false_frac": fa,
            "consistency": recall - fa, "events_matched": ";".join(sorted(caught))}


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_ENV_DIR)
    ap.add_argument("--file", default=_DEFAULT_FILE)
    ap.add_argument("--sheet", default=_DEFAULT_SHEET)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--dsm-path", default=None)
    # the pre-declared operating point (paper Methods)
    ap.add_argument("--fwer", type=float, default=0.05, help="VOCD family-wise budget")
    ap.add_argument("--alpha", type=float, default=0.05, help="alarm-method alpha")
    ap.add_argument("--hazard", type=float, default=30.0, help="BOCD / Dm-BOCD / CHAMP")
    ap.add_argument("--min-seg", type=int, default=5)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    ap.add_argument("--tol", type=int, default=15, help="headline event tolerance (days)")
    ap.add_argument("--tols", default="20,30", help="extra tolerances for the summary")
    # a-priori method settings
    ap.add_argument("--champ-gamma", type=float, default=0.05)
    ap.add_argument("--champ-em-iters", type=int, default=2)
    ap.add_argument("--champ-max-particles", type=int, default=None, help="None = exact")
    ap.add_argument("--champ-lag", type=int, default=20)
    ap.add_argument("--champ-confirm", type=int, default=None, help="default = --min-seg")
    ap.add_argument("--nunc-window", type=int, default=50)
    ap.add_argument("--nunc-sims", type=int, default=20000)
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full")
    # optional robustness curve (never changes the reported point)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--alpha-grid", default="1e-4,1e-3,1e-2,0.05,0.1,0.3")
    ap.add_argument("--hazard-grid", default="5,10,30,100,300")
    ap.add_argument("--sweep-dm", action="store_true", help="include Dm-BOCD in --sweep (slow)")
    # outputs
    ap.add_argument("--out", default="results/env/env_v3_pos_neg.csv")
    ap.add_argument("--summary-out", default="results/env/env_v3_summary.csv")
    ap.add_argument("--series-out", default="results/env/env_v3_series.csv")
    ap.add_argument("--sweep-out", default="results/env/env_v3_sweep.csv")
    args = ap.parse_args()
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    tols = [args.tol] + [int(v) for v in args.tols.split(",") if v.strip()]

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

    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    series = {}
    for group, cols in (("POS", POSITIVE), ("NEG", NEGATIVE)):
        for col in cols:
            days, y = load_pooled_daily(path, args.sheet, col)
            if days is None or len(y) < 3 * args.min_seg:
                print(f"[{group}] {col}: insufficient data — skipped")
                continue
            series[(group, col)] = (days, np.asarray(y, float))
    print(f"env change-points — {sum(g == 'POS' for g, _ in series)} positive, "
          f"{sum(g == 'NEG' for g, _ in series)} negative-control series "
          f"({min(len(v[1]) for v in series.values())}-"
          f"{max(len(v[1]) for v in series.values())} daily points)")
    print(f"rain events: {', '.join(f'{k} {v}' for k, v in RAIN_EVENTS.items())}")
    print(f"operating point (pre-declared, no selection): fwer={args.fwer:g}, "
          f"alpha={args.alpha:g}, hazard={args.hazard:g}, min_seg={args.min_seg}; "
          f"tol ±{args.tol} d (also {tols[1:]})")
    print(f"methods: {', '.join(wanted)}; twins: "
          f"{', '.join(TWINS[m][0] for m in wanted if m in TWINS)}\n")

    cps_fn = make_cps_fn(args, dsm_path)
    dates, runtime, budgets = {}, {}, {}
    for name in wanted:
        b, kind = budget_for(name, args)
        print(f"[{name}] ({kind}={b:g}) ...", end="", flush=True)
        dates[name], runtime[name] = run_method(cps_fn, name, series, b)
        budgets[name] = (b, kind)
        s = summarise(dates[name], args.tol)
        print(f" recall {s['pos_recall']:.2f}, false alarms {s['neg_false']}/{s['n_neg']}, "
              f"consistency {s['consistency']:+.2f}, {runtime[name]:.0f} ms")

    for base in wanted:
        if base not in TWINS:
            continue
        label, method = TWINS[base]
        b, kind = budgets[base]
        try:
            if method is None:
                d, rt = run_method(cps_fn, "vocd", series, b, vocd_backend=VOCD_FAST_BACKEND)
            else:
                d, rt = run_method(cps_fn, method, series, b)
        except Exception as e:
            print(f"! {label}: compiled twin unavailable ({e}); row skipped")
            continue
        nd = sum(1 for k in series if d[k] != dates[base][k])
        if nd:
            print(f"! WARNING: {base} python vs cpp CP dates differ on {nd}/{len(series)} "
                  f"series — '{label}' is NOT a pure timing twin")
        else:
            print(f"  {label}: CP dates identical to {base} on all {len(series)} series "
                  f"(only runtime differs)")
        dates[label], runtime[label], budgets[label] = d, rt, (b, kind)

    order = []
    for m in wanted:
        order.append(m)
        if m in TWINS and TWINS[m][0] in dates:
            order.append(TWINS[m][0])

    # ── summary table at every tolerance ──
    summary_rows = []
    for tol in tols:
        print(f"\n{'=' * 96}\ntolerance ±{tol} days")
        print(f"{'method':<16}{'budget':>12}{'POS recall':>11}{'POS #CP':>9}{'NEG #CP':>9}"
              f"{'NEG false':>11}{'consistency':>13}{'Runtime(ms)':>14}")
        print("-" * 96)
        for name in order:
            s = summarise(dates[name], tol)
            b, kind = budgets[name]
            print(f"{name:<16}{kind + '=' + format(b, 'g'):>12}{s['pos_recall']:>11.2f}"
                  f"{s['pos_cp']:>9}{s['neg_cp']:>9}{s['neg_false']:>8}/{s['n_neg']:<2}"
                  f"{s['consistency']:>+13.2f}{runtime[name]:>14.1f}")
            summary_rows.append({"method": name, "tol": tol, "budget_kind": kind, "budget": b,
                                 **s, "runtime_ms": round(runtime[name], 3),
                                 "min_seg": args.min_seg})
        print("-" * 96)
        rank = sorted(order, key=lambda m: -summarise(dates[m], tol)["consistency"])
        print("ranking by consistency: " + " > ".join(m for m in rank if "(cpp)" not in m))
    print("\nrecall = 4 rain events matched in TP or TN; NEG false = controls with a CP within "
          "±tol of an event;\nconsistency = recall − false-alarm fraction; runtime summed over "
          "all series. '(cpp)' rows = same CP dates, compiled.")

    # ── outputs ──
    for p in (args.out, args.summary_out, args.series_out):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "constituent", "method", "budget_kind", "budget", "n_cp",
                    "matched_events", "n_cp_near_event", "cp_dates"])
        for name in order:
            b, kind = budgets[name]
            for (group, col), cdates in dates[name].items():
                w.writerow([group, col, name, kind, b, len(cdates),
                            ";".join(events_matched(cdates, args.tol)),
                            sum(1 for cd in cdates if near_event(cd, args.tol)),
                            ";".join(d.isoformat() for d in cdates)])
    with open(args.summary_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    with open(args.series_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "constituent", "date", "value"])
        for (group, col), (days, y) in series.items():
            for d, v in zip(days, y):
                w.writerow([group, col, d.isoformat(), f"{v:.6g}"])
    print(f"\nwrote per-series CP dates -> {args.out}")
    print(f"wrote summary ({len(tols)} tolerances) -> {args.summary_out}")
    print(f"wrote pooled daily series -> {args.series_out}")

    # ── optional robustness curve ──
    if args.sweep:
        alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
        hazard_grid = [float(v) for v in args.hazard_grid.split(",") if v.strip()]
        curve = []
        print("\nrobustness sweep (reported point unchanged):")
        for name in wanted:
            if name == "dm-bocd" and not args.sweep_dm:
                continue
            grid = hazard_grid if name in HAZARD_METHODS else alpha_grid
            kind = budget_for(name, args)[1]
            print(f"  [{name}] ", end="", flush=True)
            for b in grid:
                d, rt = run_method(cps_fn, name, series, b)
                s = summarise(d, args.tol)
                curve.append({"method": name, "budget_kind": kind, "budget": b,
                              "tol": args.tol, **s, "runtime_ms": round(rt, 3)})
                print(".", end="", flush=True)
            print()
        os.makedirs(os.path.dirname(args.sweep_out) or ".", exist_ok=True)
        with open(args.sweep_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(curve[0]))
            w.writeheader()
            for r in curve:
                w.writerow(r)
        print(f"wrote robustness curve -> {args.sweep_out}")


if __name__ == "__main__":
    main()
