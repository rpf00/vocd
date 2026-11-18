"""
Env experiment — full panel, operating point CALIBRATED on TALK/TURB/DOC nulls.

Each detector is calibrated the SAME way, on the same three orthogonal negative
controls (TALK = carbonate ion, TURB = sediment, DOC = organic; the CL-free
triple, chosen to avoid the winter road-salt confound):

  Target A — the LOOSEST (most sensitive) operating point at which the detector
  produces ZERO change-points WITHIN tol days of any of the four NoVA rain
  events on ALL THREE nulls simultaneously.

Because BOCD/Dm-BOCD use a different knob than the gate methods, calibration is
knob-aware:
  vocd, np-focus, r-focus, chad  -> alpha  (loosest clean = LARGEST clean alpha)
  bocd, dm-bocd                  -> hazard (loosest clean = SMALLEST clean hazard)

The frozen knob is then applied, per method and per tol in {5,10,15,20,30}, to:
  POSITIVES  TP, TN         -> which storms are flagged (Zeta/Ida = expected),
  remaining NEGATIVE ctrls  -> near-storm CPs = spurious storm attribution (0 ideal).

BOCD/Dm-BOCD need a DSM-bocd clone (--dsm-path / VOCD_DSM_PATH); Dm-BOCD is slow
and off by default.  All detectors are the pip ``vocd`` / ``alternative_models``
panel used throughout the paper.

Usage
-----
    python examples/env_calibrated.py --data-dir data/env_monitoring
    python examples/env_calibrated.py --data-dir data/env_monitoring --dsm-path ./DSM-bocd \
        --methods vocd,bocd,np-focus,r-focus,chad
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vocd
from alternative_models import REGISTRY
from env_data import (load_pooled_daily, RAIN_EVENTS, POSITIVE, NEGATIVE,
                      _DEFAULT_ENV_DIR, _DEFAULT_FILE, _DEFAULT_SHEET)

NULLS = ["TALK", "TURB", "DOC"]
TOLS = [5, 10, 15, 20, 30]
ALPHA_GRID = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1, 7e-1]
HAZARD_GRID = [5, 10, 20, 30, 50, 80, 150]
GATE = {"vocd", "np-focus", "r-focus", "chad"}
HAZARDED = {"bocd", "dm-bocd"}
ALL_METHODS = ["vocd", "bocd", "np-focus", "r-focus", "chad"]  # dm-bocd off by default


def knob_kind(method):
    return "hazard" if method in HAZARDED else "alpha"


def detect_dates(method, days, y, knob, args):
    if method == "vocd":
        try:
            cps = [int(c) for c in vocd.detect(y, backend="python")
                   .changepoints(fwer=knob)]
        except Exception:
            cps = []
    elif method in HAZARDED:
        fn, _ = REGISTRY[method]
        cps = fn(y, mean0=None, dsm_path=args._dsm_path,
                 extract=args.bocd_extract, hazard_timescale=knob)
    else:
        fn, _ = REGISTRY[method]
        cps = fn(y, alpha=knob, min_seg=args.min_seg)
    return [days[c] for c in sorted(int(c) for c in cps if 0 < c < len(y))]


def near_storm_count(cdates, tol):
    return sum(1 for cd in cdates
               if any(abs((cd - e).days) <= tol for e in RAIN_EVENTS.values()))


def storms_hit(cdates, tol):
    return [n for n, e in RAIN_EVENTS.items()
            if any(abs((cd - e).days) <= tol for cd in cdates)]


def frozen_knob(method, ser, tol, args):
    """Loosest knob clean (0 near-storm CPs) on ALL nulls; knob-aware."""
    grid = HAZARD_GRID if method in HAZARDED else ALPHA_GRID
    clean_sets = []
    for n in NULLS:
        d, y = ser[n]
        clean_sets.append({k for k in grid
                           if near_storm_count(detect_dates(method, d, y, k, args), tol) == 0})
    common = set.intersection(*clean_sets) if clean_sets else set()
    if not common:
        return None, {n: None for n in NULLS}
    # per-null loosest (for display)
    per = {}
    for n, cs in zip(NULLS, clean_sets):
        per[n] = (max(cs) if (cs and method in GATE) else
                  (min(cs) if cs else None))
    frozen = max(common) if method in GATE else min(common)  # loosest/most-sensitive
    return frozen, per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_ENV_DIR)
    ap.add_argument("--file", default=_DEFAULT_FILE)
    ap.add_argument("--sheet", default=_DEFAULT_SHEET)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--min-seg", type=int, default=5)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    ap.add_argument("--out", default="results/env/env_calibrated.csv")
    args = ap.parse_args()
    args._dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS + ["dm-bocd"]:
            raise SystemExit(f"unknown method {m!r}")
    needs = [m for m in methods if m in HAZARDED]
    if needs and not args._dsm_path:
        print(f"! no --dsm-path/VOCD_DSM_PATH; skipping {needs}\n")
        methods = [m for m in methods if m not in HAZARDED]

    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    ser = {}
    for c in set(NULLS) | set(POSITIVE) | set(NEGATIVE):
        d, y = load_pooled_daily(path, args.sheet, c)
        if d is not None:
            ser[c] = (d, y)
    for n in NULLS:
        if n not in ser:
            sys.exit(f"null {n} missing/empty")
    remaining = [c for c in NEGATIVE if c not in NULLS and c in ser]

    print("ENV — full panel, calibrated on TALK/TURB/DOC (Target A: 0 near-storm CPs)")
    print(f"rain events: {list(RAIN_EVENTS)}   tol: {TOLS}   nulls: {NULLS}")
    print(f"remaining negative controls scored for false alarms: {remaining}\n")

    rows = []
    for method in methods:
        kk = knob_kind(method)
        print(f"================  {method}  (knob: {kk})  ================")
        hdr = "  ".join(f"{n:>8}" for n in NULLS)
        print(f"{'tol':>4} | {hdr} | {'frozen':>8} | {'POS hits (TP/TN)':<26} | "
              f"{'NEG false':>9}")
        print("-" * 90)
        for tol in TOLS:
            frozen, per = frozen_knob(method, ser, tol, args)
            pos_hits, neg_fa = [], 0
            if frozen is not None:
                for pc in POSITIVE:
                    if pc in ser:
                        for s in storms_hit(detect_dates(method, *ser[pc], frozen, args), tol):
                            pos_hits.append(f"{pc}:{s}")
                for nc in remaining:
                    neg_fa += near_storm_count(
                        detect_dates(method, *ser[nc], frozen, args), tol)

            def f(v):
                return f"{v:.4g}" if v is not None else "none"
            ph = ", ".join(pos_hits) if pos_hits else "(none)"
            print(f"{tol:>4} | {'  '.join(f'{f(per[n]):>8}' for n in NULLS)} | "
                  f"{f(frozen):>8} | {ph:<26} | {neg_fa:>9}")
            rows.append({"method": method, "knob": kk, "tol": tol,
                         **{f"clean_{n}": per[n] for n in NULLS},
                         "frozen": frozen, "pos_hits": ";".join(pos_hits),
                         "neg_false_alarms": neg_fa})
        print()

    print("Read: the 3 null columns are each null's loosest clean knob "
          "(agreement=robust);\nfrozen = loosest knob clean on ALL 3 nulls "
          "(largest alpha / smallest hazard).\nPOS hits = storms flagged in "
          "TP/TN at the frozen knob (Zeta/Ida expected);\nNEG false = near-storm "
          "CPs on the remaining controls (spurious; 0 ideal).\nStable frozen knob "
          "+ POS hits on Zeta/Ida + low NEG false = a physics-consistent,\n"
          "non-p-hacked detection. Everyone's negative here => the record's "
          "signal is\nbelow the detection floor, not a tuning artefact.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = (["method", "knob", "tol"] + [f"clean_{n}" for n in NULLS]
            + ["frozen", "pos_hits", "neg_false_alarms"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
