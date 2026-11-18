"""
Env operating-point calibration on ORTHOGONAL null triples (Target A).

For a chosen triple of mechanistically-orthogonal negative-control constituents,
find VOCD's LOOSEST storm-clean alpha: the LARGEST fwer at which VOCD produces
ZERO change-points WITHIN tol days of any of the four NoVA rain events on that
null (Target A — "0 NEAR-STORM CPs", not "0 CPs anywhere": TURB/DOC/COD flush
transiently at storms, so a CP away from storms is a real regime, not a false
alarm, and must not force alpha down).

Reports, per triple and per tol in {5,10,15,20,30}:
  - each null's loosest storm-clean alpha (agreement across the 3 = robust),
  - the frozen alpha = min across the triple (clean on all three),
  - applying the frozen alpha to TP/TN: which storms are flagged
    (Zeta/Ida are the literature-expected ones),
  - false alarms on the remaining negative controls (near-storm CPs; 0 ideal).

Runs the three candidate triples by default so you can see whether the frozen
alpha AGREES across different orthogonal null choices (robustness to null
selection).  VOCD only (alpha-gated); pip vocd MW/KS gate.

Usage
-----
    python examples/env_calibrate_triples.py --data-dir data/env_monitoring
    python examples/env_calibrate_triples.py --triples "CL,TURB,FIELDPH"
    python examples/env_calibrate_triples.py --triples "CL,TURB,FIELDPH; TALK,TURB,DOC"
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
from env_data import (load_pooled_daily, RAIN_EVENTS, POSITIVE, NEGATIVE,
                      _DEFAULT_ENV_DIR, _DEFAULT_FILE, _DEFAULT_SHEET)

TOLS = [5, 10, 15, 20, 30]
ALPHA_GRID = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1, 7e-1]
DEFAULT_TRIPLES = "CL,TURB,FIELDPH; CL,COD,FIELDPH; TALK,TURB,DOC"


def vocd_dates(days, y, alpha):
    try:
        cps = [int(c) for c in vocd.detect(y, backend="python").changepoints(fwer=alpha)]
    except Exception:
        cps = []
    return [days[c] for c in sorted(cps) if 0 < c < len(y)]


def near_storm_count(cdates, tol):
    return sum(1 for cd in cdates
               if any(abs((cd - e).days) <= tol for e in RAIN_EVENTS.values()))


def storms_hit(cdates, tol):
    return [n for n, e in RAIN_EVENTS.items()
            if any(abs((cd - e).days) <= tol for cd in cdates)]


def loosest_clean(days, y, tol):
    """Target A: largest alpha with 0 NEAR-STORM CPs on this null (or None)."""
    ok = [a for a in ALPHA_GRID
          if near_storm_count(vocd_dates(days, y, a), tol) == 0]
    return max(ok) if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_ENV_DIR)
    ap.add_argument("--file", default=_DEFAULT_FILE)
    ap.add_argument("--sheet", default=_DEFAULT_SHEET)
    ap.add_argument("--triples", default=DEFAULT_TRIPLES,
                    help="';'-separated triples of null column names")
    ap.add_argument("--out", default="results/env/env_calibration_triples.csv")
    args = ap.parse_args()

    triples = [[c.strip() for c in t.split(",") if c.strip()]
               for t in args.triples.split(";") if t.strip()]
    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    # preload every series we might touch
    need = set(POSITIVE) | set(NEGATIVE)
    for t in triples:
        need |= set(t)
    ser = {}
    for c in need:
        d, y = load_pooled_daily(path, args.sheet, c)
        if d is not None:
            ser[c] = (d, y)
    for t in triples:
        miss = [c for c in t if c not in ser]
        if miss:
            sys.exit(f"triple {t}: missing/empty columns {miss}")

    print("VOCD env calibration — Target A (loosest alpha with 0 NEAR-STORM CPs)")
    print(f"rain events: {list(RAIN_EVENTS)}   tol sweep: {TOLS} days")
    print(f"alpha grid: {ALPHA_GRID[0]:g}..{ALPHA_GRID[-1]:g}\n")

    rows = []
    frozen_by_tol = {}          # tol -> list of frozen alphas across triples
    for triple in triples:
        others = [c for c in NEGATIVE if c not in triple and c in ser]
        print(f"================  nulls = {triple}  ================")
        hdr_nulls = "  ".join(f"{n:>8}" for n in triple)
        print(f"{'tol':>4} | {hdr_nulls} | {'frozen α':>9} | "
              f"{'POS hits (TP/TN)':<28} | {'NEG false':>9}")
        print("-" * 92)
        for tol in TOLS:
            per = {n: loosest_clean(*ser[n], tol) for n in triple}
            vals = [v for v in per.values() if v is not None]
            frozen = min(vals) if vals else None
            frozen_by_tol.setdefault(tol, []).append(frozen)
            pos_hits, neg_fa = [], 0
            if frozen is not None:
                for pc in POSITIVE:
                    if pc in ser:
                        for s in storms_hit(vocd_dates(*ser[pc], frozen), tol):
                            pos_hits.append(f"{pc}:{s}")
                for nc in others:
                    neg_fa += near_storm_count(vocd_dates(*ser[nc], frozen), tol)

            def f(v):
                return f"{v:.4g}" if v is not None else "none"
            fh = f(frozen)
            ph = ", ".join(pos_hits) if pos_hits else "(none)"
            print(f"{tol:>4} | {'  '.join(f'{f(per[n]):>8}' for n in triple)} | "
                  f"{fh:>9} | {ph:<28} | {neg_fa:>9}")
            rows.append({"nulls": "+".join(triple), "tol": tol,
                         **{f"alpha_{n}": per[n] for n in triple},
                         "frozen_alpha": frozen, "pos_hits": ";".join(pos_hits),
                         "neg_false_alarms": neg_fa})
        print()

    # cross-triple agreement of the frozen alpha
    print("Frozen-alpha agreement across triples (robustness to null choice):")
    print(f"  {'tol':>4} | frozen alphas per triple")
    for tol in TOLS:
        fa = frozen_by_tol.get(tol, [])
        show = ", ".join("none" if v is None else f"{v:.4g}" for v in fa)
        agree = ("AGREE" if len({None if v is None else round(v, 12) for v in fa}) == 1
                 else "differ")
        print(f"  {tol:>4} | {show:<28} [{agree}]")
    print()
    print("Reading it: the per-null alphas within a triple should agree (robust "
          "calibration);\nthe frozen alpha should also agree ACROSS triples "
          "(robust to which orthogonal nulls\nwe chose). POS hits at the frozen "
          "alpha are the storms VOCD actually flags in TP/TN\n(Zeta/Ida = "
          "literature-expected); NEG false = spurious near-storm CPs on the other "
          "controls.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = ["nulls", "tol"]
    allcols = keys + sorted({k for r in rows for k in r if k not in keys})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=allcols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
