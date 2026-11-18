"""
Environmental experiment — negative-control CALIBRATION of the operating point.

Motivation: our alpha (detector budget) and tol (event-match window) were
arbitrary.  Rather than hand-pick them, we CALIBRATE alpha on constituents the
literature says have NO persistent storm response, then FREEZE it and apply
blind to the remaining controls and to the positives (TP, TN).  tol is NOT
calibrated (a null carries no timing information); it is swept across the
literature's post-storm window: 5, 10, 15, 20, 30 days.

Calibration nulls — chosen to be mechanistically ORTHOGONAL so agreement across
them is meaningful, not one signal echoed three times:
    SO4  — dissolved weathering ion; dilution/chemostatic (Godsey et al. 2009).
           The truly INERT null: no persistent AND ~no transient storm response.
    TSS  — suspended sediment; transport-limited FLUSH (C-Q hysteresis).
    DOC  — dissolved organic carbon; organic FLUSH (Raymond & Saiers 2010).
NB: TSS and DOC are null only for *persistent* shifts — they DO flush transiently
at a storm, so a near-storm CP on them may be a real transient, not a false
alarm.  We therefore report the loosest storm-clean alpha PER NULL and show the
three side by side; agreement = robust, and an SO4-vs-(TSS/DOC) disagreement is
expected and informative rather than forced into consensus.

"Loosest storm-clean alpha" for a null = the LARGEST alpha (most sensitive
setting) at which the detector produces ZERO change-points within tol of any of
the four NoVA rain events on that null.  The frozen alpha = the min across the
three nulls (clean on all three).  We then apply the frozen alpha to TP/TN
(hope: a CP near Zeta / Ida) and to the remaining controls (hope: still clean).

Calibrates the alpha-gated methods (vocd + np-focus, r-focus, chad for
comparison).  BOCD/Dm-BOCD use a different knob (hazard) and are out of scope
for this alpha calibration.

Usage
-----
    python examples/env_calibrate.py --data-dir data/env_monitoring
    python examples/env_calibrate.py --methods vocd,np-focus,r-focus,chad
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_data import (load_pooled_daily, RAIN_EVENTS, POSITIVE, NEGATIVE,
                      _DEFAULT_ENV_DIR, _DEFAULT_FILE, _DEFAULT_SHEET)
import vocd
from alternative_models import REGISTRY

GATE_METHODS = ["vocd", "np-focus", "r-focus", "chad"]
NULLS = ["SO4", "TSS", "DOC"]     # ion / sediment / organic — orthogonal
TOLS = [5, 10, 15, 20, 30]
ALPHA_GRID = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1, 7e-1]


def detect_dates(method, days, y, alpha, min_seg):
    if method == "vocd":
        try:
            cps = [int(c) for c in vocd.detect(y, backend="python")
                   .changepoints(fwer=alpha)]
        except Exception:
            cps = []
    else:
        fn, _ = REGISTRY[method]
        cps = fn(y, alpha=alpha, min_seg=min_seg)
    cps = sorted(int(c) for c in cps if 0 < c < len(y))
    return [days[c] for c in cps]


def near_storm_count(cdates, tol):
    return sum(1 for cd in cdates
               if any(abs((cd - e).days) <= tol for e in RAIN_EVENTS.values()))


def storms_hit(cdates, tol):
    return [n for n, e in RAIN_EVENTS.items()
            if any(abs((cd - e).days) <= tol for cd in cdates)]


def loosest_clean_alpha(method, days, y, tol, min_seg):
    """Largest alpha giving 0 near-storm CPs on this (null) series, or None."""
    clean = [a for a in ALPHA_GRID
             if near_storm_count(detect_dates(method, days, y, a, min_seg), tol) == 0]
    return max(clean) if clean else None


def corr_check(series):
    """Pairwise Pearson & Spearman on common-date daily medians."""
    from scipy.stats import pearsonr, spearmanr
    print("Null orthogonality check (pairwise, on common sampling days):")
    keys = list(series)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            da, va = series[a]
            db, vb = series[b]
            ma = {d: v for d, v in zip(da, va)}
            common = [d for d in db if d in ma]
            if len(common) < 5:
                print(f"  {a:5s} vs {b:5s}: <5 common days — n/a")
                continue
            xa = np.array([ma[d] for d in common])
            xb = np.array([vb[list(db).index(d)] for d in common])
            r = pearsonr(xa, xb)[0]
            rho = spearmanr(xa, xb)[0]
            flag = "  (correlated!)" if abs(rho) > 0.6 else ""
            print(f"  {a:5s} vs {b:5s}: n={len(common):3d}  Pearson r={r:+.2f}  "
                  f"Spearman rho={rho:+.2f}{flag}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_ENV_DIR)
    ap.add_argument("--file", default=_DEFAULT_FILE)
    ap.add_argument("--sheet", default=_DEFAULT_SHEET)
    ap.add_argument("--methods", default=",".join(GATE_METHODS))
    ap.add_argument("--min-seg", type=int, default=5)
    ap.add_argument("--out", default="results/env/env_calibration.csv")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in GATE_METHODS:
            raise SystemExit(f"{m!r} is not an alpha-gated method; choose from "
                             f"{GATE_METHODS} (bocd/dm-bocd use hazard)")
    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    # load everything once
    ser = {}
    for col in set(NULLS) | set(POSITIVE) | set(NEGATIVE):
        d, y = load_pooled_daily(path, args.sheet, col)
        if d is not None:
            ser[col] = (d, y)

    print("ENV OPERATING-POINT CALIBRATION via orthogonal negative controls")
    print(f"nulls: {NULLS} (ion / sediment / organic)   rain events: "
          f"{list(RAIN_EVENTS)}")
    print(f"tol sweep: {TOLS} days    alpha grid: {ALPHA_GRID[0]:g}..{ALPHA_GRID[-1]:g}\n")

    corr_check({k: ser[k] for k in NULLS if k in ser})

    remaining = [c for c in NEGATIVE if c not in NULLS and c in ser]
    rows = []
    for method in methods:
        print(f"================  {method}  ================")
        print(f"{'tol':>4} | {'  '.join(f'{n:>7}' for n in NULLS)} | "
              f"{'frozen α':>9} | {'POS hits (TP/TN)':<26} | {'NEG false':>9}")
        print("-" * 88)
        for tol in TOLS:
            per_null = {}
            for n in NULLS:
                if n in ser:
                    d, y = ser[n]
                    per_null[n] = loosest_clean_alpha(method, d, y, tol, args.min_seg)
                else:
                    per_null[n] = None
            vals = [v for v in per_null.values() if v is not None]
            # frozen = clean on ALL nulls that admit a clean alpha
            frozen = min(vals) if vals else None
            # apply frozen to positives + remaining controls
            pos_hits, neg_fa = [], 0
            if frozen is not None:
                for pc in POSITIVE:
                    if pc in ser:
                        d, y = ser[pc]
                        cds = detect_dates(method, d, y, frozen, args.min_seg)
                        for s in storms_hit(cds, tol):
                            pos_hits.append(f"{pc}:{s}")
                for nc in remaining:
                    d, y = ser[nc]
                    neg_fa += near_storm_count(
                        detect_dates(method, d, y, frozen, args.min_seg), tol)
            def fmt(v):
                return f"{v:>7.4g}" if v is not None else f"{'none':>7}"
            fh = f"{frozen:.4g}" if frozen is not None else "none"
            ph = ", ".join(pos_hits) if pos_hits else "(none)"
            print(f"{tol:>4} | {'  '.join(fmt(per_null[n]) for n in NULLS)} | "
                  f"{fh:>9} | {ph:<26} | {neg_fa:>9}")
            rows.append({"method": method, "tol": tol,
                         **{f"clean_alpha_{n}": per_null[n] for n in NULLS},
                         "frozen_alpha": frozen, "pos_hits": ";".join(pos_hits),
                         "neg_false_alarms": neg_fa})
        print()

    print("Reading it:")
    print("- the three α columns are each null's loosest storm-clean alpha; if "
          "they AGREE,\n  the calibration is robust. SO4 is the inert null; "
          "TSS/DOC flush transiently\n  at storms, so they may admit a stricter "
          "(smaller) clean alpha — an expected,\n  informative disagreement, not "
          "forced into consensus.")
    print("- frozen α = min across nulls (clean on all three). POS hits = "
          "storms flagged in\n  TP/TN at the frozen α (Zeta/Ida are the "
          "literature-expected ones). NEG false =\n  near-storm CPs on the other "
          "9 controls (0 ideal).")
    print("- a physics-consistent, non-p-hacked result = POS hits on Zeta/Ida "
          "with NEG false\n  staying low, STABLE across the tol band.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = (["method", "tol"] + [f"clean_alpha_{n}" for n in NULLS]
            + ["frozen_alpha", "pos_hits", "neg_false_alarms"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
