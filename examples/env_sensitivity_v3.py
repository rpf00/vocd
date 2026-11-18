"""
Environmental-monitoring sensitivity analyses, v3 — Supplementary Table 1 and
the Methods / Results numbers of the shifted-date null and the direction check.

Reads ONLY the outputs of ``env_data_v3.py`` (no detector is re-run):
  --pos-neg   env_v3_pos_neg.csv   per (group, constituent, method) CP dates
  --series    env_v3_series.csv    pooled daily-median series

and writes, to --out-dir (default ``results/env``):

* ``env_v3_null.csv``       SHIFTED-DATE NULL.  The four storm dates are moved
  together by every circular offset D (days) over the record, keeping their
  spacing, with |D| > 2 x tol so no shifted window overlaps its true one; the
  CP dates are left untouched.  Per method: observed consistency, null mean,
  and the % of shifts whose consistency the observed one exceeds (strictly,
  and ties included).
* ``env_v3_direction.csv``  DIRECTION AND PERSISTENCE.  For every CP on a
  positive (TP, TN) that matches a storm, the median of the observations in
  the --persist days after the CP vs the --persist days before it; the match is
  physics-consistent only if the shift is positive (sustained increase).
* ``env_v3_counts_table.csv``  SUPPLEMENTARY TABLE 1.  Per method: CPs on the
  positives, CPs on the controls, storm-adjacent CPs on the controls, CPs on
  the controls away from storms, and controls with a storm-adjacent CP.

The event dates and the +/-tol matching are those of ``env_data.py``
(|CP date - event date| <= tol days), restated here so the script stays
standalone.

Usage::

    python examples/env_sensitivity_v3.py
    python examples/env_sensitivity_v3.py --tol 15 --persist 30
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

EVENTS = {"Isaias": "2020-08-04", "Zeta": "2020-10-29",
          "Elsa": "2021-07-09", "Ida": "2021-09-01"}
PANEL = ["vocd", "bocd", "dm-bocd", "focus", "np-focus", "nunc", "chad", "champ"]


# ── data ──────────────────────────────────────────────────────────────

def load_cps(path):
    """method -> {(group, constituent): [CP dates]}"""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        ds = [] if pd.isna(r["cp_dates"]) else \
            [pd.Timestamp(x) for x in str(r["cp_dates"]).split(";") if x.strip()]
        out.setdefault(r["method"], {})[(r["group"], r["constituent"])] = ds
    return out


def near(d, evs, tol):
    return [k for k, e in evs.items() if abs((d - e).days) <= tol]


def consistency(cps, evs, tol):
    caught, fa, n_neg = set(), 0, 0
    for (g, _), ds in cps.items():
        if g == "POS":
            for d in ds:
                caught.update(near(d, evs, tol))
        else:
            n_neg += 1
            fa += int(any(near(d, evs, tol) for d in ds))
    return len(caught) / len(evs) - fa / n_neg


# ── analyses ──────────────────────────────────────────────────────────

def shifted_null(cps_by, evs, t0, t1, tol, methods):
    L = (t1 - t0).days + 1
    shifts = [D for D in range(1, L) if min(D, L - D) > 2 * tol]
    rows = []
    for m in methods:
        cps = cps_by[m]
        obs = consistency(cps, evs, tol)
        null = np.array([consistency(cps, {k: t0 + pd.Timedelta(days=((e - t0).days + D) % L)
                                           for k, e in evs.items()}, tol) for D in shifts])
        rows.append({"method": m, "tol": tol, "n_shifts": len(shifts),
                     "consistency": obs, "null_mean": float(null.mean()),
                     "pct_exceeds": 100.0 * float(np.mean(obs > null)),
                     "pct_exceeds_or_ties": 100.0 * float(np.mean(obs >= null))})
    return rows


def direction(cps_by, series, evs, tol, persist, methods):
    rows = []
    for m in methods:
        for (g, c), ds in cps_by[m].items():
            if g != "POS":
                continue
            y = series[c]
            for d in ds:
                ev = near(d, evs, tol)
                if not ev:
                    continue
                w = pd.Timedelta(days=persist)
                before = y[(y.index >= d - w) & (y.index < d)]
                after = y[(y.index >= d) & (y.index < d + w)]
                mb, ma = float(before.median()), float(after.median())
                rows.append({"method": m, "constituent": c, "cp_date": d.date(),
                             "event": ";".join(ev),
                             "lag_days": min((d - evs[k]).days for k in ev),
                             "median_before": mb, "n_before": len(before),
                             "median_after": ma, "n_after": len(after),
                             "change_pct": 100.0 * (ma / mb - 1.0) if mb else np.nan,
                             "positive_shift": ma > mb})
    return rows


def counts_table(cps_by, evs, tol, methods):
    rows = []
    for m in methods:
        pos = sum(len(ds) for (g, _), ds in cps_by[m].items() if g == "POS")
        neg_ds = [(c, ds) for (g, c), ds in cps_by[m].items() if g != "POS"]
        neg = sum(len(ds) for _, ds in neg_ds)
        adj = sum(1 for _, ds in neg_ds for d in ds if near(d, evs, tol))
        ctrl = sum(1 for _, ds in neg_ds if any(near(d, evs, tol) for d in ds))
        rows.append({"method": m, "cps_positives": pos, "cps_controls": neg,
                     "cps_controls_storm_adjacent": adj,
                     "cps_controls_away_from_storms": neg - adj,
                     "controls_with_storm_adjacent_cp": ctrl, "n_controls": len(neg_ds)})
    return rows


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos-neg", default="results/env/env_v3_pos_neg.csv")
    ap.add_argument("--series", default="results/env/env_v3_series.csv")
    ap.add_argument("--tol", type=int, default=15, help="event matching tolerance (days)")
    ap.add_argument("--persist", type=int, default=30,
                    help="days before / after a matched CP for the direction check")
    ap.add_argument("--out-dir", default="results/env")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cps_by = load_cps(args.pos_neg)
    methods = [m for m in PANEL if m in cps_by]          # Python rows (twins share CPs)
    s = pd.read_csv(args.series, parse_dates=["date"])
    series = {c: d.set_index("date")["value"].sort_index() for c, d in s.groupby("constituent")}
    t0, t1 = s["date"].min(), s["date"].max()
    evs = {k: pd.Timestamp(v) for k, v in EVENTS.items()}
    print(f"record {t0.date()} to {t1.date()}, tol = +/-{args.tol} d, methods {methods}\n")

    null = shifted_null(cps_by, evs, t0, t1, args.tol, methods)
    print(f"SHIFTED-DATE NULL ({null[0]['n_shifts']} circular shifts, |D| > {2 * args.tol} d)")
    for r in null:
        print(f"  {r['method']:9s} consistency {r['consistency']:+.3f}  null mean "
              f"{r['null_mean']:+.3f}  exceeds {r['pct_exceeds']:5.1f}% "
              f"(ties incl. {r['pct_exceeds_or_ties']:5.1f}%)")
    write(os.path.join(args.out_dir, "env_v3_null.csv"), null)

    dirn = direction(cps_by, series, evs, args.tol, args.persist, methods)
    print(f"\nDIRECTION CHECK (median {args.persist} d after vs before each matched CP)")
    for r in dirn:
        print(f"  {r['method']:9s} {r['constituent']} {r['cp_date']} ({r['event']}, "
              f"{r['lag_days']:+d} d): {r['median_before']:.4g} -> {r['median_after']:.4g} "
              f"({r['change_pct']:+.0f}%) {'UP' if r['positive_shift'] else 'DOWN'}")
    if dirn:
        write(os.path.join(args.out_dir, "env_v3_direction.csv"), dirn)

    tab = counts_table(cps_by, evs, args.tol, methods)
    print("\nSUPPLEMENTARY TABLE 1 (CP counts per method)")
    for r in tab:
        print(f"  {r['method']:9s} pos {r['cps_positives']:3d}  controls {r['cps_controls']:3d} "
              f"(storm-adjacent {r['cps_controls_storm_adjacent']:3d}, away "
              f"{r['cps_controls_away_from_storms']:3d})  controls flagged "
              f"{r['controls_with_storm_adjacent_cp']}/{r['n_controls']}")
    write(os.path.join(args.out_dir, "env_v3_counts_table.csv"), tab)


if __name__ == "__main__":
    main()
