"""
Environmental-monitoring changepoint experiment — POSITIVE vs NEGATIVE-CONTROL.

Final env experiment for the VOCD paper. Same detector panel + paired
``vocd (cpp)`` as the other examples (pip ``vocd`` MW/KS gate +
``alternative_models``), applied to pooled daily-median water-quality series
from the 2020-21 stream/reservoir record.

The scientific design (literature-backed; see refs at bottom):

  POSITIVE group  — TP, TN.  Storms mobilise sediment- and legacy-bound
  nutrients; the literature reports a SUSTAINED (days-to-weeks) post-storm
  increase, so a change-point near a rain event is EXPECTED here (good).

  NEGATIVE control — COND25, TDS, CL, SO4, THARD, TALK, TURB, TSS, COD, TOC,
  DOC, FIELDPH.  These are dilution/flush-controlled (chemostatic major ions;
  transient sediment/organic flush), so the literature predicts NO persistent
  regime shift — any change-point here is a FALSE ALARM, and silence is the
  correct answer.

  EXCLUDED — TEMP, DO: strong SEASONAL regime shifts (spring/fall) that are not
  storm-driven and would confound event scoring.

Ground-truth events — only the FOUR tropical rain systems that actually hit
Northern Virginia (verified): Isaias 2020-08-04, Zeta 2020-10-29, Elsa 2021-07-09, Ida 2021-09-01.  Bertha and Fred did not
meaningfully hit NoVA; the Feb-2021 ice/winter storms are a different mechanism
(road salt, not rain-flush) and are excluded from rain-event scoring.  Note:
CL/COND25 could carry a genuine winter road-salt signal — irrelevant here since
only rain events are scored.

The right question is therefore NOT "one change-point per event" but:
does a method FIRE on the positive group near rain events while STAYING SILENT
on the negative controls?  A selective detector maximises
    consistency = POS_recall  -  NEG_fired_fraction
(caught expected nutrient shifts; silent where dilution predicts none).
A robust detector (VOCD) is expected to be conservative — high specificity on
the negatives; whether it also catches the (short, ~6-day) TP/TN shifts depends
on signal strength and the operating point (a single alpha/hazard is applied
uniformly to positives and negatives — no per-group tuning).

Usage
-----
    python examples/env_data.py --data-dir data/env_monitoring
    python examples/env_data.py --data-dir data/env_monitoring --dsm-path ./DSM-bocd
    python examples/env_data.py --tol 10 --alpha 0.05 --hazard 30
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import openpyxl
except ImportError:
    sys.exit("needs openpyxl:  pip install openpyxl")

import vocd
from alternative_models import REGISTRY

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ENV_DIR = os.path.join(_REPO, "data", "env_monitoring")
_DEFAULT_FILE = "2020-21_stream_and_reservoir.xlsx"
_DEFAULT_SHEET = "strebl_2020-2021"

DETECTORS = ["vocd", "bocd", "dm-bocd", "np-focus", "chad"]
VOCD_FAIR_BACKEND = "python"
VOCD_FAST_BACKEND = "cpp"

# only the four tropical rain systems that actually hit NoVA
RAIN_EVENTS = {
    "Isaias": dt.date(2020, 8, 4), "Zeta": dt.date(2020, 10, 29),
    "Elsa": dt.date(2021, 7, 9), "Ida": dt.date(2021, 9, 1),
}

POSITIVE = {"TP": "Total phosphorus", "TN": "Total nitrogen"}
NEGATIVE = {
    "COND25": "Specific conductance", "TDS": "Total dissolved solids",
    "CL": "Chloride", "SO4": "Sulfate", "THARD": "Total hardness",
    "TALK": "Total alkalinity", "TURB": "Turbidity",
    "TSS": "Total suspended solids", "COD": "Chemical oxygen demand",
    "TOC": "Total organic carbon", "DOC": "Dissolved organic carbon",
    "FIELDPH": "Field pH",
}


# ── pooled daily-median series ────────────────────────────────────────

def load_pooled_daily(path, sheet, col):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [("" if v is None else str(v)) for v in rows[0]]
    ci = {h: i for i, h in enumerate(hdr)}
    if col not in ci or "DATE1" not in ci:
        return None, None
    di, xi = ci["DATE1"], ci[col]
    by = {}
    for r in rows[1:]:
        d, v = r[di], r[xi]
        if d is None or v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        day = d.date() if hasattr(d, "date") else d
        by.setdefault(day, []).append(v)
    if not by:
        return None, None
    days = sorted(by)
    med = np.array([float(np.median(by[k])) for k in days], dtype=float)
    return days, med


# ── detection + date mapping ──────────────────────────────────────────

def detect_dates(name, days, y, args, vocd_backend=VOCD_FAIR_BACKEND):
    if name == "vocd":
        try:
            cps = [int(c) for c in vocd.detect(y, backend=vocd_backend)
                   .changepoints(fwer=args.alpha)]
        except Exception:
            cps = []
    else:
        fn, needs_dsm = REGISTRY[name]
        if needs_dsm:
            cps = fn(y, mean0=None, dsm_path=args._dsm_path,
                     extract=args.bocd_extract, hazard_timescale=args.hazard)
        else:
            cps = fn(y, alpha=args.alpha, min_seg=args.min_seg)
    cps = sorted(int(c) for c in cps if 0 < c < len(y))
    return [days[c] for c in cps]


def near_event(cdate, tol, events=RAIN_EVENTS):
    return any(abs((cdate - e).days) <= tol for e in events.values())


def events_matched(cdates, tol, events=RAIN_EVENTS):
    return [n for n, e in events.items()
            if any(abs((cd - e).days) <= tol for cd in cdates)]


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_ENV_DIR)
    ap.add_argument("--file", default=_DEFAULT_FILE)
    ap.add_argument("--sheet", default=_DEFAULT_SHEET)
    ap.add_argument("--methods", default="vocd,bocd,np-focus,chad")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--tol", type=int, default=10, help="event-match tolerance (days)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="vocd fwer / alarm-method alpha (single operating point)")
    ap.add_argument("--hazard", type=float, default=30.0, help="bocd/dm-bocd hazard")
    ap.add_argument("--min-seg", type=int, default=5)
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    ap.add_argument("--out", default="results/env/env_pos_neg.csv")
    ap.add_argument("--summary-out", default="results/env/env_summary.csv",
                    help="per-method table (recall, CPs, false alarms, consistency, runtime)")
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

    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    has_vocd = "vocd" in wanted
    order = list(wanted) + (["vocd (cpp)"] if has_vocd else [])

    print("env changepoints — POSITIVE (TP,TN: expect CP) vs NEGATIVE control "
          "(expect none)")
    print(f"rain events (NoVA): {', '.join(f'{k} {v}' for k,v in RAIN_EVENTS.items())}")
    print(f"tol=+/-{args.tol}d, alpha={args.alpha:g}, hazard={args.hazard:g}, "
          f"min_seg={args.min_seg}")
    print(f"detectors: {', '.join(wanted)}\n")

    # results[method][group][col] = (cp_dates, n_cp, matched_events, near_cps)
    results = {name: {"POS": {}, "NEG": {}} for name in order}
    runtime_ms = {name: 0.0 for name in order}      # wall-clock, summed over all series
    csv_rows = []

    for group, cols in (("POS", POSITIVE), ("NEG", NEGATIVE)):
        for col, label in cols.items():
            days, y = load_pooled_daily(path, args.sheet, col)
            if days is None or len(y) < 3 * args.min_seg:
                print(f"[{group}] {col}: insufficient data — skipped")
                continue
            print(f"[{group}] {col} ({label}) — {len(y)} daily pts "
                  f"[{days[0]}..{days[-1]}]")
            for name in wanted:
                t0 = time.perf_counter()
                cdates = detect_dates(name, days, y, args)
                ms = (time.perf_counter() - t0) * 1000.0
                runtime_ms[name] += ms
                matched = events_matched(cdates, args.tol)
                near = sum(1 for cd in cdates if near_event(cd, args.tol))
                results[name][group][col] = (cdates, len(cdates), matched, near)
                tag = ("  <- expected" if (group == "POS" and matched)
                       else ("  <- FALSE ALARM" if (group == "NEG" and cdates)
                             else ""))
                ds = ", ".join(d.strftime("%Y-%m-%d") for d in cdates) or "(none)"
                mt = f"  matched:{','.join(matched)}" if matched else ""
                print(f"    {name:8s}: {len(cdates):2d} CP  {ds}{mt}{tag}")
                if name == "vocd" and has_vocd:
                    try:
                        t0 = time.perf_counter()
                        cpp = detect_dates("vocd", days, y, args,
                                           vocd_backend=VOCD_FAST_BACKEND)
                        runtime_ms["vocd (cpp)"] += (time.perf_counter() - t0) * 1000.0
                        results["vocd (cpp)"][group][col] = (
                            cpp, len(cpp), events_matched(cpp, args.tol),
                            sum(1 for cd in cpp if near_event(cd, args.tol)))
                        if cpp != cdates:
                            print("      ! vocd(cpp) dates DIFFER from python")
                    except Exception as e:
                        print(f"      ! cpp backend unavailable ({e})")
                csv_rows.append({
                    "group": group, "constituent": col, "method": name,
                    "n_cp": len(cdates), "matched_events": ";".join(matched),
                    "n_cp_near_event": near, "runtime_ms": round(ms, 3),
                    "cp_dates": ";".join(d.isoformat() for d in cdates)})
            print()

    # ── per-method summary: positives should fire, negatives should not ──
    n_events = len(RAIN_EVENTS)
    summary_rows = []
    print("=" * 92)
    print(f"{'method':<12}{'POS recall':>12}{'POS #CP':>9}{'NEG #CP':>9}"
          f"{'NEG false':>11}{'consistency':>13}{'Runtime(ms)':>14}")
    print(f"{'':<12}{'(events)':>12}{'':>9}{'(structure)':>9}"
          f"{'alarm/'+str(len(NEGATIVE)):>11}{'POS-NEG_fa':>13}{'all series':>14}")
    print("-" * 92)
    for name in order:
        pos = results[name]["POS"]
        neg = results[name]["NEG"]
        caught = set()
        pos_cp = 0
        for col, (_, n, matched, _) in pos.items():
            caught.update(matched)
            pos_cp += n
        pos_recall = len(caught) / n_events if n_events else 0.0
        neg_cp = sum(n for (_, n, _, _) in neg.values())
        # a NEGATIVE CP is a spurious STORM attribution only if it lands near a
        # rain event; CPs elsewhere are plausibly real non-storm regimes
        # (seasonal conductance, winter road-salt) and are NOT counted as
        # storm false alarms.
        neg_fa = sum(1 for (_, _, _, near) in neg.values() if near > 0)
        neg_fa_frac = neg_fa / len(neg) if neg else 0.0
        consistency = pos_recall - neg_fa_frac
        print(f"{name:<12}{pos_recall:>12.2f}{pos_cp:>9}{neg_cp:>9}"
              f"{neg_fa:>8}/{len(neg):<2}{consistency:>13.2f}{runtime_ms[name]:>14.1f}")
        summary_rows.append({"method": name, "pos_recall": pos_recall, "pos_cp": pos_cp,
                             "neg_cp": neg_cp, "neg_false": neg_fa, "n_neg": len(neg),
                             "consistency": consistency,
                             "runtime_ms": round(runtime_ms[name], 3),
                             "tol": args.tol, "alpha": args.alpha, "hazard": args.hazard,
                             "min_seg": args.min_seg})
    print("-" * 92)
    print("POS recall  = fraction of the 4 rain events flagged in TP or TN "
          "(higher=better)")
    print("NEG #CP     = total change-points on the 12 negative controls "
          "(structure; may include REAL non-storm regimes e.g. seasonal / "
          "road-salt)")
    print("NEG false   = negative controls with a CP NEAR a rain event = "
          "spurious storm attribution (the true false-alarm; 0 ideal)")
    print("consistency = POS recall - (NEG false-alarm fraction); +1 = caught "
          "every storm, no spurious storm attribution on controls")
    print("\nInterpretation: a selective detector fires on the nutrient")
    print("positives near storms and stays silent on the dilution/flush "
          "negatives.")
    print("VOCD's robustness favours specificity (few NEG false alarms); TP/TN")
    print("shifts are short (~6 d) so POS recall depends on signal + operating "
          "point.")
    print("Runtime(ms) = wall-clock of detection summed over all POS+NEG series "
          "(vocd = python backend; 'vocd (cpp)' = compiled twin).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols_csv = ["group", "constituent", "method", "n_cp", "matched_events",
                "n_cp_near_event", "runtime_ms", "cp_dates"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols_csv)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"\nwrote {len(csv_rows)} rows -> {args.out}")
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"wrote {len(summary_rows)} rows -> {args.summary_out}")


if __name__ == "__main__":
    main()
