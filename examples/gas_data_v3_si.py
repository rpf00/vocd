"""
MOX gas-sensor drift-correction benchmark, v3 — Supplementary Information
(diacetyl 0.1 ppm and 1 ppm; Supplementary Figs. 3 and 4).

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)
    + the oracle (true cycle_stage baseline; the non-causal ceiling)

Same task, data, correction, scoring, panel and plumbing as ``gas_data_v3.py``
(imported verbatim from it, which imports the task from ``gas_data.py``): on
each day t > 1 a detector's FIRST change-point on the raw cycle R_t(tau) marks
the baseline -> gas onset; the baseline before it is re-anchored to day 1 by
one constant; the corrected cycle is scored against the day-1 cycle by MAE
(and RMSE) over tau, averaged over the test days.  What this file changes:

* Several analytes in one run (``--analytes``, default both diacetyl
  concentrations).  Names are matched to ``gas_data.ANALYTE_LABELS`` ignoring
  case and punctuation, so ``Diacetyl_0.1ppm`` (the file prefix in
  ``data/gas_sensor_worner2025/Day N/``) and ``Diacetyl_0_1ppm`` both work.
* Budgets TRANSFERRED from ethanol by default.  Each detector runs at its
  ethanol min-MAE operating point, read from ``--budgets-from``
  (``gas_v3_best.csv``) and optionally overridden by ``--budgets-extra``
  (e.g. the single-point CHAMP / Dm-BOCD run), without any re-tuning.  This
  mirrors the ISFET Na+/K+ design (budgets fixed on H+) and avoids a second
  min-MAE selection on SI-only data.  ``--sweep`` instead re-runs the v3
  protocol on each analyte (its own min-MAE point, plateau centre, edge flag).
* Every output of ``gas_data_v3.py`` is written per analyte to ``--out-dir``
  (default ``results/gas_si``) under the same names with the analyte tag, e.g.
  ``gas_v3_diacetyl_0_1ppm_best.csv``, so ``vocd_fig4_gas.py`` can plot them
  unchanged.  ``gas_v3_si_summary.csv`` collects every analyte's rows with the
  margin to the oracle.

A-priori settings are those of ``gas_data_v3.py`` (min_seg = 30, NUNC window
60, CHAMP gamma = 0.05, confirmation = min_seg, lag = 20, 100 particles).

Usage::

    python examples/gas_data_v3_si.py --dsm-path ../DSM-bocd \\
        --budgets-from results/gas/gas_v3_best.csv
    python examples/gas_data_v3_si.py --dsm-path ../DSM-bocd \\
        --budgets-from results/gas/gas_v3_best.csv \\
        --budgets-extra results/gas/gas_v3_best_champ_dm.csv
    python examples/gas_data_v3_si.py --dsm-path ../DSM-bocd --sweep   # per-analyte selection
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from alternative_models import REGISTRY       # noqa: E402
import gas_data_v3 as v3                      # noqa: E402  (verbatim panel + task)

ANALYTE_LABELS = v3.ANALYTE_LABELS
PANEL, HAZARD_METHODS = v3.PANEL, v3.HAZARD_METHODS
TWINS, TWIN_METHODS = v3.TWINS, v3.TWIN_METHODS
METRIC_KEYS = v3.METRIC_KEYS
SUMMARY_COLS = ["analyte", "method", "budget_kind", "budget", "budget_source"] + METRIC_KEYS \
    + ["oracle_margin_pct"]


# ── analytes ──────────────────────────────────────────────────────────

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_analyte(name):
    """Match a user-given analyte name to a gas_data.ANALYTE_LABELS key,
    ignoring case and punctuation (keys and, for a dict, labels)."""
    if name in ANALYTE_LABELS:
        return name
    items = (ANALYTE_LABELS.items() if isinstance(ANALYTE_LABELS, dict)
             else [(k, k) for k in ANALYTE_LABELS])
    hits = [k for k, lab in items if _norm(name) in (_norm(k), _norm(lab))]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"analyte {name!r} not found (or ambiguous) in gas_data.ANALYTE_LABELS "
                     f"{list(ANALYTE_LABELS)}; add it there next to Ethanol")


def analyte_tag(key):
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


# ── budgets ───────────────────────────────────────────────────────────

def budget_kind(name):
    if name == "vocd":
        return "fwer"
    return "hazard" if name in HAZARD_METHODS else "alpha"


def load_budgets(path, extra=None):
    """method -> budget from v3 best CSV(s); rows of ``extra`` override."""
    out = {}
    for p in (path, extra):
        if not p:
            continue
        if not os.path.exists(p):
            raise SystemExit(f"budget file not found: {p}  (run gas_data_v3.py --sweep first)")
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                b = (r.get("budget") or "").strip()
                if b:                                  # oracle rows carry no budget
                    out[r["method"]] = float(b)
    return out


def transferred_budgets(wanted, args):
    src = load_budgets(args.budgets_from, args.budgets_extra)
    budgets, sources = {}, {}
    for m in wanted:
        if m == "dm-bocd" and args.dm_hazard is not None:
            budgets[m], sources[m] = args.dm_hazard, "--dm-hazard"
        elif m == "champ" and args.champ_hazard is not None:
            budgets[m], sources[m] = args.champ_hazard, "--champ-hazard"
        elif m in src:
            budgets[m], sources[m] = src[m], "ethanol"
        else:
            budgets[m] = args.hazard if m in HAZARD_METHODS else args.alpha
            sources[m] = "fallback"
            print(f"! {m}: not in the ethanol budget file(s); fallback "
                  f"{budget_kind(m)}={budgets[m]:g}")
    return budgets, sources


# ── one analyte ───────────────────────────────────────────────────────

def sweep_select(cps_fn, name, days, args, best_by, curve):
    """The gas_data_v3.py sweep for one detector: min-MAE plateau centre."""
    alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
    hazard_grid = [float(v) for v in args.hazard_grid.split(",") if v.strip()]
    if name == "dm-bocd":
        kind = "hazard"
        if args.dm_hazard is not None:
            grid, src = [args.dm_hazard], "--dm-hazard"
        elif args.dm_hazard_grid.strip().lower() == "inherit":
            grid, src = [best_by.get("bocd", 300.0)], "inherited from bocd best"
        else:
            grid = [float(v) for v in args.dm_hazard_grid.split(",") if v.strip()]
            src = "own grid"
        print(f"  [{name}] (hazard, {src}) ", end="", flush=True)
    elif name in HAZARD_METHODS:
        grid, kind = hazard_grid, "hazard"
        print(f"  [{name}] (hazard) ", end="", flush=True)
    else:
        grid, kind = alpha_grid, budget_kind(name)
        print(f"  [{name}] ({kind}) ", end="", flush=True)
    evals = []
    for b in grid:
        r = v3.eval_method(cps_fn, name, days, b, args)
        curve.append({"method": name, "budget_kind": kind, "budget": b,
                      **{k: r[k] for k in METRIC_KEYS}})
        evals.append((b, r))
        print(".", end="", flush=True)
    b_star, r_star, mn, n_plat = v3.best_point(evals)
    plateau = "" if n_plat == 1 else f" ({n_plat}-pt plateau)"
    edge = " [EDGE of grid]" if len(grid) > 1 and b_star in (grid[0], grid[-1]) else ""
    print(f" min MAE={mn:.2f} @ {kind}={b_star:g}{plateau}{edge}")
    return b_star, r_star


def run_analyte(analyte, wanted, cps_fn, args, budgets_t, sources_t):
    tag = analyte_tag(analyte)
    print(f"\n── {analyte} " + "─" * 60)
    days = v3.load_all_days(args.data_dir, analyte, args.replicate, args.sensor)
    print(f"  {len(days)} days ({len(days) - 1} test days), ~{len(days[0][1])} pts/cycle")

    rows, budgets, sources, curve = {}, {}, {}, []
    if not args.no_oracle:
        rows["oracle"] = v3.eval_method(cps_fn, "oracle", days, None, args)

    proc = list(wanted)
    if "bocd" in proc and "dm-bocd" in proc:          # Dm-BOCD may inherit BOCD's best
        proc.remove("dm-bocd")
        proc.insert(proc.index("bocd") + 1, "dm-bocd")
    best_by = {}
    for name in proc:
        if args.sweep:
            b, r = sweep_select(cps_fn, name, days, args, best_by, curve)
            src = "own sweep"
        else:
            b, src = budgets_t[name], sources_t[name]
            print(f"  [{name}] {budget_kind(name)}={b:g} ({src}) ...", end="", flush=True)
            r = v3.eval_method(cps_fn, name, days, b, args)
            print(f" MAE={r['mae']:.2f}, missed {r['n_missed']}")
        rows[name], budgets[name], sources[name], best_by[name] = r, b, src, b

    for base in wanted:
        if base in TWINS and base in rows:
            tw = v3.paired_row(cps_fn, base, days, budgets[base], args, rows[base])
            if tw:
                label = TWINS[base][0]
                rows[label], budgets[label], sources[label] = tw, budgets[base], sources[base]

    det_order = [m for m in v3.insert_twins(wanted) if m in rows]
    order = det_order + (["oracle"] if "oracle" in rows else [])
    v3.print_table(order, rows, budgets)

    # outputs, same files as gas_data_v3.py with the analyte tag
    out = lambda kind: os.path.join(args.out_dir, f"gas_v3_{tag}_{kind}.csv")
    v3.write_rows(out("best"), order, rows, budgets)
    print(f"wrote {len(order)} rows -> {out('best')}")
    if curve:
        v3.write_curve(out("sweep"), curve)
        print(f"wrote {len(curve)} curve rows -> {out('sweep')}")
    show_days = [int(v) for v in args.show_days.split(",") if v.strip()]
    v3.write_series(out("series"), order, rows, days, show_days, args.resid_day, args.sample_hz)
    v3.write_onsets(out("onsets"), det_order, rows)
    print(f"wrote onsets -> {out('onsets')}")
    v3.write_resid_all(out("resid_all"), [m.strip() for m in args.resid_methods.split(",")],
                       rows, days, args.sample_hz)

    ora = rows["oracle"]["mae"] if "oracle" in rows else np.nan
    summary = []
    for name in order:
        r = rows[name]
        summary.append({"analyte": analyte, "method": name,
                        "budget_kind": budget_kind(name.split(" ")[0]) if name != "oracle" else "",
                        "budget": budgets.get(name, ""),
                        "budget_source": sources.get(name, ""),
                        **{k: r[k] for k in METRIC_KEYS},
                        "oracle_margin_pct": (r["mae"] / ora - 1.0) * 100.0
                        if np.isfinite(ora) and ora > 0 else np.nan})
    worked = [s for s in summary if s["method"] in wanted and s["n_missed"] == 0]
    if worked and np.isfinite(ora):
        top = max(s["oracle_margin_pct"] for s in worked)
        print(f"  detectors with no missed day: {', '.join(s['method'] for s in worked)}; "
              f"all within {top:.1f}% of the oracle ({ora / 1e3:.2f} kOhm)")
    return summary


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=v3._DEFAULT_GAS_DIR)
    ap.add_argument("--analytes", default="Diacetyl_0.1ppm,Diacetyl_1ppm",
                    help="comma list, matched to gas_data.ANALYTE_LABELS")
    ap.add_argument("--sensor", type=int, default=0, help="sensor id (0-based -> R1)")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--sample-hz", type=float, default=1.0, help="for the t_min column")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=PANEL)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--min-seg", type=int, default=30,
                    help="min samples between change-points (per cycle)")
    ap.add_argument("--bocd-extract", choices=["causal", "upstream"], default="causal")
    # budgets: transferred from ethanol (default) or re-selected (--sweep)
    ap.add_argument("--budgets-from", default="results/gas/gas_v3_best.csv",
                    help="ethanol best-operating-point CSV from gas_data_v3.py --sweep")
    ap.add_argument("--budgets-extra", default=None,
                    help="optional CSV whose rows OVERRIDE --budgets-from "
                         "(e.g. the single-point CHAMP / Dm-BOCD run)")
    ap.add_argument("--champ-hazard", type=float, default=None,
                    help="single hazard for CHAMP, overriding the budget files")
    ap.add_argument("--dm-hazard", type=float, default=None,
                    help="single hazard for Dm-BOCD, overriding budget files and grids")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="fallback fwer/alpha for a method missing from the budget files")
    ap.add_argument("--hazard", type=float, default=300.0,
                    help="fallback hazard (samples) for a method missing from the budget files")
    ap.add_argument("--sweep", action="store_true",
                    help="re-select each detector's min-MAE point per analyte (v3 protocol)")
    ap.add_argument("--alpha-grid",
                    default="1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,5e-1,7e-1")
    ap.add_argument("--hazard-grid", default="100,300,600,1000,2000,5000,10000",
                    help="length-prior timescales (samples) swept for BOCD and CHAMP")
    ap.add_argument("--dm-hazard-grid", default="100,300,1000,3000,10000",
                    help="coarse grid swept for Dm-BOCD on its own (slow); "
                         "'inherit' = BOCD's best (the old protocol)")
    # CHAMP / NUNC a-priori settings (identical to gas_data_v3.py)
    ap.add_argument("--champ-gamma", type=float, default=0.05)
    ap.add_argument("--champ-em-iters", type=int, default=2)
    ap.add_argument("--champ-max-particles", type=int, default=100,
                    help="particle cap (the published algorithm); exact = 0 is O(n^3)")
    ap.add_argument("--champ-lag", type=int, default=20)
    ap.add_argument("--champ-confirm", type=int, default=None, help="default = --min-seg")
    ap.add_argument("--nunc-window", type=int, default=None,
                    help="NUNC rolling window; default max(50, 2*min_seg) (= 60 here)")
    ap.add_argument("--nunc-sims", type=int, default=20000)
    ap.add_argument("--vocd-cpp", choices=["hybrid", "gate", "full"], default="full")
    # outputs
    ap.add_argument("--out-dir", default="results/gas_si")
    ap.add_argument("--resid-methods", default="vocd",
                    help="methods written to the resid_all file (comma-separated)")
    ap.add_argument("--show-days", default="1,15,30,39",
                    help="day numbers whose raw cycles go to the series file")
    ap.add_argument("--resid-day", type=int, default=39,
                    help="day whose reference - corrected residual goes to the series file")
    args = ap.parse_args()
    dsm_path = args.dsm_path or os.environ.get("VOCD_DSM_PATH")
    if args.champ_max_particles is not None and args.champ_max_particles <= 0:
        args.champ_max_particles = None
    if args.nunc_window is None:
        args.nunc_window = max(50, 2 * args.min_seg)
    os.makedirs(args.out_dir, exist_ok=True)

    analytes = [resolve_analyte(a.strip()) for a in args.analytes.split(",") if a.strip()]

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

    budgets_t, sources_t = ({}, {}) if args.sweep else transferred_budgets(wanted, args)

    if args.vocd_cpp == "full":
        import vocd._fastdetect as fastdetect
        fastdetect.enable()
    elif args.vocd_cpp == "gate":
        import vocd._fastgate as fastgate
        fastgate.enable()
    if any(m in TWINS for m in wanted):
        from alternative_models import _cppalarms
        _cppalarms._load()                   # one-off compile, outside timers

    cps_fn = v3.make_cps_fn(args, dsm_path)
    print(f"SI gas benchmark: sensor R{args.sensor + 1}, replicate {args.replicate}, "
          f"analytes {analytes}")
    print(f"methods: {', '.join(wanted)}; twins: "
          f"{', '.join(TWINS[m][0] for m in wanted if m in TWINS)}; vocd C++ engine = {args.vocd_cpp}")
    if args.sweep:
        print("SWEEP mode — each analyte re-selects every detector's MIN-MAE operating point.")
    else:
        print("TRANSFER mode — budgets fixed on ethanol, no re-tuning: "
              + ", ".join(f"{m}={budgets_t[m]:g}" for m in wanted))
    if "nunc" in wanted:
        print(f"nunc : window={args.nunc_window} (= max(2 x VOCD gate window, 2 x min_seg))")
    if "champ" in wanted:
        print(f"champ: gamma={args.champ_gamma}, lag={args.champ_lag}, "
              f"confirm={args.champ_confirm or args.min_seg}, "
              f"{'exact' if not args.champ_max_particles else f'max_particles={args.champ_max_particles}'}")

    summary = []
    for analyte in analytes:
        summary += run_analyte(analyte, wanted, cps_fn, args, budgets_t, sources_t)

    p = os.path.join(args.out_dir, "gas_v3_si_summary.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {len(summary)} rows ({len(analytes)} analytes) -> {p}")
    print("mean MAE / RMSE (Ohm) of the corrected cycle vs the day-1 cycle over the test "
          "days; CI = 95%\nmoving-block bootstrap over days (5-day blocks, 2000 resamples). "
          "'miss' = days with no usable\nonset (no correction applied). '(cpp)' rows reuse "
          "their Python row's budget and onsets.\noracle uses the true cycle_stage "
          "(non-causal ceiling).")


if __name__ == "__main__":
    main()
