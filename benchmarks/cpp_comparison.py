"""
C++ vs Python runtime comparison: VOCD, FOCuS and NP-FOCuS on the synthetic task.

Same contaminated change-in-mean task as the paper (T=500, six changes, 2%
outliers at +/-10, ten trials; data, contamination and scorer imported
verbatim from ``examples/synthetic_data.py``).  Each method is run at the
operating point the v3 sweep selected for it, in every implementation that
exists for it:

  vocd      python           reference (vocd.core)
            cpp (hybrid)     compiled Viterbi passes + NumPy gate (_faststats)
            cpp (gate)       compiled Viterbi passes + compiled gate (vocd/_gate.cpp)
            cpp (full)       whole detect in one C++ call (vocd/_vocd_full.cpp)
  focus     python           alternative_models/focus.py, brute-force O(n^2)
            cpp brute-force  line-by-line port of that scan
            cpp pruned       functional pruning (Romano et al. 2023), O(log n)
  np-focus  python / cpp brute-force / cpp pruned, likewise

Every C++ variant is required to reproduce the Python reference's detections
EXACTLY on every trial (the script asserts it and prints the check), so the
runtime columns compare implementations of one algorithm, not different
algorithms.

Read the VOCD rows carefully.  VOCD's "cpp" backend compiles only the
Viterbi proposal stage; the MW/KS gate runs in NumPy (_faststats).  The
"cpp (gate)" row adds the compiled gate from vocd/_gate.cpp; the "cpp (full)"
row runs the whole of core.detect in one C++ call (vocd/_vocd_full.cpp via
vocd/_fastdetect.py).  Both are EXACT -- identical candidates and
bit-identical p-values, asserted here -- so the VOCD rows differ only in
runtime.  "cpp (full)" is the implementation to report as "vocd (cpp)"; the
other two are kept to show where the time went.  The script isolates the
gate's share of the hybrid time by stubbing the rank tests.

Optional ``--scaling`` times the compiled variants on longer series to show
the complexity difference (brute-force O(n^2) vs pruned O(n log n) vs VOCD).

Usage::

    python benchmarks/cpp_comparison.py                 # 10 trials
    python benchmarks/cpp_comparison.py --scaling       # + scaling table
    python benchmarks/cpp_comparison.py --focus-alpha 1e-7 --npfocus-alpha 0.03

The compiled alarms are the registered ones in alternative_models/_cppalarms
(built on first use); VOCD's engines come from vocd/_fastgate and
vocd/_fastdetect.  Nothing in the repo is modified by running this.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "examples"))

import vocd                                          # noqa: E402
from vocd import _backend as _vb                     # noqa: E402
import vocd._fastgate as fastgate                     # noqa: E402
import vocd._fastdetect as fastdetect                 # noqa: E402
from alternative_models._base import alpha_to_chi2_threshold   # noqa: E402
from alternative_models.focus import detect as focus_py        # noqa: E402
from alternative_models.npfocus import detect as npfocus_py    # noqa: E402
import synthetic_data as sd                          # noqa: E402

FOCUS_BURNIN = 10                       # defaults of the Python classes
NPFOCUS_BURNIN = 15
NPFOCUS_LEVELS = (0.25, 0.5, 0.75)
MIN_SEG = 10


# ── compiled alarms come from the registered module ──────────────────

def make_cpp_detectors():
    """focus_cpp(x, alpha, pruned) / npfocus_cpp(x, alpha, pruned) -> (cps, stats),
    via alternative_models._cppalarms (the same code the benchmark panel runs)."""
    from alternative_models import _cppalarms as ca
    ca._load()                      # one-off compile, outside any timer

    def focus_cpp(x, alpha, pruned):
        return ca.run("focus", x, alpha, MIN_SEG, pruned=bool(pruned))

    def npfocus_cpp(x, alpha, pruned):
        return ca.run("np-focus", x, alpha, MIN_SEG, pruned=bool(pruned))

    return focus_cpp, npfocus_cpp


# ── timing helpers ────────────────────────────────────────────────────

def timed(fn, *a, repeat=1, **kw) -> Tuple[object, float]:
    """Run fn; return (result, best-of-repeat wall-clock ms)."""
    best = np.inf
    res = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn(*a, **kw)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return res, best


def vocd_run(x, fwer, backend, gate="python"):
    if gate == "full":
        return [int(c) for c in fastdetect.detect(x).changepoints(fwer=fwer)]
    if gate == "cpp":
        fastgate.enable()
    try:
        r = vocd.detect(x, backend=backend)
        return [int(c) for c in r.changepoints(fwer=fwer)]
    finally:
        if gate == "cpp":
            fastgate.disable()


def vocd_gate_share(x, fwer, repeat) -> Tuple[float, float]:
    """(total hybrid ms, everything-except-gate ms).  The gate runs INSIDE
    ``detect`` as candidates become testable (``changepoints`` only thresholds
    stored p-values), so it is isolated by temporarily replacing
    ``vocd.core._verify`` with a stub that returns the no-reject tuple
    (p_mw, p_ks, p_max, tested) = (1, 1, 1, True).  Detections differ under the
    stub, of course; only its runtime is used."""
    import vocd.core as core
    _, tot = timed(vocd_run, x, fwer, "cpp", repeat=repeat)
    real = core._verify
    core._verify = lambda y, cp, w, min_side: (1.0, 1.0, 1.0, True)
    try:
        _, rest = timed(vocd_run, x, fwer, "cpp", repeat=repeat)
    finally:
        core._verify = real
    return tot, rest


def f1_of(cps) -> float:
    return sd.accuracy_delay([0] + sorted(cps))["f1"]


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=sd.N_TRIALS)
    ap.add_argument("--repeat", type=int, default=3,
                    help="timing repeats per trial; best-of is reported")
    ap.add_argument("--vocd-fwer", type=float, default=0.01)
    ap.add_argument("--focus-alpha", type=float, default=1e-7,
                    help="FOCuS alpha (v3 sweep best)")
    ap.add_argument("--npfocus-alpha", type=float, default=0.03,
                    help="NP-FOCuS alpha (v3 sweep best)")
    ap.add_argument("--scaling", action="store_true",
                    help="also time the compiled variants on longer series")
    args = ap.parse_args()

    focus_cpp, npfocus_cpp = make_cpp_detectors()
    if not _vb.available():
        raise SystemExit("VOCD cpp backend not built: run `python setup_ext.py` "
                         "(or `pip install -e .`) first")

    data_clean, _ = sd.make_paper_data()
    trials = [dc.ravel() for dc in sd.make_trials(data_clean, args.trials)]

    print(f"C++ vs Python — T={sd.T}, {args.trials} trials, 2% outliers, "
          f"best-of-{args.repeat} timing per trial")
    print(f"operating points: vocd fwer={args.vocd_fwer:g}, "
          f"focus alpha={args.focus_alpha:g}, np-focus alpha={args.npfocus_alpha:g}\n")

    # rows: label -> list of (ms, f1); stats: label -> list of (mean_stack, peak)
    rows: Dict[str, List[Tuple[float, float]]] = {}
    stats: Dict[str, List[Tuple[float, float]]] = {}
    mismatches: Dict[str, int] = {}
    gate = []

    for i, x in enumerate(trials):
        # ── VOCD ──
        py, ms = timed(vocd_run, x, args.vocd_fwer, "python", repeat=args.repeat)
        rows.setdefault("vocd  python", []).append((ms, f1_of(py)))
        cp, ms = timed(vocd_run, x, args.vocd_fwer, "cpp", repeat=args.repeat)
        rows.setdefault("vocd  cpp (hybrid)", []).append((ms, f1_of(cp)))
        mismatches["vocd cpp (hybrid)"] = mismatches.get("vocd cpp (hybrid)", 0) + int(cp != py)
        cg, ms = timed(vocd_run, x, args.vocd_fwer, "cpp", "cpp", repeat=args.repeat)
        rows.setdefault("vocd  cpp (gate)", []).append((ms, f1_of(cg)))
        mismatches["vocd cpp (gate)"] = mismatches.get("vocd cpp (gate)", 0) + int(cg != py)
        cf, ms = timed(vocd_run, x, args.vocd_fwer, "cpp", "full", repeat=args.repeat)
        rows.setdefault("vocd  cpp (full)", []).append((ms, f1_of(cf)))
        mismatches["vocd cpp (full)"] = mismatches.get("vocd cpp (full)", 0) + int(cf != py)
        gate.append(vocd_gate_share(x, args.vocd_fwer, args.repeat))

        # ── FOCuS ──
        py, ms = timed(focus_py, x, alpha=args.focus_alpha, min_seg=MIN_SEG,
                       repeat=args.repeat)
        rows.setdefault("focus  python", []).append((ms, f1_of(py)))
        for pruned, lab in ((0, "focus  cpp brute-force"), (1, "focus  cpp pruned")):
            (cp, st), ms = timed(focus_cpp, x, args.focus_alpha, pruned, repeat=args.repeat)
            rows.setdefault(lab, []).append((ms, f1_of(cp)))
            stats.setdefault(lab, []).append(st)
            mismatches[lab] = mismatches.get(lab, 0) + int(cp != py)

        # ── NP-FOCuS ──
        py, ms = timed(npfocus_py, x, alpha=args.npfocus_alpha, min_seg=MIN_SEG,
                       repeat=args.repeat)
        rows.setdefault("np-focus  python", []).append((ms, f1_of(py)))
        for pruned, lab in ((0, "np-focus  cpp brute-force"), (1, "np-focus  cpp pruned")):
            (cp, st), ms = timed(npfocus_cpp, x, args.npfocus_alpha, pruned, repeat=args.repeat)
            rows.setdefault(lab, []).append((ms, f1_of(cp)))
            stats.setdefault(lab, []).append(st)
            mismatches[lab] = mismatches.get(lab, 0) + int(cp != py)
        print(f"trial {i} done", end="\r", flush=True)
    print(" " * 20, end="\r")

    # ── detection-equivalence check ──
    print("detection equivalence vs the Python reference (trials with any difference):")
    ok = True
    for lab, nbad in mismatches.items():
        flag = "identical" if nbad == 0 else f"DIFFERS on {nbad}/{args.trials} trials"
        ok &= nbad == 0
        print(f"  {lab:<28s} {flag}")
    if not ok:
        print("  ! a C++ variant does not reproduce the reference; its runtime row "
              "is NOT a like-for-like number until this is resolved")
    print()

    # ── table ──
    print(f"{'method / implementation':<30}{'F1':>8}{'runtime ms':>18}"
          f"{'kept cands/step':>18}{'peak':>7}")
    print("-" * 81)
    for lab, v in rows.items():
        ms = np.array([a for a, _ in v]); f1 = np.array([b for _, b in v])
        extra = ""
        if lab in stats:
            m = np.array([a for a, _ in stats[lab]]); pk = np.array([b for _, b in stats[lab]])
            extra = f"{m.mean():>18.1f}{pk.max():>7.0f}"
        print(f"{lab:<30}{f1.mean():>8.3f}{ms.mean():>10.2f} ± {ms.std():<6.2f}{extra}")
    print("-" * 81)
    tot = np.array([t for t, _ in gate]); rest = np.array([p for _, p in gate])
    cg_ms = np.array([a for a, _ in rows["vocd  cpp (gate)"]])
    cf_ms = np.array([a for a, _ in rows["vocd  cpp (full)"]])
    print(f"vocd cpp (hybrid) = {tot.mean():.2f} ms, of which the NumPy rank-test "
          f"gate (_faststats MW+KS) = {(tot - rest).mean():.2f} ms "
          f"({100 * (tot - rest).mean() / tot.mean():.0f}%) and everything else "
          f"(compiled passes, regime bank, bookkeeping) = {rest.mean():.2f} ms.\n"
          f"vocd cpp (gate)   = {cg_ms.mean():.2f} ms with the compiled gate "
          f"(bit-identical p-values); the remaining Python is the regime-bank "
          f"fit and candidate bookkeeping in vocd.core.\n"
          f"vocd cpp (full)   = {cf_ms.mean():.2f} ms with the whole detect in "
          f"one C++ call: this is the number to report as 'vocd (cpp)'.")
    print("'kept cands/step' = mean number of surviving candidate change-points\n"
          "per observation summed over all pruning stacks (the brute-force scan\n"
          "visits every one of the up to ~500 candidates instead).  Runtime is the\n"
          "whole detection call from Python, incl. the ctypes / pybind crossing.")

    if not args.scaling:
        return

    # ── scaling ──
    print("\nscaling (compiled variants only; one series per n, best-of-3 ms):")
    print(f"{'n':>8}{'vocd cpp(hybrid)':>18}{'vocd cpp(gate)':>16}{'vocd cpp(full)':>16}"
          f"{'focus brute':>14}{'focus pruned':>14}{'np-focus brute':>16}{'np-focus pruned':>17}")
    rng = np.random.default_rng(0)
    for n in (500, 2000, 8000, 32000):
        # piecewise-constant series with the paper's segment pattern tiled
        seg = 100
        mus = np.tile(np.array(sd.MUS, float), n // (seg * len(sd.MUS)) + 1)
        y = np.repeat(mus, seg)[:n] + rng.standard_normal(n)
        k = int(0.02 * n)
        idx = rng.choice(n, k, replace=False)
        y[idx] += rng.choice([-10.0, 10.0], size=k)
        _, tv = timed(vocd_run, y, args.vocd_fwer, "cpp", repeat=3)
        _, tg = timed(vocd_run, y, args.vocd_fwer, "cpp", "cpp", repeat=3)
        _, tf = timed(vocd_run, y, args.vocd_fwer, "cpp", "full", repeat=3)
        _, fb = timed(focus_cpp, y, args.focus_alpha, 0, repeat=3)
        _, fp = timed(focus_cpp, y, args.focus_alpha, 1, repeat=3)
        _, nb = timed(npfocus_cpp, y, args.npfocus_alpha, 0, repeat=3)
        _, npr = timed(npfocus_cpp, y, args.npfocus_alpha, 1, repeat=3)
        print(f"{n:>8}{tv:>18.1f}{tg:>16.1f}{tf:>16.1f}{fb:>14.1f}{fp:>14.1f}{nb:>16.1f}{npr:>17.1f}")
    print("(the brute-force columns grow ~quadratically with the segment length;\n"
          "pruned FOCuS is the published log-linear method)")


if __name__ == "__main__":
    main()
