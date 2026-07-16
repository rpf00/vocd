"""
Synthetic benchmark: VOCD vs STD-BOCD vs DSM-BOCD.

Replicates the protocol of

    Altamirano, Briol & Knoblauch (2023),
    "Robust and Scalable Bayesian Online Changepoint Detection", ICML.
    https://github.com/maltamiranomontero/DSM-bocd   (MIT licence)

as laid out in that repository's ``notebooks/accuracy_delay.ipynb``: the data
generator, the contamination procedure, the model priors and the scoring
function are reproduced verbatim, so the baselines are compared on their own
terms rather than on ours.

The baselines are not vendored.  Clone the repo and point this script at it::

    git clone https://github.com/maltamiranomontero/DSM-bocd.git
    python benchmarks/synthetic_benchmark.py --dsm-path ./DSM-bocd

Without ``--dsm-path`` only VOCD is reported, so the script always runs.

Reproducibility notes
---------------------
*Data*: ``np.random.seed(3000)`` then one ``np.random.normal(mu, 1)`` draw per
step, walking the segment means.  This is the legacy global RNG, not a
``Generator``; the two give different streams from the same seed, so the legacy
call is required to land on the same series.

*Contamination*: ``np.random.seed(54321)`` once, then ten contaminated copies
drawn back-to-back from the running global state.  Trial i depends on trial
i-1: the trials are not independently seeded and cannot be generated out of
order.  This procedure reproduces the published STD-BOCD row
(PPV 0.670+/-0.105, TPR 0.933+/-0.082).

*Scoring*: ``accuracy_delay`` below is the reference implementation, copied
as-is.  It is **not** equivalent to ``vocd.metrics.evaluate``:

  - Matching is many-to-one.  Several predictions near one true change-point
    each count as a separate TP, so clustered detections inflate TP instead of
    costing precision.
  - TP counts *predictions* while FN counts *true* change-points, so the two
    terms of TPR = TP/(TP+FN) are different kinds of object.
  - Hits use a strict ``< window``, misses a strict ``> window``; a prediction
    at exactly ``window`` is neither.
  - ``cps_pred[0]`` is skipped, because ``find_cp`` returns a leading 0
    sentinel.  VOCD's output is prefixed with 0 here so every method is scored
    alike.

``--metric strict`` switches to the one-to-one matching in ``vocd.metrics``,
which is harsher on clustered detections.  The two metrics are not comparable;
``paper`` is the default because it is what the baselines' published figures
were produced with.

Spread is the population standard deviation (``np.var``, ddof=0), matching the
reference notebook.

Usage::

    python benchmarks/synthetic_benchmark.py --dsm-path ./DSM-bocd
    python benchmarks/synthetic_benchmark.py --trials 3 --methods vocd,std-bocd
    python benchmarks/synthetic_benchmark.py --metric strict
    python benchmarks/synthetic_benchmark.py --fwer 0.01      # legacy operating point
    python benchmarks/synthetic_benchmark.py --fdr 0.05 --correction by

Choosing the VOCD budget
------------------------
The legacy pipeline used a fixed alpha=1e-4 on every candidate.  That is not a
false-discovery rate; it is an attempt to say "essentially no false alarms",
which is a *family-wise* statement.  The matching budget is therefore
``--fwer``, not a small ``--fdr``.

Bonferroni tests at ``fwer / K`` and this benchmark yields K ~ 46 candidates
per trial, so ``--fwer 0.01`` lands on roughly the same threshold as alpha=1e-4
and reproduces the legacy row (#CPs 6.0+/-0.0, PPV 1.000, TPR 1.000).
``--fdr 0.001`` reaches the same place, but only by picking a number that hits
a known target -- which is the calibration the FDR framing exists to avoid.
Prefer stating a budget you can defend before seeing the data.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Callable, Dict, List, Sequence

import numpy as np

import vocd
from vocd.metrics import evaluate as strict_evaluate

# ── experiment constants (verbatim from accuracy_delay.ipynb) ─────────
T = 500
CPS_TRUE = [100, 200, 300, 350, 400, 450, 600]   # 600 is a sentinel past T
MUS = [10, 5, 0, 10, 5, 12, 0]
WINDOW = 10
N_TRIALS = 10
SEED_DATA = 3000
SEED_CONTAM = 54321
CONTAM_FRAC = 0.02
CONTAM_SPIKE = 10.0

HAZARD_TIMESCALE = 300
OMEGA = 0.1            # DSM-BOCD robustness parameter (requires calibration)

# VOCD error budget, set from the CLI (see --fdr / --fwer / --alpha).
VOCD_BUDGET: Dict = {"fdr": 0.05, "method": "bh"}
# VOCD backend, set from the CLI (see --backend).  Both backends produce
# identical output; only runtime differs.
VOCD_BACKEND: str = "auto"


# ── data (verbatim) ───────────────────────────────────────────────────


def make_paper_data(n: int = T, seed: int = SEED_DATA):
    """Piecewise-constant Gaussian series, legacy global RNG."""
    np.random.seed(seed)
    data = np.zeros((n, 1))
    mu = MUS[0]
    i = 0
    for t in range(n):
        if t == CPS_TRUE[i]:
            mu = MUS[i + 1]
            i += 1
        data[t, 0] = np.random.normal(mu, 1)
    return data, float(np.mean(data))


def contaminate(data: np.ndarray, frac: float = CONTAM_FRAC,
                spike: float = CONTAM_SPIKE) -> np.ndarray:
    """One contaminated copy, drawn from the *running* global RNG state."""
    n = data.shape[0]
    i_obs = np.random.choice(np.arange(0, n, 1), int(frac * n), replace=False)
    out = data.copy()
    j = np.random.choice([1, -1], size=len(i_obs))
    out[i_obs, 0] = out[i_obs, 0] + j * spike
    return out


def make_trials(data: np.ndarray, n_trials: int = N_TRIALS) -> List[np.ndarray]:
    """Seed once, then draw the trials back-to-back (order matters)."""
    np.random.seed(SEED_CONTAM)
    return [contaminate(data) for _ in range(n_trials)]


# ── scoring (verbatim from accuracy_delay.ipynb) ──────────────────────


def accuracy_delay(cps_pred: Sequence[int], cps_true: Sequence[int],
                   window: int = WINDOW):
    """Reference scoring function, reproduced as-is.  See module docstring."""
    TP = 0
    FP = 0
    FN = 0
    delays = []
    for cp in cps_pred[1:]:
        if np.min(np.abs(np.array(cps_true) - cp)) < window:
            TP += 1
            delays.append(np.min(np.abs(np.array(cps_true) - cp)))
        else:
            FP += 1
    for cp in np.array(cps_true)[:-1]:
        if np.min(np.abs(np.array(cps_pred) - cp)) > window:
            FN += 1
    ppv = TP / (TP + FP) if (TP + FP) else 0.0
    tpr = TP / (TP + FN) if (TP + FN) else 0.0
    return [ppv, tpr], delays


def score(cps_pred: Sequence[int], metric: str) -> dict:
    """``cps_pred`` includes the leading 0 sentinel, as ``find_cp`` returns."""
    if metric == "paper":
        (ppv, tpr), delays = accuracy_delay(cps_pred, CPS_TRUE, WINDOW)
        delay = float(np.mean(delays)) if delays else np.nan
    else:
        m = strict_evaluate(list(cps_pred[1:]), CPS_TRUE[:-1], tol=WINDOW)
        ppv, tpr, delay = m["ppv"], m["tpr"], m["delay"]
    f1 = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) else 0.0
    return {"n_detected": len(cps_pred) - 1, "ppv": ppv, "tpr": tpr,
            "f1": f1, "delay": delay}


# ── baselines ─────────────────────────────────────────────────────────


def load_dsm_bocd(path: str) -> Dict:
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise SystemExit(f"--dsm-path not found: {path}")
    sys.path.insert(0, path)
    try:
        from bocpd import bocpd
        from hazard import ConstantHazard
        from models import DSMGaussian, Gaussian
        from utils.find_cp import find_cp
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            f"could not import DSM-bocd from {path!r}: {e}\n"
            "  git clone https://github.com/maltamiranomontero/DSM-bocd.git"
        )
    return {"bocpd": bocpd, "ConstantHazard": ConstantHazard,
            "Gaussian": Gaussian, "DSMGaussian": DSMGaussian,
            "find_cp": find_cp}


# ── detectors: (data, mean0, mods) -> cps *with* leading 0 sentinel ───


def run_vocd(data: np.ndarray, mean0: float, mods: Dict) -> List[int]:
    r = vocd.detect(data.ravel(), backend=VOCD_BACKEND)
    return [0] + [int(c) for c in r.changepoints(**VOCD_BUDGET)]


def run_std_bocd(data: np.ndarray, mean0: float, mods: Dict) -> List[int]:
    model = mods["Gaussian"](mu0=mean0, kappa0=1, alpha0=1, omega0=1)
    R = mods["bocpd"](data, mods["ConstantHazard"](HAZARD_TIMESCALE), model)
    return list(mods["find_cp"](R))


def run_dsm_bocd(data: np.ndarray, mean0: float, mods: Dict) -> List[int]:
    # Priors and weighting function verbatim from accuracy_delay.ipynb.
    mean_mu0, var_mu0 = mean0, 1
    mean_Sigma0, var_Sigma0 = 1, 1

    mu0 = np.array([[mean_mu0 / var_mu0], [1 / var_mu0]])
    Sigma0 = np.eye(2)
    Sigma0[0, 0] = mean_Sigma0 / var_Sigma0
    Sigma0[1, 1] = 1 / var_Sigma0

    def m(x):
        return np.array([(1 + x ** 2) ** (-1 / 2)])

    def grad_m(x):
        return np.array([[-x / ((1 + x ** 2) ** (3 / 2))]])

    model = mods["DSMGaussian"](data=data, m=m, grad_m=grad_m,
                                omega=OMEGA, mu0=mu0, Sigma0=Sigma0)
    R = mods["bocpd"](data, mods["ConstantHazard"](HAZARD_TIMESCALE), model)
    return list(mods["find_cp"](R))


METHODS: Dict[str, Callable] = {
    "vocd": run_vocd,
    "std-bocd": run_std_bocd,
    "dsm-bocd": run_dsm_bocd,
}
NEEDS_DSM = {"std-bocd", "dsm-bocd"}


# ── harness ───────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="VOCD vs BOCD baselines")
    ap.add_argument("--dsm-path", default=None,
                    help="path to a clone of maltamiranomontero/DSM-bocd")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--methods", default="vocd,std-bocd,dsm-bocd")
    ap.add_argument("--metric", choices=["paper", "strict"], default="paper",
                    help="'paper' = reference accuracy_delay (many-to-one); "
                         "'strict' = vocd.metrics one-to-one matching")
    # VOCD error budget: at most one of these.
    ap.add_argument("--fdr", type=float, default=None,
                    help="target false-discovery rate (default 0.05)")
    ap.add_argument("--fwer", type=float, default=None,
                    help="target family-wise error rate (Bonferroni).  "
                         "fwer=0.01 reproduces the legacy alpha=1e-4 operating "
                         "point, since Bonferroni tests at fwer/K and K~46 here")
    ap.add_argument("--alpha", type=float, default=None,
                    help="raw per-test threshold, no correction.  Controls no "
                         "global error rate; for continuity with fixed-alpha "
                         "results only")
    ap.add_argument("--correction", choices=["bh", "by"], default="bh",
                    help="procedure for --fdr; 'by' is valid under arbitrary "
                         "dependence (test windows overlap), 'bh' assumes "
                         "independence or PRDS")
    ap.add_argument("--backend", choices=["auto", "python", "cpp"],
                    default="auto",
                    help="VOCD Viterbi backend.  Output is identical either "
                         "way; only runtime changes.  'cpp' raises if the "
                         "extension is not built")
    ap.add_argument("--out", default="benchmarks/results/synthetic.csv")
    args = ap.parse_args()

    if sum(x is not None for x in (args.fdr, args.fwer, args.alpha)) > 1:
        raise SystemExit("give at most one of --fdr, --fwer, --alpha")
    global VOCD_BUDGET, VOCD_BACKEND
    VOCD_BACKEND = args.backend
    if args.fwer is not None:
        VOCD_BUDGET = {"fwer": args.fwer}
        budget_desc = f"fwer={args.fwer:g} (Bonferroni)"
    elif args.alpha is not None:
        VOCD_BUDGET = {"alpha": args.alpha}
        budget_desc = f"alpha={args.alpha:g} (uncorrected)"
    else:
        q = 0.05 if args.fdr is None else args.fdr
        VOCD_BUDGET = {"fdr": q, "method": args.correction}
        budget_desc = f"fdr={q:g} ({args.correction.upper()})"

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in wanted:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; choose from {list(METHODS)}")

    mods: Dict = {}
    if any(m in NEEDS_DSM for m in wanted):
        if args.dsm_path is None:
            print("! no --dsm-path given; skipping BOCD baselines.\n"
                  "  git clone https://github.com/maltamiranomontero/DSM-bocd.git\n")
            wanted = [m for m in wanted if m not in NEEDS_DSM]
        else:
            mods = load_dsm_bocd(args.dsm_path)
    if not wanted:
        raise SystemExit("no methods left to run")

    data_clean, mean0 = make_paper_data()
    trials = make_trials(data_clean, args.trials)

    print(f"synthetic benchmark (DSM-BOCD protocol) — T={T}, "
          f"{len(CPS_TRUE) - 1} true CPs at {CPS_TRUE[:-1]}")
    print(f"{CONTAM_FRAC:.0%} outliers @ +/-{CONTAM_SPIKE:g}, "
          f"{args.trials} trials, window={WINDOW}, mean0={mean0:.3f}")
    from vocd import _backend as _vb
    print(f"metric: {args.metric}   VOCD budget: {budget_desc}   "
          f"backend: {_vb.resolve(args.backend)}")
    print(f"methods: {', '.join(wanted)}\n")

    rows: List[dict] = []
    for name in wanted:
        fn = METHODS[name]
        print(f"[{name}] ", end="", flush=True)
        for i, dc in enumerate(trials):
            t0 = time.perf_counter()
            cps = fn(dc, mean0, mods)
            ms = (time.perf_counter() - t0) * 1000.0
            r = score(cps, args.metric)
            r.update(method=name, trial=i, runtime_ms=ms)
            rows.append(r)
            print(".", end="", flush=True)
        print(" done")

    def agg(name: str, key: str):
        v = np.array([r[key] for r in rows if r["method"] == name], dtype=float)
        # population std (ddof=0), as in the reference notebook
        return np.nanmean(v), np.sqrt(np.nanvar(v))

    print(f"\n{'method':<10}{'#CPs':>12}{'PPV':>14}{'TPR':>14}"
          f"{'F1':>14}{'Delay':>13}{'Runtime (ms)':>16}")
    print("-" * 93)
    for name in wanted:
        n, p, t = agg(name, "n_detected"), agg(name, "ppv"), agg(name, "tpr")
        f1, d, rt = agg(name, "f1"), agg(name, "delay"), agg(name, "runtime_ms")
        print(f"{name:<10}"
              f"{n[0]:>7.1f}±{n[1]:<4.1f}"
              f"{p[0]:>8.3f}±{p[1]:<5.3f}"
              f"{t[0]:>8.3f}±{t[1]:<5.3f}"
              f"{f1[0]:>8.3f}±{f1[1]:<5.3f}"
              f"{d[0]:>8.2f}±{d[1]:<4.2f}"
              f"{rt[0]:>10.1f}±{rt[1]:<5.1f}")
    print("-" * 93)
    print(f"metric={args.metric}; spread is population std over {args.trials} "
          "trials.\nF1 is the mean of per-trial F1, so it is not exactly the F1 "
          "implied by the\nmean PPV and mean TPR in the same row (F1 is "
          "nonlinear).  Runtime is wall-clock\nfor the whole detection call, "
          "measured identically for every method here.")

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cols = ["method", "trial", "n_detected", "ppv", "tpr", "f1",
            "delay", "runtime_ms"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
