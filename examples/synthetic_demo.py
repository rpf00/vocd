"""
VOCD on the synthetic benchmark.

Reproduces the setup from the paper notebook: 500 samples, 6 true
change-points, 2% symmetric +/-10 sigma outlier contamination.

Run from the repository root:

    python examples/synthetic_demo.py
"""

import time

import numpy as np

import vocd
from vocd.datasets import synthetic_trials
from vocd.metrics import evaluate

TOL = 10
N_TRIALS = 10


def main() -> None:
    trials, true_cps = synthetic_trials(n_trials=N_TRIALS)
    print(f"synthetic benchmark: T={trials[0].size}, "
          f"{len(true_cps)} true CPs at {true_cps}, 2% outliers @ +/-10\n")

    # ── 1. score once ────────────────────────────────────────────────
    t0 = time.perf_counter()
    r = vocd.detect(trials[0])
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"detect() on trial 0: {elapsed:.0f} ms   {r!r}\n")

    print("candidate table (p_max is what the correction consumes):")
    print(f"  {'cp':>5}  {'p_mw':>10}  {'p_ks':>10}  {'p_max':>10}  tested")
    for row in r.table():
        print(f"  {row['cp']:>5}  {row['p_mw']:>10.2e}  {row['p_ks']:>10.2e}  "
              f"{row['p_max']:>10.2e}  {row['tested']}")

    # ── 2. re-threshold for free ─────────────────────────────────────
    print("\nsame scoring, different error budgets (no rescoring):")
    for label, kw in [
        ("fdr=0.05  (BH)", dict(fdr=0.05)),
        ("fdr=0.01  (BH)", dict(fdr=0.01)),
        ("fdr=0.05  (BY)", dict(fdr=0.05, method="by")),
        ("fwer=0.01 (Bonf)", dict(fwer=0.01)),
    ]:
        t0 = time.perf_counter()
        cps = r.changepoints(**kw)
        us = (time.perf_counter() - t0) * 1e6
        m = evaluate(cps, true_cps, tol=TOL)
        print(f"  {label:<18} {str(list(cps)):<34} "
              f"F1={m['f1']:.3f}  ({us:.0f} us)")

    # ── 3. aggregate over trials ─────────────────────────────────────
    print(f"\naggregate over {N_TRIALS} contaminated trials (fdr=0.05, tol={TOL}):")
    rows, times = [], []
    for y in trials:
        t0 = time.perf_counter()
        res = vocd.detect(y)
        times.append((time.perf_counter() - t0) * 1000)
        rows.append(evaluate(res.changepoints(fdr=0.05), true_cps, tol=TOL))

    def ms(k):
        v = np.array([r[k] for r in rows], dtype=float)
        return np.nanmean(v), np.nanstd(v)

    for k in ("n_detected", "ppv", "tpr", "f1", "delay"):
        mu, sd = ms(k)
        print(f"  {k:<11} {mu:6.3f} +/- {sd:.3f}")
    print(f"  {'runtime_ms':<11} {np.mean(times):6.1f} +/- {np.std(times):.1f}")

    # ── 4. online vs offline ─────────────────────────────────────────
    print("\nonline vs offline (one engine):")
    det = vocd.VOCD()
    live = []
    for x in trials[0]:
        live.extend(det.update(x))
    det.flush()

    same_scores = (
        np.array_equal(det.result.candidates, r.candidates)
        and np.allclose(det.result.p_max, r.p_max, equal_nan=True)
    )
    print(f"  scoring identical            : {same_scores}")
    print(f"  streaming update() emitted   : {live}")
    print(f"  detect() at same alpha       : {list(r.changepoints(alpha=det.online_alpha))}")
    print("  note: the scored candidates and p-values are identical; the emitted")
    print("  lists can differ only where near-duplicates are merged, because a")
    print("  live alarm cannot be retracted in favour of a later, stronger one.")


if __name__ == "__main__":
    main()
