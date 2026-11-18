"""
Twitter flash crash, v3 — the paper panel on the DSM-bocd benchmark
(Supplementary Information).

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)

Data: Dow Jones Industrial Average, minute 'open' prices on 23 April 2013, the
day the hacked Associated Press Twitter account caused a one-minute crash and
rebound (DJI.txt, subset to that day exactly as in the DSM-bocd Flashcrash
notebook).  Univariate, no ground truth: the output is the CPs each method
places.  The crash is a transient outlier, so a robust detector should place
no CP on it.

The panel runs on the standardized series at one declared operating point
(``_dsm_v3``).  Dm-BOCD is the authors' configuration from the notebook:
DSMGaussian on the standardized series, prior mu0 = (0, 1), Sigma0 =
diag(10, 1), b = 20, K = 50, hazard 300, np.random.seed(100), causal
extraction, and omega = 0.1 (the synthetic notebook's value, as in every other
experiment).  ``--estimate-omega`` restores the notebook's estimate instead:
OmegaEstimatorGaussian on observations 100-199, lr 1e-11, 2000 iterations,
priors [1, 1, 100, 100], rounded to 5 decimals.

    python examples/flashcrash_data_v3.py --data-dir data/flashcrash --dsm-path ../DSM-bocd
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _dsm_v3 as h                                            # noqa: E402
from _dsm_problems import load_dsm, load_estimators, m1, grad_m1  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, day):
    import pandas as pd
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    dji = pd.read_csv(path, header=None)
    dji.columns = ["date", "open", "high", "low", "close"]
    dji["date"] = pd.to_datetime(dji["date"], errors="coerce")
    flash = dji[dji["date"].dt.date.astype(str) == day]
    if len(flash) == 0:
        sys.exit(f"no rows for --day {day}; check the date in DJI.txt")
    labels = [d.strftime("%H:%M") for d in flash["date"]]
    return flash["open"].to_numpy(float), labels


def make_dm_bocd(data_scaled, args, dsm_path, hazard=300):
    def run():
        d = load_dsm(dsm_path)
        mu0 = np.array([[0.0 / 1.0], [1.0 / 1.0]])            # mean_mu0 = 0, var_mu0 = 1
        Sigma0 = np.eye(2); Sigma0[0, 0] = 10.0 / 1.0; Sigma0[1, 1] = 1.0 / 1.0
        omega_ms = None
        if args.estimate_omega:
            EstG, _ = load_estimators(dsm_path)
            t0 = time.perf_counter()
            est = EstG(data_scaled[100:200], m1, grad_m1, mu0, Sigma0)
            omegas, _ = est.omega(0.1, lr=1e-11, niter=2000, prior_parameters=[1, 1, 100, 100])
            omega_ms = (time.perf_counter() - t0) * 1000.0
            omega = float(np.round(float(omegas[-1]), 5))
        else:
            omega = float(args.omega)
        model = d["DSMGaussian"](data_scaled, m1, grad_m1, omega, mu0, Sigma0, b=20)
        np.random.seed(h.SEED)
        t0 = time.perf_counter()
        R = d["bocpd"](data_scaled, d["ConstantHazard"](hazard), model, K=50, verbose=False)
        cps = d["find_cp_causal"](R)
        ms = (time.perf_counter() - t0) * 1000.0
        return cps, ms, dict(hazard=hazard, omega=omega, omega_ms=omega_ms)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data", "flashcrash"))
    ap.add_argument("--file", default="DJI.txt")
    ap.add_argument("--day", default="2013-04-23", help="flash-crash day (AP Twitter hack)")
    h.add_common_args(ap)
    args = ap.parse_args()
    dsm_path, wanted = h.setup(args)

    raw, labels = load(os.path.join(args.data_dir, args.file), args.day)
    y = h.standardize(raw)
    h.run_benchmark("flashcrash", y, labels, raw, args, dsm_path, wanted,
                    dm_bocd_fn=make_dm_bocd(y.reshape(-1, 1), args, dsm_path))


if __name__ == "__main__":
    main()
