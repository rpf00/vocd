"""
UK 10-year bond yield, v3 — the paper panel on the DSM-bocd benchmark
(Supplementary Information).

    VOCD vs BOCD, Dm-BOCD, FOCuS, NP-FOCuS, NUNC, CHAD, CHAMP
    + the compiled timing twins vocd (cpp), focus (cpp), np-focus (cpp)

Data: daily UK 10-year government bond yield, 2018-2022 ('Price' column),
prepared EXACTLY as in the DSM-bocd Bond notebook: the CSV is ordered
newest-first, so it is sorted by date, then resampled to calendar days with
linear interpolation.  (The previous script skipped both steps and ran every
method on the series backwards.)  Univariate, no ground truth: the output is
the CPs each method places, which the DSM-bocd paper relates to UK political
and Covid-19 events (``EVENTS`` below).

The panel runs on the standardized series at one declared operating point
(``_dsm_v3``).  Dm-BOCD is the authors' configuration from the notebook: a
Gamma model (DSMGamma) on the RAW positive yields, prior mu0 = (0, 1),
Sigma0 = diag(50, 3), b = 20, K = 50, hazard 100, causal extraction, and
omega = 0.1 (the synthetic notebook's value, as in every other experiment);
np.random.seed(100) is set for reproducibility as in the other two notebooks.
``--estimate-omega`` restores the notebook's estimate instead:
OmegaEstimatorGamma on the first 100 observations, lr 1e-4, 1000 iterations.

    python examples/bond_data_v3.py --data-dir data/bond --dsm-path ../DSM-bocd
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

EVENTS = {   # the DSM-bocd notebook's reference events (for the SI figure)
    "2019-05-24": "May resignation", "2019-07-24": "Johnson sworn in",
    "2020-02-21": "first Covid case in EU", "2020-04-15": "first Covid wave in UK",
    "2021-01-08": "third Covid wave in UK", "2021-07-19": "restrictions end",
    "2021-12-06": "Plan B measures", "2022-09-06": "Truss sworn in",
}


def load(path):
    import pandas as pd
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    s = df["Price"].resample("D").mean().interpolate("linear")   # notebook: resample('D').interpolate
    labels = [d.strftime("%Y-%m-%d") for d in s.index]
    return s.to_numpy(float), labels


def make_dm_bocd(data_raw, args, dsm_path, hazard=100):
    def run():
        d = load_dsm(dsm_path)
        mu0 = np.asarray([[0.0], [1.0]])                       # k_mu0 = 0, theta_mu0 = 1
        Sigma0 = np.eye(2); Sigma0[0, 0] = 50.0; Sigma0[1, 1] = 3.0
        omega_ms = None
        if args.estimate_omega:
            _, EstGa = load_estimators(dsm_path)
            t0 = time.perf_counter()
            est = EstGa(data_raw[:100], m1, grad_m1, mu0, Sigma0)
            omegas, _ = est.omega(1.0, lr=0.0001, niter=1000)
            omega_ms = (time.perf_counter() - t0) * 1000.0
            omega = float(omegas[-1])
        else:
            omega = float(args.omega)
        model = d["DSMGamma"](data_raw, m1, grad_m1, omega, mu0, Sigma0, b=20)
        np.random.seed(h.SEED)
        t0 = time.perf_counter()
        R = d["bocpd"](data_raw, d["ConstantHazard"](hazard), model, K=50, verbose=False)
        cps = d["find_cp_causal"](R)
        ms = (time.perf_counter() - t0) * 1000.0
        return cps, ms, dict(hazard=hazard, omega=omega, omega_ms=omega_ms)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data", "bond"))
    ap.add_argument("--file", default="United Kingdom 10-Year Bond Yield Historical Data.csv")
    h.add_common_args(ap)
    args = ap.parse_args()
    dsm_path, wanted = h.setup(args)

    raw, labels = load(os.path.join(args.data_dir, args.file))
    y = h.standardize(raw)
    h.run_benchmark("bond", y, labels, raw, args, dsm_path, wanted,
                    dm_bocd_fn=make_dm_bocd(raw.reshape(-1, 1), args, dsm_path))


if __name__ == "__main__":
    main()
