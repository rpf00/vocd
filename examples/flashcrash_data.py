"""
Flash-crash change-point detection — VOCD panel + author's faithful Dm-BOCD.

2010 Dow-Jones flash-crash minute 'open' prices (DJI.txt), as in the DSM-bocd
Flashcrash notebook.  Univariate.  No ground truth.

Dm-BOCD = author's config (DSMGaussian, StandardScaler'd, estimated omega via
OmegaEstimatorGaussian priors [1,1,100,100], b=20, hazard=300).  Rest = our panel.

    python examples/flashcrash_data.py --data-dir data/flashcrash --dsm-path ./DSM-bocd
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsm_problems import (load_dsm, load_estimators, our_panel_1d,
                           print_cps, m1, grad_m1, OUR_METHODS)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data", "flashcrash"))
    ap.add_argument("--file", default="DJI.txt")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--omega", type=float, default=0.1)
    ap.add_argument("--day", default="2013-04-23",
                    help="flash-crash day to subset (author uses 2013-04-23)")
    ap.add_argument("--methods", default=",".join(OUR_METHODS + ["dm-bocd"]))
    args = ap.parse_args()

    import pandas as pd
    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    DJI = pd.read_csv(path, header=None)
    DJI.columns = ["date", "open", "high", "low", "close"]
    DJI["date"] = pd.to_datetime(DJI["date"], errors="coerce")
    # author subsets to the single flash-crash day (intraday)
    flash = DJI[DJI["date"].dt.date.astype(str) == args.day]
    if len(flash) == 0:
        sys.exit(f"no rows for --day {args.day}; check the date in DJI.txt")
    data = flash["open"].values
    T = len(data)
    data = data.reshape([T, 1])
    from sklearn import preprocessing
    data_scaled = preprocessing.StandardScaler().fit_transform(data)
    print(f"flashcrash: {T} points")

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    our = [m for m in wanted if m in OUR_METHODS]
    cps = our_panel_1d(data_scaled[:, 0], dsm_path=args.dsm_path, methods=our)

    if "dm-bocd" in wanted:
        d = load_dsm(args.dsm_path)
        mu0 = np.array([[0.0 / 1.0], [1.0 / 1.0]])          # mean_mu0=0,var_mu0=1
        Sigma0 = np.eye(2); Sigma0[0, 0] = 10.0 / 1.0; Sigma0[1, 1] = 1.0 / 1.0
        if args.omega is None:
            print("estimating omega (OmegaEstimatorGaussian, niter=2000)…", flush=True)
            EstG, _ = load_estimators(args.dsm_path)
            est = EstG(data_scaled[100:200], m1, grad_m1, mu0, Sigma0)
            omegas, _ = est.omega(0.1, lr=1e-11, niter=2000, prior_parameters=[1, 1, 100, 100])
            omega = float(np.round(omegas[-1], 5))
        else:
            omega = args.omega
        print(f"  omega = {omega}")
        model = d["DSMGaussian"](data_scaled, m1, grad_m1, omega, mu0, Sigma0, b=20)
        R = d["bocpd"](data_scaled, d["ConstantHazard"](300), model, K=50, verbose=False)
        cps["dm-bocd"] = sorted(int(c) for c in d["find_cp_causal"](R) if 0 < c < T)

    print_cps("flashcrash", {m: cps[m] for m in wanted if m in cps})


if __name__ == "__main__":
    main()
