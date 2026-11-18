"""
Well-log change-point detection — VOCD panel + author's faithful Dm-BOCD.

Nuclear-magnetic well-log signal (Fearnhead & Clifford data), as in the DSM-bocd
Well-log notebook.  Univariate.  No ground truth: output = change-points each
method predicts.

Dm-BOCD is the AUTHOR's config (DSMGaussian, StandardScaler'd data, estimated
omega via OmegaEstimatorGaussian, his priors, b=20, hazard=100).  The rest is
OUR panel (vocd, bocd, np-focus, r-focus, chad) at synthetic defaults.

    python examples/welllog_data.py --data-dir data/well-log --dsm-path ./DSM-bocd
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
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data", "well-log"))
    ap.add_argument("--file", default="well.txt")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--omega", type=float, default=0.1,
                    help="Dm-BOCD omega (default 0.1 = synthetic value; --omega None-> estimate)")
    ap.add_argument("--methods", default=",".join(OUR_METHODS + ["dm-bocd"]))
    args = ap.parse_args()

    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    data = np.loadtxt(path)
    T = len(data)
    data = data.reshape([T, 1])
    from sklearn import preprocessing
    data_scaled = preprocessing.StandardScaler().fit_transform(data)
    print(f"well-log: {T} points")

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    our = [m for m in wanted if m in OUR_METHODS]
    cps = our_panel_1d(data_scaled[:, 0], dsm_path=args.dsm_path, methods=our)

    if "dm-bocd" in wanted:
        d = load_dsm(args.dsm_path)
        mu0 = np.array([[0.0 / 0.1], [1.0 / 0.1]])          # mean_mu0=0,var_mu0=0.1
        Sigma0 = np.eye(2); Sigma0[0, 0] = 1.0 / 0.01; Sigma0[1, 1] = 1.0 / 0.01
        if args.omega is None:
            print("estimating omega (OmegaEstimatorGaussian, niter=15000)…", flush=True)
            EstG, _ = load_estimators(args.dsm_path)
            est = EstG(data_scaled[100:300], m1, grad_m1, mu0, Sigma0)
            omegas, _ = est.omega(0.01, lr=1e-11, niter=15000, prior_parameters=[1, 1, 15, 20])
            omega = omegas[-1]
        else:
            omega = args.omega
        print(f"  omega = {omega}")
        model = d["DSMGaussian"](data_scaled, m1, grad_m1, omega, mu0, Sigma0, b=20)
        R = d["bocpd"](data_scaled, d["ConstantHazard"](100), model, K=50, verbose=False)
        cps["dm-bocd"] = sorted(int(c) for c in d["find_cp_causal"](R) if 0 < c < T)

    print_cps("well-log", {m: cps[m] for m in wanted if m in cps})


if __name__ == "__main__":
    main()
