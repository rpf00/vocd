"""
Bond-yield change-point detection — VOCD panel + author's faithful Dm-BOCD.

UK 10-Year Bond Yield ('Price' column), as in the DSM-bocd Bond notebook.
Univariate.  No ground truth.

Dm-BOCD = author's config: a GAMMA model (DSMGamma) on the RAW (unscaled)
positive yields, omega estimated via OmegaEstimatorGamma, his priors, b=20,
hazard=100.  Rest = our panel (run on the same raw series; the rank-based gates
are scale-invariant).

    python examples/bond_data.py --data-dir data/bond --dsm-path ./DSM-bocd
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
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data", "bond"))
    ap.add_argument("--file",
                    default="United Kingdom 10-Year Bond Yield Historical Data.csv")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--omega", type=float, default=0.1)
    ap.add_argument("--methods", default=",".join(OUR_METHODS + ["dm-bocd"]))
    args = ap.parse_args()

    import pandas as pd
    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    df = pd.read_csv(path)
    data = df["Price"].values
    T = len(data)
    data = data.reshape((T, 1))
    print(f"bond: {T} points")

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    our = [m for m in wanted if m in OUR_METHODS]
    cps = our_panel_1d(data[:, 0], dsm_path=args.dsm_path, methods=our)

    if "dm-bocd" in wanted:
        d = load_dsm(args.dsm_path)
        mu0 = np.asarray([[0.0], [1.0]])                    # k_mu0=0, theta_mu0=1
        Sigma0 = np.eye(2); Sigma0[0, 0] = 50.0; Sigma0[1, 1] = 3.0  # k_Sigma0, theta_Sigma0
        if args.omega is None:
            print("estimating omega (OmegaEstimatorGamma, niter=1000)…", flush=True)
            _, EstGa = load_estimators(args.dsm_path)
            est = EstGa(data[:100], m1, grad_m1, mu0, Sigma0)
            omegas, _ = est.omega(1.0, lr=0.0001, niter=1000)
            omega = omegas[-1]
        else:
            omega = args.omega
        print(f"  omega = {omega}")
        model = d["DSMGamma"](data, m1, grad_m1, omega, mu0, Sigma0, b=20)
        R = d["bocpd"](data, d["ConstantHazard"](100), model, K=50, verbose=False)
        cps["dm-bocd"] = sorted(int(c) for c in d["find_cp_causal"](R) if 0 < c < T)

    print_cps("bond", {m: cps[m] for m in wanted if m in cps})


if __name__ == "__main__":
    main()
