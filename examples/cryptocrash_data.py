"""
Crypto-crash change-point detection — VOCD panel + author's faithful Dm-BOCD.

FTX + BTC daily 'Open' (2022-10-01 .. 2023-01-01), as in the DSM-bocd
Cryptocrash notebook.  BIVARIATE.

Per your instruction: our univariate panel (vocd, bocd, np-focus, r-focus, chad)
is run PER CHANNEL (FTX and BTC separately).  Dm-BOCD is the author's config —
a MULTIVARIATE model (DSMMultivariateGaussian, d=2) run jointly on both channels,
omega=0.1 fixed, his priors, b=40, hazard=50 — so it produces ONE joint
change-point set.  No ground truth.

    python examples/cryptocrash_data.py --data-dir data/cryptocrash --dsm-path ./DSM-bocd
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsm_problems import load_dsm, our_panel_1d, print_cps, m2, grad_m2, OUR_METHODS

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START, END = "2022-10-01", "2023-01-01"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data", "cryptocrash"))
    ap.add_argument("--ftx-file", default="FTT-USD.csv")
    ap.add_argument("--btc-file", default="BTC-USD.csv")
    ap.add_argument("--dsm-path", default=None)
    ap.add_argument("--methods", default=",".join(OUR_METHODS + ["dm-bocd"]))
    args = ap.parse_args()

    import pandas as pd
    from sklearn import preprocessing
    p1 = os.path.join(args.data_dir, args.ftx_file)
    p2 = os.path.join(args.data_dir, args.btc_file)
    for p in (p1, p2):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
    ftx = pd.read_csv(p1); ftx["Date"] = pd.to_datetime(ftx["Date"]); ftx.sort_values("Date", inplace=True)
    ftx = ftx[(ftx["Date"] > START) & (ftx["Date"] < END)]
    btc = pd.read_csv(p2); btc["Date"] = pd.to_datetime(btc["Date"]); btc.sort_values("Date", inplace=True)
    btc = btc[(btc["Date"] > START) & (btc["Date"] < END)]
    df = pd.merge(ftx[["Date", "Open"]], btc[["Date", "Open"]], on="Date")
    df.columns = ["date", "ftx", "btc"]
    data = df[["ftx", "btc"]].values
    T = len(data)
    data_scaled = preprocessing.StandardScaler().fit_transform(data)
    print(f"cryptocrash: {T} days ({df['date'].iloc[0].date()} .. {df['date'].iloc[-1].date()})")

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    our = [m for m in wanted if m in OUR_METHODS]

    # our univariate panel, per channel
    for ch, idx in (("FTX", 0), ("BTC", 1)):
        cps = our_panel_1d(data_scaled[:, idx], dsm_path=args.dsm_path, methods=our)
        print_cps(f"cryptocrash [{ch}] (our univariate panel)", cps)

    # author's Dm-BOCD: multivariate, joint
    if "dm-bocd" in wanted:
        d = load_dsm(args.dsm_path)
        mu0 = np.asarray([[0.0 / 1], [1.0 / 1], [0.0 / 1], [1.0 / 1]])
        Sigma0 = np.eye(4)
        Sigma0[0, 0] = 5.0 / 2; Sigma0[1, 1] = 1.0 / 1
        Sigma0[2, 2] = 5.0 / 2; Sigma0[3, 3] = 1.0 / 1
        model = d["DSMMultivariateGaussian"](data_scaled, m2, grad_m2, 0.1, mu0, Sigma0, d=2, b=40)
        R = d["bocpd"](data_scaled, d["ConstantHazard"](50), model, K=50, verbose=False)
        cps_dm = sorted(int(c) for c in d["find_cp_causal"](R) if 0 < c < T)
        print_cps("cryptocrash [FTX+BTC joint] (author Dm-BOCD, multivariate)",
                  {"dm-bocd": cps_dm})


if __name__ == "__main__":
    main()
