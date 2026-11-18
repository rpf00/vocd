"""
Reproduce the negative-control correlation matrix + least-correlated triples.

Standalone check for the env experiment: computes pairwise Spearman correlations
among every negative-control constituent (pooled daily median, on common
sampling days), then ranks all 3-null triples by their lowest maximum pairwise
|rho| — i.e. finds the most mutually-orthogonal sets of nulls.

Run from vocd/examples/ .  Prints the same tables produced during analysis so
you can confirm the numbers locally.

Usage
-----
    python examples/env_null_corr.py --data-dir data/env_monitoring
    python examples/env_null_corr.py --min-common 30 --top 12 --method spearman
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    sys.exit("needs scipy:  pip install scipy")

from env_data import (load_pooled_daily, NEGATIVE,
                      _DEFAULT_ENV_DIR, _DEFAULT_FILE, _DEFAULT_SHEET)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=_DEFAULT_ENV_DIR)
    ap.add_argument("--file", default=_DEFAULT_FILE)
    ap.add_argument("--sheet", default=_DEFAULT_SHEET)
    ap.add_argument("--min-common", type=int, default=30,
                    help="min shared sampling days for a valid correlation")
    ap.add_argument("--top", type=int, default=12, help="how many triples to list")
    ap.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    args = ap.parse_args()

    path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    corr = spearmanr if args.method == "spearman" else pearsonr
    lab = "Spearman rho" if args.method == "spearman" else "Pearson r"

    # load every negative-control candidate as {date: value}
    ser = {}
    for c in NEGATIVE:
        d, y = load_pooled_daily(path, args.sheet, c)
        if d is not None:
            ser[c] = {dd: vv for dd, vv in zip(d, y)}
    cols = list(ser)
    print(f"candidates ({len(cols)}): {cols}\n")

    def pair(a, b):
        common = [d for d in ser[a] if d in ser[b]]
        if len(common) < args.min_common:
            return None, len(common)
        xa = np.array([ser[a][d] for d in common])
        xb = np.array([ser[b][d] for d in common])
        return float(corr(xa, xb)[0]), len(common)

    print(f"=== pairwise {lab} (n >= {args.min_common}) ===")
    rho = {}
    for a, b in itertools.combinations(cols, 2):
        r, n = pair(a, b)
        rho[(a, b)] = rho[(b, a)] = r
        if r is not None:
            print(f"  {a:7s} {b:7s}  = {r:+.2f}   n={n}")
    print()

    print(f"=== 3-null triples ranked by lowest max |{lab}| (most orthogonal) ===")
    triples = []
    for combo in itertools.combinations(cols, 3):
        rs = [rho.get((combo[i], combo[j])) for i, j in [(0, 1), (0, 2), (1, 2)]]
        if any(r is None for r in rs):
            continue
        triples.append((max(abs(r) for r in rs),
                        float(np.mean([abs(r) for r in rs])), combo, rs))
    triples.sort()
    for maxa, meana, combo, rs in triples[:args.top]:
        print(f"  {str(combo):40s} max|rho|={maxa:.2f}  mean|rho|={meana:.2f}  "
              f"pairs={[f'{r:+.2f}' for r in rs]}")
    if not triples:
        print("  (no triple had all pairs above the min-common threshold)")


if __name__ == "__main__":
    main()
