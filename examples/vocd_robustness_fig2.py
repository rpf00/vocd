"""
Fig 2 — VOCD robustness (VOCD only).

Panels, all on the contaminated synthetic benchmark (ground truth known):

  A  USER-KNOB heatmap: mean F1 over (error budget fwer x gate window verify_w),
     at 2% outliers.  Broad plateau = robust, not knife-edge.
  B  DATA-DIFFICULTY: mean F1 vs outlier fraction {0,2,5,10,20}%, at defaults.
     Series length, change-point locations and segment means are IDENTICAL to
     the original synthetic example (only the outlier fraction changes).
  C  INTERNAL-DEFAULTS sensitivity (one-at-a-time), shown at TWO contamination
     levels side by side (2% and a stress level, default 10%).  Every tunable is
     swept alone with all others at default; the bar spans the F1 range over its
     sweep.  A parameter flat at BOTH levels is genuinely inert (a simplification
     candidate); one flat at 2% but active under stress has a real job.  A
     pre-registered threshold (default 0.05 F1 swing) marks "sensitive" (red).

Every PassConfig field is swept for BOTH passes, plus the detect()-level knobs
verify_w / dedup_w / min_side.  CRUCIALLY, each pass is perturbed only within its
ROLE-APPROPRIATE range (PPV in precision-space, TPR in recall-space) so neither is
pushed into the other's regime.  init_window is a start-up constant, shown for
completeness.

Saved to results/heatmap_robust/ (figure + per-panel CSVs).

    python examples/vocd_robustness_fig2.py
    python examples/vocd_robustness_fig2.py --stress-frac 0.10 --sens-threshold 0.05
"""
from __future__ import annotations
import argparse, csv, dataclasses, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vocd
from vocd.core import PassConfig, DEFAULT_PPV, DEFAULT_TPR
from synthetic_data import (make_paper_data, contaminate, accuracy_delay,
                            SEED_CONTAM, CONTAM_SPIKE)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "results", "heatmap_robust")

VERIFY_W = [10, 15, 20, 25, 30, 40, 50]
FWER = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
OUTLIER_FRACS = [0.0, 0.02, 0.05, 0.10, 0.20]
DEFAULT_FWER = 0.01
DEFAULT_VERIFY_W = 25
DEFAULT_DEDUP_W = 15
DEFAULT_MIN_SIDE = 5

# ── ROLE-APPROPRIATE one-at-a-time sweep set ──────────────────────────
# Each pass is perturbed ONLY within its own regime, so neither is ever pushed
# into the other's territory (which would confound "is this pass redundant?"):
#   PPV = precision pass  -> strong penalty, slow confirmation, long dwell
#   TPR = recall pass     -> little/no penalty, fast confirmation, short dwell
# Directional params (pen_base, confirm_k, min_dwell) get regime-specific bands
# around each pass's default; non-directional ones (S, huber_delta, merge_window,
# init_window) vary around their own default.  Defaults:
#   PPV: S=9  confirm_k=5 huber=2 pen_base=5 min_dwell=5 merge=7 init=100
#   TPR: S=5  confirm_k=1 huber=2 pen_base=0 min_dwell=0 merge=7 init=100
PPV_GRIDS = {                       # precision regime (cautious)
    "S":            [6, 9, 12, 16],
    "confirm_k":    [3, 5, 8, 12],      # slow confirmation
    "huber_delta":  [1.5, 2.0, 3.0, 5.0],
    "pen_base":     [3.0, 5.0, 8.0, 12.0],  # strong penalty (never 0)
    "min_dwell":    [3, 5, 8, 12],      # long dwell
    "merge_window": [5, 7, 10, 15],
    "init_window":  [75, 100, 150, 200],
}
TPR_GRIDS = {                       # recall regime (eager)
    "S":            [3, 5, 8],
    "confirm_k":    [1, 2, 3],          # fast confirmation
    "huber_delta":  [1.5, 2.0, 3.0, 5.0],
    "pen_base":     [0.0, 0.5, 1.0, 2.0],   # little/no penalty (never large)
    "min_dwell":    [0, 1, 2, 3],       # short dwell
    "merge_window": [5, 7, 10, 15],
    "init_window":  [75, 100, 150, 200],
}
# gate / merge knobs are not pass-specific -> role-agnostic
TOP_FIELDS = [
    ("verify_w",  [15, 20, 25, 30, 40, 50]),
    ("dedup_w",   [5, 10, 15, 20, 30]),
    ("min_side",  [3, 5, 8, 12]),
]


def build_sweeps():
    """(label, where, field, values); PPV & TPR use role-appropriate ranges."""
    out = []
    for field, vals in PPV_GRIDS.items():
        out.append((f"ppv.{field}", "ppv", field, vals))
    for field, vals in TPR_GRIDS.items():
        out.append((f"tpr.{field}", "tpr", field, vals))
    for field, vals in TOP_FIELDS:
        out.append((field, "top", field, vals))
    return out


def trials_at(frac, n_trials):
    data, _ = make_paper_data()
    np.random.seed(SEED_CONTAM)
    return [contaminate(data, frac=frac, spike=CONTAM_SPIKE) for _ in range(n_trials)]


def mean_f1(trials, *, ppv=DEFAULT_PPV, tpr=DEFAULT_TPR, verify_w=DEFAULT_VERIFY_W,
            dedup_w=DEFAULT_DEDUP_W, min_side=DEFAULT_MIN_SIDE, fwer=DEFAULT_FWER):
    f1 = []
    for dc in trials:
        r = vocd.detect(dc.ravel(), ppv=ppv, tpr=tpr, verify_w=verify_w,
                        dedup_w=dedup_w, min_side=min_side)
        cps = [int(c) for c in r.changepoints(fwer=fwer)]
        f1.append(accuracy_delay([0] + sorted(cps))["f1"])
    return float(np.mean(f1))


def f1_with(where, field, val, trials):
    kw = dict(ppv=DEFAULT_PPV, tpr=DEFAULT_TPR, verify_w=DEFAULT_VERIFY_W,
              dedup_w=DEFAULT_DEDUP_W, min_side=DEFAULT_MIN_SIDE, fwer=DEFAULT_FWER)
    if where == "ppv":
        kw["ppv"] = dataclasses.replace(DEFAULT_PPV, **{field: val})
    elif where == "tpr":
        kw["tpr"] = dataclasses.replace(DEFAULT_TPR, **{field: val})
    else:
        kw[field] = val
    return mean_f1(trials, **kw)


def panel_C(trials, sweeps):
    rows = []
    for label, where, field, values in sweeps:
        vf = [(v, f1_with(where, field, v, trials)) for v in values]
        f1s = [f for _, f in vf]
        rows.append((label, min(f1s), max(f1s), max(f1s) - min(f1s), vf))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--stress-frac", type=float, default=0.10,
                    help="second contamination level for Panel C (default 0.10)")
    ap.add_argument("--sens-threshold", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "fig2_robustness.png"))
    args = ap.parse_args()
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("needs matplotlib")
    os.makedirs(OUTDIR, exist_ok=True)

    base = trials_at(0.02, args.trials)
    stress = trials_at(args.stress_frac, args.trials)
    sweeps = build_sweeps()

    # Panel A
    print("Panel A: user-knob heatmap (2% outliers) …")
    A = np.array([[mean_f1(base, verify_w=w, fwer=f) for f in FWER] for w in VERIFY_W])
    with open(os.path.join(OUTDIR, "panelA_userknobs.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(["verify_w", "fwer", "mean_f1"])
        for i, w in enumerate(VERIFY_W):
            for j, f in enumerate(FWER):
                wtr.writerow([w, f, A[i, j]])

    # Panel B
    print("Panel B: outlier-fraction robustness …")
    B = [mean_f1(trials_at(fr, args.trials)) for fr in OUTLIER_FRACS]
    with open(os.path.join(OUTDIR, "panelB_outliers.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(["outlier_frac", "mean_f1"])
        for fr, v in zip(OUTLIER_FRACS, B):
            wtr.writerow([fr, v])

    # Panel C at two levels
    print(f"Panel C: full internal sweep at 2% and {args.stress_frac:.0%} …")
    C2 = panel_C(base, sweeps)
    Cs = panel_C(stress, sweeps)
    f1_def2, f1_defS = mean_f1(base), mean_f1(stress)
    with open(os.path.join(OUTDIR, "panelC_internal.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(["level", "param", "value", "mean_f1"])
        for lvl, rows in (("2pct", C2), (f"{int(args.stress_frac*100)}pct", Cs)):
            for label, _, _, _, vf in rows:
                for v, f1 in vf:
                    wtr.writerow([lvl, label, v, f1])

    # ── figure: A, B on top row; C spans a tall bottom row ────────────
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.35, wspace=0.28)

    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(A, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axA.set_xticks(range(len(FWER))); axA.set_xticklabels([f"{f:g}" for f in FWER], rotation=45)
    axA.set_yticks(range(len(VERIFY_W))); axA.set_yticklabels(VERIFY_W)
    axA.set_xlabel("error budget (fwer)"); axA.set_ylabel("gate window (verify_w)")
    axA.set_title("A · user knobs — mean F1 (2% outliers)")
    for i in range(len(VERIFY_W)):
        for j in range(len(FWER)):
            axA.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                     color="white" if A[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)

    axB = fig.add_subplot(gs[0, 1])
    axB.plot([f * 100 for f in OUTLIER_FRACS], B, "o-", color="#2ca02c", lw=2)
    axB.set_xlabel("outlier fraction (%)"); axB.set_ylabel("mean F1")
    axB.set_ylim(0, 1.02); axB.grid(alpha=0.3); axB.set_title("B · data difficulty (defaults)")
    for x, y in zip([f * 100 for f in OUTLIER_FRACS], B):
        axB.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8)

    axC = fig.add_subplot(gs[1, :])
    labels = [r[0] for r in C2]
    y = np.arange(len(labels))
    off = 0.18
    for k in range(len(labels)):
        lo2, hi2, r2 = C2[k][1], C2[k][2], C2[k][3]
        loS, hiS, rS = Cs[k][1], Cs[k][2], Cs[k][3]
        c2 = "#d62728" if r2 > args.sens_threshold else "#9e9e9e"
        cS = "#1f77b4" if rS > args.sens_threshold else "#9e9e9e"
        axC.plot([lo2, hi2], [k + off, k + off], color=c2, lw=6, solid_capstyle="round")
        axC.plot([loS, hiS], [k - off, k - off], color=cS, lw=6, solid_capstyle="round")
    axC.axvline(f1_def2, color="#d62728", ls="--", lw=1, label=f"default F1 @2%={f1_def2:.2f}")
    axC.axvline(f1_defS, color="#1f77b4", ls="--", lw=1,
                label=f"default F1 @{int(args.stress_frac*100)}%={f1_defS:.2f}")
    axC.set_yticks(y); axC.set_yticklabels(labels, fontsize=8)
    lo_all = min(min(r[1] for r in C2), min(r[1] for r in Cs))
    axC.set_xlim(min(0.7, lo_all - 0.03), 1.01)
    axC.set_xlabel("mean F1 range over one-at-a-time sweep")
    axC.set_title(f"C · internal defaults, full set — upper bar = 2% (red if swing>"
                  f"{args.sens_threshold}), lower bar = {int(args.stress_frac*100)}% "
                  f"(blue if swing>{args.sens_threshold}); grey = stable")
    axC.legend(fontsize=8, loc="lower left")
    axC.grid(axis="x", alpha=0.3)

    fig.suptitle("VOCD robustness — parameters and data regime (synthetic, "
                 "ground truth)", y=0.995, fontsize=13)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    print(f"default F1: 2%={f1_def2:.3f}, {int(args.stress_frac*100)}%={f1_defS:.3f}\n")
    print(f"{'param':14s} {'swing@2%':>9} {'swing@'+str(int(args.stress_frac*100))+'%':>10}  note")
    for k in range(len(labels)):
        r2, rS = C2[k][3], Cs[k][3]
        note = ("inert at both — simplification candidate"
                if (r2 <= args.sens_threshold and rS <= args.sens_threshold)
                else ("dormant at 2%, ACTIVE under stress — has a job"
                      if r2 <= args.sens_threshold < rS else "sensitive"))
        print(f"{labels[k]:14s} {r2:>9.2f} {rS:>10.2f}  {note}")


if __name__ == "__main__":
    main()
