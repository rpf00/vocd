# `alternative_models/`

The comparison detectors of the VOCD manuscript: everything in the panel except VOCD itself, which is the installed `vocd` package and is called directly by the experiment scripts.

## Design rule: online, no lookahead

Every detector here obeys one rule:

> each decision at time `t` uses only `x[:t+1]`. No method sees the whole signal before committing a changepoint.

## Common contract

Each module exposes

```python
detect(x: np.ndarray, **kwargs) -> list[int]
```

returning sorted, unique changepoint indices. The experiment scripts read the detectors from one registry:

```python
from alternative_models import REGISTRY     # name -> (callable, needs_dsm)
```

## Methods

| Registry name | Module | Method | Implementation | Reference |
|---|---|---|---|---|
| `bocd` | `bocd.py` | BOCD | wrapper around the DSM-bocd code, not re-implemented | Adams & MacKay, arXiv:0710.3742 (2007) |
| `dm-bocd` | `dm_bocd.py` | Dm-BOCD | wrapper around the DSM-bocd code, not re-implemented | Altamirano, Briol & Knoblauch, *ICML* (2023) |
| `focus` | `focus.py` | FOCuS | our re-implementation | Romano, Eckley, Fearnhead & Rigaill, *J. Mach. Learn. Res.* **24**, 1–36 (2023) |
| `np-focus` | `npfocus.py` | NP-FOCuS | our re-implementation | Romano, Eckley & Fearnhead, *IEEE Trans. Signal Process.* **72**, 594–606 (2024) |
| `nunc` | `nunc.py` | NUNC (Local) | our re-implementation | Austin, Romano, Eckley & Fearnhead, *Comput. Stat. Data Anal.* **177**, 107551 (2023) |
| `chad` | `chad.py` | CHAD | our re-implementation | Moen, arXiv:2504.09573 |
| `champ` | `champ.py` | CHAMP | our re-implementation | Niekum, Osentoski, Atkeson & Barto, *IEEE ICRA* (2015) |
| `focus-cpp`, `np-focus-cpp` | `_cppalarms.py`, `_cpp_alarms.cpp` | FOCuS, NP-FOCuS | compiled C++ with functional pruning | as above |

Shared plumbing: `_base.py` (alarm interface, threshold map, robust scale), `_restart.py` (restart driver), `_dsm.py` (causal read-off for the Bayesian methods).

None of the re-implementations is the authors' code.

### BOCD and Dm-BOCD are called, not modified

Both wrappers import `bocpd`, the models and `find_cp` from a clone of [`maltamiranomontero/DSM-bocd`](https://github.com/maltamiranomontero/DSM-bocd):

```bash
git clone https://github.com/maltamiranomontero/DSM-bocd.git
export VOCD_DSM_PATH=./DSM-bocd      # or pass --dsm-path to the experiment scripts
```

The one thing the wrappers change is how changepoints are read off the run-length matrix. The upstream `find_cp` scans the full matrix and can delete an earlier changepoint using later rows, which is not online. The wrappers therefore default to `extract="causal"` (commit on arrival, past data only) and offer `extract="upstream"` to reproduce the published read-off. Nothing else in BOCD or Dm-BOCD is modified.

### Alarm-style methods and the restart driver

FOCuS, NP-FOCuS, NUNC and CHAD are alarm-style: they emit a stopping time, not a segmentation. `_restart.run_restart` drives them online. It feeds one observation at a time, commits the estimated change location when the statistic crosses the threshold, then resets and re-feeds only the post-change observations already seen. A shared `min_seg` sets both the minimum gap between changepoints and the per-segment burn-in.

### CHAMP

CHAMP is an online MAP segmentation method and does not use the restart driver. Its changepoints are committed causally with the same spacing rule as the BOCD wrapper, plus a confirmation step. Its swept parameter is the length-prior timescale, on the same hazard grid as BOCD.

### Compiled FOCuS and NP-FOCuS

`focus-cpp` and `np-focus-cpp` implement functional pruning in C++ (`_cpp_alarms.cpp`, built next to the file on first use with any C++17 compiler). They return the same changepoints as the Python versions, which the experiment scripts assert on every run, and differ only in runtime.

## Budgets

- **Alarm methods** take a family-wise level `alpha`. `_base.alpha_to_chi2_threshold` maps it to a statistic threshold, splitting the budget across the signal length and, where relevant, the number of parallel tests (the quantile grid of NP-FOCuS, the window grid of CHAD).
- **BOCD, Dm-BOCD and CHAMP** take a hazard (length-prior) timescale.
- **VOCD** takes its family-wise budget `fwer`.

In the synthetic, ISFET and gas-sensor experiments each method's budget is swept over a grid and its best point is reported. In the environmental experiment every method runs at one operating point fixed in advance.

## Where the re-implementations make their own choices

These choices are ours and are not taken from the original papers:

- **Threshold map.** The alarm threshold comes from a Gaussian-GLR approximation of the family-wise level, not from each paper's average-run-length calibration.
- **NP-FOCuS quantile grid.** Three quantiles (0.25, 0.5, 0.75), estimated once per segment from the burn-in.
- **CHAD.** A dyadic window schedule wrapping a Mann–Whitney two-sample test (Kolmogorov–Smirnov optional).
- **NUNC.** The Local variant, with the threshold calibrated by Monte Carlo simulation under the null. The window is fixed in advance at twice VOCD's gate window, or twice `min_seg` if that is larger.
- **CHAMP.** A constant-mean Gaussian segment model with a fixed outlier ratio `gamma = 0.05`, a geometric length prior, and deterministic pruning in place of stratified resampling when a particle cap is set.

## Run

```bash
python examples/synthetic_data_v3.py --sweep --dsm-path ./DSM-bocd
python examples/synthetic_data_v3.py --methods vocd,focus,np-focus,nunc,chad,champ    # no DSM-bocd needed
```
