# `alternative_models/`

Benchmark detectors for the VOCD synthetic comparison — everything in the panel
**except VOCD itself** (which is the installed `vocd` package and is called
directly by the harness).

## Design bar: fair, online, no lookahead

Every detector here obeys one rule:

> each decision at time `t` uses only `x[:t+1]` — no method sees the whole
> series before committing a change-point.

That rule drives every choice below. It also means the numbers are produced
under **one shared, a-priori error budget** (`alpha`), never per-method tuning
on the test series.

## Common contract

Each module exposes:

```python
detect(x: np.ndarray, **kwargs) -> list[int]
```

returning sorted, unique change-point indices **without** the leading-`0`
sentinel. The harness (`examples/synthetic_data.py`) prepends `0` so all methods
are scored alike by the reference `accuracy_delay` scorer.

The registry the harness reads:

```python
from alternative_models import REGISTRY
# name -> (callable, needs_dsm)
```

## Methods

| Module        | Method    | Status | Follows |
|---------------|-----------|--------|---------|
| `bocd.py`     | BOCD      | **wrapper** — calls DSM-bocd upstream, not reimplemented | Adams & MacKay (2007) |
| `dm_bocd.py`  | Dm-BOCD   | **wrapper** — calls DSM-bocd upstream, not reimplemented | Altamirano, Briol & Knoblauch (2023) |
| `focus.py`    | FOCuS     | reference reimplementation (not authors' code) | Romano, Eckley, Fearnhead & Rigaill (2023), *JMLR* 24(81) |
| `rfocus.py`   | R-FOCuS   | reference reimplementation (not authors' code) | Fearnhead & Rigaill (2019), *JASA* 114(525) + FOCuS recursion |
| `npfocus.py`  | NP-FOCuS  | reference reimplementation (not authors' code) | Romano, Eckley & Fearnhead (2024), *IEEE TSP* |
| `chad.py`     | CHAD      | reference reimplementation, in spirit (not authors' code) | Moen (2026) — *confirm citation against the version you cite* |

`focus.py` and `_base.py` are shared plumbing (FOCuS is R-FOCuS's parent class;
the shared statistics live in `_base`). Vanilla FOCuS is **not** in the final
panel — just omit it from `--methods`.

### BOCD / Dm-BOCD are called, not touched

Both wrappers import the real `bocpd`, `Gaussian`, `DSMGaussian`, `find_cp` from
a clone of [`maltamiranomontero/DSM-bocd`](https://github.com/maltamiranomontero/DSM-bocd)
so the Bayesian baselines run on their own terms:

```bash
git clone https://github.com/maltamiranomontero/DSM-bocd.git
export VOCD_DSM_PATH=./DSM-bocd      # or pass --dsm-path
```

The **one** thing the wrapper changes is how change-points are read off the
run-length matrix. Upstream `find_cp` scans the full matrix and can *delete* an
earlier change-point using later rows — i.e. it revises the past with the
future, which is not online. The wrappers therefore default to
`extract="causal"` (commit-on-arrival, past-only) to meet the no-lookahead bar,
and offer `extract="upstream"` to reproduce the published operating point.
Nothing else in BOCD/Dm-BOCD is modified.

## Alarm-style methods and `run_restart`

FOCuS, R-FOCuS, NP-FOCuS and CHAD are natively **alarm-style**: they emit a
stopping time, not a segmentation. `_restart.run_restart` drives them online:
feed one sample at a time, commit the estimated change location the moment the
statistic crosses the a-priori threshold, then reset and re-feed **only the
already-seen** post-change samples (no future data). A shared `min_seg` sets
both the minimum gap between change-points and the per-segment burn-in used to
estimate scale. All four share this driver and the same `alpha`, so their
outputs are commensurable with each other and with VOCD's native segmentation.

## Shared error budget

`_base.alpha_to_chi2_threshold` maps the family-wise `alpha` to a statistic
threshold, splitting the budget across the horizon (series length — a scalar,
not a peek at values) and, where relevant, the number of parallel tests
(quantile grid for NP-FOCuS, window grid for CHAD, via Bonferroni). This is the
alarm-method analogue of VOCD's FWER budget.

## Caveats to confirm before publication

- **Operating points are a shared a-priori rule, not calibrated results.** At a
  given `alpha` an alarm method may over- or under-detect (e.g. R-FOCuS can
  under-detect). That is the fair-by-construction setup working as intended. If
  you later want each method at its own best-honest operating point, use a
  **single symmetric calibration protocol** for all of them — not per-method
  fiddling.
- **Threshold map** (`alpha_to_chi2_threshold`) is a Gaussian-GLR approximation,
  not each paper's exact average-run-length calibration.
- **Grid defaults** — CHAD's dyadic window schedule and NP-FOCuS's quantile grid
  are reasonable defaults chosen here, flagged in-code; confirm against the
  papers.
- **CHAD's wrapped test is an explicit choice** — report it as `CHAD(MW)` /
  `CHAD(KS)`; CHAD's behaviour is a property of the test it schedules, not the
  schedule alone.
- None of the above touches BOCD, Dm-BOCD, or VOCD.

## Run

```bash
python examples/synthetic_data.py --dsm-path ./DSM-bocd --alpha 0.01
python examples/synthetic_data.py --methods vocd,np-focus,r-focus,chad   # no DSM needed
```
