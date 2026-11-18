# VOCD — Viterbi Online Changepoint Detection

Robust and fast changepoint detection for streaming sensor data, with a focus on reference-free drift compensation in field chemical sensing.

This repository contains the `vocd` package, the comparison detectors, and the scripts that reproduce every figure and table of the manuscript *"Robust and Low-latency Online Changepoint Detection for Sensor Drift Compensation"*.

- **Data:** https://doi.org/10.5281/zenodo.23388462 (see [`data/README.md`](data/README.md))
- **License:** MIT

---

## What it is

VOCD detects changepoints (CPs) in a univariate data stream in two stages:

1. **Proposal.** Two causal Viterbi passes decode the stream against a small bank of regimes (a robust mean and variance each, fitted on the first observations). A precision-tuned pass (PPV) proposes a CP only after a new regime has won several consecutive steps, and a recall-tuned pass (TPR) proposes one at the first step. Their union over-proposes by design.
2. **Verification.** Every proposal is checked by a distribution-free gate: a Mann–Whitney and a Kolmogorov–Smirnov test on the `verify_w` observations before and after it. Both tests are rank-based, so a single outlier, whatever its size, cannot make a proposal pass. The candidate is kept if `p_max = max(p_MW, p_KS)` survives the user's error budget (family-wise via Bonferroni, false-discovery via Benjamini–Hochberg/Yekutieli, or a raw per-test threshold).

VOCD needs no sampling, no learning rate and no random seed: its output is deterministic and bit-identical across runs and platforms. The proposals and the gate are causal, with a detection delay of at most the gate window (`verify_w`, 25 observations by default). The Python reference implementation and the compiled C++ engines return the same CPs and differ only in runtime.

## Install

```bash
git clone https://github.com/rpf00/vocd
cd vocd
pip install -e .            # NumPy + SciPy; builds the C++ Viterbi backend if a compiler and pybind11 are available
pip install -e ".[dev,paper]"
pytest                      # unit tests and Python/C++ equivalence tests
```

Requirements: Python ≥ 3.9, NumPy and SciPy. The paper's results were produced with Python 3.11, NumPy 2.4 and SciPy 1.17. The C++ parts are optional: without a compiler VOCD runs in pure Python with identical output, only slower. Use the editable install (`pip install -e .`), since the full C++ engine (`vocd/_fastdetect.py`) compiles its source next to the package on first use (any C++17 compiler; set `CXX` to override).

The experiment and figure scripts additionally need `matplotlib` and `openpyxl`.

## Quick start

```python
import numpy as np
import vocd

y = np.loadtxt("my_signal.txt")           # any 1-D array of observations

# Score every proposal once, then apply any error budget without rescoring
r = vocd.detect(y)
cps = r.changepoints(fwer=0.01)           # family-wise budget (Bonferroni), as in the paper
cps = r.changepoints(fdr=0.05)            # false-discovery budget (the default)
cps = r.changepoints(fdr=0.01, method="by")   # Benjamini–Yekutieli
r.table()                                 # every candidate with p_MW, p_KS and p_max

# Streaming: one observation at a time
det = vocd.VOCD()
for x in stream:
    for cp in det.update(x):              # CPs confirmed at this step
        ...
det.flush()                               # end of stream: score what is still pending
det.result.changepoints(fwer=0.01)        # budget applied to everything scored so far
```

Two things to know about the two modes:

- `VOCDResult.changepoints(fwer=...)` and `(fdr=...)` apply the budget to all `K` candidates of the signal, so the threshold is final only once the signal is complete. This is the mode used in the paper.
- `VOCD.update` emits immediately, using a fixed per-test threshold (`online_alpha = 1e-4`), because a growing candidate list cannot be corrected without retracting earlier alarms. Use it when you need an alarm the moment a CP is confirmed.

## Using VOCD on your own data

1. **Pass a 1-D signal.** Missing values should be removed or filled beforehand. Irregular sampling is fine; all windows are counted in observations, not in time.
2. **Choose the error budget.** It is the only setting most users need. `fwer=0.01` is a strict default; looser budgets (`0.05` to `0.3`) suit short or sparse signals.
3. **Adjust the gate window only if needed.** `verify_w` sets both the detection delay and the shortest segment VOCD can resolve; it is the one internal parameter that affects accuracy appreciably. `dedup_w` merges CPs that describe the same event.
4. **Short signals are supported.** If a signal ends before `init_window` (100) observations, the regime bank is fitted on all available observations. A candidate whose right window is incomplete at the end of the signal is tested on the truncated window (at least `min_side` = 5 observations).

```python
r = vocd.detect(y, verify_w=40, dedup_w=20)             # slower-changing signal
ppv = vocd.PassConfig(S=12, confirm_k=8, pen_base=8.0, min_dwell=8)
r = vocd.detect(y, ppv=ppv)                              # stricter precision pass
r = vocd.detect(y, backend="python")                    # force the reference implementation
```

For drift compensation, the CPs define the segments on which a drift model is fitted. `examples/isfet_data_v3.py` (piecewise-linear drift of a potentiometric sensor, re-run causally on each day's prefix) and `examples/gas_data_v3.py` (baseline re-anchoring of a cyclic gas-sensor measurement) are complete, reusable templates.

VOCD assumes piecewise-stationary segments. On a signal that trends instead of switching between regimes, it will segment the trend.

### Default parameters

`vocd.DEFAULT_PPV`, `vocd.DEFAULT_TPR`:

| Parameter | PPV pass | TPR pass |
|---|---|---|
| Regimes `S` | 9 | 5 |
| Confirmation steps `confirm_k` | 5 | 1 |
| Huber knee `huber_delta` | 2.0 | 2.0 |
| Switching penalty `pen_base` | 5.0 | 0.0 |
| Minimum dwell `min_dwell` | 5 | 0 |
| Merge window `merge_window` | 7 | 7 |
| Initialization window `init_window` | 100 | 100 |

Gate: `verify_w = 25`, `dedup_w = 15`, `min_side = 5`.

## Main results (manuscript)

- **Contaminated synthetic benchmark** (500 points, 6 mean shifts, 2% outliers): F1 = 1.000 on every trial, matched only by CHAMP and NUNC, at 20.8 ms per trial in Python and 0.37 ms in C++, against 15.2 s for Dm-BOCD. F1 stays above 0.96 up to 10% outliers.
- **ISFET drift compensation** (90-day pH probe at a river-water intake): 9.7 mV RMSE against a co-located reference, statistically matching every other detector and a weekly two-point calibration while using the reference only once, to set the initial offset. On the Na⁺ channel, it removes a curved drift that a single straight line leaves behind.
- **MOX gas sensor baseline re-anchoring** (year-long electronic-nose record, ethanol): within 1% of an oracle that reads the true stage labels, in 27 ms (C++) for the full record.
- **Environmental screening** (two years of water quality, 22 stations, 4 tropical storms): the only detector whose event recall exceeds its false-alarm fraction. It recovers one of the four storms and places a storm-adjacent CP on 2 of 12 negative-control constituents.
- **Additional benchmarks** (Twitter flash crash, UK 10-year bond yield, well-log): compared with every detector and across error budgets.
- **Scaling:** constant cost per observation, about 33 µs in Python and 0.3 µs in C++.

Among the eight online detectors analyzed, VOCD is the only one that statistically matches the best ones on the outlier benchmark and on both drift-compensation tasks while raising few false alarms in environmental screening, at millisecond runtime.

## Reproducing the paper

### 1. Install and get the data

```bash
pip install -e ".[dev]" matplotlib openpyxl
python data/fetch_data.py          # downloads the Zenodo archives and unpacks them into data/
```

### 2. Get the Bayesian baselines

BOCD and Dm-BOCD are run from the authors' own code, which is not redistributed here:

```bash
git clone https://github.com/maltamiranomontero/DSM-bocd.git ../DSM-bocd
export VOCD_DSM_PATH=../DSM-bocd   # or pass --dsm-path to each script
```

Without it, the two BOCD rows are skipped and everything else runs. DSM-bocd has its own requirements (see its repository); in particular, its Dm-BOCD estimators need `jax`.

### 3. Run the experiments, then the figures

Experiment scripts write CSVs to `results/`; figure scripts read them back without recomputing anything. Every script accepts `--help`.

```bash
# Fig. 3 — ISFET, H+ channel
python examples/isfet_data_v3.py --sweep --data-dir data/isfet --dsm-path ../DSM-bocd
python examples/vocd_fig3_isfet.py

# Fig. 4 — MOX gas sensor, ethanol
python examples/gas_data_v3.py --sweep --analyte Ethanol --dsm-path ../DSM-bocd
python examples/vocd_fig4_gas.py

# Fig. 5 — environmental screening (one pre-declared operating point; --sweep adds the robustness curve)
python examples/env_data_v3.py --data-dir data/env_monitoring --dsm-path ../DSM-bocd --sweep
python examples/vocd_fig5_env.py

# Supplementary Fig. 10 — runtime scaling
python examples/scaling_v3.py
python examples/vocd_figS_scaling.py
```

| Figure / table | Experiment script(s) | Figure / table script |
|---|---|---|
| Fig. 1 (overview) | — | `examples/vocd_fig1_natcomms.py` |
| Fig. 2 (synthetic) | `examples/synthetic_data_v3.py` | `examples/vocd_fig2.py`, `examples/vocd_robustness_fig2.py` |
| Fig. 3 (ISFET, H⁺) | `examples/isfet_data_v3.py` | `examples/vocd_fig3_isfet.py` |
| Fig. 4 (MOX gas, ethanol) | `examples/gas_data_v3.py` | `examples/vocd_fig4_gas.py` |
| Fig. 5 (environmental) | `examples/env_data_v3.py` | `examples/vocd_fig5_env.py` |
| Supp. Figs. 1–2 (ISFET Na⁺, K⁺) | `examples/si_isfet_na_k.py` | `examples/vocd_figS_isfet_na_k.py` |
| Supp. Figs. 3–4 (diacetyl) | `examples/gas_data_v3_si.py` | `examples/vocd_figS_gas_diacetyl.py` |
| Supp. Figs. 5–6, Supp. Tables 3–5 (environmental) | `examples/env_data_v3.py --sweep`, `examples/env_sensitivity_v3.py` | `examples/vocd_figS_env_sweep.py`, `examples/vocd_figS_env_null.py`, `examples/vocd_tableS_env.py` |
| Supp. Figs. 7–9 (additional benchmarks) | `examples/{flashcrash,bond,welllog}_data_v3.py` (shared: `examples/_dsm_v3.py`) | `examples/vocd_figS_dsm.py` |
| Supp. Fig. 10 (scaling) | `examples/scaling_v3.py` | `examples/vocd_figS_scaling.py` |

### Operating points used in the paper

| Experiment | VOCD budget | How it was chosen |
|---|---|---|
| Synthetic benchmark | `fwer = 0.01` | best mean F1 over the budget grid |
| ISFET | `fwer = 0.01` | minimum RMSE over the budget grid |
| MOX gas sensor | `fwer = 0.3` | minimum MAE over the budget grid (on ethanol; reused unchanged for diacetyl) |
| Environmental screening | `fwer = 0.05` | fixed before scoring, no selection |

Every comparison detector is treated the same way: its own budget (alarm level or hazard) is swept over a grid and the best point is reported, except in the environmental experiment, where all operating points are fixed in advance.

### Notes on runtime

- All runtimes in the paper are wall-clock, single-threaded, on an Apple M2. Absolute numbers will differ on other machines; ratios should not.
- Dm-BOCD is the slow row: about a minute per grid point on the ISFET signal and minutes per grid point on the gas-sensor signal.
- Each `(cpp)` row re-runs a detector at its Python row's operating point, and the scripts assert that the detections are identical.

## Repository layout

```
vocd/                 the package: core.py is the specification; _viterbi.cpp, _fastgate.py,
                      _fastdetect.py and _vocd_full.cpp are compiled accelerators that reproduce it exactly
alternative_models/   comparison detectors behind one REGISTRY
examples/             experiment scripts (*_data_v3.py) and figure/table scripts (vocd_fig*.py, vocd_tableS*.py);
                      the *_data.py files hold the data loaders and scoring that the v3 scripts import
benchmarks/           backend profiling, scaling and Python/C++ equivalence checks
tests/                unit and equivalence tests (pytest)
data/                 README and download script; the datasets live on Zenodo
results/, si_isfet/   CSV outputs of the experiment scripts and the figures built from them
```

## Comparison detectors

All detectors are called through one `REGISTRY` in `alternative_models/`, with identical inputs and causal extraction:

- **BOCD** and **Dm-BOCD**: the authors' DSM-bocd code (Altamirano, Briol & Knoblauch, 2023), called, not re-implemented.
- **FOCuS**, **NP-FOCuS**, **NUNC**, **CHAD** and **CHAMP**: our re-implementations following the original papers. FOCuS and NP-FOCuS also run as compiled C++ with functional pruning.

## Data

The datasets are archived on Zenodo: https://doi.org/10.5281/zenodo.23388462. See [`data/README.md`](data/README.md) for the contents, the expected folder layout and the original sources.

## Citation

If you use VOCD, please cite the manuscript:

> Pires Ferreira, R., Ding, R., Alvi, D., Shi, X., Pu, H., Chen, Y. & Chen, J. Robust and Low-latency Online Changepoint Detection for Sensor Drift Compensation. (2026).

and, if you use the data, the Zenodo record (https://doi.org/10.5281/zenodo.23388462) together with the original sources listed in `data/README.md`.

## License

MIT. See [`LICENSE`](LICENSE).
