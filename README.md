# VOCD — Viterbi Online Changepoint Detection

Robust, low-latency changepoint detection for streaming sensor data, with a focus on reference-free drift compensation in field chemical sensing.

---

## What it is

VOCD detects changepoints (CPs) in a univariate data stream in two stages:

1. **Proposal.** Two causal Viterbi passes decode the stream against a small bank of regimes (a robust mean and variance each, fitted on the first observations). A precision-tuned pass (PPV, positive predictive value) proposes a CP only after a new regime has won several consecutive steps, and a recall-tuned pass (TPR, true positive rate) proposes one at the first step. Their union over-proposes by design.
2. **Verification.** Every proposal is checked by a distribution-free gate: a Mann-Whitney and a Kolmogorov-Smirnov test on the `verify_w` observations before and after it. Both tests are rank-based, so a single outlier, whatever its size, cannot make a proposal pass. The candidate is kept if `p_max = max(p_MW, p_KS)` survives the user's error budget (family-wise via Bonferroni, false-discovery via Benjamini-Hochberg/Yekutieli, or a raw per-test threshold).

VOCD needs no sampling, no learning rate and no random seed: its output is deterministic and bit-identical across runs and platforms. Detection is causal, with a fixed latency equal to the gate window. The Python reference implementation and the optional C++ backend (pybind11) return the same CPs and differ only in runtime.

## Install

```bash
git clone https://github.com/rpf00/vocd
cd vocd
pip install -e .            # NumPy + SciPy; builds the C++ backend if a compiler and pybind11 are available
pip install -e ".[dev]"     # + pytest
pytest                      # unit tests and Python/C++ equivalence tests
```

## Quick start

```python
import vocd

# offline: score every proposal once, then apply any error budget for free
r = vocd.detect(y)
r.changepoints(fwer=0.01)                 # family-wise budget (Bonferroni), as in the paper
r.changepoints(fdr=0.05)                  # false-discovery budget (default)
r.changepoints(fdr=0.01, method="by")     # Benjamini-Yekutieli, no rescoring needed

# online: one observation at a time
det = vocd.VOCD()
for x in stream:
    for cp in det.update(x):
        ...
```

Default pass parameters (`vocd.DEFAULT_PPV`, `vocd.DEFAULT_TPR`):

| Parameter | PPV pass | TPR pass |
|---|---|---|
| Regimes `S` | 9 | 5 |
| Confirmation steps `confirm_k` | 5 | 1 |
| Huber knee `huber_delta` | 2.0 | 2.0 |
| Switching penalty `pen_base` | 5.0 | 0.0 |
| Minimum dwell `min_dwell` | 5 | 0 |
| Merge window `merge_window` | 7 | 7 |
| Initialization window `init_window` | 100 | 100 |

Gate: `verify_w = 25` (also the detection latency), `dedup_w = 15`, `min_side = 5`.

## Main results (manuscript)

- **Contaminated synthetic benchmark** (DSM-BOCD protocol, 500 points, 6 mean shifts, 2% outliers): F1 = 1.000 on every trial, matched only by CHAMP and NUNC, at 20.8 ms per trial in Python and 0.37 ms in C++, against 15.2 s for Dm-BOCD. F1 stays above 0.96 up to 10% outliers.
- **ISFET drift compensation** (90-day pH probe, river-water intake): about 9.7 mV RMSE against a co-located reference that VOCD never sees, matching every detector and a weekly two-point calibration, two to nearly four orders of magnitude faster than Dm-BOCD. On the Na⁺ channel, it removes a curved drift that a single straight line leaves behind.
- **MOX gas sensor baseline re-anchoring** (year-long electronic-nose record, ethanol): within 0.6% of an oracle that reads the true stage labels, in 27 ms (C++) for the full record.
- **Environmental monitoring** (two years of water quality, 22 stations, 4 tropical storms): the only detector with positive consistency (storm recall above the false-alarm fraction on 12 negative-control constituents).
- **Classical CD benchmarks** (Twitter flash crash, UK 10-year bond yield, well-log): segmentation stable across a 500-fold range of error budgets, and the sparsest segmentation of the well-log strata.
- **Scaling:** constant cost per observation, about 33 µs in Python and 0.3 µs in C++.

## Reproducing the paper

Experiment scripts write CSVs to `results/`, and figure scripts read them back without recomputing anything.

| Figure / table | Experiment script(s) | Figure / table script |
|---|---|---|
| Fig. 2 (synthetic) | `examples/synthetic_data_v3.py` | `examples/vocd_fig2_*.py` |
| Fig. 3 (ISFET, H⁺) | `examples/isfet_data_v3.py` | `examples/vocd_fig3_isfet.py` |
| Fig. 4 (MOX gas, ethanol) | `examples/gas_data_v3.py` | `examples/vocd_fig4_gas.py` |
| Fig. 5 (environmental) | `examples/env_data_v3.py` | `examples/vocd_fig5_env.py` |
| Supp. Figs. 1–2 (ISFET Na⁺, K⁺) | `examples/si_isfet_na_k.py` | `examples/vocd_figS_isfet_na_k.py` |
| Supp. Figs. 3–4 (diacetyl) | `examples/gas_data_v3_si.py` | `examples/vocd_figS_gas_diacetyl.py` |
| Supp. Figs. 5–6, Supp. Tables (environmental) | `examples/env_data_v3.py --sweep`, `examples/env_sensitivity_v3.py` | `examples/vocd_figS_env_sweep.py`, `examples/vocd_figS_env_null.py`, `examples/vocd_tableS_env.py`, `examples/vocd_tableS_env_constituents.py` |
| Supp. Figs. 7–9 (classical benchmarks) | `examples/{flashcrash,bond,welllog}_data_v3.py` (shared: `examples/_dsm_v3.py`) | `examples/vocd_figS_dsm.py` |
| Supp. Fig. 10 (scaling) | `examples/scaling_v3.py` | `examples/vocd_figS_scaling.py` |
| Supp. Tables (methods survey) | — | `examples/vocd_tableS_methods.py` |

Comparison detectors live in `alternative_models/` and are called through one `REGISTRY` with identical inputs and causal extraction: BOCD and Dm-BOCD from the authors' DSM-bocd code (pass `--dsm-path` to a clone of https://github.com/maltamiranomontero/DSM-bocd), and our re-implementations of FOCuS, NP-FOCuS, NUNC, CHAD and CHAMP following their original papers (FOCuS and NP-FOCuS also as compiled C++).

## Data

All datasets used in the paper are public:

- ISFET record: Margarit-Taulé et al., *Sens. Actuators B Chem.* 353, 131123 (2022).
- MOX gas sensor record: Wörner, Eimler & Pein-Hackelbusch, *Sci. Data* 12, 1628 (2025).
- Environmental record: Occoquan Watershed Monitoring Laboratory (OWML, Virginia Tech), 2020–2021.
- Classical benchmarks: DSM-bocd repository (Altamirano, Briol & Knoblauch, ICML 2023).

Processed data and results are archived on Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX

## Citation

If you use VOCD, please cite the accompanying manuscript (citation to be added on publication).

## License

MIT
