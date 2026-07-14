# VOCD — Viterbi Online Changepoint Detection

Robust, real-time changepoint detection for streaming sensor data, with a focus on automated drift correction.

---

## 1. What it is

VOCD (Viterbi Online Changepoint Detection) is a changepoint detection algorithm for streaming time-series data. It frames the problem as maximum a posteriori (MAP) decoding over a piecewise-stationary hidden Markov model and uses the Viterbi algorithm to recover the most probable sequence of signal regimes.

Unlike Bayesian online changepoint detection (BOCD) and its robust variants, VOCD needs no variational approximations and no parameter self-adaptation. The result is exact MAP inference via dynamic programming, giving inherent outlier resistance while remaining fast enough for real-time, causal (online) use on live data streams.

VOCD is fully software-based and platform-agnostic — it runs on the raw signal from an existing sensor and requires no hardware modification, extra reference sensors, or recalibration circuitry.

## 2. Why it matters

Sensor signals in real-world deployments drift. Gradual or abrupt baseline shifts — from membrane fouling, leaching, aging, or environmental change — corrupt measurement integrity across healthcare, environmental monitoring, and industrial automation. Correcting for drift usually means periodic manual recalibration or dedicated reference hardware, both of which are costly and don't scale to large sensor networks.

Existing changepoint detection methods force a tradeoff:

- **BOCD** (Adams & MacKay, 2007) is fast but fragile — a handful of outliers can fool it.
- **Dm-BOCD** (Altamirano et al., 2023) is robust but roughly 10× slower than BOCD and requires ω-calibration.

VOCD breaks this tradeoff. By using Viterbi decoding over a piecewise-stationary HMM, it delivers robustness *and* speed in a single method with no calibration step. That makes continuous, automated, software-only regime-change detection practical for the kind of long-term sensing deployments where manual inspection currently dominates.

## 3. Main results / highlights

| Method | Robustness | Speed | Notes |
|---|---|---|---|
| BOCD | Fragile (PPV ~0.6 on contaminated data) | Fast | Fooled by outliers |
| Dm-BOCD | Robust | 10× slower than BOCD | Requires ω-calibration |
| **VOCD** | **Robust** | **60× faster than Dm-BOCD** | No calibration; exact MAP via DP |

- **60× speed advantage** over the closest robust competitor (Dm-BOCD), at *superior* detection accuracy.
- **Outlier-resistant by construction** — robustness comes from the model formulation, not from added tuning.
- **No calibration required** — avoids the ω hyperparameter that Dm-BOCD depends on.
- **Beats Dm-BOCD across standard benchmarks** (including the well-log / NMR benchmarks from Altamirano et al., 2023).
- **Validated across multiple domains** — algorithmic benchmarks, financial time series (Twitter flash crash, FTX/crypto crash, UK bond yields), geophysical data, and — most importantly — real sensor platforms.
- **Drift correction on real sensors** — applied to ISFET water-quality sensors (90 days of real river water) and gas sensors (16 sensors, 6 gases, 36 months of drift), including a fully unsupervised, reference-free deployment mode.

## 4. Applications

- **Environmental monitoring** — automated drift correction and regime-change detection in long-term water-quality sensing (pH, dissolved oxygen, conductivity), scalable across large public sensor networks.
- **Predictive maintenance** — fault-onset detection in industrial and vibration sensors using the same changepoint mechanism.
- **Chemical and gas sensing** — membrane-based sensors that drift over months of continuous operation.
- **Environmental surveillance** — detecting genuine physical events (contamination events, remediation, seasonal shifts) rather than just correcting instrument artifacts.
- **General real-time signal processing** — any streaming setting where regime changes must be caught online, robustly, and cheaply.

---

## Status

Research code accompanying the VOCD manuscript (working title: *Robust Viterbi Online Changepoint Detection for Sensor Signal Drift Correction*), targeted at **Nature Sensors**.

## References

- Altamirano, Briol & Knoblauch (2023). *Robust and Scalable Bayesian Online Changepoint Detection.* ICML 2023.
- Adams & MacKay (2007). *Bayesian Online Changepoint Detection.*
- Margarit-Taulé et al. (2022). *Sensors & Actuators B: Chemical* 353, 131123 (ISFET drift dataset).
- Vergara et al. (2012). *Chemical gas sensor drift compensation using classifier ensembles.* Sensors & Actuators B.
