"""CHAMP — online MAP changepoint detection with per-segment ML fits.

Reference reimplementation following

    Niekum, Osentoski, Atkeson & Barto (2015), "Online Bayesian changepoint
    detection for articulated motion models", IEEE ICRA
    (CHAMP, CMU-RI-TR-14-10),

which builds on the online MAP (Viterbi) recursion of

    Fearnhead & Liu (2007), "On-line inference for multiple changepoint
    problems", JRSS-B 69(4), 589-605.

Not the authors' code (the released CHAMP is a ROS service that only runs on
batch data).

CHAMP is a genuine online MAP *segmentation* method — the closest published
relative of VOCD — not an alarm detector, so it does NOT go through
``_restart.run_restart``.  At each time t it maintains, for every candidate
start s of the current segment, the MAP probability of a changepoint at s,
and recovers the Viterbi path of changepoints.  Its two defining choices,
both kept here:

* the segment evidence is the MAXIMUM-LIKELIHOOD fit of a candidate model
  penalised by BIC, instead of a marginal likelihood (this is what lets
  CHAMP use any model fittable by ML);
* robustness is MODEL-BASED: the segment likelihood is an inlier/outlier
  mixture with a fixed outlier ratio ``gamma`` (the MLESAC model), fitted
  by EM from a robust (median/MAD) start.  ``gamma`` must be set a priori.

Candidate model set.  The benchmark is a change-in-mean problem, so the
model set is a single constant-mean Gaussian (unknown mean and variance,
k=2 free parameters per segment).

Length prior.  Default is a geometric prior with expected length
``hazard_timescale``, truncated below ``min_len`` — exactly BOCD's constant
hazard with a minimum run length — so CHAMP is swept over the SAME hazard
grid as BOCD and the two share one prior family.  CHAMP's own truncated-
Gaussian length prior is available via ``length_prior="gaussian"``.

Particles.  CHAMP caps the number of tracked candidate starts with a
particle filter.  With ``max_particles=None`` (default) every candidate is
kept and the recursion is EXACT: on T=500 this is cheap, and it means the
reported accuracy is CHAMP's ceiling, with no particle approximation to
blame.  A positive ``max_particles`` keeps the candidates with the largest
open-segment posterior weight from the previous step (the Fearnhead-Liu
particle weight), always retaining starts that became eligible this step;
this is deterministic pruning in place of the paper's stratified
resampling.

Causal read-off.  Changepoints are committed with the SAME spacing rule the
harness applies to BOCD (``_dsm.find_cp_causal``) plus a stabilisation
criterion: the start ``s_last`` of the current segment on the MAP path is
committed once it has been the MAP start for ``confirm`` consecutive steps
(default = ``min_len``, the analogue of VOCD's ``kc``) AND it lies more
than ``lag`` points past the last committed changepoint; it is then never
revised.  Without the confirmation step the minimum-length constraint makes
the method commit the newest *allowed* start a few points BEFORE the true
boundary becomes a candidate (an artefact, since a causal method cannot
anticipate a change), and lets a single spike commit a spurious segment
that later data would have revised away.
``extract="final"`` instead returns the MAP path at t=n (retrospective,
NOT causal) for reference only, analogous to ``bocd(extract="upstream")``.

The recursion (log domain, 0-indexed; segment [s, t) has length L = t - s)::

    logP[0] = 0
    closed(s) = logP[s] + log beta(L) + BIC(s, t)      # segment ends at t
    open(s)   = logP[s] + log P(len >= L) + BIC(s, t)  # segment still running
    logP[t]   = max_s closed(s);   back[t] = argmax_s closed(s)
    s_last    = argmax_s open(s)   # start of the current segment on the MAP path

Each step costs O(|S| * t) for the vectorised segment fits over all
candidate starts S, i.e. O(n^2) per observation and O(n^3) per series in
the exact mode.  At n=500 that is a few seconds per trial in NumPy.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.stats import norm

_LOG2PI = float(np.log(2.0 * np.pi))


def _log_norm_pdf(y: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    z = (y - mu) / sig
    return -0.5 * z * z - np.log(sig) - 0.5 * _LOG2PI


class CHAMP:
    def __init__(self, hazard_timescale: float = 300.0, min_len: int = 10,
                 gamma: float = 0.05, em_iters: int = 2,
                 max_particles: Optional[int] = None,
                 length_prior: str = "geometric",
                 prior_mu: Optional[float] = None,
                 prior_sigma: Optional[float] = None,
                 sigma_floor: float = 1e-3, k_params: int = 2):
        self.hazard = float(hazard_timescale)
        self.min_len = int(min_len)
        self.gamma = float(gamma)
        self.em_iters = int(em_iters)
        self.max_particles = int(max_particles) if max_particles else None
        self.length_prior = length_prior
        self.sigma_floor = float(sigma_floor)
        self.k = int(k_params)
        if length_prior == "geometric":
            p = 1.0 / self.hazard
            self._logp, self._log1mp = np.log(p), np.log1p(-p)
        elif length_prior == "gaussian":
            # CHAMP's truncated-Gaussian length prior; defaults centre it on
            # the hazard timescale so the two priors are comparable.
            self.pmu = float(prior_mu) if prior_mu is not None else self.hazard
            self.psig = float(prior_sigma) if prior_sigma is not None \
                else 0.5 * self.hazard
            # normaliser over L >= min_len (continuity-corrected)
            self._logZ = float(norm.logsf(self.min_len - 0.5, self.pmu, self.psig))
        else:
            raise ValueError("length_prior must be 'geometric' or 'gaussian'")

    # ── length prior ───────────────────────────────────────────────────

    def _log_beta(self, L: np.ndarray) -> np.ndarray:
        """log P(length = L) for L >= min_len."""
        if self.length_prior == "geometric":
            return (L - self.min_len) * self._log1mp + self._logp
        return norm.logpdf(L, self.pmu, self.psig) - self._logZ

    def _log_surv(self, L: np.ndarray) -> np.ndarray:
        """log P(length >= L) for L >= min_len (the open-segment term)."""
        if self.length_prior == "geometric":
            return (L - self.min_len) * self._log1mp
        return norm.logsf(L - 0.5, self.pmu, self.psig) - self._logZ

    # ── segment evidence: BIC-penalised ML fit of the outlier mixture ──

    def _segment_bic(self, Y: np.ndarray, S: np.ndarray, rng: float) -> np.ndarray:
        """BIC-penalised mixture log-likelihood of segment Y[s:t] for every s in S.

        Vectorised over candidate starts: row i of the (|S| x t) workspace is
        Y masked to [S[i], t).  Inlier N(mu, sig^2) with weight (1 - gamma),
        outlier uniform over the causal data range with weight gamma; (mu, sig)
        are the EM maximisers of that mixture from a median/MAD start.
        """
        t = Y.size
        L = (t - S).astype(float)
        mask = np.arange(t)[None, :] >= S[:, None]            # (|S|, t)
        Yb = np.broadcast_to(Y[None, :], mask.shape)
        Ynan = np.where(mask, Yb, np.nan)

        mu = np.nanmedian(Ynan, axis=1)
        mad = np.nanmedian(np.abs(Ynan - mu[:, None]), axis=1)
        sig = np.maximum(1.4826 * mad, self.sigma_floor)

        g = self.gamma
        u = 1.0 / max(rng, 1e-6)                                # outlier density
        for _ in range(self.em_iters):
            N = np.exp(_log_norm_pdf(Yb, mu[:, None], sig[:, None])) * mask
            r = (1.0 - g) * N / ((1.0 - g) * N + g * u) * mask   # inlier resp.
            wsum = r.sum(axis=1) + 1e-12
            mu = (r * Yb).sum(axis=1) / wsum
            var = (r * (Yb - mu[:, None]) ** 2).sum(axis=1) / wsum
            sig = np.maximum(np.sqrt(var), self.sigma_floor)

        N = np.exp(_log_norm_pdf(Yb, mu[:, None], sig[:, None]))
        ll = np.where(mask, np.log((1.0 - g) * N + g * u), 0.0).sum(axis=1)
        return ll - 0.5 * self.k * np.log(L)

    # ── online MAP recursion ───────────────────────────────────────────

    def run(self, x: np.ndarray, extract: str = "causal", lag: int = 20,
            confirm: Optional[int] = None) -> List[int]:
        confirm = int(confirm) if confirm else self.min_len
        y = np.asarray(x, dtype=float).ravel()
        n = y.size
        logP = np.full(n + 1, -np.inf)
        logP[0] = 0.0
        back = np.full(n + 1, -1, dtype=int)
        open_w = np.full(n + 1, np.nan)             # last open-segment weight per start
        committed: List[int] = []
        last = 0
        s_last = 0
        stable = 0                                  # consecutive steps s_last unchanged
        ymin = ymax = y[0]

        for t in range(1, n + 1):
            ymin, ymax = min(ymin, y[t - 1]), max(ymax, y[t - 1])
            if t < self.min_len:
                continue
            S = np.arange(0, t - self.min_len + 1)
            S = S[np.isfinite(logP[S])]            # reachable starts only
            if S.size == 0:
                continue
            if self.max_particles and S.size > self.max_particles:
                # Fearnhead-Liu particle weight = posterior of "current segment
                # started at s" = last step's open-segment score.  Starts that
                # became eligible this step have no weight yet and are kept.
                w = open_w[S]
                w = np.where(np.isnan(w), np.inf, w)
                keep = np.argsort(w)[-self.max_particles:]
                S = np.sort(S[keep])
            L = (t - S).astype(float)
            base = logP[S] + self._segment_bic(y[:t], S, ymax - ymin)
            closed = base + self._log_beta(L)
            opened = base + self._log_surv(L)
            open_w[S] = opened
            i = int(np.argmax(closed))
            logP[t], back[t] = closed[i], int(S[i])
            s_new = int(S[int(np.argmax(opened))])
            stable = stable + 1 if s_new == s_last else 1
            s_last = s_new

            if extract == "causal":
                # spacing rule of _dsm.find_cp_causal + stabilisation (confirm)
                if (stable >= confirm and s_last > last + lag
                        and s_last not in committed):
                    committed.append(s_last)
                    last = s_last

        if extract == "final":                     # retrospective MAP path
            path, s = [], s_last
            while s > 0:
                path.append(s)
                s = int(back[s])
            committed = path

        return sorted(set(int(c) for c in committed if c > 0))


def detect(x: np.ndarray, hazard_timescale: float = 300.0, min_seg: int = 10,
           gamma: float = 0.05, em_iters: int = 2,
           max_particles: Optional[int] = None, extract: str = "causal",
           lag: int = 20, confirm: Optional[int] = None,
           length_prior: str = "geometric",
           prior_mu: Optional[float] = None,
           prior_sigma: Optional[float] = None) -> list:
    det = CHAMP(hazard_timescale=hazard_timescale, min_len=min_seg,
                gamma=gamma, em_iters=em_iters, max_particles=max_particles,
                length_prior=length_prior, prior_mu=prior_mu,
                prior_sigma=prior_sigma)
    return det.run(np.asarray(x, float).ravel(), extract=extract, lag=lag,
                   confirm=confirm)
