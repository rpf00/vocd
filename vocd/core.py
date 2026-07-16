"""
VOCD — Viterbi Online Change-point Detection.

Reference implementation.  Pure NumPy/SciPy, no compiled dependencies.
This module is the *specification* of the method: any accelerated backend
must reproduce its output exactly.

Architecture (propose-and-verify)
---------------------------------
1. PROPOSE.  Two causal Viterbi passes decode the signal against a bank of
   S regimes under a Huber emission loss.  One pass is tuned for precision,
   one for recall.  Their union over-proposes candidate change-points.
2. VERIFY.  Each proposal is tested with two-sample Mann-Whitney U and
   Kolmogorov-Smirnov tests on the windows either side of the candidate.
   Robustness comes from here: the Viterbi passes fire on outliers, but a
   lone spike shifts neither the location nor the shape of the local
   distribution, so both tests fail to reject and the candidate is dropped.
3. DECIDE.  The K p-values are thresholded under a multiple-testing
   correction chosen by the user's error budget (see ``_correction``).
4. MERGE.  Surviving detections within ``dedup_w`` bars describe the same
   event; the one with the strongest evidence is kept.  Merging happens
   *after* selection, so the location with the evidence is the one reported
   and FDR is still controlled over the K tested hypotheses.

Steps 1-2 are the expensive part and do not depend on the error budget, so
they run once; steps 3-4 are instantaneous and can be re-run at any level.

Online and offline
------------------
There is a single engine.  ``VOCD.update(x)`` feeds one observation and is
strictly causal.  ``detect(y)`` is a loop over ``update`` plus a flush, and
returns the same change-points -- the only difference is *when* you learn of
them.  A candidate at tau cannot be verified until ``tau + verify_w``
observations exist; this is detection latency, not lookahead.

Determinism
-----------
Fully deterministic.  There is no random seed: regime initialisation uses a
median/MAD k-medians whose empty-cluster rule picks the point farthest from
all current centres.  Output is bit-identical across runs and platforms.

The two Viterbi passes are hyperparameters of the *proposal* stage, not of
the answer.  Over-proposing is intended: the gate is what decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

from . import _backend
from ._correction import select
from ._faststats import ks2samp_p, mannwhitney_p

__all__ = ["PassConfig", "VOCDResult", "VOCD", "detect", "DEFAULT_PPV", "DEFAULT_TPR"]

_VAR_FLOOR = 1e-12
_CLUSTER_VAR_FLOOR = 1e-3


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PassConfig:
    """
    Configuration for one Viterbi proposal pass.

    S : number of regimes in the bank.  Granularity of the proposer, *not* a
        claim about how many regimes the signal has -- over-proposing is
        filtered downstream by the gate.
    confirm_k : consecutive steps a new state must win before a change-point
        is emitted.  Also the emission latency of this pass.
    huber_delta : Huber loss knee, in standardised residual units.  Residuals
        beyond this are penalised linearly rather than quadratically.
    pen_base : additive cost of switching regime.
    min_dwell : minimum steps in a regime before it may be left.
    merge_window : minimum gap between successive change-points from this pass.
    init_window : observations used to initialise the regime bank.
    """

    S: int = 9
    confirm_k: int = 5
    huber_delta: float = 2.0
    pen_base: float = 5.0
    min_dwell: int = 5
    merge_window: int = 7
    init_window: int = 100


#: Precision-tuned pass: strict penalty, slow confirmation.
DEFAULT_PPV = PassConfig(
    S=9, confirm_k=5, huber_delta=2.0, pen_base=5.0,
    min_dwell=5, merge_window=7, init_window=100,
)

#: Recall-tuned pass: no penalty, instant confirmation.
DEFAULT_TPR = PassConfig(
    S=5, confirm_k=1, huber_delta=2.0, pen_base=0.0,
    min_dwell=0, merge_window=7, init_window=100,
)


# ══════════════════════════════════════════════════════════════════════
# Primitives
# ══════════════════════════════════════════════════════════════════════


def _emission(y_t: float, mu: np.ndarray, var: np.ndarray, delta: float) -> np.ndarray:
    """
    Negative log-likelihood of ``y_t`` under each regime, with the quadratic
    term replaced by a Huber loss.  Vectorised over regimes.

    Returns an array of shape (S,).
    """
    var = np.maximum(var, _VAR_FLOOR)
    r = (y_t - mu) / np.sqrt(var)
    a = np.abs(r)
    huber = np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    return huber + 0.5 * np.log(2.0 * np.pi * var)


def _init_regimes(init_data: np.ndarray, S: int, iters: int = 20):
    """
    Initialise the regime bank by median/MAD k-medians.

    Deterministic by construction: when a cluster empties, its centre is
    reseeded to the observation farthest from every current centre (ties
    broken by lowest index) rather than to a random draw.

    Returns (mu, var), each of shape (S,).
    """
    d = np.asarray(init_data, dtype=float).ravel()
    n = d.size

    if n == 0:
        return np.zeros(S), np.ones(S)

    if n < S:
        mu = np.linspace(d.min(), d.max(), S)
        var = np.full(S, float(d.var()) + 1e-6)
        return mu, var

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad < 1e-6:
        mad = float(d.std()) + 1e-6

    centers = med + np.linspace(-2.0 * mad, 2.0 * mad, S)

    for _ in range(iters):
        z = np.argmin((d[:, None] - centers[None, :]) ** 2, axis=1)
        for s in range(S):
            m = z == s
            if m.any():
                centers[s] = np.median(d[m])
            else:
                # Deterministic reseed: farthest point from its nearest centre.
                nearest = np.min((d[:, None] - centers[None, :]) ** 2, axis=1)
                centers[s] = d[int(np.argmax(nearest))]

    z = np.argmin((d[:, None] - centers[None, :]) ** 2, axis=1)
    mu = centers.copy()
    var = np.empty(S, dtype=float)
    global_var = float(d.var()) + 1e-6

    for s in range(S):
        m = z == s
        if int(m.sum()) > 1:
            cluster_mad = float(np.median(np.abs(d[m] - mu[s])))
            var[s] = max((1.4826 * cluster_mad) ** 2, _CLUSTER_VAR_FLOOR)
        else:
            var[s] = global_var

    return mu, np.maximum(var, _CLUSTER_VAR_FLOOR)


# ══════════════════════════════════════════════════════════════════════
# Stage 1 — causal Viterbi proposal pass
# ══════════════════════════════════════════════════════════════════════


class _ViterbiPass:
    """
    One causal Viterbi pass over a regime bank.

    Feed observations with ``update(x)``; it returns any change-point
    locations confirmed at this step.  Carries ``dp`` and ``dwell`` forward,
    so per-step cost is O(S^2) with no re-decoding of history.

    The first ``init_window`` observations are buffered to fit the regime
    bank, then replayed through the recurrence.  Change-points inside the
    burn-in are therefore all emitted at once, at ``t == init_window``.
    """

    def __init__(self, cfg: PassConfig):
        self.cfg = cfg
        self.S = cfg.S
        self._buf: List[float] = []
        self._ready = False

        self.mu: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.dp: Optional[np.ndarray] = None
        self.dwell: Optional[np.ndarray] = None

        self.t = 0
        self.confirmed_state = 0
        self.last_cp = 0
        self._pending_state: Optional[int] = None
        self._pending_count = 0
        self._pending_cp: Optional[int] = None

    # -- lifecycle ---------------------------------------------------

    def _start(self) -> List[int]:
        """Fit the regime bank on the buffer and replay it."""
        self.mu, self.var = _init_regimes(np.asarray(self._buf, dtype=float), self.S)
        self.dp = np.zeros(self.S, dtype=float)
        self.dwell = np.ones(self.S, dtype=int)
        self._ready = True

        out: List[int] = []
        for v in self._buf:
            cp = self._step(v)
            if cp is not None:
                out.append(cp)
        self._buf = []
        return out

    def update(self, x: float) -> List[int]:
        """Feed one observation.  Returns change-points confirmed at this step."""
        if not self._ready:
            self._buf.append(float(x))
            if len(self._buf) < self.cfg.init_window:
                return []
            return self._start()

        cp = self._step(float(x))
        return [] if cp is None else [cp]

    def flush(self) -> List[int]:
        """Force initialisation on a short series (T < init_window)."""
        if not self._ready and self._buf:
            return self._start()
        return []

    # -- recurrence --------------------------------------------------

    def _step(self, x: float) -> Optional[int]:
        cfg = self.cfg
        S = self.S
        dp, dwell = self.dp, self.dwell

        emis = _emission(x, self.mu, self.var, cfg.huber_delta)
        idx = np.arange(S)

        # Vectorised over the full S x S transition matrix: cost[k, s] is the
        # cost of arriving in s from k.  Identical arithmetic and identical
        # tie-breaking (argmin takes the lowest index) to the per-state loop
        # this replaces; it just avoids S NumPy round-trips per observation.
        eye = np.eye(S, dtype=bool)

        # k -> s allowed when k has dwelled long enough and s explains this
        # observation strictly better than k does.  Staying is always allowed,
        # so every column has a finite entry.
        allowed = ((dwell >= cfg.min_dwell)[:, None]
                   & ((emis[:, None] - emis[None, :]) > 0.0))
        allowed |= eye

        cost = dp[:, None] + np.where(eye, 0.0, cfg.pen_base)
        cost = np.where(allowed, cost, np.inf)

        best_k = np.argmin(cost, axis=0)          # ties -> lowest index
        new_dp = cost[best_k, idx] + emis
        new_dwell = np.where(best_k == idx, dwell + 1, 1)

        self.dp, self.dwell = new_dp, new_dwell

        best_state = int(np.argmin(new_dp))
        t = self.t
        self.t += 1

        cp_out: Optional[int] = None
        if best_state != self.confirmed_state:
            if self._pending_state == best_state:
                self._pending_count += 1
            else:
                self._pending_state = best_state
                self._pending_count = 1
                self._pending_cp = t

            if self._pending_count >= cfg.confirm_k:
                if (self._pending_cp - self.last_cp) >= cfg.merge_window:
                    cp_out = self._pending_cp
                    self.last_cp = self._pending_cp
                self.confirmed_state = self._pending_state
                self._pending_state = None
                self._pending_count = 0
                self._pending_cp = None
        else:
            self._pending_state = None
            self._pending_count = 0
            self._pending_cp = None

        return cp_out


# ══════════════════════════════════════════════════════════════════════
# Stage 3 — verification gate
# ══════════════════════════════════════════════════════════════════════


def _verify(y: np.ndarray, cp: int, w: int, min_side: int):
    """
    Two-sample MW + KS test across ``cp``.

    Returns (p_mw, p_ks, p_max, tested).

    ``p_max = max(p_mw, p_ks)`` is the p-value of the intersection ("both
    tests reject") rule.  It is a valid, mildly conservative p-value: the
    event {p_max <= t} is contained in {p_mw <= t}, so under H0 its
    probability is at most t.  This is the single p-value per hypothesis
    that the multiple-testing correction consumes.

    A candidate with fewer than ``min_side`` observations on either side is
    untestable and is reported with ``tested=False`` and ``p_max=1.0``; it is
    never selected.  (The legacy pipeline accepted such candidates unchecked.)
    """
    T = y.size
    A = y[max(0, cp - w): cp]
    B = y[cp: min(T, cp + w)]

    if A.size < min_side or B.size < min_side:
        return np.nan, np.nan, 1.0, False

    p_mw = mannwhitney_p(A, B)
    p_ks = ks2samp_p(A, B)
    return p_mw, p_ks, max(p_mw, p_ks), True


def _verify_reference(y: np.ndarray, cp: int, w: int, min_side: int):
    """
    The same gate, computed straight through SciPy.

    Kept as a permanent oracle: ``_verify`` must agree with this to numerical
    precision (see tests/test_equivalence.py).  SciPy defines correctness here;
    the fast path only removes its dispatch overhead.
    """
    T = y.size
    A = y[max(0, cp - w): cp]
    B = y[cp: min(T, cp + w)]
    if A.size < min_side or B.size < min_side:
        return np.nan, np.nan, 1.0, False
    p_mw = float(mannwhitneyu(A, B, alternative="two-sided")[1])
    p_ks = float(ks_2samp(A, B)[1])
    return p_mw, p_ks, max(p_mw, p_ks), True


# ══════════════════════════════════════════════════════════════════════
# Result
# ══════════════════════════════════════════════════════════════════════


def _merge_selected(cps: np.ndarray, p: np.ndarray, dedup_w: int) -> np.ndarray:
    """
    Collapse selected change-points closer than ``dedup_w`` into one, keeping
    the strongest evidence (smallest p) in each run.

    Applied *after* selection: it only ever removes discoveries, so FDR
    control over the tested hypotheses is preserved.
    """
    if cps.size == 0 or dedup_w <= 0:
        return cps
    order = np.argsort(cps, kind="mergesort")
    cps, p = cps[order], p[order]

    keep_idx = [0]
    for i in range(1, cps.size):
        j = keep_idx[-1]
        if cps[i] - cps[j] < dedup_w:
            if p[i] < p[j]:
                keep_idx[-1] = i
        else:
            keep_idx.append(i)
    return cps[np.array(keep_idx, dtype=int)]


@dataclass
class VOCDResult:
    """
    Scored candidate change-points.

    Scoring is done once.  Thresholding is free, so re-query at any error
    budget without recomputing::

        r = detect(y)
        r.changepoints(fdr=0.05)
        r.changepoints(fdr=0.01, method="by")
        r.changepoints(fwer=0.01)
    """

    candidates: np.ndarray          # (K,) int   every tested proposal
    p_mw: np.ndarray                # (K,) float
    p_ks: np.ndarray                # (K,) float
    p_max: np.ndarray               # (K,) float  intersection p-value
    tested: np.ndarray              # (K,) bool
    raw_proposals: np.ndarray       # union of both passes
    n_obs: int
    dedup_w: int = 15
    params: dict = field(default_factory=dict)

    def changepoints(
        self,
        *,
        fdr: Optional[float] = None,
        fwer: Optional[float] = None,
        alpha: Optional[float] = None,
        method: str = "bh",
        merge: bool = True,
    ) -> np.ndarray:
        """
        Change-points accepted at the given error budget.

        Exactly one of ``fdr``, ``fwer``, ``alpha`` may be given; the default
        is ``fdr=0.05``.  ``alpha`` applies a raw per-test threshold with no
        correction and controls no global error rate -- it exists to reproduce
        fixed-threshold results.

        The correction is applied over tested candidates only.  Surviving
        detections within ``dedup_w`` are then merged unless ``merge=False``.
        """
        if fdr is None and fwer is None and alpha is None:
            fdr = 0.05
        if self.candidates.size == 0:
            return np.zeros(0, dtype=int)

        mask = np.zeros(self.candidates.size, dtype=bool)
        live = self.tested
        if live.any():
            mask[live] = select(
                self.p_max[live], fdr=fdr, fwer=fwer, alpha=alpha, method=method
            )

        cps, ps = self.candidates[mask], self.p_max[mask]
        return _merge_selected(cps, ps, self.dedup_w) if merge else cps

    def table(self) -> List[dict]:
        """Per-candidate detail, for inspection."""
        return [
            {
                "cp": int(self.candidates[i]),
                "p_mw": float(self.p_mw[i]),
                "p_ks": float(self.p_ks[i]),
                "p_max": float(self.p_max[i]),
                "tested": bool(self.tested[i]),
            }
            for i in range(self.candidates.size)
        ]

    def __repr__(self) -> str:
        return (
            f"VOCDResult(n_obs={self.n_obs}, "
            f"raw_proposals={self.raw_proposals.size}, "
            f"tested={int(self.tested.sum())}/{self.candidates.size})"
        )


# ══════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════


class VOCD:
    """
    Streaming VOCD detector.

    Feed observations one at a time::

        det = VOCD()
        for x in stream:
            for cp in det.update(x):
                print("change-point at", cp)

    ``update`` emits using a fixed per-test ``online_alpha``.  A growing
    candidate set makes FDR control ill-posed online -- Benjamini-Hochberg
    can *retract* a discovery when later candidates arrive, which is not
    possible once an alarm has been raised.  Online emission therefore
    controls no global error rate.  For FDR control use ``detect``, or read
    ``det.result`` at the end of the stream.

    Parameters
    ----------
    ppv, tpr : PassConfig for the precision- and recall-tuned passes.
    verify_w : half-window for the two-sample tests.  Also the detection
        latency: a change at tau is verifiable at tau + verify_w.
    dedup_w : detections closer than this describe one event and are merged
        after selection.  Set it to the shortest segment you need to resolve;
        it must be smaller than the gap between change-points you want kept
        apart.
    min_side : minimum observations either side of a candidate for the tests
        to run.
    online_alpha : per-test threshold on p_max for ``update``.
    """

    def __init__(
        self,
        ppv: PassConfig = DEFAULT_PPV,
        tpr: PassConfig = DEFAULT_TPR,
        *,
        verify_w: int = 25,
        dedup_w: int = 15,
        min_side: int = 5,
        online_alpha: float = 1e-4,
        backend: str = "auto",
    ):
        self.ppv_cfg = ppv
        self.tpr_cfg = tpr
        self.verify_w = int(verify_w)
        self.dedup_w = int(dedup_w)
        self.min_side = int(min_side)
        self.online_alpha = float(online_alpha)

        self.backend = _backend.resolve(backend)
        self._ppv = _backend.make_pass(ppv, self.backend)
        self._tpr = _backend.make_pass(tpr, self.backend)

        # Observations live in a geometrically grown NumPy buffer rather than
        # a list.  The gate needs an array view of the history on every step;
        # rebuilding one from a list would be O(T) per step, i.e. O(T^2) over a
        # run -- unnoticeable at T=500 and ruinous at the lengths real sensor
        # records reach.  Appending here is amortised O(1) and the view is free.
        self._ybuf = np.empty(1024, dtype=float)
        self._n = 0
        self._pending: List[int] = []      # proposals awaiting their gate time
        self._seen: set = set()            # dedupe identical locations
        self._scored: List[dict] = []
        self._last_emit: Optional[int] = None

    # -- internals ---------------------------------------------------

    def _add_proposal(self, loc: int) -> None:
        if loc not in self._seen:
            self._seen.add(loc)
            self._pending.append(loc)

    def _score_ready(self, t: int, final: bool = False) -> List[int]:
        """Test any proposal whose verification window has filled."""
        y = self._ybuf[:self._n]
        emitted: List[int] = []
        still: List[int] = []

        for loc in self._pending:
            if not final and t < loc + self.verify_w:
                still.append(loc)
                continue

            p_mw, p_ks, p_max, tested = _verify(y, loc, self.verify_w, self.min_side)
            self._scored.append(
                {"cp": loc, "p_mw": p_mw, "p_ks": p_ks, "p_max": p_max, "tested": tested}
            )

            if tested and p_max <= self.online_alpha:
                # Causal merge: an alarm already raised cannot be retracted, so
                # a near-duplicate is simply suppressed rather than replaced.
                if self._last_emit is None or (loc - self._last_emit) >= self.dedup_w:
                    emitted.append(loc)
                    self._last_emit = loc

        self._pending = still
        return emitted

    # -- public ------------------------------------------------------

    def update(self, x: float) -> List[int]:
        """
        Feed one observation.

        Returns change-point locations confirmed at this step (possibly
        empty).  Locations are historical: a change at tau is reported at
        roughly tau + verify_w.
        """
        if self._n == self._ybuf.size:
            self._ybuf = np.resize(self._ybuf, self._ybuf.size * 2)
        self._ybuf[self._n] = x
        self._n += 1
        t = self._n - 1

        for loc in self._ppv.update(x):
            self._add_proposal(loc)
        for loc in self._tpr.update(x):
            self._add_proposal(loc)

        return self._score_ready(t)

    def flush(self) -> List[int]:
        """
        End of stream: initialise if the series was shorter than
        ``init_window``, then score every outstanding proposal on whatever
        data exists.  Candidates near the end use truncated windows.
        """
        for loc in self._ppv.flush():
            self._add_proposal(loc)
        for loc in self._tpr.flush():
            self._add_proposal(loc)
        return self._score_ready(self._n - 1, final=True)

    @property
    def result(self) -> VOCDResult:
        """Everything scored so far, as a re-thresholdable result."""
        recs = sorted(self._scored, key=lambda r: r["cp"])
        n = len(recs)
        arr = lambda k, dt: (
            np.array([r[k] for r in recs], dtype=dt) if n else np.zeros(0, dtype=dt)
        )
        return VOCDResult(
            candidates=arr("cp", int),
            p_mw=arr("p_mw", float),
            p_ks=arr("p_ks", float),
            p_max=arr("p_max", float),
            tested=arr("tested", bool),
            raw_proposals=np.array(sorted(self._seen), dtype=int),
            n_obs=self._n,
            dedup_w=self.dedup_w,
            params={
                "ppv": self.ppv_cfg,
                "tpr": self.tpr_cfg,
                "verify_w": self.verify_w,
                "dedup_w": self.dedup_w,
                "min_side": self.min_side,
                "backend": self.backend,
            },
        )


def detect(
    y: Sequence[float],
    ppv: PassConfig = DEFAULT_PPV,
    tpr: PassConfig = DEFAULT_TPR,
    *,
    verify_w: int = 25,
    dedup_w: int = 15,
    min_side: int = 5,
    backend: str = "auto",
) -> VOCDResult:
    """
    Offline detection over a complete series.

    A loop over the same causal engine ``VOCD.update`` uses, followed by a
    flush.  Proposals verifiable within the series are scored identically to
    the streaming path; the flush additionally scores proposals near the end
    on truncated windows, which a live stream could not yet have seen.

    Returns a :class:`VOCDResult`.  Call ``.changepoints(fdr=...)`` on it.
    """
    y = np.asarray(y, dtype=float).ravel()
    det = VOCD(ppv, tpr, verify_w=verify_w, dedup_w=dedup_w,
               min_side=min_side, backend=backend)
    for x in y:
        det.update(x)
    det.flush()
    return det.result
