"""
Fast invariants.  These run in CI on every commit; they assert properties,
not benchmark numbers.  Benchmarks live in benchmarks/ and are not tests.
"""

import numpy as np
import pytest

import vocd
from vocd._correction import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
)
from vocd.datasets import contaminate, make_synthetic


# ── determinism ────────────────────────────────────────────────────────


def test_detect_is_deterministic():
    y, _ = make_synthetic()
    a = vocd.detect(y)
    b = vocd.detect(y)
    assert np.array_equal(a.candidates, b.candidates)
    assert np.array_equal(a.p_max, b.p_max, equal_nan=True)
    assert np.array_equal(a.changepoints(fdr=0.05), b.changepoints(fdr=0.05))


# ── online / offline equivalence ───────────────────────────────────────


def test_streaming_scores_identically_to_batch():
    """The engine is one engine: streaming must score exactly as batch does."""
    y = contaminate(make_synthetic()[0], seed=1)

    offline = vocd.detect(y)

    det = vocd.VOCD()
    for x in y:
        det.update(x)
    det.flush()
    online = det.result

    assert np.array_equal(offline.candidates, online.candidates)
    assert np.allclose(offline.p_max, online.p_max, equal_nan=True)


def test_streaming_is_causal():
    """update() may never emit a change-point it cannot yet have verified."""
    y = contaminate(make_synthetic()[0], seed=2)
    det = vocd.VOCD()
    for t, x in enumerate(y):
        for cp in det.update(x):
            assert cp + det.verify_w <= t + 1, (
                f"emitted cp={cp} at t={t}: needs data up to {cp + det.verify_w}"
            )


# ── correction ─────────────────────────────────────────────────────────


def test_bh_matches_reference_on_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # Thresholds are 0.05*i/8 = .00625 .0125 .01875 .025 ... ; the largest i
    # with p_(i) <= thresh_i is i=2, so the first two are rejected.
    assert list(benjamini_hochberg(p, 0.05)) == [True, True] + [False] * 6

    # Every p-value tiny -> all rejected; every p-value large -> none.
    assert benjamini_hochberg(np.full(5, 1e-9), 0.05).all()
    assert not benjamini_hochberg(np.full(5, 0.9), 0.05).any()


def test_by_is_more_conservative_than_bh():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.05, 40)
    assert benjamini_yekutieli(p, 0.05).sum() <= benjamini_hochberg(p, 0.05).sum()


def test_bonferroni_is_most_conservative():
    p = np.array([0.001, 0.01, 0.02, 0.03])
    assert bonferroni(p, 0.05).sum() <= benjamini_hochberg(p, 0.05).sum()


def test_stricter_budget_is_nested():
    """Tightening the budget may only remove change-points, never add."""
    y = contaminate(make_synthetic()[0], seed=3)
    r = vocd.detect(y)
    loose = set(r.changepoints(fdr=0.10, merge=False).tolist())
    tight = set(r.changepoints(fdr=0.01, merge=False).tolist())
    assert tight <= loose


def test_select_rejects_ambiguous_budget():
    y, _ = make_synthetic()
    r = vocd.detect(y)
    with pytest.raises(ValueError):
        r.changepoints(fdr=0.05, fwer=0.05)


# ── behaviour ──────────────────────────────────────────────────────────


def test_p_max_is_the_intersection_rule():
    y = contaminate(make_synthetic()[0], seed=4)
    r = vocd.detect(y)
    live = r.tested
    assert np.allclose(r.p_max[live], np.maximum(r.p_mw[live], r.p_ks[live]))


def test_recovers_a_clear_changepoint():
    rng = np.random.default_rng(11)
    y = np.concatenate([rng.normal(0.0, 1.0, 250), rng.normal(4.0, 1.0, 250)])
    found = vocd.detect(y).changepoints(fdr=0.05)
    assert np.any(np.abs(found - 250) <= 10), f"missed the change at 250: {found}"


def test_recovers_the_paper_benchmark_changepoints():
    """
    The benchmark series from the DSM-BOCD protocol (legacy global RNG).
    All six change-points must be recovered at the legacy fixed threshold.

    Note this series is *not* interchangeable with one drawn from
    ``np.random.default_rng(3000)``: same seed, different stream, different
    signal.  On some default_rng draws the 5 -> 0 change at t=200 sits wholly
    outside the burn-in regime bank and is missed, since the bank is fitted on
    the first ``init_window`` samples and never adapts.  That is a real
    limitation, but it is not exercised by this benchmark.
    """
    np.random.seed(3000)
    cps_true, mus = [100, 200, 300, 350, 400, 450, 600], [10, 5, 0, 10, 5, 12, 0]
    y = np.zeros(500)
    mu, i = mus[0], 0
    for t in range(500):
        if t == cps_true[i]:
            mu = mus[i + 1]
            i += 1
        y[t] = np.random.normal(mu, 1)

    found = vocd.detect(y).changepoints(alpha=1e-4)
    for t in cps_true[:-1]:
        assert np.any(np.abs(found - t) <= 10), f"missed cp at {t}: {found}"


def test_frozen_bank_misses_regimes_outside_the_burn_in():
    """
    The regime bank is fitted on the first ``init_window`` observations and
    never adapts.  A transition between two levels that both sit outside the
    burn-in range moves no state, so no proposal fires and the gate never
    sees it.

    This series (``default_rng(3000)``) exercises that: burn-in covers only
    the first segment (mean 10), giving a bank spanning roughly [8.4, 11.6],
    and the 5 -> 0 change at t=200 falls wholly outside it.

    The paper benchmark series does *not* exercise this -- see
    ``test_recovers_the_paper_benchmark_changepoints``, where all six are
    found.  The limitation is real but data-dependent, which is precisely why
    both tests exist.  Adaptive re-initialisation would fix it; if this test
    starts failing, that work has landed and the benchmarks need regenerating.
    """
    y, _ = make_synthetic()
    found = vocd.detect(y).changepoints(fdr=0.05)
    assert not np.any(np.abs(found - 200) <= 10)
    # ...while changes touching the burn-in range are still found.
    for t in (300, 350, 400, 450):
        assert np.any(np.abs(found - t) <= 10), f"missed cp at {t}"


def test_outliers_alone_do_not_create_changepoints():
    """A constant signal with spikes has no change-points.  This is the
    robustness claim: the Viterbi fires, the gate refuses."""
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 1.0, 500)
    idx = rng.choice(500, 10, replace=False)
    y[idx] += rng.choice([-1.0, 1.0], 10) * 10.0

    found = vocd.detect(y).changepoints(fdr=0.05)
    assert found.size == 0, f"false alarms on pure-outlier signal: {found}"


def test_flat_signal_yields_nothing():
    found = vocd.detect(np.zeros(300)).changepoints(fdr=0.05)
    assert found.size == 0


def test_short_series_does_not_crash():
    y = np.random.default_rng(6).normal(size=20)
    r = vocd.detect(y)
    assert r.changepoints(fdr=0.05).size >= 0


def test_merge_keeps_strongest():
    y = contaminate(make_synthetic()[0], seed=7)
    r = vocd.detect(y)
    merged = r.changepoints(fdr=0.05, merge=True)
    assert np.all(np.diff(merged) >= r.dedup_w)
