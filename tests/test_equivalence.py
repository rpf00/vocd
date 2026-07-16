"""
Equivalence tests.

VOCD has three optional accelerations, each of which must be exactly as correct
as the thing it replaces:

  1. ``_faststats.mannwhitney_p`` / ``ks2samp_p``  vs  SciPy
  2. ``core._verify``                             vs  ``core._verify_reference``
  3. the compiled Viterbi backend                 vs  ``core._ViterbiPass``

The references are the specification.  These tests fuzz the fast paths against
them on randomised input, so a speedup that changes an answer fails CI rather
than quietly moving the published numbers.

Tests requiring the compiled extension skip when it is not built, so this file
passes either way.
"""

import numpy as np
import pytest
from scipy.stats import ks_2samp, mannwhitneyu

import vocd
from vocd import _backend
from vocd._faststats import ks2samp_p, mannwhitney_p
from vocd.core import _verify, _verify_reference
from vocd.datasets import contaminate, make_synthetic

requires_ext = pytest.mark.skipif(
    not _backend.available(),
    reason="compiled backend not built (python setup_ext.py)",
)


# ── 1. fast stats vs SciPy ────────────────────────────────────────────


def test_fast_stats_match_scipy_under_fuzz():
    """
    Randomised n, ties, and effect sizes.  SciPy switches between exact and
    asymptotic distributions depending on sample size and ties; the fast paths
    must follow it across every branch, not just the common one.
    """
    rng = np.random.default_rng(0)
    worst_mw = worst_ks = 0.0

    for _ in range(400):
        n1, n2 = rng.integers(5, 60, 2)
        a = rng.normal(0.0, 1.0, n1)
        b = rng.normal(rng.uniform(0.0, 4.0), 1.0, n2)
        if rng.random() < 0.3:          # force ties
            a = np.round(a, 1)
            b = np.round(b, 1)

        worst_mw = max(worst_mw, abs(
            mannwhitney_p(a, b) - mannwhitneyu(a, b, alternative="two-sided")[1]))
        worst_ks = max(worst_ks, abs(ks2samp_p(a, b) - ks_2samp(a, b)[1]))

    assert worst_mw < 1e-12, f"MW drifted from SciPy by {worst_mw:.3e}"
    assert worst_ks < 1e-12, f"KS drifted from SciPy by {worst_ks:.3e}"


def test_fast_stats_match_scipy_in_the_far_tail():
    """
    The thresholds live in the far tail, so agreement there is what matters.
    An asymptotic KS approximation passes a loose check and fails this one.
    """
    rng = np.random.default_rng(1)
    for shift in (2.0, 3.0, 4.0, 5.0):
        a = rng.normal(0.0, 1.0, 50)
        b = rng.normal(shift, 1.0, 50)
        p_fast, p_ref = ks2samp_p(a, b), ks_2samp(a, b)[1]
        assert p_fast == pytest.approx(p_ref, rel=1e-9, abs=0.0)


def test_verify_matches_the_scipy_reference():
    y = contaminate(make_synthetic()[0], seed=1)
    for cp in (100, 150, 200, 300, 350, 400, 450):
        fast = _verify(y, cp, 25, 5)
        ref = _verify_reference(y, cp, 25, 5)
        assert fast[3] == ref[3]                       # tested flag
        for i in range(3):
            assert fast[i] == pytest.approx(ref[i], rel=1e-9, nan_ok=True)


# ── 2. compiled backend vs the Python reference ───────────────────────


@requires_ext
def test_backends_agree_on_the_paper_benchmark():
    np.random.seed(3000)
    cps_true, mus = [100, 200, 300, 350, 400, 450, 600], [10, 5, 0, 10, 5, 12, 0]
    y = np.zeros(500)
    mu, i = mus[0], 0
    for t in range(500):
        if t == cps_true[i]:
            mu = mus[i + 1]
            i += 1
        y[t] = np.random.normal(mu, 1)

    a = vocd.detect(y, backend="python")
    b = vocd.detect(y, backend="cpp")
    assert np.array_equal(a.raw_proposals, b.raw_proposals)
    assert np.array_equal(a.candidates, b.candidates)
    assert np.allclose(a.p_max, b.p_max, equal_nan=True)


@requires_ext
def test_backends_agree_under_fuzz():
    """
    Random signals, including ones with no change-points and ones with heavy
    contamination.  The compiled pass carries state across calls, so a
    divergence would compound rather than stay local.
    """
    rng = np.random.default_rng(7)
    for trial in range(12):
        n_seg = int(rng.integers(1, 6))
        segs = [rng.normal(rng.uniform(-8, 8), rng.uniform(0.5, 2.0),
                           int(rng.integers(60, 200)))
                for _ in range(n_seg)]
        y = np.concatenate(segs)
        if rng.random() < 0.5:
            idx = rng.choice(y.size, max(1, y.size // 50), replace=False)
            y[idx] += rng.choice([-1.0, 1.0], idx.size) * 10.0

        a = vocd.detect(y, backend="python")
        b = vocd.detect(y, backend="cpp")
        assert np.array_equal(a.raw_proposals, b.raw_proposals), (
            f"trial {trial}: proposals diverged")
        assert np.allclose(a.p_max, b.p_max, equal_nan=True)


@requires_ext
def test_backends_agree_when_streaming():
    """Equivalence must hold step by step, not only in batch."""
    y = contaminate(make_synthetic()[0], seed=3)

    out = {}
    for be in ("python", "cpp"):
        det = vocd.VOCD(backend=be)
        emitted = []
        for x in y:
            emitted.extend(det.update(x))
        det.flush()
        out[be] = (emitted, det.result)

    assert out["python"][0] == out["cpp"][0]
    assert np.array_equal(out["python"][1].candidates, out["cpp"][1].candidates)


@requires_ext
def test_backends_agree_on_a_short_series():
    """Shorter than init_window: exercises the flush path in both backends."""
    y = np.random.default_rng(8).normal(size=40)
    a = vocd.detect(y, backend="python")
    b = vocd.detect(y, backend="cpp")
    assert np.array_equal(a.raw_proposals, b.raw_proposals)


# ── 3. dispatch ───────────────────────────────────────────────────────


def test_auto_backend_resolves_to_something_real():
    assert _backend.resolve("auto") in ("python", "cpp")
    assert _backend.resolve("python") == "python"


def test_explicit_cpp_raises_when_unavailable_rather_than_degrading():
    """
    A silent fallback would make a timing run report Python numbers under a
    C++ label.  Better to fail loudly.
    """
    if _backend.available():
        assert _backend.resolve("cpp") == "cpp"
    else:
        with pytest.raises(RuntimeError):
            _backend.resolve("cpp")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        _backend.resolve("fortran")


def test_result_records_which_backend_ran():
    r = vocd.detect(make_synthetic()[0], backend="python")
    assert r.params["backend"] == "python"
