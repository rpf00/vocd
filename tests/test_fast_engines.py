"""
The compiled engines must reproduce ``vocd.core`` exactly.

``tests/test_equivalence.py`` covers the pybind11 Viterbi backend.  This file
covers the two ctypes engines that are built on first use:

* ``vocd._fastgate``   -- the compiled verification gate
* ``vocd._fastdetect`` -- the whole of ``detect`` in one C++ call

Both are skipped when no C++ compiler is available.
"""
import numpy as np
import pytest

import vocd
from vocd.datasets import contaminate, make_synthetic

fastdetect = pytest.importorskip("vocd._fastdetect")
if not fastdetect.available():
    pytest.skip("no C++ compiler: full engine not built", allow_module_level=True)


def _same(a, b):
    assert np.array_equal(a.candidates, b.candidates)
    assert np.array_equal(a.tested, b.tested)
    assert np.array_equal(a.p_max, b.p_max)
    for budget in (1e-4, 0.01, 0.05, 0.3):
        assert np.array_equal(a.changepoints(fwer=budget), b.changepoints(fwer=budget))


def test_full_engine_matches_python_on_the_paper_benchmark():
    y, _ = make_synthetic()
    for seed in range(5):
        yc = contaminate(y, frac=0.02, seed=seed)
        _same(vocd.detect(yc, backend="python"), fastdetect.detect(yc))


def test_full_engine_matches_python_under_fuzz():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(120, 900))
        k = int(rng.integers(1, 6))
        levels = rng.normal(0.0, 5.0, size=k + 1)
        cuts = np.sort(rng.choice(np.arange(20, n - 20), size=k, replace=False))
        y = np.repeat(levels, np.diff(np.concatenate([[0], cuts, [n]])))
        y = y + rng.normal(0.0, 1.0, size=n)
        _same(vocd.detect(y, backend="python"), fastdetect.detect(y))


@pytest.mark.parametrize("n", [9, 30, 60, 99, 100, 101])
def test_full_engine_matches_python_on_short_signals(n):
    """Signals shorter than ``init_window``: the bank is fitted at flush."""
    rng = np.random.default_rng(n)
    y = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(6, 1, n - n // 2)])
    _same(vocd.detect(y, backend="python"), fastdetect.detect(y))


def test_enable_routes_cpp_backend_and_disable_restores():
    y, _ = make_synthetic()
    ref = vocd.detect(y, backend="python")
    fastdetect.enable()
    try:
        assert fastdetect.enabled()
        _same(ref, vocd.detect(y, backend="cpp"))
        _same(ref, vocd.detect(y, backend="python"))
    finally:
        fastdetect.disable()
    assert not fastdetect.enabled()
