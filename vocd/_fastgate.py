"""
Compiled verification gate for VOCD.

``vocd.core._verify`` runs the Mann-Whitney and Kolmogorov-Smirnov tests
once per candidate through ``vocd._faststats`` (NumPy).  With the Viterbi
passes compiled, that gate is ~80% of VOCD's remaining runtime.  This module
provides the same gate in C++ (``_gate.cpp``) and a switch to use it::

    import vocd._fastgate as fastgate
    fastgate.enable()      # vocd.detect(...) now verifies in C++
    fastgate.disable()     # back to _faststats

The compiled gate is EXACT, not an approximation.  ``_gate.cpp`` reproduces
the gate's arithmetic in the same floating-point order for every branch the
gate takes, so p-values are bit-identical and detections do not change:

* both windows > 8 points (every candidate tested inside a series, and most
  tail candidates): ``_faststats``' tie-corrected asymptotic MW and SciPy's
  exact two-sample KS (square lattice when the windows are equal, the
  Hodges/Viehmann lattice recursion when the right window is truncated);
* a window of <= 8 points without ties (tail candidates on the final flush):
  SciPy's exact small-sample MW distribution.

The one case left to Python -- a window of <= 8 points WITH ties -- uses
SciPy's own asymptotic MW through its private functions, skipping only the
validation wrapper (SciPy's ``ndtr`` is cephes', not libm's, so it is kept
rather than re-implemented).  ``benchmarks/cpp_comparison.py`` asserts
identical detections; the branch-by-branch p-value checks are in its
development history.

The library is built on first use next to this file (any C++17 compiler; set
``CXX`` to override) and loaded with ctypes.
"""
from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

from . import core as _core
from ._faststats import ks2samp_p as _ks_py, mannwhitney_p as _mw_py

try:  # SciPy's own MW asymptotic pieces, minus its validation wrapper
    from scipy import special as _special
    from scipy.stats._mannwhitneyu import _get_mwu_z
    from scipy.stats._stats_py import _rankdata
except Exception:  # pragma: no cover
    _special = _get_mwu_z = _rankdata = None

__all__ = ["enable", "disable", "enabled", "verify", "available"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_gate.cpp")
_LIB = os.path.join(_HERE, "_gate.so")
_lib = None
_orig_verify = _core._verify
_dp = ctypes.POINTER(ctypes.c_double)


def _load():
    global _lib
    if _lib is not None:
        return _lib
    newest = max(os.path.getmtime(_SRC),
                 os.path.getmtime(os.path.join(_HERE, "_gate_impl.hpp")))
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < newest:
        cxx = os.environ.get("CXX", "c++")
        subprocess.run([cxx, "-O2", "-std=c++17", "-ffp-contract=off", "-shared", "-fPIC",
                        "-o", _LIB, _SRC], check=True)
    L = ctypes.CDLL(_LIB)
    L.vocd_gate.restype = ctypes.c_int
    L.vocd_gate.argtypes = [_dp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, _dp]
    _lib = L
    return L


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


_out = np.zeros(2, dtype=np.float64)
_out_p = _out.ctypes.data_as(_dp)


def verify(y: np.ndarray, cp: int, w: int, min_side: int):
    """Drop-in replacement for ``vocd.core._verify`` (same signature/return)."""
    L = _load()
    if y.dtype != np.float64 or not y.flags["C_CONTIGUOUS"]:
        y = np.ascontiguousarray(y, dtype=np.float64)
    code = L.vocd_gate(y.ctypes.data_as(_dp), y.size, int(cp), int(w),
                       int(min_side), _out_p)
    if code == 1:
        p_mw, p_ks = float(_out[0]), float(_out[1])
        return p_mw, p_ks, max(p_mw, p_ks), True
    T = y.size
    A = y[max(0, cp - w): cp]
    B = y[cp: min(T, cp + w)]
    if A.size < min_side or B.size < min_side:
        return np.nan, np.nan, 1.0, False
    if code == 2:
        p_mw = _mw_small_ties(A, B)
        p_ks = float(_out[1])
        return p_mw, p_ks, max(p_mw, p_ks), True
    p_mw = _mw_py(A, B)                   # code 0: unchanged Python path
    p_ks = _ks_py(A, B)
    return p_mw, p_ks, max(p_mw, p_ks), True


def _mw_small_ties(A: np.ndarray, B: np.ndarray) -> float:
    """SciPy's mannwhitneyu(method='asymptotic') for a tiny sample with ties,
    computed with SciPy's own private functions but without its validation
    wrapper (which is what costs ~1 ms).  Bit-identical to the public call."""
    if _get_mwu_z is None:
        return _mw_py(A, B)
    n1, n2 = A.size, B.size
    xy = np.concatenate((A, B))
    ranks, t = _rankdata(xy, "average", return_ties=True)
    R1 = np.sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = np.maximum(U1, U2)
    z = _get_mwu_z(U, n1, n2, t, continuity=True, xp=np)
    return float(np.clip(2 * _special.ndtr(-z), 0.0, 1.0))


def enable() -> None:
    """Route ``vocd.core._verify`` through the compiled gate."""
    _load()
    _core._verify = verify


def disable() -> None:
    _core._verify = _orig_verify


def enabled() -> bool:
    return _core._verify is verify
