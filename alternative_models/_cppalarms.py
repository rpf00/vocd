"""
Compiled FOCuS and NP-FOCuS (functional pruning) for the benchmark panel.

``focus_detect`` / ``npfocus_detect`` have the same contract as every other
entry in ``REGISTRY`` -- ``detect(x, alpha, min_seg) -> sorted change-point
indices`` -- and are registered as ``"focus-cpp"`` and ``"np-focus-cpp"``.
They are timing twins of ``focus.py`` / ``npfocus.py``: same statistic, same
restart driver, same threshold, bit-identical detections (the synthetic
driver asserts this on every run), only the runtime differs.

The default is the PUBLISHED algorithm: functional pruning (Romano, Eckley,
Fearnhead & Rigaill 2023; Romano, Eckley & Fearnhead 2024), O(log n)
candidates per step.  ``pruned=False`` selects a line-by-line C++ port of
the Python references' brute-force O(n^2) scan; it exists only as the
equivalence oracle used by ``benchmarks/cpp_comparison.py`` and is not a
benchmark row.

The alarm threshold is computed here, inside the timed call, with the same
``alpha_to_chi2_threshold`` the Python classes use in ``__init__``.  The
library (``_cpp_alarms.cpp``) is built next to this file on first use with
``c++`` (``CXX`` overrides) and loaded with ctypes.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from typing import List, Optional, Tuple

import numpy as np

from ._base import alpha_to_chi2_threshold

__all__ = ["focus_detect", "npfocus_detect", "run", "available"]

FOCUS_BURNIN = 10                 # defaults of focus.FOCuS / npfocus.NPFOCuS
NPFOCUS_BURNIN = 15
NPFOCUS_LEVELS = (0.25, 0.5, 0.75)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_cpp_alarms.cpp")
_LIB = os.path.join(_HERE, "_cpp_alarms.so")
_lib = None
_dp = ctypes.POINTER(ctypes.c_double)
_ip = ctypes.POINTER(ctypes.c_int)
_levels = np.asarray(NPFOCUS_LEVELS, dtype=np.float64)


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
        cxx = os.environ.get("CXX", "c++")
        subprocess.run([cxx, "-O2", "-std=c++17", "-ffp-contract=off", "-shared",
                        "-fPIC", "-o", _LIB, _SRC], check=True)
    L = ctypes.CDLL(_LIB)
    L.focus_detect.restype = ctypes.c_int
    L.focus_detect.argtypes = [_dp, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, _ip,
                               ctypes.c_int, _dp]
    L.npfocus_detect.restype = ctypes.c_int
    L.npfocus_detect.argtypes = [_dp, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_int, ctypes.c_int, _dp, ctypes.c_int,
                                 ctypes.c_int, _ip, ctypes.c_int, _dp]
    _lib = L
    return L


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def run(kind: str, x: np.ndarray, alpha: float, min_seg: int = 10,
        pruned: bool = True, burnin: Optional[int] = None
        ) -> Tuple[List[int], Tuple[float, float]]:
    """Low-level entry: returns (change-points, (mean kept candidates per
    step over all pruning stacks, peak kept in one stack))."""
    L = _load()
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    n = x.size
    out = np.zeros(max(n, 1), dtype=np.int32)
    st = np.zeros(2, dtype=np.float64)
    if kind == "focus":
        h = alpha_to_chi2_threshold(alpha, n, n_tests=1)
        m = L.focus_detect(x.ctypes.data_as(_dp), n, h, float(alpha), int(min_seg),
                           int(burnin if burnin is not None else FOCUS_BURNIN),
                           int(pruned), out.ctypes.data_as(_ip), n,
                           st.ctypes.data_as(_dp))
    elif kind == "np-focus":
        h = alpha_to_chi2_threshold(alpha, n, n_tests=len(NPFOCUS_LEVELS))
        m = L.npfocus_detect(x.ctypes.data_as(_dp), n, h, float(alpha), int(min_seg),
                             int(burnin if burnin is not None else NPFOCUS_BURNIN),
                             _levels.ctypes.data_as(_dp), len(NPFOCUS_LEVELS),
                             int(pruned), out.ctypes.data_as(_ip), n,
                             st.ctypes.data_as(_dp))
    else:
        raise ValueError(kind)
    return [int(c) for c in out[:m]], (float(st[0]), float(st[1]))


def focus_detect(x: np.ndarray, alpha: float = 0.01, min_seg: int = 10) -> list:
    """FOCuS, functional pruning, C++.  Twin of ``focus.detect``."""
    return run("focus", x, alpha, min_seg, pruned=True)[0]


def npfocus_detect(x: np.ndarray, alpha: float = 0.01, min_seg: int = 10) -> list:
    """NP-FOCuS, functional pruning, C++.  Twin of ``npfocus.detect``."""
    return run("np-focus", x, alpha, min_seg, pruned=True)[0]
