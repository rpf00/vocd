"""
Fully compiled VOCD detection.

``vocd.detect`` with the ``cpp`` backend compiles only the Viterbi passes;
the gate, the regime-bank fit and the per-observation bookkeeping stay in
Python and dominate the remaining runtime.  This module runs the whole of
``core.detect`` in one C++ call (``_vocd_full.cpp``) and returns the same
``VOCDResult``::

    import vocd._fastdetect as fastdetect
    r = fastdetect.detect(y)             # VOCDResult, identical to vocd.detect(y)
    r.changepoints(fwer=0.01)

    fastdetect.enable()                  # vocd.detect(..., backend="cpp") now
                                         # uses the full C++ engine
    fastdetect.disable()

The result is EXACT: ``_vocd_full.cpp`` mirrors ``vocd/core.py`` with NumPy's
arithmetic reproduced where the last bit can matter (linspace, pairwise
summation in var/std, median, the emission's log term, first-index argmin),
and scores candidates with the same gate as ``vocd/_fastgate.py``.  The one
case still computed through SciPy -- a window of <= 8 points with ties --
is handled exactly as in ``_fastgate``.  ``benchmarks/cpp_comparison.py``
asserts identical detections on every run.

Falls back to the Python engine when a regime bank cannot be fitted here
(series shorter than the bank, i.e. ``T < S``), so behaviour never differs.
The library is built on first use next to this file (any C++17 compiler;
``CXX`` overrides).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Optional, Sequence

import numpy as np

import vocd as _pkg
from . import core as _core
from ._faststats import ks2samp_p as _ks_py, mannwhitney_p as _mw_py
from ._fastgate import _mw_small_ties
from .core import DEFAULT_PPV, DEFAULT_TPR, PassConfig, VOCDResult

__all__ = ["detect", "enable", "disable", "enabled", "available"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_vocd_full.cpp")
_HDR = os.path.join(_HERE, "_gate_impl.hpp")
_LIB = os.path.join(_HERE, "_vocd_full.so")
_lib = None
_orig_detect = _core.detect
_buf_cap = 0
_bufs = None
_dp = ctypes.POINTER(ctypes.c_double)
_ip = ctypes.POINTER(ctypes.c_int)


def _load():
    global _lib
    if _lib is not None:
        return _lib
    newest = max(os.path.getmtime(_SRC), os.path.getmtime(_HDR))
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < newest:
        cxx = os.environ.get("CXX", "c++")
        subprocess.run([cxx, "-O2", "-std=c++17", "-ffp-contract=off", "-shared", "-fPIC",
                        "-o", _LIB, _SRC], check=True)
    L = ctypes.CDLL(_LIB)
    L.vocd_detect_full.restype = ctypes.c_int
    L.vocd_detect_full.argtypes = (
        [_dp, ctypes.c_int]
        + [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
           ctypes.c_int, ctypes.c_int, ctypes.c_int] * 2
        + [ctypes.c_int, ctypes.c_int, _ip, _dp, _dp, _ip, ctypes.c_int])
    _lib = L
    return L


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def _cfg_args(c: PassConfig):
    return (int(c.S), int(c.confirm_k), float(c.huber_delta), float(c.pen_base),
            int(c.min_dwell), int(c.merge_window), int(c.init_window))


def detect(y: Sequence[float], ppv: PassConfig = DEFAULT_PPV,
           tpr: PassConfig = DEFAULT_TPR, *, verify_w: int = 25,
           dedup_w: int = 15, min_side: int = 5,
           backend: Optional[str] = None) -> VOCDResult:
    """Same contract as ``vocd.core.detect``; ``backend`` is accepted and
    ignored (this is the full C++ engine)."""
    global _buf_cap, _bufs
    L = _load()
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    n = y.size
    cap = max(n, 1)
    if cap > _buf_cap:                      # output scratch, grown on demand
        _buf_cap = cap
        _bufs = (np.zeros(cap, dtype=np.int32), np.zeros(cap, dtype=np.float64),
                 np.zeros(cap, dtype=np.float64), np.zeros(cap, dtype=np.int32))
    out_cp, out_pmw, out_pks, out_code = _bufs

    K = L.vocd_detect_full(
        y.ctypes.data_as(_dp), n, *_cfg_args(ppv), *_cfg_args(tpr),
        int(verify_w), int(min_side),
        out_cp.ctypes.data_as(_ip), out_pmw.ctypes.data_as(_dp),
        out_pks.ctypes.data_as(_dp), out_code.ctypes.data_as(_ip), cap)
    if K < 0:   # bank could not be fitted here: identical behaviour via Python
        return _orig_detect(y, ppv, tpr, verify_w=verify_w, dedup_w=dedup_w,
                            min_side=min_side, backend="python")

    cps = out_cp[:K].astype(int)
    p_mw = out_pmw[:K].copy()
    p_ks = out_pks[:K].copy()
    code = out_code[:K]
    tested = np.ones(K, dtype=bool)

    # the rare candidates the C++ gate hands back (see _gate_impl.hpp::gate)
    for i in np.flatnonzero(code != 1):
        cp = int(cps[i])
        A = y[max(0, cp - verify_w): cp]
        B = y[cp: min(n, cp + verify_w)]
        if A.size < min_side or B.size < min_side:
            p_mw[i], p_ks[i], tested[i] = np.nan, np.nan, False
        elif code[i] == 2:
            p_mw[i] = _mw_small_ties(A, B)         # p_ks already computed
        else:
            p_mw[i] = _mw_py(A, B)
            p_ks[i] = _ks_py(A, B)

    p_max = np.where(tested, np.maximum(p_mw, p_ks), 1.0)
    return VOCDResult(
        candidates=cps, p_mw=p_mw, p_ks=p_ks, p_max=p_max, tested=tested,
        raw_proposals=cps.copy(), n_obs=int(n), dedup_w=int(dedup_w),
        params={"ppv": ppv, "tpr": tpr, "verify_w": int(verify_w),
                "dedup_w": int(dedup_w), "min_side": int(min_side),
                "backend": "cpp-full"},
    )


def _routed_detect(y, ppv: PassConfig = DEFAULT_PPV, tpr: PassConfig = DEFAULT_TPR,
                   *, verify_w: int = 25, dedup_w: int = 15, min_side: int = 5,
                   backend: str = "auto") -> VOCDResult:
    if backend in ("cpp", "auto"):
        return detect(y, ppv, tpr, verify_w=verify_w, dedup_w=dedup_w,
                      min_side=min_side)
    return _orig_detect(y, ppv, tpr, verify_w=verify_w, dedup_w=dedup_w,
                        min_side=min_side, backend=backend)


def enable() -> None:
    """Route ``vocd.detect(..., backend="cpp" | "auto")`` to the full C++ engine.
    ``backend="python"`` is untouched, so the fair Python row is unaffected."""
    _load()
    _core.detect = _routed_detect
    _pkg.detect = _routed_detect


def disable() -> None:
    _core.detect = _orig_detect
    _pkg.detect = _orig_detect


def enabled() -> bool:
    return _pkg.detect is _routed_detect
