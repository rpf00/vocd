"""
Backend selection for the Viterbi proposal pass.

``vocd.core`` is the specification.  The compiled backend in ``_viterbi.cpp``
is an accelerator that must reproduce it exactly; ``tests/test_equivalence.py``
enforces that on randomised input.  If the extension is not built, everything
still works -- just slower.

Only the proposal stage is compiled.  The verification gate stays in Python so
SciPy remains the definition of the p-values (see ``_faststats``).

Choosing a backend::

    vocd.detect(y)                     # auto: compiled if available
    vocd.detect(y, backend="python")   # force the reference
    vocd.detect(y, backend="cpp")      # force compiled; raises if unbuilt

Build the extension in place with::

    python setup_ext.py

or ``pip install -e .``, which does it as part of the build.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["available", "resolve", "make_pass", "backend_name"]

try:
    from . import _viterbi as _ext
except Exception:  # pragma: no cover - extension simply not built
    _ext = None


def available() -> bool:
    """True when the compiled backend can be used."""
    return _ext is not None


def resolve(backend: str = "auto") -> str:
    """
    Map a requested backend onto one that exists.

    'auto' prefers the compiled pass and falls back silently.  An explicit
    'cpp' raises rather than degrading quietly: if a caller asked for it by
    name, timing results that silently came from Python would be misleading.
    """
    if backend == "python":
        return "python"
    if backend == "cpp":
        if not available():
            raise RuntimeError(
                "backend='cpp' requested but the extension is not built.\n"
                "Build it with:  python setup_ext.py    (or pip install -e .)"
            )
        return "cpp"
    if backend == "auto":
        return "cpp" if available() else "python"
    raise ValueError(f"unknown backend {backend!r}; expected "
                     "'auto', 'python' or 'cpp'")


def backend_name(backend: str = "auto") -> str:
    """Resolved backend name, for reporting."""
    return resolve(backend)


def make_pass(cfg, backend: str = "auto"):
    """
    Build a Viterbi pass for ``cfg`` on the resolved backend.

    Both backends expose the same tiny interface -- ``update(x) -> list[int]``
    and ``flush() -> list[int]`` -- so the engine above them is backend-blind
    and the streaming and batch paths stay identical.
    """
    from .core import _ViterbiPass  # local import: core imports this module

    if resolve(backend) == "python":
        return _ViterbiPass(cfg)

    return _ext.ViterbiPass(
        S=cfg.S,
        pen_base=float(cfg.pen_base),
        min_dwell=int(cfg.min_dwell),
        init_window=int(cfg.init_window),
        merge_window=int(cfg.merge_window),
        huber_delta=float(cfg.huber_delta),
        confirm_k=int(cfg.confirm_k),
    )
