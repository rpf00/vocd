"""
VOCD — Viterbi Online Change-point Detection.

Robust, deterministic change-point detection for streaming and archived
time series.  Two Viterbi passes over-propose candidates; nonparametric
Mann-Whitney and Kolmogorov-Smirnov tests verify them; a multiple-testing
correction turns the user's error budget into a decision rule.

Offline::

    import vocd
    r = vocd.detect(y)
    r.changepoints(fdr=0.05)
    r.changepoints(fdr=0.01, method="by")   # free: no rescoring

Online::

    det = vocd.VOCD()
    for x in stream:
        for cp in det.update(x):
            ...

There is no alpha to calibrate: state the false-alarm rate you can tolerate
and the threshold follows from the data.
"""

from ._correction import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
)
from .core import (
    DEFAULT_PPV,
    DEFAULT_TPR,
    PassConfig,
    VOCD,
    VOCDResult,
    detect,
)

__version__ = "0.1.0"

__all__ = [
    "detect",
    "VOCD",
    "VOCDResult",
    "PassConfig",
    "DEFAULT_PPV",
    "DEFAULT_TPR",
    "bonferroni",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "__version__",
]
