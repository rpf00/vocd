"""
alternative_models/ — benchmark detectors for the VOCD synthetic comparison.

This package holds *thin, uniform adapters* for every method in the benchmark
except VOCD itself (which is the installed ``vocd`` package and is called
directly).  The design rule for this directory is fairness under one bar:

    every detector is ONLINE and never sees the whole series before deciding a
    change-point — each decision at time t uses only x[:t+1].

Two consequences of that rule shape the code here:

1.  BOCD / Dm-BOCD are *not* reimplemented.  They are imported from a clone of
    ``maltamiranomontero/DSM-bocd`` (point at it with ``--dsm-path`` or the
    ``VOCD_DSM_PATH`` env var), so the Bayesian baselines run on their own
    terms.  The only thing the wrapper changes is *how change-points are read
    off the run-length matrix*: the upstream ``find_cp`` scans the full matrix
    and revises past change-points using later data, which is not causal.  The
    wrapper therefore defaults to a causal, commit-on-arrival extraction and
    offers ``extract="upstream"`` to reproduce the published operating point.

2.  FOCuS / R-FOCuS / NP-FOCuS / CHAD are alarm-style: natively they emit a
    stopping time, not a segmentation.  They are reference reimplementations
    (following the cited papers, NOT the authors' code) exposed through a
    common ``OnlineAlarm`` interface and driven by ``_restart.run_restart``,
    which restarts the detector after each alarm to yield a full change-point
    set online.  All four share ONE error budget ``alpha`` (family-wise, split
    across the tests each method performs), set before seeing the data — the
    analogue of VOCD's FWER budget — so no method is individually calibrated on
    the test series.

Every detector exposes the same callable contract used by the example harness::

    detect(x: np.ndarray, **kwargs) -> list[int]

returning sorted, unique change-point indices WITHOUT the leading-0 sentinel;
the harness prepends 0 itself so all methods are scored alike.

See ``examples/synthetic_data.py`` for the driver.
"""
from __future__ import annotations

from ._base import OnlineAlarm
from ._restart import run_restart
from .bocd import detect as bocd_detect
from .dm_bocd import detect as dm_bocd_detect
from .focus import detect as focus_detect
from .rfocus import detect as rfocus_detect
from .npfocus import detect as npfocus_detect
from .chad import detect as chad_detect

# name -> (callable, needs_dsm).  VOCD is added by the harness, not here.
REGISTRY = {
    "bocd":     (bocd_detect,    True),
    "dm-bocd":  (dm_bocd_detect, True),
    "focus":    (focus_detect,   False),
    "r-focus":  (rfocus_detect,  False),
    "np-focus": (npfocus_detect, False),
    "chad":     (chad_detect,    False),
}

__all__ = ["REGISTRY", "OnlineAlarm", "run_restart"]
