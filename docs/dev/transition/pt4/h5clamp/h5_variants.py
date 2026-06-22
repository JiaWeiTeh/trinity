#!/usr/bin/env python3
"""H5 (beta/delta CLAMP-WIDTH) DIAGNOSTIC variant for the transition-trigger pt4
experiment. Production is NEVER touched — this monkeypatches the four module-level
clamp constants on ``trinity.phase1b_energy_implicit.get_betadelta`` only.

>>> WHAT THIS ISOLATES — READ FIRST <<<
H5 claims legacy's cooling-ratio crossing (Lgain-Lloss)/Lgain < 0.05 is
"predetermined" by the (beta,delta) box clamp [0,1]x[-1,0] soft-locking the
solution on the boundary (holding Lloss high). The CAUSAL test is to WIDEN the
BOX progressively and re-run the LEGACY solver. If a config's crossing is CAUSED
by the box, widening should move/vanish the crossing; if unchanged, the box is
NOT the cause for that config.

The four constants are read AT CALL TIME by the LEGACY solver:
  - grid search bounds        get_betadelta.py:969-972 (max(BETA_MIN,..), min(BETA_MAX,..), ..)
  - L-BFGS-B np.clip          get_betadelta.py:1044-1045
  - L-BFGS-B `bounds`         get_betadelta.py:1053
so setting the module attributes BEFORE start_expansion widens the whole box for
every legacy beta-delta solve in the run. (hybr ignores them entirely — it is an
unbounded scipy root('hybr') with a dMdt gate, get_betadelta.py:874.)

CAVEAT (state this in the writeup): W3 (wide-box legacy) is NOT identical to hybr.
hybr is a DIFFERENT root-finder (scipy root('hybr') on a pole-free g residual +
dMdt acceptance gate), not "legacy with an infinite box". So this sweep isolates
the role of the BOX, separate from the solver-method difference. Use
c0_<cfg>_h0.csv as the hybr (true-unbounded-root) reference, NOT as the W->inf
limit of the legacy sweep.

Widths (whole box widened symmetrically beyond the legacy default):
  W0  beta [ 0, 1]    delta [-1,  0]   (default legacy — committed c0_*_legacy.csv)
  W1  beta [-1, 2]    delta [-2,  1]
  W2  beta [-4, 4]    delta [-4,  4]
  W3  beta [-20,20]   delta [-20, 20]  (effectively unbounded box)

Single sim per process => module globals are safe (no concurrency in-process).
Call ``apply(width)`` BEFORE running the sim.
"""
import trinity.phase1b_energy_implicit.get_betadelta as gbd

# (beta_min, beta_max, delta_min, delta_max) per width id.
WIDTHS = {
    "W0": (0.0, 1.0, -1.0, 0.0),       # default legacy box (no-op vs production)
    "W1": (-1.0, 2.0, -2.0, 1.0),
    "W2": (-4.0, 4.0, -4.0, 4.0),
    "W3": (-20.0, 20.0, -20.0, 20.0),  # effectively unbounded
}

# remember the shipped defaults so _restore() is exact (and so a W0 apply is a
# provable no-op vs production).
_ORIG = (gbd.BETA_MIN, gbd.BETA_MAX, gbd.DELTA_MIN, gbd.DELTA_MAX)


def _restore() -> None:
    gbd.BETA_MIN, gbd.BETA_MAX, gbd.DELTA_MIN, gbd.DELTA_MAX = _ORIG


def apply(width: str):
    """Install the named box width onto the get_betadelta module constants.

    Returns the (beta_min, beta_max, delta_min, delta_max) tuple installed.
    """
    _restore()  # idempotent
    if width not in WIDTHS:
        raise ValueError(f"unknown width {width!r} (one of {sorted(WIDTHS)})")
    bmin, bmax, dmin, dmax = WIDTHS[width]
    gbd.BETA_MIN = bmin
    gbd.BETA_MAX = bmax
    gbd.DELTA_MIN = dmin
    gbd.DELTA_MAX = dmax
    return (bmin, bmax, dmin, dmax)
