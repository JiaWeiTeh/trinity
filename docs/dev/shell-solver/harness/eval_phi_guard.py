#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantify the V2 phi-guard shell-ODE fix across regimes (production untouched).

WHAT THIS DOES
--------------
The shell-ODE RHS (``trinity/shell_structure/get_shellODE.py``) carries a
finite-radius POLE in its ``+nShell**2`` recombination term (lines 97,100). Past
the ionization front the integrated state runs to ``inf`` -- but shell_structure
truncates each 1000-pt slice at the first ``phi<=1e-9`` / mass-limited row
(``shell_structure.py:182``), so the overflow lives entirely in the DISCARDED
post-front tail while LSODA still floods ``t+h=t`` warnings.

V2 phi-guard (``get_shellODE_phiguard`` in get_shellODE_variants.py) freezes the
derivatives once the integrated ``phi<=0`` so the integrator never grinds into the
pole. This harness, per config:

  * MONKEYPATCHES the production RHS to phi-guard so the HOST run consumes the fix
    (``god.get_shellODE = get_shellODE_phiguard``) -- this is what clears the
    real-run flood.
  * MONKEYPATCHES ``scipy.integrate.odeint`` to capture each shell solve ONCE and
    run, side by side, the BASELINE (production RHS) and the PHI-GUARD solve, then
    compare n/phi/tau over the physically-USED region (``_phys_cutoff``: up to the
    first ``phi<=1e-9`` / first non-finite row -- everything production keeps).

PER CONFIG it records (aggregated): overflow_warns_total (host-run flood, target
0), nonfinite_tail_solves (target 0), used-region max rel diff of n/phi/tau vs
baseline (the phi-guard is NOT an identity -- this is the front shift), and
per-solve wall time phi-guard vs baseline odeint.

CONFIGS (degenerate -> realistic): simple_cluster, sfe0.6, probe_typical_hybr,
steep, dense_flat, mock_hybr.

REPRODUCE
---------
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/eval_phi_guard.py            # all configs
    python docs/dev/shell-solver/harness/eval_phi_guard.py sfe0.6     # one config

Writes docs/dev/shell-solver/data/eval_phi_guard.csv. Per config we cap captures
at CAP_N (default 40) ionized solves to keep wall time bounded; the flood/overflow
totals are aggregated over the captured solves (which span the early energy phase
where the pole is hottest). Authored 2026-06-18; python 3.11, numpy 1.26.4,
scipy 1.17.1.
"""

import os
import sys
import csv
import time
import tempfile
import warnings
import contextlib
from pathlib import Path

import numpy as np
import scipy.integrate

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

from docs.dev.shell_solver_harness import _shim  # noqa: E402  (see _ensure_import)
