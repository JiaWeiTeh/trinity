"""One deterministic T_init sweep over a single captured bubble state.

Loads a state dumped by ``TRINITY_BUBBLE_STATE_DUMP`` (via the bubble_audit
``load_state`` reconstructor), then for each candidate boundary temperature
``T_init`` re-runs the *full* production pipeline ``get_bubbleproperties_pure``
(R1/Pb -> fsolve dMdt -> initial conditions -> ODE -> luminosity) and records
the resulting luminosities.

``T_init`` is the single coupled knob ``bl._T_INIT_BOUNDARY`` (anchor + fsolve
floor + penalty); we monkeypatch that module attribute so all three sites move
together, exactly as a production change would.

This script does ONE pass and prints a JSON blob to stdout. Determinism is
proven by the driver (run_sweep.py) running this in many *separate* pinned
processes and asserting bit-identical output -- a single in-process loop could
hide cross-process FP nondeterminism (the BLAS-threading flake this whole
exercise is about).

Usage:
    python sweep.py <state.pkl> [base.param]
Thread pinning must be set in the ENV by the caller (before this process's
numpy import), not here.
"""
from __future__ import annotations

import os
import sys
import json

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (_REPO, os.path.join(_REPO, "tools", "bubble_audit")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from audit import load_state  # noqa: E402  (tools/bubble_audit/audit.py)
import trinity.bubble_structure.bubble_luminosity as bl  # noqa: E402
import trinity._functions.operations as operations  # noqa: E402

# Candidate boundary temperatures [K]. 3e4 is the production baseline and MUST
# be present (the sensitivity gate is relative to it).
T_INIT_GRID = [1.5e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4]
BASELINE = 3.0e4


def _thread_env():
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


def run_one(params, dumped_dMdt, t_init):
    """Full-pipeline bubble solve at a given boundary temperature.

    Returns (record_dict). ``status='ok'`` on success; otherwise the exception
    type name is recorded and the luminosities are None -- a crash is data, not
    a silent pass.
    """
    bl._T_INIT_BOUNDARY = float(t_init)
    # Warm-start fsolve from the dumped converged value every cell, so each
    # T_init starts from the same point (fair + deterministic).
    params["bubble_dMdt"].value = float(dumped_dMdt)
    try:
        props = bl.get_bubbleproperties_pure(params)
    except (bl.BubbleSolverError, operations.MonotonicError) as e:
        return {"t_init": t_init, "status": type(e).__name__, "L_total": None}
    except Exception as e:  # noqa: BLE001 -- surface anything else, don't pass
        return {"t_init": t_init, "status": f"OTHER:{type(e).__name__}:{e}",
                "L_total": None}

    L = float(props.bubble_LTotal)
    Tarr = np.asarray(props.bubble_T_arr, dtype=float)
    return {
        "t_init": t_init,
        "status": "ok",
        "L_total": L,
        "L_total_hex": float.hex(L),          # exact-bit determinism comparison
        "L1": float(props.bubble_L1Bubble),
        "L2": float(props.bubble_L2Conduction),
        "L3": float(props.bubble_L3Intermediate),
        "dMdt": float(props.bubble_dMdt),
        "T_min": float(np.nanmin(Tarr)),       # < t_init means the boundary dip
        "T_max": float(np.nanmax(Tarr)),
        "monotonic": bool(operations.monotonic(Tarr)),
    }


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    state_path = argv[0]
    base = argv[1] if len(argv) > 1 else None
    kwargs = {} if base is None else {"base_param": base}
    params, inputs, ref, meta = load_state(state_path, **kwargs)
    dumped_dMdt = float(inputs["bubble_dMdt"])

    cells = [run_one(params, dumped_dMdt, t) for t in T_INIT_GRID]

    # Fidelity: at the baseline, the re-solved dMdt must reproduce the dumped
    # converged value (the .pkl was made by current code, so this is a true
    # ground truth; T_array reproduction is checked bit-exact by bubble_audit).
    base_cell = next(c for c in cells if c["t_init"] == BASELINE)
    fid = None
    if base_cell["status"] == "ok":
        fid = abs(base_cell["dMdt"] - dumped_dMdt) / max(abs(dumped_dMdt), 1e-300)

    out = {
        "state": os.path.basename(state_path),
        "thread_env": _thread_env(),
        "dumped_dMdt": dumped_dMdt,
        "fidelity_dMdt_rel": fid,
        "cells": cells,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
