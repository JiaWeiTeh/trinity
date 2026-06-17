#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase/time probe: WHERE in the run do the first MAX_PROBE shell-structure ODE
solves (the "/40" the variant harness captures) actually happen? Records the
evolution phase + simulation time + calling phase-module for each shell solve,
so we know whether the capture-replay evidence covers energy/implicit only, or
also transition/momentum.

REPRODUCE
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/phase_probe.py [param-file]   # default: simple_cluster
"""
import os
import sys
import inspect
from pathlib import Path
from collections import Counter

import numpy as np
import scipy.integrate

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

MAX_PROBE = int(os.environ.get("PROBE_N", "40"))  # how many shell solves to sample
_REAL = scipy.integrate.odeint
_rec = []

_PHASE_FILES = {
    "run_energy_phase.py": "energy",
    "run_energy_implicit_phase.py": "implicit",
    "run_transition_phase.py": "transition",
    "run_momentum_phase.py": "momentum",
}


class _Done(Exception):
    pass


def _caller_phase():
    for fr in inspect.stack():
        base = os.path.basename(fr.filename)
        if base in _PHASE_FILES:
            return _PHASE_FILES[base]
    return "?"


def _patched(func, y0, t, args=(), **kw):
    if len(_rec) >= MAX_PROBE:
        raise _Done()
    params = args[2] if len(args) >= 3 else None
    phase_field = ""
    t_now = float("nan")
    if params is not None:
        try:
            phase_field = params["current_phase"].value
        except Exception:
            pass
        try:
            t_now = float(params["t_now"].value)
        except Exception:
            pass
    is_ion = bool(args[1]) if len(args) >= 2 else None
    _rec.append({"idx": len(_rec), "caller_phase": _caller_phase(),
                 "phase_field": phase_field, "t_now": t_now, "is_ion": is_ion})
    return _REAL(func, y0, t, args=args, **kw)


def main():
    param = (Path(sys.argv[1]) if len(sys.argv) > 1
             else TRINITY_ROOT / "param" / "simple_cluster.param")
    import logging
    logging.disable(logging.CRITICAL)
    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main as tmain

    params = read_param.read_param(str(param))
    chk = validate_gmc_from_params(params)
    if not chk.valid:
        raise SystemExit("GMC invalid: " + "; ".join(chk.errors))

    scipy.integrate.odeint = _patched
    try:
        tmain.start_expansion(params)
    except _Done:
        pass
    except SystemExit:
        pass
    finally:
        scipy.integrate.odeint = _REAL

    logging.disable(logging.NOTSET)
    by_caller = Counter(r["caller_phase"] for r in _rec)
    by_field = Counter(r["phase_field"] for r in _rec)
    ts = [r["t_now"] for r in _rec if not np.isnan(r["t_now"])]
    print("=" * 64, file=sys.stderr)
    print(f"phase probe: {param.name}  ({len(_rec)} shell solves captured)",
          file=sys.stderr)
    print(f"  by calling phase-module: {dict(by_caller)}", file=sys.stderr)
    print(f"  by params['current_phase']: {dict(by_field)}", file=sys.stderr)
    if ts:
        print(f"  t_now range: {min(ts):.6e} .. {max(ts):.6e} Myr", file=sys.stderr)
    print("=" * 64, file=sys.stderr)


if __name__ == "__main__":
    main()
