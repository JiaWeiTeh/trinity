#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase/time probe: WHERE in the run do the shell-structure ODE solves happen, and
how many solves / how much wall time does it take to reach the implicit and
transition phases? Answers "is the /40 capture enough?" and "can we get a solve
into the transition phase?".

For each shell solve it records the evolution phase (from params['current_phase']
AND from the calling phase-module on the stack), the sim time t_now, and the
wall-clock elapsed. It:
  - prints a line to stderr EVERY TIME the phase changes (so you see
    energy -> implicit -> transition -> momentum live), and
  - appends every record to data/phase_map_<config>.csv, FLUSHED per row, so the
    map survives even if a wall-time timeout kills the process.

REPRODUCE
    cd /home/user/trinity
    PROBE_N=20000 python docs/dev/shell-solver/harness/phase_probe.py [param-file]
    # default param: simple_cluster ; default PROBE_N: 40
"""
import os
import sys
import csv
import time
import inspect
from pathlib import Path
from collections import Counter

import numpy as np
import scipy.integrate

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_PROBE = int(os.environ.get("PROBE_N", "40"))
_REAL = scipy.integrate.odeint
_rec = []
_t0 = None
_last_phase = None
_csv_fh = None
_csv_w = None

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
    global _t0, _last_phase
    if _t0 is None:
        _t0 = time.time()
    if len(_rec) >= MAX_PROBE:
        raise _Done()
    params = args[2] if len(args) >= 3 else None
    phase_field, t_now = "", float("nan")
    if params is not None:
        try:
            phase_field = params["current_phase"].value
        except Exception:
            pass
        try:
            t_now = float(params["t_now"].value)
        except Exception:
            pass
    caller = _caller_phase()
    wall = time.time() - _t0
    row = {"idx": len(_rec), "caller_phase": caller, "phase_field": phase_field,
           "t_now": t_now, "wall_s": round(wall, 2),
           "is_ion": int(bool(args[1])) if len(args) >= 2 else -1}
    _rec.append(row)
    _csv_w.writerow(row)
    _csv_fh.flush()
    phase = caller if caller != "?" else phase_field
    if phase != _last_phase:
        print(f"  PHASE -> {phase:11s} at solve #{len(_rec)-1}  "
              f"t_now={t_now:.4e} Myr  wall={wall:.1f}s", file=sys.stderr, flush=True)
        _last_phase = phase
    return _REAL(func, y0, t, args=args, **kw)


def main():
    global _csv_fh, _csv_w
    param = (Path(sys.argv[1]) if len(sys.argv) > 1
             else TRINITY_ROOT / "param" / "simple_cluster.param")
    csv_path = DATA_DIR / f"phase_map_{param.stem}.csv"
    _csv_fh = open(csv_path, "w", newline="")
    _csv_w = csv.DictWriter(_csv_fh, fieldnames=["idx", "caller_phase",
             "phase_field", "t_now", "wall_s", "is_ion"])
    _csv_w.writeheader()

    import logging
    logging.disable(logging.CRITICAL)
    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main as tmain

    params = read_param.read_param(str(param))
    chk = validate_gmc_from_params(params)
    if not chk.valid:
        raise SystemExit("GMC invalid: " + "; ".join(chk.errors))

    print(f"probing {param.name} (PROBE_N={MAX_PROBE}) -> {csv_path.name}",
          file=sys.stderr, flush=True)
    scipy.integrate.odeint = _patched
    try:
        tmain.start_expansion(params)
    except (_Done, SystemExit):
        pass
    finally:
        scipy.integrate.odeint = _REAL
        _csv_fh.flush()
        _csv_fh.close()

    logging.disable(logging.NOTSET)
    by_caller = Counter(r["caller_phase"] for r in _rec)
    ts = [r["t_now"] for r in _rec if not np.isnan(r["t_now"])]
    print("=" * 64, file=sys.stderr)
    print(f"phase probe: {param.name}  ({len(_rec)} shell solves)", file=sys.stderr)
    print(f"  shell solves per phase: {dict(by_caller)}", file=sys.stderr)
    if ts:
        print(f"  t_now: {min(ts):.4e} .. {max(ts):.4e} Myr  "
              f"(wall {_rec[-1]['wall_s']}s)", file=sys.stderr)
    print(f"  full map -> {csv_path}", file=sys.stderr)
    print("=" * 64, file=sys.stderr)


if __name__ == "__main__":
    main()
