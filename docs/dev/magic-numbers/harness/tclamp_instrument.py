#!/usr/bin/env python3
"""M1/M2 instrumentation for the net_coolingcurve `T<1e4` clamp (AUDIT finding #1).

NON-INVASIVE: wraps `get_dudt` (every bubble-ODE RHS eval) and
`_solve_bubble_structure` (every accepted bubble profile) to RECORD ONLY -- both
call through to the original, so the run behaves bit-identically to production.
Drives one real run via `main.start_expansion`, then writes two committed
artifacts under <out>/:
  <tag>_summary.json  scalar counters + log10(T) histogram (M1) + accepted-profile
                      T_min stats/histogram (M2)
  <tag>_lowT.csv      every get_dudt call with T < 3e4 K (t_now, log10T, ndens, phi)

Robust to a wall-clock `timeout` kill: a SIGTERM/SIGINT handler flushes first.

Usage:
  python docs/dev/magic-numbers/harness/tclamp_instrument.py <param> <tag> [out_dir]
Bound non-terminating stiff runs with `timeout` so they still flush + return:
  timeout 180 python docs/dev/magic-numbers/harness/tclamp_instrument.py \
      param/simple_cluster.param simple_cluster docs/dev/magic-numbers/data

ponytail: counters + a log10(T) histogram instead of per-call rows (millions of
calls); only the < 3e4 K excursions (the question) are stored row-by-row, capped.
"""
import os
import sys
import json
import math
import atexit
import signal

import numpy as np

sys.path.insert(0, os.getcwd())

PARAM = sys.argv[1]
TAG = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "docs/dev/magic-numbers/data"
os.makedirs(OUT, exist_ok=True)

TABLE_MIN = 10 ** 3.5      # 3162.28 K -- the true non-CIE table min (measured, not the comment's 3.99)
BOUNDARY = 3e4             # _T_INIT_BOUNDARY: a valid bubble profile is >= this everywhere

state = {
    "param": PARAM, "tag": TAG,
    "calls": 0, "n_below_1e4": 0, "n_below_3162": 0, "n_below_3e4": 0,
    "min_T": math.inf, "min_T_t": None,
    "hist": {},                       # log10(T) bin (0.05 wide) -> count   (M1)
    "accepted_solves": 0, "accepted_min_T": math.inf,
    "accepted_below_1e4": 0, "accepted_below_3e4": 0,
    "accepted_minT_hist": {},         # accepted-profile T_min, log10 bin -> count  (M2)
}
LOWT_CAP = 300000
lowT_rows = []


def _bin(x, w=0.05):
    return round(math.floor(x / w) * w, 4)


def flush():
    s = dict(state)
    s["min_T"] = None if s["min_T"] is math.inf else s["min_T"]
    s["accepted_min_T"] = None if s["accepted_min_T"] is math.inf else s["accepted_min_T"]
    with open(os.path.join(OUT, f"{TAG}_summary.json"), "w") as f:
        json.dump(s, f, indent=2, default=str)
    with open(os.path.join(OUT, f"{TAG}_lowT.csv"), "w") as f:
        f.write("t_now,log10T,ndens_au,phi_au\n")
        for r in lowT_rows:
            f.write(f"{r[0]},{r[1]:.5f},{r[2]:.6e},{r[3]:.6e}\n")
    frac = (s["n_below_1e4"] / s["calls"]) if s["calls"] else 0.0
    print(f"[tclamp/{TAG}] calls={s['calls']} below_1e4={s['n_below_1e4']} ({frac:.2%}) "
          f"below_3162={s['n_below_3162']} below_3e4={s['n_below_3e4']} "
          f"min_T={s['min_T']} | accepted_solves={s['accepted_solves']} "
          f"accepted_min_T={s['accepted_min_T']} accepted_below_1e4={s['accepted_below_1e4']} "
          f"accepted_below_3e4={s['accepted_below_3e4']} -> {OUT}/{TAG}_*")


atexit.register(flush)


def _sig(signum, frame):
    flush()
    os._exit(0)


signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)

# ---- install non-invasive wrappers (record, then call through) ----
import trinity.cooling.net_coolingcurve as ncc
import trinity.bubble_structure.bubble_luminosity as bl

_orig_get_dudt = ncc.get_dudt


def wrapped_get_dudt(age, ndens, T, phi, params_dict):
    # Record BEFORE calling: the original mutates ndens/phi in place (/=).
    try:
        t_now = float(params_dict['t_now'].value)
    except Exception:
        t_now = float("nan")
    Tf = float(T)
    state["calls"] += 1
    if Tf > 0:
        b = _bin(math.log10(Tf))
        state["hist"][b] = state["hist"].get(b, 0) + 1
    if Tf < state["min_T"]:
        state["min_T"] = Tf
        state["min_T_t"] = t_now
    if Tf < 1e4:
        state["n_below_1e4"] += 1
    if Tf < TABLE_MIN:
        state["n_below_3162"] += 1
    if Tf < BOUNDARY:
        state["n_below_3e4"] += 1
        if len(lowT_rows) < LOWT_CAP:
            lowT_rows.append((repr(t_now), math.log10(Tf) if Tf > 0 else -99.0,
                              float(ndens), float(phi)))
    return _orig_get_dudt(age, ndens, T, phi, params_dict)


ncc.get_dudt = wrapped_get_dudt

_orig_solve = bl._solve_bubble_structure


def wrapped_solve(*a, **k):
    out = _orig_solve(*a, **k)
    try:
        psoln = out[0]
        if psoln is not None and len(psoln):
            tmin = float(np.min(psoln[:, 1]))     # col 1 = T (bubble_luminosity.py:609)
            state["accepted_solves"] += 1
            if tmin < state["accepted_min_T"]:
                state["accepted_min_T"] = tmin
            if tmin < 1e4:
                state["accepted_below_1e4"] += 1
            if tmin < BOUNDARY:
                state["accepted_below_3e4"] += 1
            if tmin > 0:
                bb = _bin(math.log10(tmin))
                state["accepted_minT_hist"][bb] = state["accepted_minT_hist"].get(bb, 0) + 1
    except Exception:
        pass
    return out


bl._solve_bubble_structure = wrapped_solve

# ---- drive one real run ----
from trinity._input import read_param
from trinity import main

params = read_param.read_param(PARAM)
# Optional 4th arg overrides path2output so concurrent runs (configs that share
# the default output dir) don't clobber each other's dictionary.jsonl.
if len(sys.argv) > 4:
    os.makedirs(sys.argv[4], exist_ok=True)
    params['path2output'].value = sys.argv[4]
main.start_expansion(params)
