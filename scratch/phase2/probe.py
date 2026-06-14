#!/usr/bin/env python3
"""Phase 2.1/2.2 probe, v2 (shadow, scratch-only — never shipped).

v2 fixes two v1 design faults found in the field: (a) results are written
per evaluation (one jsonl line each, flushed), so progress is observable
and a crash loses at most one point; (b) every evaluation runs under a
SIGALRM wall-time cap — exotic (beta, delta) corners can hang LSODA for
30+ minutes, and the alarm fires at the next Python RHS callback. The
timeout raises a BaseException subclass so get_residual_pure's
`except Exception` plateau handler cannot swallow it.

Per probe segment: 2.1 transects around the accepted point (noise floor),
then a 7x7 wide-box scan (beta [-1,2] x delta [-1,0.5]) ordered
center-out so the cheap physical region lands first. Each line records
both metric components, dMdt validity (Phase-2 abort contract), and wall
time. Production trajectory unaffected.

Usage: python scratch/phase2/probe.py <param> <out.jsonl> [seg,seg,...]
"""
import json
import signal
import sys
import time

import numpy as np

import trinity.phase1b_energy_implicit.run_energy_implicit_phase as RIP
from trinity import main as tmain
from trinity._input import read_param
from trinity.phase1b_energy_implicit.get_betadelta import (
    get_residual_detailed,
    get_residual_pure,
)

DEFAULT_SEGMENTS = [2, 8, 20, 45, 90, 160]
TRANSECT_OFFSETS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SCAN_BETA = np.linspace(-1.0, 2.0, 7)
SCAN_DELTA = np.linspace(-1.0, 0.5, 7)
POINT_TIMEOUT_S = 20


class _ProbeDone(Exception):
    """Raised after the last probe segment to end the run early."""


class _PointTimeout(BaseException):
    """BaseException so the plateau handler in get_residual_pure
    (`except Exception`) cannot swallow the per-point timeout."""


def _alarm(signum, frame):
    raise _PointTimeout()


def _eval_point(beta, delta, params, dMdt_seed):
    t0 = time.perf_counter()
    signal.alarm(POINT_TIMEOUT_S)
    try:
        Eres, Tres, props = get_residual_pure(
            beta, delta, params, return_bubble_props=True,
            dMdt_guess=dMdt_seed)
    except _PointTimeout:
        return dict(beta=float(beta), delta=float(delta), ok=False,
                    timeout=True, wall=time.perf_counter() - t0)
    finally:
        signal.alarm(0)
    rec = dict(beta=float(beta), delta=float(delta), ok=props is not None,
               f_E=float(Eres), f_T=float(Tres),
               wall=time.perf_counter() - t0)
    if props is not None:
        det = get_residual_detailed(beta, delta, params, bubble_props=props)
        dm = float(props.bubble_dMdt)
        rec.update(E1=float(det.Edot_from_beta),
                   E2=float(det.Edot_from_balance),
                   T1=float(det.T_bubble), T0=float(det.T0),
                   Lg=float(det.L_gain), dMdt=dm,
                   dmdt_ok=bool(np.isfinite(dm) and dm > 0))
    return rec


def _emit(out, kind, seg, t_now, rec):
    # numpy scalars (np.bool_, np.float64) are not all JSON-serializable;
    # .item() converts any of them to the native Python type.
    payload = {'kind': kind, 'segment': int(seg), 't_now': float(t_now), **rec}
    out.write(json.dumps(
        payload, default=lambda o: o.item() if hasattr(o, 'item') else str(o)
    ) + '\n')
    out.flush()


def _make_wrapper(orig, out, targets):
    last = max(targets)
    state = {'seg': 0}

    def wrapper(beta_guess, delta_guess, params, method='grid'):
        res = orig(beta_guess, delta_guess, params, method)
        state['seg'] += 1
        seg = state['seg']
        if seg in targets:
            t_now = params['t_now'].value
            seed = None
            if res.bubble_properties is not None:
                dm = float(res.bubble_properties.bubble_dMdt)
                if np.isfinite(dm) and dm > 0:
                    seed = dm
            _emit(out, 'accept', seg, t_now,
                  dict(beta=res.beta, delta=res.delta,
                       total_residual=res.total_residual,
                       converged=res.converged))
            for axis in ('beta', 'delta'):
                for sign in (1.0, -1.0):
                    for off in TRANSECT_OFFSETS:
                        b = res.beta + (sign * off if axis == 'beta' else 0.0)
                        d = res.delta + (sign * off if axis == 'delta' else 0.0)
                        rec = _eval_point(b, d, params, seed)
                        rec.update(axis=axis, off=sign * off)
                        _emit(out, 'transect', seg, t_now, rec)
            pts = sorted(
                ((float(b), float(d)) for b in SCAN_BETA for d in SCAN_DELTA),
                key=lambda bd: (bd[0] - res.beta) ** 2 + (bd[1] - res.delta) ** 2)
            for b, d in pts:
                _emit(out, 'scan', seg, t_now, _eval_point(b, d, params, seed))
            if seg >= last:
                raise _ProbeDone()
        return res

    return wrapper


def main():
    parampath, outpath = sys.argv[1], sys.argv[2]
    targets = (set(int(s) for s in sys.argv[3].split(','))
               if len(sys.argv) > 3 else set(DEFAULT_SEGMENTS))
    signal.signal(signal.SIGALRM, _alarm)
    params = read_param.read_param(parampath)
    orig = RIP.solve_betadelta_pure
    with open(outpath, 'w') as out:
        RIP.solve_betadelta_pure = _make_wrapper(orig, out, targets)
        try:
            tmain.start_expansion(params)
        except _ProbeDone:
            print(f'probe complete: last target segment reached -> {outpath}')
        finally:
            RIP.solve_betadelta_pure = orig


if __name__ == '__main__':
    main()
