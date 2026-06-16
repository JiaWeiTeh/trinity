#!/usr/bin/env python3
"""Phase 2.3 four-arm shadow harness (scratch-only — never shipped).

Runs arms A-D side by side at each sampled implicit segment. Production
advances on its own (arm-A) result exactly as before; arms B-D are
shadow-evaluated from the same warm start and logged. One jsonl line per
arm per segment, flushed per line (crash loses at most one record).

Arms (docs/dev/archive/betadelta/HYBR_PLAN.md 2.3):
  A  control — the production call itself (validates the harness).
  B  metric  — production 5x5 grid/box/cap, but g-ranked and g-thresholded.
     The L-BFGS-B fallback is NOT re-implemented; records would_fallback
     when best g > LBFGSB_FALLBACK_THRESHOLD (where production would fire it).
  C  cap+bounds — ±0.02 window *iterating* (re-center on best until interior
     or 10 rescans) inside wide rails beta [-2,5], delta [-2,1]; old box
     demoted to a logged warn flag.
  D  hybr — scipy.optimize.root(method='hybr') on the g vector, unbounded,
     eps=3e-4 (from the measured ~1e-7 noise floor: h* ~ sqrt(eps_f)),
     xtol=1e-8, factor=0.1, maxfev=30. Out-of-box roots accepted + flagged.

Metrics: f = ((E1-E2)/E1)^2 + ((T1-T0)/T0)^2 (production); g replaces the
first component's denominator with L_gain (pole-free). Both reported for
every arm. Threshold 1e-4 on the sum of squares for both.

Abort contract: an evaluation is a failure (not a candidate) if the
structure solve fails (props None / (100,100) plateau), the point times
out (20 s SIGALRM), or dMdt is non-finite or <= 0. Grid arms skip failed
points (as production does); inside hybr a failure aborts the arm's
segment via _ArmAbort (BaseException: the plateau handler in
get_residual_pure catches only Exception).

Usage: python docs/dev/archive/betadelta/diagnostics/arms.py <param> <out.jsonl> [stride]
  stride 2 = arms at every 2nd implicit segment (cost cap: plan 2.3 allows
  a stratified ~50% subset when the baseline exceeds ~30 min).
"""
import json
import signal
import sys
import time

import numpy as np
from scipy.optimize import root as scipy_root

import trinity.phase1b_energy_implicit.get_betadelta as GB
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as RIP
from trinity import main as tmain
from trinity._input import read_param

THR = GB.RESIDUAL_THRESHOLD          # 1e-4, for f and g alike
EARLY = GB.GRID_EARLY_EXIT_RESIDUAL  # THR/10
WIDE_BETA = (-2.0, 5.0)              # arm C rails
WIDE_DELTA = (-2.0, 1.0)
HYBR_OPTS = dict(xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4)
POINT_TIMEOUT_S = 20
C_MAX_RESCANS = 10
C_WALL_BUDGET_S = 240  # cost cap: C stops rescanning past this and records it


class _ArmAbort(BaseException):
    """Evaluation failure inside an arm; BaseException so get_residual_pure's
    `except Exception` plateau handler cannot swallow it mid-hybr."""


class _PointTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _PointTimeout()


def _in_old_box(b, d):
    return (GB.BETA_MIN <= b <= GB.BETA_MAX) and (GB.DELTA_MIN <= d <= GB.DELTA_MAX)


class _Eval:
    """Counts evaluations and threads the dMdt warm start within a segment."""

    def __init__(self, params, seed):
        self.params = params
        self.seed = seed
        self.Lm = float(params['Lmech_total'].value)  # g denominator: per-segment constant
        self.n = 0

    def __call__(self, beta, delta):
        """Returns dict(fE, fT, gE, gT, f, g) or raises _ArmAbort/_PointTimeout."""
        self.n += 1
        signal.alarm(POINT_TIMEOUT_S)
        try:
            fE, fT, props = GB.get_residual_pure(
                float(beta), float(delta), self.params,
                return_bubble_props=True, dMdt_guess=self.seed)
        finally:
            signal.alarm(0)
        if props is None:
            raise _ArmAbort(f'structure failure at ({beta:.4f},{delta:.4f})')
        dm = float(props.bubble_dMdt)
        if not (np.isfinite(dm) and dm > 0):
            raise _ArmAbort(f'invalid dMdt={dm} at ({beta:.4f},{delta:.4f})')
        self.seed = dm
        det = GB.get_residual_detailed(float(beta), float(delta), self.params,
                                       bubble_props=props)
        gE = (det.Edot_from_beta - det.Edot_from_balance) / self.Lm
        gT = float(fT)  # T component identical in f and g
        return dict(fE=float(fE), fT=float(fT), gE=float(gE), gT=float(gT),
                    f=float(fE) ** 2 + float(fT) ** 2,
                    g=float(gE) ** 2 + float(gT) ** 2)


def _grid_axes(bc, dc, lo_b, hi_b, lo_d, hi_d):
    b0, b1 = max(lo_b, bc - GB.GRID_EPSILON), min(hi_b, bc + GB.GRID_EPSILON)
    d0, d1 = max(lo_d, dc - GB.GRID_EPSILON), min(hi_d, dc + GB.GRID_EPSILON)
    return np.linspace(b0, b1, GB.GRID_SIZE), np.linspace(d0, d1, GB.GRID_SIZE)


def _scan_order():
    c = (GB.GRID_SIZE - 1) // 2
    return sorted(((i, j) for i in range(GB.GRID_SIZE) for j in range(GB.GRID_SIZE)),
                  key=lambda ij: ((ij[0] - c) ** 2 + (ij[1] - c) ** 2, ij[0], ij[1]))


def _grid_pass(ev, bc, dc, rails, best):
    """One 5x5 center-out g-ranked pass around (bc, dc); returns best, updated."""
    (lo_b, hi_b), (lo_d, hi_d) = rails
    bs, ds = _grid_axes(bc, dc, lo_b, hi_b, lo_d, hi_d)
    for i, j in _scan_order():
        b, d = float(bs[i]), float(ds[j])
        if best is not None and abs(b - best['beta']) < 1e-12 \
                and abs(d - best['delta']) < 1e-12:
            continue
        try:
            r = ev(b, d)
        except (_ArmAbort, _PointTimeout):
            continue  # grid arms skip failed points, as production does
        except Exception:
            continue
        if best is None or r['g'] < best['g']:
            best = dict(beta=b, delta=d, **r)
        if r['g'] < EARLY:
            break
    return best


def _arm_B(ev, bg, dg, r0):
    best = dict(beta=bg, delta=dg, **r0)
    if r0['g'] < THR:
        return best, dict(short_circuit=True)
    best = _grid_pass(ev, bg, dg,
                      ((GB.BETA_MIN, GB.BETA_MAX), (GB.DELTA_MIN, GB.DELTA_MAX)),
                      best)
    return best, dict(short_circuit=False,
                      would_fallback=best['g'] > GB.LBFGSB_FALLBACK_THRESHOLD)


def _arm_C(ev, bg, dg, r0):
    best = dict(beta=bg, delta=dg, **r0)
    if r0['g'] < THR:
        return best, dict(short_circuit=True, rescans=0)
    rails = (WIDE_BETA, WIDE_DELTA)
    bc, dc = bg, dg
    rescans = 0
    t0 = time.perf_counter()
    budget_exceeded = False
    for rescans in range(1, C_MAX_RESCANS + 1):
        best = _grid_pass(ev, bc, dc, rails, best)
        if best['g'] < THR:
            break
        if time.perf_counter() - t0 > C_WALL_BUDGET_S:
            budget_exceeded = True
            break
        on_edge = (abs(abs(best['beta'] - bc) - GB.GRID_EPSILON) < 1e-9
                   or abs(abs(best['delta'] - dc) - GB.GRID_EPSILON) < 1e-9)
        if not on_edge:
            break  # optimum interior to the window: converged as far as C goes
        bc, dc = best['beta'], best['delta']
    return best, dict(short_circuit=False, rescans=rescans,
                      budget_exceeded=budget_exceeded,
                      left_old_box=not _in_old_box(best['beta'], best['delta']))


def _arm_D(ev, bg, dg, r0):
    if r0['g'] < THR:
        return dict(beta=bg, delta=dg, **r0), dict(short_circuit=True, ier=0)
    cache = {}

    def gvec(x):
        r = ev(x[0], x[1])
        cache[(float(x[0]), float(x[1]))] = r
        return [r['gE'], r['gT']]

    sol = scipy_root(gvec, [bg, dg], method='hybr', options=HYBR_OPTS)
    b, d = float(sol.x[0]), float(sol.x[1])
    r = cache.get((b, d)) or ev(b, d)
    return dict(beta=b, delta=d, **r), dict(
        short_circuit=False, ier=int(sol.status), success=bool(sol.success),
        left_old_box=not _in_old_box(b, d))


def _emit(out, arm, seg, t_now, sol, extra, n_evals, wall, error=None):
    rec = dict(arm=arm, segment=int(seg), t_now=float(t_now),
               n_evals=int(n_evals), wall=round(wall, 3))
    if sol is not None:
        rec.update(beta=sol['beta'], delta=sol['delta'],
                   f=sol['f'], g=sol['g'],
                   conv_f=sol['f'] < THR, conv_g=sol['g'] < THR)
    if error is not None:
        rec['error'] = error
    rec.update(extra or {})
    out.write(json.dumps(
        rec, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
    out.flush()


def _make_wrapper(orig, out, stride):
    state = {'seg': 0}

    def wrapper(beta_guess, delta_guess, params, method='grid'):
        t0 = time.perf_counter()
        res = orig(beta_guess, delta_guess, params, method)
        wall_A = time.perf_counter() - t0
        state['seg'] += 1
        seg = state['seg']
        if seg % stride:
            return res
        t_now = params['t_now'].value
        g_A = float('nan')
        if res.bubble_properties is not None:
            det = GB.get_residual_detailed(res.beta, res.delta, params,
                                           bubble_props=res.bubble_properties)
            gE = ((det.Edot_from_beta - det.Edot_from_balance)
                  / float(params['Lmech_total'].value))
            g_A = gE ** 2 + float(res.T_residual) ** 2
        _emit(out, 'A', seg, t_now,
              dict(beta=res.beta, delta=res.delta,
                   f=res.total_residual, g=g_A),
              dict(short_circuit=res.iterations == 0),
              res.iterations, wall_A)
        seed = None
        if res.bubble_properties is not None:
            dm = float(res.bubble_properties.bubble_dMdt)
            if np.isfinite(dm) and dm > 0:
                seed = dm
        for arm, fn in (('B', _arm_B), ('C', _arm_C), ('D', _arm_D)):
            ev = _Eval(params, seed)
            t0 = time.perf_counter()
            try:
                r0 = ev(beta_guess, delta_guess)
                sol, extra = fn(ev, beta_guess, delta_guess, r0)
                _emit(out, arm, seg, t_now, sol, extra, ev.n,
                      time.perf_counter() - t0)
            except (_ArmAbort, _PointTimeout, Exception) as e:
                _emit(out, arm, seg, t_now, None,
                      None, ev.n, time.perf_counter() - t0,
                      error=f'{type(e).__name__}: {e}')
        return res

    return wrapper


def main():
    parampath, outpath = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    signal.signal(signal.SIGALRM, _alarm)
    params = read_param.read_param(parampath)
    orig = RIP.solve_betadelta_pure
    with open(outpath, 'w') as out:
        RIP.solve_betadelta_pure = _make_wrapper(orig, out, stride)
        try:
            tmain.start_expansion(params)
        finally:
            RIP.solve_betadelta_pure = orig
    print(f'arms complete -> {outpath}')


if __name__ == '__main__':
    main()
