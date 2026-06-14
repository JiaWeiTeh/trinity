#!/usr/bin/env python3
"""One-off: traverse to mock segment 1, then evaluate arm D's claimed root
(0.23009, -0.35672) under different dMdt seeds to check for false zeros /
seed-dependence. Scratch-only."""
import signal
import sys

import numpy as np

import trinity.phase1b_energy_implicit.get_betadelta as GB
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as RIP
from trinity import main as tmain
from trinity._input import read_param

B, D = 0.23009298088845212, -0.3567204150943865


class _Done(Exception):
    pass


class _PointTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _PointTimeout()


def _eval(tag, beta, delta, params, seed):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(20)
    try:
        fE, fT, props = GB.get_residual_pure(beta, delta, params,
                                             return_bubble_props=True,
                                             dMdt_guess=seed)
    except _PointTimeout:
        print(f'{tag}: TIMEOUT (20s)', flush=True)
        return None
    finally:
        signal.alarm(0)
    dm = float(props.bubble_dMdt) if props is not None else float('nan')
    det = None
    if props is not None:
        det = GB.get_residual_detailed(beta, delta, params, bubble_props=props)
    print(f'{tag}: fE={fE:.6e} fT={fT:.6e} dMdt={dm:.6e}'
          + (f' E1={det.Edot_from_beta:.6e} E2={det.Edot_from_balance:.6e}'
             f' T1={det.T_bubble:.6e} T0={det.T0:.6e}' if det else ' (no props)'),
          flush=True)
    return dm


def main():
    params = read_param.read_param(sys.argv[1])
    orig = RIP.solve_betadelta_pure
    state = {'n': 0}

    def wrapper(bg, dg, p, method='grid'):
        res = orig(bg, dg, p, method)
        state['n'] += 1
        if state['n'] == 1:
            seed = None
            if res.bubble_properties is not None:
                dmp = float(res.bubble_properties.bubble_dMdt)
                if np.isfinite(dmp) and dmp > 0:
                    seed = dmp
            print(f'accepted: beta={res.beta} delta={res.delta} seed={seed}',
                  flush=True)
            dm1 = _eval('seed=accepted ', B, D, p, seed)
            _eval('seed=None     ', B, D, p, None)
            _eval('seed=prev-eval', B, D, p, dm1)
            # neighborhood: is the zero isolated or a plateau?
            for db, dd in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3),
                           (0.05, 0), (0, 0.05)):
                _eval(f'offset({db:+.0e},{dd:+.0e})', B + db, D + dd, p, dm1)
            raise _Done()
        return res

    RIP.solve_betadelta_pure = wrapper
    try:
        tmain.start_expansion(params)
    except _Done:
        print('done')
    finally:
        RIP.solve_betadelta_pure = orig


if __name__ == '__main__':
    main()
