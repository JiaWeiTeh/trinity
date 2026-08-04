#!/usr/bin/env python3
"""Patched TRINITY runner for the phase-1a investigation.

Usage: python runner.py <param file>

Env vars (all optional; values in Myr unless noted):
  TRIN_SEG_DUR          run_energy_phase.SEGMENT_DURATION
  TRIN_TFINAL           run_energy_phase.TFINAL_ENERGY_PHASE
  TRIN_DT_EXIT          run_energy_phase.DT_EXIT_THRESHOLD
  TRIN_RTOL, TRIN_ATOL  run_energy_phase.RTOL / ATOL
  TRIN_NO_EARLY_APPROX  "1" -> params['EarlyPhaseApproximation'] = False
  TRIN_1B_DT_INIT/MIN/MAX  phase 1b DT_SEGMENT_* (MIN also rescales ODE_MAX_STEP)
  TRIN_LOGSEG           eps -> phase-1a segments become log-spaced: dt = eps*t_now
                        (prototype of adaptive segmenting; replaces SEGMENT_DURATION
                        via an object whose __radd__ hooks `t_now + SEGMENT_DURATION`)
"""
import os
import sys

sys.path.insert(0, '/home/user/trinity')

from trinity._input import read_param  # noqa: E402
from trinity._functions.logging_setup import setup_logging  # noqa: E402
import trinity.phase1_energy.run_energy_phase as p1a  # noqa: E402
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as p1b  # noqa: E402


def main():
    params = read_param.read_param(sys.argv[1])

    patches = []

    def patch(mod, name, env, cast=float):
        val = os.environ.get(env)
        if val is not None:
            setattr(mod, name, cast(val))
            patches.append(f"{mod.__name__.split('.')[-1]}.{name}={cast(val):.3e}")

    patch(p1a, 'SEGMENT_DURATION', 'TRIN_SEG_DUR')
    patch(p1a, 'TFINAL_ENERGY_PHASE', 'TRIN_TFINAL')
    patch(p1a, 'DT_EXIT_THRESHOLD', 'TRIN_DT_EXIT')
    patch(p1a, 'RTOL', 'TRIN_RTOL')
    patch(p1a, 'ATOL', 'TRIN_ATOL')
    patch(p1b, 'DT_SEGMENT_INIT', 'TRIN_1B_DT_INIT')
    patch(p1b, 'DT_SEGMENT_MIN', 'TRIN_1B_DT_MIN')
    patch(p1b, 'DT_SEGMENT_MAX', 'TRIN_1B_DT_MAX')
    if os.environ.get('TRIN_1B_DT_MIN') is not None:
        p1b.ODE_MAX_STEP = p1b.DT_SEGMENT_MIN / 5
        patches.append(f"ODE_MAX_STEP={p1b.ODE_MAX_STEP:.3e}")

    if os.environ.get('TRIN_NO_EARLY_APPROX') == '1':
        params['EarlyPhaseApproximation'].value = False
        patches.append("EarlyPhaseApproximation=False")

    logseg = os.environ.get('TRIN_LOGSEG')
    if logseg is not None:
        class LogSeg:
            """Stands in for the SEGMENT_DURATION constant: `t_now + SEGMENT_DURATION`
            resolves via __radd__ to t_now*(1+eps), giving log-spaced segments."""
            def __init__(self, eps):
                self.eps = eps

            def __radd__(self, t_now):
                return t_now + max(self.eps * t_now, 1e-12)

            def __truediv__(self, k):  # retry path: SEGMENT_DURATION / 10
                return LogSeg(self.eps / k)

        p1a.SEGMENT_DURATION = LogSeg(float(logseg))
        patches.append(f"SEGMENT_DURATION=LogSeg(eps={float(logseg)})")

    logger = setup_logging(
        log_level=params['log_level'].value if 'log_level' in params else 'INFO',
        console_output=True,
        file_output=True,
        log_file_path=params['path2output'].value,
        log_file_name='trinity.log',
        use_colors=False,
    )
    logger.info(f"runner.py patches: {patches if patches else 'none (stock constants)'}")

    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    gmc_check = validate_gmc_from_params(params)
    for w in gmc_check.warnings:
        logger.warning(w)
    if not gmc_check.valid:
        for e in gmc_check.errors:
            logger.error(e)
        sys.exit("GMC validation failed")

    from trinity import main as trinity_main
    trinity_main.start_expansion(params)


if __name__ == '__main__':
    main()
