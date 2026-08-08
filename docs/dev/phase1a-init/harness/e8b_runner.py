#!/usr/bin/env python3
"""E8b: ablate the ``dt_switchon = 1e-3`` Myr early-phase R1 ramp, and measure it.

PLAN.md §8 E8b. ``get_bubbleParams.get_effective_bubble_pressure`` linearly ramps
the inner bubble radius over the first 1e-3 Myr of a run::

    dt_switchon = 1e-3
    if t <= tSF + dt_switchon:
        R1_tmp = (t - tSF) / dt_switchon * R1     # -> bubble_E2P(Eb, R2, R1_tmp, gamma)

``bubble_E2P`` divides by the shell volume ``~ (R2**3 - R1**3)``, so a suppressed
``R1`` means a LARGER volume and therefore a LOWER driving pressure. The ramp is
an *absolute* time (magic-number audit finding #2) while phase 1a runs to
3e-3 Myr, so it shapes the driving pressure across the first third of the energy
phase — the same "absolute constant vs scale-dependent physics" class as the
SEGMENT_DURATION defect this workstream fixed.

The ablation is exact and needs no copy of the function body: the ramp is gated
on ``t is not None``, so forwarding ``t=None`` disables that branch and changes
nothing else. Production source is NOT modified — this is a measurement, and
PLAN.md §8 forbids bundling the change into the phase-1a branch.

Usage (mirrors harness/patched_runner.py). Run it on TOP of the phase-1a fix,
never against stock, or the result inherits the artifact that branch removed::

    python docs/dev/phase1a-init/harness/e8b_runner.py <param file>
"""
import sys

sys.path.insert(0, '/home/user/trinity')

from trinity._input import read_param  # noqa: E402
from trinity._functions.logging_setup import setup_logging  # noqa: E402
import trinity.bubble_structure.get_bubbleParams as get_bubbleParams  # noqa: E402


def main():
    params = read_param.read_param(sys.argv[1])

    _orig = get_bubbleParams.get_effective_bubble_pressure

    def no_ramp(*args, **kwargs):
        # The ramp fires only when both t and tSF are supplied; dropping t skips it.
        kwargs['t'] = None
        return _orig(*args, **kwargs)

    get_bubbleParams.get_effective_bubble_pressure = no_ramp

    setup_logging(
        params['log_level'].value,
        params['log_console'].value,
        params['path2output'].value,
        log_file_name='trinity.log',
        use_colors=False,
    )
    import logging
    logging.getLogger(__name__).info(
        "e8b_runner: dt_switchon R1 ramp ABLATED (t=None forwarded to "
        "get_effective_bubble_pressure); production source unmodified"
    )

    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    gmc_check = validate_gmc_from_params(params)
    for w in gmc_check.warnings:
        logging.getLogger(__name__).warning(w)
    if not gmc_check.valid:
        for e in gmc_check.errors:
            logging.getLogger(__name__).error(e)
        sys.exit("GMC validation failed")

    from trinity import main as trinity_main
    trinity_main.start_expansion(params)


if __name__ == '__main__':
    main()
