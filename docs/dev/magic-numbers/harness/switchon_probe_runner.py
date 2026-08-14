#!/usr/bin/env python3
"""R2+R3 (SWEEP2_PLAN.md §4): ablate the ``dt_switchon`` R1 ramp AND instrument
the stall it protects against.

Same exact ablation as ``docs/dev/phase1a-init/harness/e8b_runner.py`` (forward
``t=None`` so the ramp branch is skipped; production source untouched), plus a
wrapper around every ``bubble_luminosity.get_bubbleproperties_pure`` call that
records wall time, the entry state, and — for any call slower than
``SLOW_CALL_S`` — a cProfile top-cumtime breakdown, so the stalling segment
names the function that grinds instead of just the fact of the grind.

Run wall-capped from outside (the stall is the finding, not a run to finish)::

    timeout 1200 python docs/dev/magic-numbers/harness/switchon_probe_runner.py <param>

Writes ``bubble_calls.csv`` (one row per bubble solve) and ``profiles.txt``
(top functions per slow call) into the run's ``path2output`` directory; commit
them to ``docs/dev/magic-numbers/data/`` after the run.
"""
import cProfile
import io
import os
import pstats
import signal
import sys
import time
import traceback

sys.path.insert(0, os.getcwd())

from trinity._input import read_param  # noqa: E402
from trinity._functions.logging_setup import setup_logging  # noqa: E402
import trinity.bubble_structure.get_bubbleParams as get_bubbleParams  # noqa: E402
import trinity.bubble_structure.bubble_luminosity as bubble_luminosity  # noqa: E402

SLOW_CALL_S = 5.0


def main():
    params = read_param.read_param(sys.argv[1])
    outdir = params['path2output'].value
    os.makedirs(outdir, exist_ok=True)

    # The first probe run showed the grind is NOT in the bubble solve (all its
    # calls returned in ~1.3 s; the wall vanished between calls), so name the
    # grinding frames directly: on SIGUSR1, print the main-thread stack.
    # Python-level handler = runs between bytecodes, no watchdog race
    # (faulthandler.dump_traceback_later segfaulted mid-walk; py-spy cannot
    # ptrace in this sandbox). Sample from outside with: kill -USR1 <pid>
    stacks = open(os.path.join(outdir, 'stacks.txt'), 'w', buffering=1)

    def dump_stack(signum, frame):
        stacks.write(f"\n===== SIGUSR1 wall={time.monotonic():.0f}s =====\n")
        traceback.print_stack(frame, file=stacks)

    signal.signal(signal.SIGUSR1, dump_stack)

    # --- the e8b ablation, verbatim in effect ---
    _orig_pressure = get_bubbleParams.get_effective_bubble_pressure

    def no_ramp(*args, **kwargs):
        kwargs['t'] = None
        return _orig_pressure(*args, **kwargs)

    get_bubbleParams.get_effective_bubble_pressure = no_ramp

    # --- instrumentation around the per-segment bubble solve ---
    _orig_bubble = bubble_luminosity.get_bubbleproperties_pure
    calls_csv = open(os.path.join(outdir, 'bubble_calls.csv'), 'w', buffering=1)
    calls_csv.write("call,t_now_Myr,R2_pc,Eb_au,Pb_au,wall_s,outcome\n")
    profiles = open(os.path.join(outdir, 'profiles.txt'), 'w', buffering=1)
    state = {'n': 0}

    def probed(p):
        state['n'] += 1
        n = state['n']
        t_now = p['t_now'].value
        entry = (f"{n},{t_now!r},{p['R2'].value!r},{p['Eb'].value!r},"
                 f"{p['Pb'].value!r}")
        prof = cProfile.Profile()
        t0 = time.monotonic()
        try:
            prof.enable()
            out = _orig_bubble(p)
            prof.disable()
            calls_csv.write(f"{entry},{time.monotonic() - t0:.2f},ok\n")
            return out
        except BaseException as e:
            prof.disable()
            calls_csv.write(f"{entry},{time.monotonic() - t0:.2f},"
                            f"raise:{type(e).__name__}\n")
            raise
        finally:
            wall = time.monotonic() - t0
            if wall > SLOW_CALL_S:
                s = io.StringIO()
                pstats.Stats(prof, stream=s).sort_stats('cumulative').print_stats(14)
                profiles.write(f"\n===== call {n} t_now={t_now:.6e} Myr "
                               f"wall={wall:.1f}s =====\n{s.getvalue()}")

    bubble_luminosity.get_bubbleproperties_pure = probed

    setup_logging(
        params['log_level'].value,
        params['log_console'].value,
        outdir,
        log_file_name='trinity.log',
        use_colors=False,
    )
    import logging
    logging.getLogger(__name__).info(
        "switchon_probe_runner: ramp ABLATED (t=None) + bubble-solve "
        "instrumentation; production source unmodified"
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
