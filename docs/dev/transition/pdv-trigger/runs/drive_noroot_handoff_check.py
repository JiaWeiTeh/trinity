#!/usr/bin/env python3
"""Runnable check for the no-physical-root => momentum handoff (KAPPA_FREEZE_MECHANISM fix #1).

Runs a full simulation with NO_ROOT_HANDOFF_STREAK monkeypatched small so the handoff is
exercised in minutes (production threshold is 50; healthy rejection bursts are <= 8 segments,
and a full local freeze takes too long for a check). Mirrors run.py::run_single; production
code untouched -- only a module attribute is set at runtime.

Usage:
    python docs/dev/transition/pdv-trigger/runs/drive_noroot_handoff_check.py <param> <streak>

Pass a param that hits the dMdt<0 gate early (simple_cluster + cooling_boost_kappa 8 rejects
from segment 2 at t~3.4e-3 Myr) and a small streak (e.g. 3). PASS = the log shows
"no_physical_root_handoff" and the run continues into the transition/momentum phases and
terminates with a proper end code instead of freezing to max_segments.
"""
import sys

PARAM = sys.argv[1]
STREAK = int(sys.argv[2])

from trinity._input import read_param
from trinity._output import header

params = read_param.read_param(PARAM)
header.show_param(params)

from trinity import main
from trinity._functions.logging_setup import setup_logging

setup_logging(
    log_level='INFO',
    console_output=True,
    file_output=True,
    log_file_path=params['path2output'].value,
    log_file_name='trinity.log',
    use_colors=True,
)

from trinity.phase1b_energy_implicit import run_energy_implicit_phase as ri
ri.NO_ROOT_HANDOFF_STREAK = STREAK
print(f"[driver] NO_ROOT_HANDOFF_STREAK monkeypatched to {ri.NO_ROOT_HANDOFF_STREAK}")

main.start_expansion(params)
print("[driver] start_expansion returned cleanly")
