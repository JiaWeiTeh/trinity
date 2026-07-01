#!/usr/bin/env python3
"""STAGE-A BASELINE worker: run ONE config end-to-end with STOCK trinity (no theta_elbadry patch).

Companion to _theta_elbadry_runner.py. Identical plumbing (separate process, stop_t >= STOP_T,
console off) but it does NOT monkeypatch effective_Lloss_from_params and keeps the production-default
transition trigger. Used to isolate whether a fate seen in the shadow run (e.g. SHELL_COLLAPSED on a
dense cloud) is introduced by imposing the El-Badry theta or is simply stock TRINITY's native fate.

Usage:  STOP_T=6 OUT_BASE=outputs/baseline_te python _baseline_runner.py <config.param> <name>
"""
import os
import sys

config = sys.argv[1]
name = sys.argv[2]
OUT_BASE = os.environ.get("OUT_BASE", "outputs/baseline_te")
STOP_T = float(os.environ.get("STOP_T", "6.0"))

from trinity._input import read_param                                # noqa: E402
from trinity._functions.logging_setup import setup_logging           # noqa: E402
from trinity import main                                            # noqa: E402

params = read_param.read_param(config)
outdir = os.path.join(OUT_BASE, name)
params["path2output"].value = outdir
# leave transition_trigger at its production default (cooling_balance)
if "stop_t" in params:
    params["stop_t"].value = max(float(params["stop_t"].value), STOP_T)
if "log_console" in params:
    params["log_console"].value = False

os.makedirs(outdir, exist_ok=True)
setup_logging(log_level="INFO", console_output=False, file_output=True,
              log_file_path=outdir, log_file_name="trinity.log", use_colors=False)
main.start_expansion(params)
print(f"[{name}] baseline done -> {outdir}")
