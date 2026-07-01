#!/usr/bin/env python3
"""CONFIRM the min_T boundary-transient stall diagnosis by relaxing the guard (monkeypatch, no prod edit).

DEBUG showed `_get_velocity_residuals` (bubble_luminosity.py:344) rejects EVERY candidate bubble whose min_T
dips a floating-point hair (~1e-4 K) below the T=3e4 K outer-boundary IC — a boundary transient, not real
sub-floor physics — so fsolve thrashes and the run stalls at t~0.003 Myr (large_diffuse, small_1e6 @ f_κ=8).

This raises `_T_INIT_BOUNDARY` guard by TOL below 3e4 (proxy for a guard tolerance; the IC shifts by TOL, a
~1e-6 relative change — immaterial). If the run now advances well past t=0.003 Myr and yields a θ_max, the
diagnosis + one-line fix are confirmed. Records θ_max like the validation runner.

Usage: FK=8 TOL=0.05 STOP_T=6 OUT_BASE=outputs/minT_confirm python _minT_tol_confirm_runner.py <cfg> <name>
"""
import atexit
import json
import os
import sys

config, name = sys.argv[1], sys.argv[2]
FK = float(os.environ.get("FK", "8.0"))
TOL = float(os.environ.get("TOL", "0.05"))
STOP_T = float(os.environ.get("STOP_T", "6.0"))
OUT_BASE = os.environ.get("OUT_BASE", "outputs/minT_confirm")

import trinity.bubble_structure.bubble_luminosity as bl              # noqa: E402
bl._T_INIT_BOUNDARY = 3.0e4 - TOL      # relax the min_T rejection floor by TOL (the confirmation)

import trinity.phase1b_energy_implicit.get_betadelta as gbd          # noqa: E402
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rip  # noqa: E402
_real = gbd.effective_Lloss_from_params
_diag = {"config": name, "f_kappa": FK, "minT_tol": TOL, "theta_max": 0.0, "n_calls": 0}


def _observe(params, Lcool, Lleak, Lmech):
    val = _real(params, Lcool, Lleak, Lmech)
    if Lmech and Lmech > 0:
        _diag["theta_max"] = max(_diag["theta_max"], val / Lmech)
        _diag["n_calls"] += 1
    return val


gbd.effective_Lloss_from_params = _observe
rip.effective_Lloss_from_params = _observe

from trinity._input import read_param                                # noqa: E402
from trinity._functions.logging_setup import setup_logging           # noqa: E402
from trinity import main                                            # noqa: E402

params = read_param.read_param(config)
outdir = os.path.join(OUT_BASE, name)
params["path2output"].value = outdir
params["cooling_boost_mode"].value = "multiplier"
params["cooling_boost_fmix"].value = FK
params["transition_trigger"].value = "cooling_balance"
if "stop_t" in params:
    params["stop_t"].value = max(float(params["stop_t"].value), STOP_T)
if "log_console" in params:
    params["log_console"].value = False
os.makedirs(outdir, exist_ok=True)


@atexit.register
def _dump():
    try:
        json.dump(_diag, open(os.path.join(outdir, "minT_confirm_diag.json"), "w"), indent=2)
    except Exception:
        pass


setup_logging(log_level="INFO", console_output=False, file_output=True,
              log_file_path=outdir, log_file_name="trinity.log", use_colors=False)
main.start_expansion(params)
print(f"[{name}] TOL={TOL} f_kappa={FK} theta_max={_diag['theta_max']:.3f} ({_diag['n_calls']} calls)")
