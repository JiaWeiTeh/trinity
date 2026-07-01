#!/usr/bin/env python3
"""CORRECT-KNOB validation: emergent θ under `cooling_boost_kappa` (the structural κ_eff, θ fully emergent).

Supersedes the `_fkappa_validation_runner.py` (multiplier) runs for validating §14: the §14 leverage/θ₀ were
fit with `cooling_boost_kappa`, which scales the Spitzer conduction coefficient INSIDE the bubble-structure ODE
(`bubble_luminosity.py:291/370/406`), so L_cool changes THROUGH the physics and θ emerges. `cooling_boost_mode`
stays 'none', so the observed θ = (L_cool+L_leak)/L_mech is the fully-emergent loss fraction. We record its PEAK
(θ_max, the first-crossing metric — 📏 θ_max standing rule) over a ≥5 Myr run.

Usage:  FK=8 STOP_T=6 OUT_BASE=outputs/kappa_val LOG_LEVEL=INFO python _kappa_validation_runner.py <cfg> <name>
"""
import atexit
import json
import os
import sys

config, name = sys.argv[1], sys.argv[2]
FK = float(os.environ.get("FK", "8.0"))
STOP_T = float(os.environ.get("STOP_T", "6.0"))
OUT_BASE = os.environ.get("OUT_BASE", "outputs/kappa_val")

import trinity.phase1b_energy_implicit.get_betadelta as gbd          # noqa: E402
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rip  # noqa: E402

_real = gbd.effective_Lloss_from_params            # mode='none' => returns L_cool + L_leak (emergent)
_diag = {"config": name, "cooling_boost_kappa": FK, "theta_max": 0.0, "theta_min": 1.0,
         "n_calls": 0, "theta_at_first_fire": None}


def _observe(params, Lcool, Lleak, Lmech):
    val = _real(params, Lcool, Lleak, Lmech)       # behaviour UNCHANGED; observe only
    if Lmech and Lmech > 0:
        theta = val / Lmech
        _diag["n_calls"] += 1
        _diag["theta_max"] = max(_diag["theta_max"], theta)
        _diag["theta_min"] = min(_diag["theta_min"], theta)
        if _diag["theta_at_first_fire"] is None and theta >= 0.95:
            _diag["theta_at_first_fire"] = theta
    return val


gbd.effective_Lloss_from_params = _observe
rip.effective_Lloss_from_params = _observe

from trinity._input import read_param                                # noqa: E402
from trinity._functions.logging_setup import setup_logging           # noqa: E402
from trinity import main                                            # noqa: E402

params = read_param.read_param(config)
outdir = os.path.join(OUT_BASE, name)
params["path2output"].value = outdir
params["cooling_boost_kappa"].value = FK           # THE structural knob (θ emerges through the structure)
params["cooling_boost_mode"].value = "none"        # keep L_loss = L_cool + L_leak (no post-hoc scaling)
params["transition_trigger"].value = "cooling_balance"
if "stop_t" in params:
    params["stop_t"].value = max(float(params["stop_t"].value), STOP_T)
if "log_console" in params:
    params["log_console"].value = False
os.makedirs(outdir, exist_ok=True)


@atexit.register
def _dump():
    try:
        json.dump(_diag, open(os.path.join(outdir, "kappa_val_diag.json"), "w"), indent=2)
    except Exception:
        pass


setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"), console_output=False, file_output=True,
              log_file_path=outdir, log_file_name="trinity.log", use_colors=False)
main.start_expansion(params)
print(f"[{name}] cooling_boost_kappa={FK} theta_max={_diag['theta_max']:.3f} "
      f"fires={'YES' if _diag['theta_max'] >= 0.95 else 'no'} ({_diag['n_calls']} calls)")
