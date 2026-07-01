#!/usr/bin/env python3
"""VALIDATE the emergent-θ calibration: run production `multiplier` (f_κ) mode and OBSERVE the emergent θ.

Corrected direction (FINDINGS §8c, F_KAPPA §14): θ is an output. This runs the *real* production path
(`cooling_boost_mode='multiplier'`, `cooling_boost_fmix=FK`) — no behaviour change — and wraps
`effective_Lloss_from_params` only to RECORD θ = L_loss/L_mech each step. Reports the PEAK emergent θ
(θ_max, the first-crossing-relevant metric) over a ≥5 Myr run, NOT θ at blowout (the flagged-questionable epoch,
F_KAPPA §10). Compares to the §14 prediction: fires iff θ_max ≥ 0.95.

Usage:  FK=8 STOP_T=6 OUT_BASE=outputs/fkappa_val python _fkappa_validation_runner.py <config.param> <name>
"""
import atexit
import json
import os
import sys

config = sys.argv[1]
name = sys.argv[2]
FK = float(os.environ.get("FK", "8.0"))
OUT_BASE = os.environ.get("OUT_BASE", "outputs/fkappa_val")
STOP_T = float(os.environ.get("STOP_T", "6.0"))
TRIGGER = os.environ.get("TRIGGER", "cooling_balance")

import trinity.phase1b_energy_implicit.get_betadelta as gbd          # noqa: E402
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rip  # noqa: E402

_real = gbd.effective_Lloss_from_params           # the REAL production function (multiplier logic lives here)
_diag = {"config": name, "f_kappa": FK, "trigger": TRIGGER,
         "theta_max": 0.0, "theta_min": 1.0, "n_calls": 0, "theta_at_first_fire": None}


def _observing_effective_Lloss(params, Lcool, Lleak, Lmech):
    val = _real(params, Lcool, Lleak, Lmech)      # call through -- behaviour UNCHANGED
    if Lmech and Lmech > 0:
        theta = val / Lmech
        _diag["n_calls"] += 1
        _diag["theta_max"] = max(_diag["theta_max"], theta)
        _diag["theta_min"] = min(_diag["theta_min"], theta)
        if _diag["theta_at_first_fire"] is None and theta >= 0.95:
            _diag["theta_at_first_fire"] = theta
    return val


gbd.effective_Lloss_from_params = _observing_effective_Lloss
rip.effective_Lloss_from_params = _observing_effective_Lloss

from trinity._input import read_param                                # noqa: E402
from trinity._functions.logging_setup import setup_logging           # noqa: E402
from trinity import main                                            # noqa: E402

params = read_param.read_param(config)
outdir = os.path.join(OUT_BASE, name)
params["path2output"].value = outdir
params["cooling_boost_mode"].value = "multiplier"
params["cooling_boost_fmix"].value = FK
params["transition_trigger"].value = TRIGGER
if "stop_t" in params:
    params["stop_t"].value = max(float(params["stop_t"].value), STOP_T)
if "log_console" in params:
    params["log_console"].value = False
os.makedirs(outdir, exist_ok=True)


@atexit.register
def _dump():
    try:
        with open(os.path.join(outdir, "fkappa_val_diag.json"), "w") as fh:
            json.dump(_diag, fh, indent=2)
    except Exception:
        pass


setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"), console_output=False, file_output=True,
              log_file_path=outdir, log_file_name="trinity.log", use_colors=False)
main.start_expansion(params)
print(f"[{name}] f_kappa={FK}  theta_max={_diag['theta_max']:.3f}  "
      f"fires={'YES' if _diag['theta_max'] >= 0.95 else 'no'}  ({_diag['n_calls']} calls)")
