#!/usr/bin/env python3
"""STAGE-A SHADOW (gated): impose El-Badry theta ONLY where radiative cooling plausibly dominates.

Extends _theta_elbadry_runner.py with a REGIME GATE (FINDINGS sec8b): El-Badry's theta is a *radiative*
loss ratio, but the high-mass turnover is PdV/inertia-driven -- imposing theta there double-counts the loss
(PdV is already in Edot_from_balance = Lgain - Lloss - 4*pi*R2^2*v2*Pb) and drives the bubble to recollapse,
reversing PR#715's momentum handoff. This gate skips the theta imposition when PdV/Lmech exceeds a threshold,
deferring PdV-dominated clouds to the default path (native cooling + the Eb<=0 -> momentum handoff).

Gate proxy: PdV/Lmech from the LAST-COMMITTED params (R2, v2, Pb) -- not the residual's trial state, but a
faithful regime indicator. effective_Lloss_from_params only receives (params, Lcool, Lleak, Lmech), so R2/v2/
Pb are read off params.

Usage:
  LDV=3 THETA_MAX=0.99 PDV_GATE=0.7 TRIGGER=cooling_balance STOP_T=6 OUT_BASE=outputs/shadow_gate \
    python _theta_elbadry_gated_runner.py <config.param> <name>
"""
import atexit
import json
import math
import os
import sys

config = sys.argv[1]
name = sys.argv[2]
LDV = float(os.environ.get("LDV", "3.0"))
THETA_MAX = float(os.environ.get("THETA_MAX", "0.99"))
PDV_GATE = float(os.environ.get("PDV_GATE", "0.7"))     # gate OFF theta when PdV/Lmech > this
TRIGGER = os.environ.get("TRIGGER", "cooling_balance")  # default: no ebpeak (ebpeak also mis-fires on PdV)
OUT_BASE = os.environ.get("OUT_BASE", "outputs/shadow_gate")
STOP_T = float(os.environ.get("STOP_T", "6.0"))
A_MIX = 3.5

import trinity.phase1b_energy_implicit.get_betadelta as gbd          # noqa: E402
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rip  # noqa: E402
import trinity._functions.unit_conversions as cvt                    # noqa: E402
from trinity.cloud_properties.density_profile import get_density_profile  # noqa: E402

_diag = {"n_calls": 0, "n_gated": 0, "n_resolved_wins": 0, "theta_min": 1.0, "theta_max_seen": 0.0,
         "pdv_ratio_max": 0.0, "config": name, "lambda_dv": LDV, "theta_max": THETA_MAX,
         "pdv_gate": PDV_GATE, "trigger": TRIGGER, "A_mix": A_MIX}


def _val(params, key):
    p = params.get(key, None)
    v = getattr(p, "value", None) if p is not None else None
    return v if isinstance(v, (int, float)) else None


def _theta_elbadry_gated(params, Lcool, Lleak, Lmech):
    resolved = Lcool + Lleak
    R2 = _val(params, "R2")
    if not R2 or R2 <= 0 or LDV <= 0:
        return resolved
    # --- REGIME GATE: skip theta where PdV dominates the budget (defer to default handoff) ---
    v2 = _val(params, "v2")
    Pb = _val(params, "Pb")
    if Lmech and Lmech > 0 and v2 is not None and Pb is not None:
        pdv = 4.0 * math.pi * R2 ** 2 * v2 * Pb
        ratio = pdv / Lmech
        _diag["pdv_ratio_max"] = max(_diag["pdv_ratio_max"], ratio)
        if ratio > PDV_GATE:
            _diag["n_calls"] += 1
            _diag["n_gated"] += 1
            return resolved            # PdV-dominated -> DON'T impose theta
    # --- otherwise impose the El-Badry theta (radiative regime) ---
    n_amb = float(get_density_profile(R2, params)) * cvt.ndens_au2cgs
    if not (n_amb > 0):
        return resolved
    X = A_MIX * (LDV * n_amb) ** 0.5
    theta = X / (11.0 / 5.0 + X)
    if theta > THETA_MAX:
        theta = THETA_MAX
    boosted = theta * Lmech
    _diag["n_calls"] += 1
    if resolved > boosted:
        _diag["n_resolved_wins"] += 1
    _diag["theta_min"] = min(_diag["theta_min"], theta)
    _diag["theta_max_seen"] = max(_diag["theta_max_seen"], theta)
    return max(resolved, boosted)


gbd.effective_Lloss_from_params = _theta_elbadry_gated
rip.effective_Lloss_from_params = _theta_elbadry_gated

from trinity._input import read_param                                # noqa: E402
from trinity._functions.logging_setup import setup_logging           # noqa: E402
from trinity import main                                            # noqa: E402

params = read_param.read_param(config)
outdir = os.path.join(OUT_BASE, name)
params["path2output"].value = outdir
params["transition_trigger"].value = TRIGGER
if "stop_t" in params:
    params["stop_t"].value = max(float(params["stop_t"].value), STOP_T)
if "log_console" in params:
    params["log_console"].value = False
os.makedirs(outdir, exist_ok=True)


@atexit.register
def _dump_diag():
    _diag["gated_frac"] = (_diag["n_gated"] / _diag["n_calls"]) if _diag["n_calls"] else float("nan")
    try:
        with open(os.path.join(outdir, "theta_gated_diag.json"), "w") as fh:
            json.dump(_diag, fh, indent=2)
    except Exception:
        pass


setup_logging(log_level="INFO", console_output=False, file_output=True,
              log_file_path=outdir, log_file_name="trinity.log", use_colors=False)
main.start_expansion(params)
print(f"[{name}] gated done -> {outdir}  (gated {_diag['n_gated']}/{_diag['n_calls']}, "
      f"pdv_ratio_max {_diag['pdv_ratio_max']:.2f}, theta {_diag['theta_min']:.3f}-{_diag['theta_max_seen']:.3f})")
