#!/usr/bin/env python3
"""STAGE-A SHADOW worker: run ONE config end-to-end with the theta_elbadry logic MONKEYPATCHED in.

THETA_ELBADRY_SPEC.md §8 Stage A. This changes NO production code -- it patches
`effective_Lloss_from_params` at runtime (in BOTH modules that bind the name) to the §3 logic, then runs the
config via the normal pipeline. Launched once per config as a SEPARATE process (trinity leaks module-global
state in-process), so the 8 configs don't contaminate each other.

Usage:  LDV=3 THETA_MAX=0.99 OUT_BASE=outputs/shadow_te python _theta_elbadry_runner.py <config.param> <name>
"""
import atexit
import json
import os
import sys

config = sys.argv[1]
name = sys.argv[2]
LDV = float(os.environ.get("LDV", "3.0"))            # pc.km/s (calibrated)
THETA_MAX = float(os.environ.get("THETA_MAX", "0.99"))
OUT_BASE = os.environ.get("OUT_BASE", "outputs/shadow_te")
STOP_T = float(os.environ.get("STOP_T", "6.0"))      # >= 5 Myr per the standing rule
A_MIX = 3.5                                          # El-Badry+2019 Eq 37 fit

# --- monkeypatch effective_Lloss_from_params BEFORE the run --------------------------------------
import trinity.phase1b_energy_implicit.get_betadelta as gbd          # noqa: E402
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rip  # noqa: E402
import trinity._functions.unit_conversions as cvt                    # noqa: E402
from trinity.cloud_properties.density_profile import get_density_profile  # noqa: E402

_diag = {"n_calls": 0, "n_resolved_wins": 0, "theta_min": 1.0, "theta_max_seen": 0.0,
         "config": name, "lambda_dv": LDV, "theta_max": THETA_MAX, "A_mix": A_MIX}


def _theta_elbadry_effective_Lloss(params, Lcool, Lleak, Lmech):
    """The §3 theta_elbadry logic (forced ON for the shadow; production gates it on the mode)."""
    r2p = params.get("R2", None)
    R2 = r2p.value if (r2p is not None and getattr(r2p, "value", 0)) else 0.0
    if R2 <= 0 or LDV <= 0:
        return Lcool + Lleak
    n_amb = float(get_density_profile(R2, params)) * cvt.ndens_au2cgs   # pc^-3 -> cm^-3
    if not (n_amb > 0):
        return Lcool + Lleak
    X = A_MIX * (LDV * n_amb) ** 0.5
    theta = X / (11.0 / 5.0 + X)
    if theta > THETA_MAX:
        theta = THETA_MAX
    boosted = theta * Lmech
    resolved = Lcool + Lleak
    _diag["n_calls"] += 1
    if resolved > boosted:
        _diag["n_resolved_wins"] += 1            # §6 gate: where TRINITY's resolved theta wins the max()
    _diag["theta_min"] = min(_diag["theta_min"], theta)
    _diag["theta_max_seen"] = max(_diag["theta_max_seen"], theta)
    return max(resolved, boosted)


gbd.effective_Lloss_from_params = _theta_elbadry_effective_Lloss
rip.effective_Lloss_from_params = _theta_elbadry_effective_Lloss

# --- run the config via the normal pipeline ------------------------------------------------------
from trinity._input import read_param                                # noqa: E402
from trinity._functions.logging_setup import setup_logging           # noqa: E402
from trinity import main                                            # noqa: E402

params = read_param.read_param(config)
outdir = os.path.join(OUT_BASE, name)
params["path2output"].value = outdir
params["transition_trigger"].value = "cooling_balance,ebpeak"        # PdV-inclusive pairing (SPEC §5)
if "stop_t" in params:
    params["stop_t"].value = max(float(params["stop_t"].value), STOP_T)
if "log_console" in params:
    params["log_console"].value = False

os.makedirs(outdir, exist_ok=True)


@atexit.register
def _dump_diag():
    _diag["resolved_wins_frac"] = (_diag["n_resolved_wins"] / _diag["n_calls"]
                                   if _diag["n_calls"] else float("nan"))
    try:
        with open(os.path.join(outdir, "theta_elbadry_diag.json"), "w") as fh:
            json.dump(_diag, fh, indent=2)
    except Exception:
        pass


setup_logging(log_level="INFO", console_output=False, file_output=True,
              log_file_path=outdir, log_file_name="trinity.log", use_colors=False)
main.start_expansion(params)
print(f"[{name}] done -> {outdir}  (theta range {_diag['theta_min']:.3f}-{_diag['theta_max_seen']:.3f}, "
      f"resolved-wins {_diag['n_resolved_wins']}/{_diag['n_calls']})")
