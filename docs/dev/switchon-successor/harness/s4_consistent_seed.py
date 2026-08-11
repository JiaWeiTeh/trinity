#!/usr/bin/env python3
"""Batch 4b (docs/dev/switchon-successor/PLAN.md §5, "Batch 4 pre-registration"):
candidate S4 — fix the handover state instead of ramping `R1`, and run with **no
ramp at all**.

The pre-registration derives, and `harness/s4_seed_anatomy.py` verifies on all
five configs, that the handover's work rate is algebraic:

    PdV / Lmech = 2 (v2/v_wind) / (R1/R2)**2       (R1 at ram-pressure balance)

`E0` is absent from it, so the only lever is `v2/v_wind` — which `get_y0` pins at
exactly 1 by handing over the free-streaming wind terminal speed. Two variants
change **only the returned `v0`**:

    similarity   v0 = (3/5) * v_wind      -> PdV/Lmech = 1.588  (predicted to fail)
    sustain      v0 = (x**2/2) * v_wind   -> PdV/Lmech = 1.000  (the equality)

`r0`, `E0`, `T0` and `dt_phase0` are left exactly as phase 0 computed them, so
phase 0's published behaviour is unchanged; `x` for the `sustain` variant is read
from the run's own `solve_R1` at its own seed, not from a stored number.
Production source is NOT modified — both the seed and the ramp are patched here.

    python docs/dev/switchon-successor/harness/s4_consistent_seed.py \\
        --config simple_cluster --variant sustain --stop-t 0.02 --workdir <dir>
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))


def load_screen():
    spec = importlib.util.spec_from_file_location(
        "screen", os.path.join(REPO, "docs", "dev", "screen", "screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True, choices=("similarity", "sustain"))
    p.add_argument("--stop-t", type=float, default=0.02)
    p.add_argument("--workdir", required=True)
    a = p.parse_args()

    screen = load_screen()
    workdir = os.path.abspath(a.workdir)
    os.makedirs(workdir, exist_ok=True)
    param = os.path.join(workdir, "p.param")
    screen.write_param(screen.CONFIGS[a.config], param, a.stop_t, "screen")
    os.chdir(workdir)

    sys.path.insert(0, REPO)
    import logging
    from trinity._input import read_param
    from trinity._functions.logging_setup import setup_logging
    from trinity.phase0_init import get_InitPhaseParam
    from trinity.sps import update_feedback
    import trinity.bubble_structure.get_bubbleParams as get_bubbleParams

    params = read_param.read_param(param)

    # --- the seed patch -----------------------------------------------------
    orig_y0 = get_InitPhaseParam.get_y0
    seed_note = {}

    def patched_y0(p_):
        t0, r0, v0, E0, T0 = orig_y0(p_)
        fb = update_feedback.get_current_sps_feedback(t0, p_)
        R1 = get_bubbleParams.solve_R1(r0, E0, fb.Lmech_total, fb.v_mech_total)
        x = R1 / r0
        factor = 0.6 if a.variant == "similarity" else x ** 2 / 2.0
        v_new = factor * fb.v_mech_total
        seed_note.update(x=x, factor=factor, v0_head=v0, v0_new=v_new,
                         v_wind=fb.v_mech_total,
                         pdv_over_L=2.0 * (v_new / fb.v_mech_total) / x ** 2)
        logging.getLogger(__name__).info(
            f"S4[{a.variant}]: R1/R2={x:.6f}, v0 {v0:.6e} -> {v_new:.6e} pc/Myr "
            f"({factor:.6f} x v_wind); PdV/Lmech at seed = "
            f"{seed_note['pdv_over_L']:.6f}; r0/E0/T0/dt_phase0 unchanged")
        return t0, r0, v_new, E0, T0

    get_InitPhaseParam.get_y0 = patched_y0

    # --- the ramp ablation --------------------------------------------------
    orig_pb = get_bubbleParams.get_effective_bubble_pressure

    def no_ramp(*args, **kw):
        kw = dict(kw)
        kw["t"] = None                    # the 1e-3 Myr ramp never engages
        return orig_pb(*args, **kw)

    get_bubbleParams.get_effective_bubble_pressure = no_ramp

    setup_logging(params["log_level"].value, False, params["path2output"].value,
                  log_file_name="trinity.log", use_colors=False)
    logging.getLogger(__name__).info(
        f"s4_consistent_seed: config={a.config}, variant={a.variant}, ramp DISABLED; "
        f"production source unmodified")

    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    check = validate_gmc_from_params(params)
    if not check.valid:
        sys.exit(f"GMC validation failed: {check.errors}")

    from trinity import main as trinity_main
    try:
        trinity_main.start_expansion(params)
    finally:
        meta = os.path.join(params["path2output"].value, "metadata.json")
        term = {}
        if os.path.exists(meta):
            term = json.load(open(meta)).get("termination") or {}
        print(f"{a.config}/{a.variant}: {term.get('exit_code')} {term.get('outcome')} | "
              f"seed x={seed_note.get('x')} factor={seed_note.get('factor')} "
              f"PdV/L={seed_note.get('pdv_over_L')}")


if __name__ == "__main__":
    sys.exit(main())
