#!/usr/bin/env python3
"""Batch 2 (docs/dev/switchon-successor/PLAN.md §5): candidate S1 — replace the
fixed 1e-3 Myr ramp window with the physical one the code already computes.

`dt_switchon = 1e-3` Myr is an absolute clock. The free-expansion time
`dt_phase0` — the moment the swept-up mass equals the ejected wind mass, i.e.
when the termination shock actually forms — is computed per run in
`phase0_init/get_InitPhaseParam.py` and spans 0.0115-1.96 yr across the screen
configs, so the shipped ramp runs 500-87,000x longer than the physics it
imitates. S1 sets the window to `k * dt_phase0` instead.

`dt_phase0` is not persisted to params, but `t0 = tSF + dt_phase0` is the run's
first integration time, so it is recovered from the first RHS call — no
production change is needed to test the candidate.

Batch 1 (PLAN §3, D1) predicts this fails, and says why: at the point the
solution reaches the Weaver work partition the ramp is still ~99.94%
suppressing, so closing it at `k*dt_phase0` releases `R1` while the state still
needs the suppression. Running it anyway converts that expectation into a
measurement -- which is the whole reason this candidate is cheap and first.

    python docs/dev/switchon-successor/harness/s1_physical_clock.py \\
        --config simple_cluster --k 1.0 --stop-t 0.02 --workdir <dir>
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
    p.add_argument("--k", type=float, default=1.0, help="window = k * dt_phase0")
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
    from trinity._input import read_param
    from trinity._functions.logging_setup import setup_logging
    import trinity.bubble_structure.get_bubbleParams as get_bubbleParams

    params = read_param.read_param(param)
    orig = get_bubbleParams.get_effective_bubble_pressure
    state = {"t0": None}

    def s1(*args, **kw):
        phase = kw.get("current_phase", args[0] if args else None)
        t, tSF = kw.get("t"), kw.get("tSF")
        if phase in ("momentum", "transition") or t is None or tSF is None:
            return orig(*args, **kw)
        if state["t0"] is None:                 # first call is at t0 = tSF + dt_phase0
            state["t0"] = t
            dt0 = t - tSF
            import logging
            logging.getLogger(__name__).info(
                f"S1: dt_phase0 = {dt0:.6e} Myr ({dt0 * 1e6:.4g} yr); window = "
                f"k*dt_phase0 = {a.k * dt0:.6e} Myr, vs the shipped 1e-3 Myr "
                f"({1e-3 / (a.k * dt0):.4g}x shorter)")
        tmin = a.k * (state["t0"] - tSF)
        kw = dict(kw)
        if tmin > 0 and t <= tSF + tmin:
            kw["R1"] = (t - tSF) / tmin * kw["R1"]
        kw["t"] = None                          # disable the shipped 1e-3 ramp
        return orig(*args, **kw)

    get_bubbleParams.get_effective_bubble_pressure = s1

    setup_logging(params["log_level"].value, False, params["path2output"].value,
                  log_file_name="trinity.log", use_colors=False)
    import logging
    logging.getLogger(__name__).info(
        f"s1_physical_clock: config={a.config} k={a.k}; production source unmodified")

    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    check = validate_gmc_from_params(params)
    if not check.valid:
        sys.exit(f"GMC validation failed: {check.errors}")

    from trinity import main as trinity_main
    try:
        trinity_main.start_expansion(params)
    finally:
        out = os.path.join(params["path2output"].value, "metadata.json")
        if os.path.exists(out):
            term = (json.load(open(out)).get("termination") or {})
            print(f"{a.config}: {term.get('exit_code')} {term.get('outcome')}")


if __name__ == "__main__":
    sys.exit(main())
