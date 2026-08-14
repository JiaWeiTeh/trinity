#!/usr/bin/env python3
"""Batch 3 (docs/dev/switchon-successor/PLAN.md §5): candidate S2 — replace the
*clock* with a state criterion that has no free constant.

D2 retired the "better clock" family: window length is not the controlling
variable, the config's own early state is. D1 named the mechanism — the seed
does PdV work 4.85x faster than the Weaver partition implies, and the ramp is
what stops that over-work from draining `Eb`.

S2 therefore suppresses `R1` **only while the bubble would do work faster than
the wind supplies energy**, and lets go permanently the moment it would not:

    Pb_sustain = (Lmech_total - L_cool - L_leak) / (4*pi*R2**2 * v2)
    Pb_eff     = min(Pb_unramped, Pb_sustain)      while the limiter binds
    Pb_eff     = Pb_unramped                        once released (latched)

Properties that matter for the bars:

* **No free constant** (N3). The threshold is the equality `PdV = net gain`,
  which is the energy equation's own zero — not a tuned number, not a
  dimensionless factor imported from a wind-only solution (§0.3 forbids using
  Weaver's 6/11 as a target).
* **Self-releasing and scale-free.** Nothing references an absolute time,
  energy or length; a run releases when its own state allows.
* **Inert once released**, by the latch. That is deliberate: without it the
  limiter would also forbid `Eb` from *ever* declining in phase 1a, which would
  mask genuine energy-driven collapses. The latch confines it to the initial
  relaxation, which is the only thing the ramp was ever protecting.

`L_cool` and `v2` are read from `params` (frozen per segment, as all the other
driving terms in phase 1a already are). Production source is NOT modified.

    python docs/dev/switchon-successor/harness/s2_state_trigger.py \\
        --config simple_cluster --stop-t 0.02 --workdir <dir>
"""
import argparse
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
FOURPI = 4.0 * math.pi


def load_screen():
    spec = importlib.util.spec_from_file_location(
        "screen", os.path.join(REPO, "docs", "dev", "screen", "screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
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
    state = {"released": False, "t_release": None, "n_limited": 0}

    def value(key, default=0.0):
        item = params.get(key)
        v = getattr(item, "value", None) if item is not None else None
        return default if v is None else v

    def s2(*args, **kw):
        phase = kw.get("current_phase", args[0] if args else None)
        t, tSF = kw.get("t"), kw.get("tSF")
        if phase in ("momentum", "transition") or t is None or tSF is None:
            return orig(*args, **kw)

        kw = dict(kw)
        kw["t"] = None                       # disable the shipped 1e-3 Myr ramp
        pb_unramped = orig(*args, **kw)
        if state["released"]:
            return pb_unramped

        R2, v2 = kw["R2"], value("v2")
        gain = kw.get("Lmech_total") or value("Lmech_total")
        net_gain = gain - value("bubble_LTotal") - value("bubble_Leak")
        if v2 <= 0 or R2 <= 0 or net_gain <= 0:
            return pb_unramped               # nothing to sustain; do not limit

        pb_sustain = net_gain / (FOURPI * R2 ** 2 * v2)
        if pb_unramped <= pb_sustain:        # the bubble can take full pressure
            state["released"] = True
            state["t_release"] = t
            import logging
            logging.getLogger(__name__).info(
                f"S2: limiter released at t={t:.6e} Myr after {state['n_limited']} "
                f"limited calls (Pb_unramped {pb_unramped:.4e} <= Pb_sustain "
                f"{pb_sustain:.4e})")
            return pb_unramped
        state["n_limited"] += 1
        return pb_sustain

    get_bubbleParams.get_effective_bubble_pressure = s2

    setup_logging(params["log_level"].value, False, params["path2output"].value,
                  log_file_name="trinity.log", use_colors=False)
    import logging
    logging.getLogger(__name__).info(
        f"s2_state_trigger: config={a.config}; sustainability limiter with a one-way "
        f"latch, no free constant; production source unmodified")

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
        rel = state["t_release"]
        print(f"{a.config}: {term.get('exit_code')} {term.get('outcome')} | "
              f"released_at={rel if rel is None else f'{rel:.4e}'} Myr | "
              f"limited_calls={state['n_limited']}")


if __name__ == "__main__":
    sys.exit(main())
