"""Offline audit harness for the bubble-luminosity solver (Phase 0).

Loads a state dumped by ``TRINITY_BUBBLE_STATE_DUMP``, reconstructs the full
``params`` -- including the runtime cooling cubes, which are not picklable
(a function-local class) and so are rebuilt *deterministically* from the
bundled tables via ``read_param`` + ``get_coolingStructure`` -- and verifies
that re-running the structure solve reproduces the dumped ``T_array``.

This is the faithful, reproducible test bed for the correctness audit
(Phases 1-3). It changes no production code.

Usage:
    python tools/bubble_audit/audit.py <state.pkl> [base.param]
    python tools/bubble_audit/audit.py <dir_of_states>/ [base.param]
"""
from __future__ import annotations

import os
import sys
import glob
import pickle

import numpy as np
import scipy.integrate
import scipy.interpolate

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from trinity._input.read_param import read_param  # noqa: E402
import trinity.cooling.non_CIE.read_cloudy as non_CIE  # noqa: E402
import trinity.bubble_structure.bubble_luminosity as bl  # noqa: E402

_DEFAULT_BASE = os.path.join(_REPO, "param", "cloud_example_PL.param")


def load_state(pkl_path: str, base_param: str = _DEFAULT_BASE):
    """Reconstruct (params, inputs, ref_arrays) from a dumped bubble state.

    The cooling cubes (skipped in the dump) are rebuilt deterministically from
    the dumped t_now / ZCloud / SB99_rotation / cooling paths.
    """
    with open(pkl_path, "rb") as fh:
        state = pickle.load(fh)

    params = read_param(base_param)
    # override every dumped scalar/array param value onto the fresh params
    n_set = 0
    for k, v in state["param_values"].items():
        if k in params:
            params[k].value = v
            n_set += 1

    # rebuild CIE cooling (loadtxt + interp1d) exactly as main.py does
    logT, logL = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    if "cStruc_cooling_CIE_logLambda" in params:
        params["cStruc_cooling_CIE_logLambda"].value = logL
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logL, kind="linear"
    )

    # rebuild non-CIE cubes deterministically (depends only on t_now/Z/rot/path)
    cooling_nonCIE, heating_nonCIE, net = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = net

    inputs = {k: state[k] for k in
              ("R1", "Pb", "bubble_dMdt", "bubble_r_Tb", "r2Prime", "initial_conditions")}
    ref = {k: state[k] for k in ("r_array", "v_array", "T_array", "dTdr_array")}
    meta = {"n_param_set": n_set, "n_skipped": len(state["skipped_param_keys"]),
            "skipped": state["skipped_param_keys"]}
    return params, inputs, ref, meta


def gate_check(params, inputs, ref):
    """Re-run the current structure solve from the dumped inputs and compare
    against the dumped T_array. A faithful reconstruction matches to ~FP.
    """
    R1, Pb, r2P = inputs["R1"], inputs["Pb"], inputs["r2Prime"]
    ic = [float(x) for x in inputs["initial_conditions"]]

    r = bl._create_radius_grid(R1, r2P)
    psoln = scipy.integrate.odeint(
        bl._get_bubble_ODE, ic, r, args=(params, Pb), tfirst=True
    )
    T_new = psoln[:, 1]
    T_ref = ref["T_array"]

    same_grid = (len(r) == len(ref["r_array"]) and
                 np.allclose(r, ref["r_array"], rtol=0, atol=0))
    n = min(len(T_new), len(T_ref))
    denom = np.maximum(np.abs(T_ref[:n]), 1e-300)
    rel = np.abs(T_new[:n] - T_ref[:n]) / denom
    return {
        "grid_len_new": len(T_new), "grid_len_ref": len(T_ref),
        "grid_identical": bool(same_grid),
        "T_max_rel": float(rel.max()), "T_median_rel": float(np.median(rel)),
    }


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    target = argv[0]
    base = argv[1] if len(argv) > 1 else _DEFAULT_BASE
    files = (sorted(glob.glob(os.path.join(target, "*.pkl")))
             if os.path.isdir(target) else [target])
    if not files:
        print(f"no .pkl states found at {target}")
        return 2
    print(f"base param: {base}\n")
    worst = 0.0
    for f in files:
        params, inputs, ref, meta = load_state(f, base)
        res = gate_check(params, inputs, ref)
        worst = max(worst, res["T_max_rel"])
        print(f"{os.path.basename(f)}")
        print(f"  params set={meta['n_param_set']} skipped(reconstructed)={meta['n_skipped']} {meta['skipped']}")
        print(f"  grid identical={res['grid_identical']} "
              f"(new={res['grid_len_new']} ref={res['grid_len_ref']})")
        print(f"  T reproduce: max_rel={res['T_max_rel']:.3e} median_rel={res['T_median_rel']:.3e}")
    print(f"\nGATE: worst T_max_rel across states = {worst:.3e}  "
          f"({'PASS' if worst < 1e-6 else 'INVESTIGATE'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
