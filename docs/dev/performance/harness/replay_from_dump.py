#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline replay of the F1 residual variants from a dumped state pickle.

Loads a state pickle produced by ``capture_replay_bubble.py`` (param_values +
skipped cooling-cube keys), reconstructs ``params`` with a lean loader that
mirrors ``tools/bubble_audit/audit.py:load_state`` (read_param(base) -> override
param_values -> rebuild CIE + non-CIE cooling cubes), then runs
``get_bubbleproperties_pure(params)`` under each variant (monkeypatching the
module-global ``BL._get_velocity_residuals``), records the 4 compared outputs +
timing, compares each variant to the baseline, and prints a small table.

This is the reproduction path: a future visit can re-check a captured state
without re-running the (slow) live host sim.

REPRODUCE
---------
    cd /home/user/trinity
    python docs/dev/performance/harness/replay_from_dump.py \
        docs/dev/performance/data/states/state_mock_hybr_implicit_0000.pkl \
        [base.param]
"""

import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import scipy.interpolate

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))
HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import trinity.bubble_structure.bubble_luminosity as BL  # noqa: E402
import residual_variants  # noqa: E402
from trinity._input.read_param import read_param  # noqa: E402
import trinity.cooling.non_CIE.read_cloudy as non_CIE  # noqa: E402

_DEFAULT_BASE = TRINITY_ROOT / "param" / "cloud_example_PL.param"
TIMING_REPS = int(os.environ.get("TIMING_REPS", "3"))


def load_state(pkl_path, base_param=_DEFAULT_BASE):
    """Reconstruct ``params`` from a dumped state, rebuilding the cooling cubes
    deterministically (mirrors tools/bubble_audit/audit.py:load_state:55-68)."""
    with open(pkl_path, "rb") as fh:
        state = pickle.load(fh)

    params = read_param(str(base_param))
    for k, v in state["param_values"].items():
        if k in params:
            params[k].value = v

    # rebuild CIE cooling (loadtxt + interp1d) exactly as main.py / audit.py do
    logT, logL = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    if "cStruc_cooling_CIE_logLambda" in params:
        params["cStruc_cooling_CIE_logLambda"].value = logL
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logL, kind="linear")

    # rebuild non-CIE cubes (depends only on t_now / Z / rotation / path)
    cooling_nonCIE, heating_nonCIE, net = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = net
    return params, state


def _props(bp):
    return {
        "bubble_dMdt": float(bp.bubble_dMdt),
        "bubble_LTotal": float(bp.bubble_LTotal),
        "bubble_T_r_Tb": float(bp.bubble_T_r_Tb),
        "bubble_mass": float(bp.bubble_mass),
    }


def _rel(a, b):
    denom = max(abs(b), 1e-300)
    if abs(a) < 1e-300 and abs(b) < 1e-300:
        return 0.0
    return abs(a - b) / denom


def _time_call(thunk):
    best = np.inf
    for _ in range(TIMING_REPS):
        t0 = time.perf_counter()
        try:
            thunk()
        except Exception:  # noqa: BLE001
            return np.nan
        best = min(best, time.perf_counter() - t0)
    return best


def replay(params):
    """Run every variant on ``params``; return a list of result dicts."""
    real_resid = BL._get_velocity_residuals
    rows, base = [], None
    for vname, vfn in residual_variants.VARIANTS.items():
        BL._get_velocity_residuals = vfn
        try:
            try:
                out = _props(BL.get_bubbleproperties_pure(params))
                t_ms = _time_call(lambda: BL.get_bubbleproperties_pure(params)) * 1e3
                ok = True
            except Exception as exc:  # noqa: BLE001
                out = {k: np.nan for k in
                       ("bubble_dMdt", "bubble_LTotal", "bubble_T_r_Tb", "bubble_mass")}
                t_ms, ok = np.nan, False
                print(f"  [variant {vname} failed: {type(exc).__name__}: "
                      f"{str(exc)[:80]}]", file=sys.stderr)
        finally:
            BL._get_velocity_residuals = real_resid
        if vname == "baseline":
            base = out
        rows.append({"variant": vname, "npts": residual_variants.VARIANT_NPTS.get(vname, ""),
                     "time_ms": t_ms, "ok": ok, **out,
                     "rel_dMdt": _rel(out["bubble_dMdt"], base["bubble_dMdt"]) if base else 0.0,
                     "rel_LTotal": _rel(out["bubble_LTotal"], base["bubble_LTotal"]) if base else 0.0,
                     "rel_T_r_Tb": _rel(out["bubble_T_r_Tb"], base["bubble_T_r_Tb"]) if base else 0.0,
                     "rel_mass": _rel(out["bubble_mass"], base["bubble_mass"]) if base else 0.0})
    return rows


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    pkl = argv[0]
    base = argv[1] if len(argv) > 1 else _DEFAULT_BASE
    params, state = load_state(pkl, base)
    print(f"state: {os.path.basename(pkl)}  config={state.get('config')} "
          f"phase={state.get('phase')}  base_param={base}")
    rows = replay(params)
    print(f"\n{'variant':>9} {'npts':>5} {'time_ms':>9} {'bubble_dMdt':>13} "
          f"{'rel_dMdt':>10} {'rel_LTotal':>11} {'rel_T_r_Tb':>11} {'rel_mass':>10} ok")
    for r in rows:
        print(f"{r['variant']:>9} {str(r['npts']):>5} {r['time_ms']:>9.2f} "
              f"{r['bubble_dMdt']:>13.5e} {r['rel_dMdt']:>10.2e} "
              f"{r['rel_LTotal']:>11.2e} {r['rel_T_r_Tb']:>11.2e} "
              f"{r['rel_mass']:>10.2e} {int(r['ok'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
