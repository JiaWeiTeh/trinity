#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture one genuinely-stiff bubble state for the dR2 robustness test (Tier 2).

The committed fixture used by test/test_dR2min_magic_number.py's mild checks is a
small cluster (dR2/R2 ~ 1e-8). The conduction-layer stiffness that motivated
WARPFIELD's dR2min floor only appears for a strong wind, where dR2/R2 falls to
~1e-10..1e-11 (docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md). This script
runs a high-feedback config, hooks the first phase-1a bubble call
(get_bubbleproperties_pure), and snapshots the scalar param state + the solved
(R1, Pb, dMdt) into a JSON fixture in the SAME format as
test/data/residual_resample_fixture.json, so the test's _build_params loader
reconstructs it (cooling cubes rebuilt) without storing big arrays.

The host run is aborted right after the first capture, so this is cheap (setup +
phase 0 + one bubble solve). The fixture is committed; the test only loads it.

REPRODUCE
---------
    cd /home/user/trinity
    python docs/dev/performance/harness/capture_stiff_dR2_state.py \
        docs/dev/performance/f1edge_lowdens_himass_hisfe.param \
        test/data/dR2_stiff_state_fixture.json
"""

import json
import sys
from pathlib import Path

import numpy as np

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

import trinity.bubble_structure.bubble_luminosity as BL  # noqa: E402

_REAL_GBP = BL.get_bubbleproperties_pure
_T_INIT = BL._T_INIT_BOUNDARY


class _CaptureDone(Exception):
    pass


def _scalar_snapshot(params):
    """All param entries whose value is a plain int/float (JSON-serialisable)."""
    real = getattr(params, "_params", params)
    out = {}
    for k in real.keys():
        try:
            v = params[k].value
        except Exception:
            continue
        if isinstance(v, (bool,)):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            out[k] = float(v)
    return out


def _make_hook(base_param, out_path, captured):
    def hook(params):
        bp = _REAL_GBP(params)  # real solve -> converged dMdt, R1, Pb
        R2 = float(params["R2"].value)
        const = 25.0 / 4.0 * (params["k_B"].value / params["mu_ion"].value
                              / params["C_thermal"].value)
        dR2 = _T_INIT ** 2.5 / (const * bp.bubble_dMdt / (4.0 * np.pi * R2 ** 2))
        captured.update({
            "_comment": ("genuinely-stiff bubble state for test_dR2min_magic_number "
                         "Tier 2; captured by capture_stiff_dR2_state.py"),
            "base_param": base_param,
            "Pb": float(bp.Pb),
            "R1": float(bp.R1),
            "dMdt_converged": float(bp.bubble_dMdt),
            "dR2_over_R2": float(dR2 / R2),
            "param_values": _scalar_snapshot(params),
        })
        with open(out_path, "w") as fh:
            json.dump(captured, fh, indent=2, sort_keys=True)
        print(f"[capture] dR2/R2={dR2 / R2:.3e}  R2={R2:.4e}  dMdt={bp.bubble_dMdt:.4e}  "
              f"-> {out_path}", file=sys.stderr, flush=True)
        raise _CaptureDone()
    return hook


def main():
    if len(sys.argv) < 3:
        print("usage: capture_stiff_dR2_state.py <config.param> <out.json>",
              file=sys.stderr)
        return 2
    base_param = sys.argv[1]
    out_path = sys.argv[2]

    import logging
    logging.disable(logging.CRITICAL)
    from trinity._input import read_param
    from trinity import main as trinity_main

    params = read_param.read_param(str(TRINITY_ROOT / base_param))
    captured = {}
    BL.get_bubbleproperties_pure = _make_hook(base_param, out_path, captured)
    try:
        trinity_main.start_expansion(params)
        print("Host run finished without a bubble call -- nothing captured.",
              file=sys.stderr)
        return 1
    except _CaptureDone:
        return 0
    finally:
        BL.get_bubbleproperties_pure = _REAL_GBP
        logging.disable(logging.NOTSET)


if __name__ == "__main__":
    raise SystemExit(main())
