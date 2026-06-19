#!/usr/bin/env python3
"""Monkeypatched candidate-fix variants for the failed-large-clouds matrix.

Production is NEVER touched. Each variant patches `get_bubbleParams` in place;
because every call site reaches the helpers via the module attribute
(`get_bubbleParams.solve_R1` / `.bubble_E2P`), patching the module propagates
everywhere (phase 1a / 1b / 1c, energy ODE, bubble solve).

Variants
--------
V0  baseline            : no patch (reference).
V1  solve_R1 R1-clamp   : clamp R1 <= R2*(1-eps) so (R2^3 - R1^3) can never
                          underflow to 0. eps=1e-6. Below the float64 cliff this
                          makes Pb ~ proportional to Eb (decaying).
V2  bubble_E2P vol-floor: floor the shell volume (r2^3 - r1^3) at eps*r2^3
                          inside bubble_E2P only. eps=1e-6.
V3  both                : V1 + V2.

Usage: `import variants; variants.apply("V1")`  (call BEFORE running the sim).
"""
import numpy as np

import trinity.bubble_structure.get_bubbleParams as gbp
import trinity._functions.unit_conversions as cvt

EPS = 1e-6

_ORIG_SOLVE_R1 = gbp.solve_R1
_ORIG_E2P = gbp.bubble_E2P


def _solve_R1_clamped(R2, Eb, Lmech_total, v_mech_total):
    R1 = _ORIG_SOLVE_R1(R2, Eb, Lmech_total, v_mech_total)
    # Keep the wind shock strictly inside the outer radius so the hot-shell
    # volume stays resolvable in float64 (the get_r1 root drives R1->R2 as Eb->0).
    R1_max = R2 * (1.0 - EPS)
    if R1 > R1_max:
        return R1_max
    return R1


def _bubble_E2P_vol_floored(Eb, r2, r1, gamma):
    r1c = r1 * cvt.pc2cm
    r2c = r2 * cvt.pc2cm
    Ebc = Eb * cvt.E_au2cgs
    vol = r2c**3 - r1c**3
    floor = EPS * r2c**3
    if vol < floor:
        vol = floor
    Pb = (gamma - 1) * Ebc / vol / (4 * np.pi / 3)
    return Pb * cvt.Pb_cgs2au


def apply(variant: str):
    """Install the named variant's monkeypatch. Returns the variant id."""
    # always restore originals first (idempotent)
    gbp.solve_R1 = _ORIG_SOLVE_R1
    gbp.bubble_E2P = _ORIG_E2P
    if variant == "V0":
        pass
    elif variant == "V1":
        gbp.solve_R1 = _solve_R1_clamped
    elif variant == "V2":
        gbp.bubble_E2P = _bubble_E2P_vol_floored
    elif variant == "V3":
        gbp.solve_R1 = _solve_R1_clamped
        gbp.bubble_E2P = _bubble_E2P_vol_floored
    else:
        raise ValueError(f"unknown variant {variant!r} (expected V0/V1/V2/V3)")
    return variant
