"""Covering-fraction leak: helper correctness, Cf=1 invariant, unit landing.

Phase A/C of the geometry-set leak (docs/dev/LEAKING_LUMINOSITIES_SKELETON.md).
The merge gate is that Cf=1 changes nothing; these tests pin the helper that
guarantees it, the self-limiting guards, and that Pb*cs*R2**2 lands in the
code luminosity unit with no hidden conversion factor.
"""
from __future__ import annotations

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
from trinity.bubble_structure.get_bubbleParams import (
    get_leak_luminosity,
    get_leak_thermal_pressure,
    bubble_E2P,
)
from trinity._input.registry import _validate_coverFraction, REGISTRY

GAMMA = 5.0 / 3.0


def test_sealed_bubble_returns_exactly_zero():
    # Cf = 1 must reproduce the sealed (Weaver) bubble byte-for-byte.
    assert get_leak_luminosity(1.0, R2=5.0, Pb=1.0, c_sound=10.0, gamma=GAMMA) == 0.0


def test_degenerate_states_do_not_inject_energy():
    # Depressurised / no-temperature states return 0, never negative.
    assert get_leak_luminosity(0.9, R2=5.0, Pb=0.0, c_sound=10.0, gamma=GAMMA) == 0.0
    assert get_leak_luminosity(0.9, R2=5.0, Pb=-1.0, c_sound=10.0, gamma=GAMMA) == 0.0
    assert get_leak_luminosity(0.9, R2=5.0, Pb=1.0, c_sound=0.0, gamma=GAMMA) == 0.0


def test_formula_matches_enthalpy_flux():
    Cf, R2, Pb, cs = 0.9, 5.0, 1.3, 12.0
    expected = GAMMA / (GAMMA - 1.0) * (1.0 - Cf) * 4.0 * np.pi * R2**2 * Pb * cs
    assert get_leak_luminosity(Cf, R2, Pb, cs, GAMMA) == pytest.approx(expected)


def test_leak_grows_as_wall_opens():
    # Smaller Cf (more open wall) -> larger leak; strictly monotone.
    vals = [get_leak_luminosity(cf, R2=5.0, Pb=1.0, c_sound=10.0, gamma=GAMMA)
            for cf in (0.99, 0.95, 0.90, 0.50)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[0] > 0.0


def test_units_land_in_code_luminosity_no_hidden_factor():
    # Build a leak in cgs, convert inputs to code units, and confirm the helper
    # reproduces the cgs result converted to code luminosity. This is the
    # assertion S2.6 asks for: Pb*cs*R2**2 -> Msun*pc**2/Myr**3 directly.
    Cf = 0.9
    Pb_cgs = 3.0e-10        # dyn/cm**2
    cs_cgs = 4.0e7          # cm/s   (~hot-bubble sound speed)
    R2_cgs = 5.0 * cvt.pc2cm  # 5 pc in cm

    leak_cgs = (GAMMA / (GAMMA - 1.0) * (1.0 - Cf)
                * 4.0 * np.pi * R2_cgs**2 * Pb_cgs * cs_cgs)  # erg/s

    leak_code = get_leak_luminosity(
        Cf,
        R2=R2_cgs * cvt.cm2pc,
        Pb=Pb_cgs * cvt.Pb_cgs2au,
        c_sound=cs_cgs * cvt.v_cms2au,
        gamma=GAMMA,
    )
    assert leak_code == pytest.approx(leak_cgs * cvt.L_cgs2au, rel=1e-10)


def test_param_default_and_unit_registered():
    spec = REGISTRY['coverFraction']
    assert spec.run_const is True
    # default is a string in the registry, parsed downstream; sealed by default.
    assert float(spec.default) == 1.0
    assert REGISTRY['bubble_Leak'].unit == 'Msun*pc**2/Myr**3'


@pytest.mark.parametrize('bad', [0.0, -0.1, 1.5, True, 'x', None])
def test_validator_rejects_out_of_range(bad):
    from trinity._input.errors import ParameterFileError
    with pytest.raises(ParameterFileError):
        _validate_coverFraction(bad, {})


@pytest.mark.parametrize('ok', [1.0, 0.99, 0.5, 0.01])
def test_validator_accepts_valid(ok):
    _validate_coverFraction(ok, {})  # must not raise


# --- leak pressure selection (hot-gas thermal Pb, not the driving pressure) ---

def test_leak_pressure_reuses_press_bubble_outside_transition():
    # Energy/implicit (any non-transition phase): press_bubble is already the
    # thermal Pb, so it is returned unchanged regardless of Eb/R1.
    for phase in ('energy', 'implicit'):
        p = get_leak_thermal_pressure(phase, Eb=123.0, R2=5.0, R1=0.5,
                                      gamma=GAMMA, press_bubble=7.0)
        assert p == 7.0


def test_leak_pressure_uses_thermal_in_transition():
    # Transition: recompute the thermal Pb (bubble_E2P), ignoring the effective
    # max(P_th, P_ram) driving pressure passed in as press_bubble.
    Eb, R2, R1 = 100.0, 5.0, 0.5
    thermal = bubble_E2P(Eb, R2, R1, GAMMA)
    assert thermal > 0.0
    inflated = thermal * 5.0  # as if ram pressure dominates: press_bubble = max(...)
    p = get_leak_thermal_pressure('transition', Eb, R2, R1, GAMMA, press_bubble=inflated)
    assert p == pytest.approx(thermal)
    assert p < inflated


def test_transition_leak_smaller_than_if_ram_pressure_were_used():
    # The fix: using thermal Pb gives a smaller (correct) leak than the effective
    # driving pressure would when ram dominates late in the transition.
    Eb, R2, R1, Cf, cs = 100.0, 5.0, 0.5, 0.9, 12.0
    thermal = bubble_E2P(Eb, R2, R1, GAMMA)
    inflated = thermal * 5.0
    p_used = get_leak_thermal_pressure('transition', Eb, R2, R1, GAMMA, inflated)
    leak_thermal = get_leak_luminosity(Cf, R2, p_used, cs, GAMMA)
    leak_inflated = get_leak_luminosity(Cf, R2, inflated, cs, GAMMA)
    assert 0.0 < leak_thermal < leak_inflated
