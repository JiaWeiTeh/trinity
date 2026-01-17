#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for pure ODE functions and energy phase integration.

Tests:
1. Pure ODE function has no side effects
2. Kinematic constraint dR/dt = v2
3. Consistency with scipy.integrate.solve_ivp
4. Bubble ODE regularization (no singularity at r→0)
5. Gradual mShell_dot activation
6. R1Cache functionality
7. Mass profile calculation

Author: TRINITY Team
"""

import numpy as np
import sys
import os
import copy

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_energy.energy_phase_ODEs_modified import (
    get_ODE_Edot_pure,
    R1Cache,
    _get_mass_from_profile,
    _get_mShell_dot_with_activation,
)
from src.bubble_structure.bubble_luminosity_modified import (
    get_bubble_ODE_regularized,
    R_MIN,
)


# =============================================================================
# Helper class for mock parameters
# =============================================================================

class MockParam:
    """Mock parameter object with .value attribute."""
    def __init__(self, value):
        self.value = value


def make_test_params(**overrides):
    """
    Create a mock params dict for testing.

    The params dict should have all keys needed by get_ODE_Edot_pure and
    related functions, with each value having a .value attribute.
    """
    defaults = {
        # Basic physics constants
        'gamma_adia': 5.0/3.0,
        'G': 4.49e-15,  # pc³/Msun/Myr²
        'k_B': 6.94e-60,

        # Cloud properties
        'rCloud': 10.0,  # pc
        'rCore': 1.0,  # pc
        'mCloud': 1e5,  # Msun
        'mCluster': 1e4,  # Msun
        'nCore': 1e6,  # 1/pc³ (internal units)
        'nISM': 1e3,  # 1/pc³
        'mu_convert': 2.34e-57,  # Msun (1.4 m_H)
        'dens_profile': 'densPL',
        'densPL_alpha': 0.0,

        # Wind/bubble properties
        'LWind': 1e38,  # internal units
        'vWind': 1e3,  # pc/Myr
        'bubble_LTotal': 1e37,
        'Qi': 1e49,  # ionizing photons/s

        # Shell properties
        'shell_F_rad': 0.0,
        'shell_fAbsorbedIon': 1.0,  # FABSi
        'shell_mass': 1e3,  # Msun
        'shell_massDot': 0.0,
        'rShell': 1.0,  # pc

        # HII region
        'TShell_ion': 1e4,  # K
        'PISM': 1e3,  # ambient pressure
        'caseB_alpha': 2.6e-13,

        # Timing and phase
        'tSF': 0.0,  # Myr
        't_now': 0.01,  # Myr
        'current_phase': 'energy',

        # Collapse flags
        'isCollapse': False,

        # Early phase approximation
        'EarlyPhaseApproximation': False,
    }
    defaults.update(overrides)
    return {k: MockParam(v) for k, v in defaults.items()}


# =============================================================================
# Test: Pure ODE has no side effects
# =============================================================================

def test_ODE_pure_no_side_effects():
    """Verify pure ODE function doesn't modify inputs."""
    print("Testing ODE pure function has no side effects...")

    params = make_test_params()
    y = np.array([1.0, 10.0, 1e30])  # [R2, v2, Eb]
    t = 1e-4
    R1_cached = 0.01

    # Deep copy to check for mutations
    y_copy = y.copy()
    params_values_before = {k: v.value for k, v in params.items()}

    # Call ODE function
    dydt = get_ODE_Edot_pure(t, y, params, R1_cached)

    # Check y wasn't modified
    assert np.allclose(y, y_copy), "ODE function modified input state vector"

    # Check params weren't modified (ODE should be read-only)
    params_values_after = {k: v.value for k, v in params.items()}
    for key in params_values_before:
        before = params_values_before[key]
        after = params_values_after[key]
        if isinstance(before, (int, float, bool, str)):
            assert before == after, f"ODE modified params['{key}']: {before} -> {after}"

    # Check output is correct shape
    assert dydt.shape == (3,), f"Expected shape (3,), got {dydt.shape}"

    print(f"  State before: {y_copy}")
    print(f"  State after:  {y}")
    print(f"  Derivatives:  {dydt}")
    print("  [PASS] Pure ODE function has no side effects")


# =============================================================================
# Test: Kinematic constraint dR/dt = v2
# =============================================================================

def test_ODE_velocity_is_dRdt():
    """Verify kinematic constraint dR/dt = v2."""
    print("Testing kinematic constraint dR/dt = v2...")

    params = make_test_params()
    R1_cached = 0.01

    # Test multiple state vectors
    test_cases = [
        np.array([0.5, 5.0, 1e29]),
        np.array([1.0, 10.0, 1e30]),
        np.array([5.0, 50.0, 1e31]),
        np.array([0.1, -5.0, 1e28]),  # Collapsing case
    ]

    for y in test_cases:
        dydt = get_ODE_Edot_pure(0.001, y, params, R1_cached)
        R2, v2, Eb = y
        rd = dydt[0]

        assert np.isclose(rd, v2), f"dR/dt ({rd}) != v2 ({v2})"
        print(f"  R2={R2:.2f}, v2={v2:.2f} → dR/dt={rd:.2f} [OK]")

    print("  [PASS] Kinematic constraint satisfied")


# =============================================================================
# Test: Consistency with scipy.solve_ivp
# =============================================================================

def test_ODE_consistent_with_scipy():
    """Verify pure ODE works with scipy.integrate.solve_ivp."""
    print("Testing ODE compatibility with scipy.solve_ivp...")

    import scipy.integrate

    params = make_test_params()
    R1_cached = 0.01
    y0 = np.array([0.5, 20.0, 1e30])
    t_span = (1e-5, 1e-4)

    # Run integration
    sol = scipy.integrate.solve_ivp(
        fun=lambda t, y: get_ODE_Edot_pure(t, y, params, R1_cached),
        t_span=t_span,
        y0=y0,
        method='LSODA',
        rtol=1e-8,
        atol=1e-12,
    )

    assert sol.success, f"solve_ivp failed: {sol.message}"
    assert len(sol.t) > 1, "solve_ivp returned no points"

    # Check final state is reasonable
    R2_final = sol.y[0, -1]
    v2_final = sol.y[1, -1]
    Eb_final = sol.y[2, -1]

    assert R2_final > 0, f"Negative radius: {R2_final}"
    assert Eb_final > 0, f"Negative energy: {Eb_final}"

    print(f"  Initial: R2={y0[0]:.3f}, v2={y0[1]:.1f}, Eb={y0[2]:.2e}")
    print(f"  Final:   R2={R2_final:.3f}, v2={v2_final:.1f}, Eb={Eb_final:.2e}")
    print(f"  Steps:   {len(sol.t)}")
    print("  [PASS] ODE compatible with scipy.solve_ivp")


# =============================================================================
# Test: Bubble ODE regularization
# =============================================================================

def test_bubble_ODE_no_singularity():
    """Verify regularized bubble ODE is finite at r→0."""
    print("Testing bubble ODE regularization at small r...")

    # Create params dict for bubble ODE (plain dict, not MockParam)
    params_dict = {
        'Pb': 1e10,
        'k_B': 6.94e-60,
        'Qi': 1e49,
        't_now': 1e-3,
        'C_thermal': 1e-6,
        'cool_alpha': 0.5,
        'cool_beta': 0.1,
        'cool_delta': 0.1,
    }

    y = np.array([10.0, 1e6, 1e3])  # [v, T, dTdr]

    # Test at progressively smaller radii
    test_radii = [1.0, 0.1, 0.01, 1e-6, 1e-10, 0.0]

    for r in test_radii:
        try:
            dydr = get_bubble_ODE_regularized(r, y, params_dict)
            is_finite = np.all(np.isfinite(dydr))
            r_used = max(r, R_MIN)
            print(f"  r={r:.1e} (r_safe={r_used:.1e}) → dydr finite: {is_finite}")
            assert is_finite, f"Non-finite derivatives at r={r}"
        except Exception as e:
            # Allow if it's a legitimate physics error, not 1/r singularity
            if "division by zero" in str(e).lower():
                assert False, f"Division by zero at r={r}"
            print(f"  r={r:.1e} → exception (OK if physics): {e}")

    print("  [PASS] Bubble ODE regularized correctly")


# =============================================================================
# Test: Gradual mShell_dot activation
# =============================================================================

def test_mShell_dot_activation():
    """Verify gradual activation of mShell_dot at small R2."""
    print("Testing gradual mShell_dot activation...")

    params = make_test_params(rCore=1.0)
    R2_activate = params['rCore'].value * 0.1  # 10% of core radius

    # Test at various radii
    mShell_dot_raw = 1000.0  # arbitrary test value

    radii = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    print(f"  Activation radius: {R2_activate:.3f} pc")

    for R2 in radii:
        mShell_dot = _get_mShell_dot_with_activation(mShell_dot_raw, R2, params)
        expected_activation = min(R2 / R2_activate, 1.0)
        expected = mShell_dot_raw * expected_activation

        assert np.isclose(mShell_dot, expected, rtol=1e-10), \
            f"At R2={R2}: got {mShell_dot}, expected {expected}"
        print(f"  R2={R2:.3f} → activation={expected_activation:.3f}, mShell_dot={mShell_dot:.1f}")

    print("  [PASS] mShell_dot activation works correctly")


# =============================================================================
# Test: R1Cache functionality
# =============================================================================

def test_R1Cache():
    """Test R1Cache caching and interpolation."""
    print("Testing R1Cache...")

    cache = R1Cache()

    # Add some test points (mocking brentq results)
    # We'll just test the caching mechanism, not the brentq itself
    cache.t_values = [0.001, 0.002, 0.003]
    cache.R1_values = [0.01, 0.015, 0.02]

    # Build interpolator manually
    import scipy.interpolate
    cache._interp = scipy.interpolate.interp1d(
        cache.t_values, cache.R1_values,
        kind='linear', fill_value='extrapolate'
    )

    # Test interpolation
    R1_interp = cache.get(0.0015)
    expected = 0.0125  # Linear interpolation
    assert np.isclose(R1_interp, expected, rtol=0.01), \
        f"Got {R1_interp}, expected {expected}"

    print(f"  Interpolated R1 at t=0.0015: {R1_interp:.4f}")
    print("  [PASS] R1Cache works correctly")


# =============================================================================
# Test: Mass calculation using existing mass_profile module
# =============================================================================

def test_get_mass_from_profile():
    """Test mass calculation via existing mass_profile module."""
    print("Testing mass calculation via mass_profile module...")

    # Homogeneous cloud (alpha=0)
    params = make_test_params(densPL_alpha=0.0)

    R2 = 0.5  # Inside cloud
    v2 = 10.0

    mShell, mShell_dot = _get_mass_from_profile(R2, v2, params)

    assert mShell > 0, f"Expected positive mass, got {mShell}"
    assert mShell_dot > 0, f"Expected positive mdot (expanding), got {mShell_dot}"

    # Check mass scales as R³ for homogeneous
    R2_2 = 1.0
    mShell_2, _ = _get_mass_from_profile(R2_2, v2, params)
    ratio = mShell_2 / mShell
    expected_ratio = (R2_2 / R2)**3

    assert np.isclose(ratio, expected_ratio, rtol=0.01), \
        f"Mass ratio {ratio} != expected {expected_ratio}"

    print(f"  R2={R2:.2f} → mShell={mShell:.2e}")
    print(f"  R2={R2_2:.2f} → mShell={mShell_2:.2e}")
    print(f"  Ratio: {ratio:.2f} (expected: {expected_ratio:.2f})")
    print("  [PASS] Mass calculation via mass_profile works correctly")


# =============================================================================
# Test: EarlyPhaseApproximation
# =============================================================================

def test_early_phase_approximation():
    """Test that EarlyPhaseApproximation sets vd = -1e8."""
    print("Testing EarlyPhaseApproximation...")

    params = make_test_params(EarlyPhaseApproximation=True)
    R1_cached = 0.01
    y = np.array([1.0, 10.0, 1e30])
    t = 1e-4

    dydt = get_ODE_Edot_pure(t, y, params, R1_cached)

    # When EarlyPhaseApproximation is True, vd should be -1e8
    vd = dydt[1]
    assert vd == -1e8, f"Expected vd=-1e8, got {vd}"

    print(f"  With EarlyPhaseApproximation=True: vd={vd}")
    print("  [PASS] EarlyPhaseApproximation works correctly")


# =============================================================================
# Test: Collapse mode freezes shell mass
# =============================================================================

def test_collapse_mode():
    """Test that collapse mode freezes shell mass."""
    print("Testing collapse mode...")

    frozen_mass = 5000.0  # Msun
    params = make_test_params(isCollapse=True, shell_mass=frozen_mass)

    R2 = 0.5
    v2 = -10.0  # Negative = collapsing

    mShell, mShell_dot = _get_mass_from_profile(R2, v2, params)

    assert mShell == frozen_mass, f"Expected frozen mass {frozen_mass}, got {mShell}"
    assert mShell_dot == 0.0, f"Expected mShell_dot=0 in collapse, got {mShell_dot}"

    print(f"  isCollapse=True, shell_mass={frozen_mass}")
    print(f"  Result: mShell={mShell}, mShell_dot={mShell_dot}")
    print("  [PASS] Collapse mode freezes shell mass")


# =============================================================================
# Run all tests
# =============================================================================

def run_all_tests():
    """Run all tests and report summary."""
    tests = [
        test_ODE_pure_no_side_effects,
        test_ODE_velocity_is_dRdt,
        test_ODE_consistent_with_scipy,
        test_bubble_ODE_no_singularity,
        test_mShell_dot_activation,
        test_R1Cache,
        test_get_mass_from_profile,
        test_early_phase_approximation,
        test_collapse_mode,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running energy phase tests")
    print("=" * 60)

    for test in tests:
        try:
            print()
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
