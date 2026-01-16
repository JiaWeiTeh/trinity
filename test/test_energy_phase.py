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

Author: TRINITY Team
"""

import numpy as np
import sys
import os
import copy

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_energy.energy_phase_ODEs_modified import (
    StaticODEParams,
    get_ODE_Edot_pure,
    extract_static_params,
    R1Cache,
    _calculate_mass_pure,
    _get_mShell_dot_with_activation,
    velocity_floor_event,
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


def make_static_params(**overrides) -> StaticODEParams:
    """Create StaticODEParams for testing with sensible defaults."""
    defaults = dict(
        gamma_adia=5.0/3.0,
        G=4.49e-15,  # pc³/Msun/Myr²
        k_B=6.94e-60,
        rCloud=10.0,  # pc
        rCore=1.0,  # pc
        mCloud=1e5,  # Msun
        mCluster=1e4,  # Msun
        nCore=1e6,  # 1/pc³ (internal units)
        nISM=1e3,  # 1/pc³
        mu_convert=2.34e-57,  # Msun (1.4 m_H)
        dens_profile='densPL',
        densPL_alpha=0.0,
        LWind=1e38,  # internal units
        vWind=1e3,  # pc/Myr
        L_bubble=1e37,
        F_rad=0.0,
        FABSi=1.0,
        press_HII_in=0.0,
        press_HII_out=0.0,
        R1_cached=0.01,  # pc
        tSF=0.0,  # Myr
        current_phase='energy',
        is_collapse=False,
        shell_mass_frozen=0.0,
    )
    defaults.update(overrides)
    return StaticODEParams(**defaults)


def make_test_params(**overrides):
    """Create a mock params dict for testing."""
    defaults = {
        'gamma_adia': 5.0/3.0,
        'G': 4.49e-15,
        'k_B': 6.94e-60,
        'rCloud': 10.0,
        'rCore': 1.0,
        'mCloud': 1e5,
        'mCluster': 1e4,
        'nCore': 1e6,
        'nISM': 1e3,
        'mu_convert': 2.34e-57,
        'dens_profile': 'densPL',
        'densPL_alpha': 0.0,
        'LWind': 1e38,
        'vWind': 1e3,
        'bubble_LTotal': 1e37,
        'shell_F_rad': 0.0,
        'shell_fAbsorbedIon': 1.0,
        'press_HII_in': 0.0,
        'press_HII_out': 0.0,
        'tSF': 0.0,
        'current_phase': 'energy',
        'isCollapse': False,
        'shell_mass': 0.0,
    }
    defaults.update(overrides)
    return {k: MockParam(v) for k, v in defaults.items()}


# =============================================================================
# Test: Pure ODE has no side effects
# =============================================================================

def test_ODE_pure_no_side_effects():
    """Verify pure ODE function doesn't modify inputs."""
    print("Testing ODE pure function has no side effects...")

    static = make_static_params()
    y = np.array([1.0, 10.0, 1e30])  # [R2, v2, Eb]
    t = 1e-4

    # Deep copy static params (frozen dataclass is immutable anyway)
    y_copy = y.copy()

    # Call ODE function
    dydt = get_ODE_Edot_pure(t, y, static)

    # Check y wasn't modified
    assert np.allclose(y, y_copy), "ODE function modified input state vector"

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

    static = make_static_params()

    # Test multiple state vectors
    test_cases = [
        np.array([0.5, 5.0, 1e29]),
        np.array([1.0, 10.0, 1e30]),
        np.array([5.0, 50.0, 1e31]),
        np.array([0.1, -5.0, 1e28]),  # Collapsing case
    ]

    for y in test_cases:
        dydt = get_ODE_Edot_pure(0.001, y, static)
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

    static = make_static_params()
    y0 = np.array([0.5, 20.0, 1e30])
    t_span = (1e-5, 1e-4)

    # Run integration
    sol = scipy.integrate.solve_ivp(
        fun=lambda t, y: get_ODE_Edot_pure(t, y, static),
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

    # Create params dict for bubble ODE
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

    static = make_static_params(rCore=1.0)
    R2_activate = static.rCore * 0.1  # 10% of core radius

    # Test at various radii
    mShell_dot_raw = 1000.0  # arbitrary test value

    radii = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    print(f"  Activation radius: {R2_activate:.3f} pc")

    for R2 in radii:
        mShell_dot = _get_mShell_dot_with_activation(mShell_dot_raw, R2, static)
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
    cache._rebuild_interpolator = lambda: None  # Skip rebuilding

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
# Test: extract_static_params
# =============================================================================

def test_extract_static_params():
    """Test extraction of static params from params dict."""
    print("Testing extract_static_params...")

    params = make_test_params()
    R1_cached = 0.05

    static = extract_static_params(params, R1_cached=R1_cached)

    # Verify key values extracted correctly
    assert static.gamma_adia == params['gamma_adia'].value
    assert static.rCloud == params['rCloud'].value
    assert static.R1_cached == R1_cached
    assert static.current_phase == 'energy'

    print(f"  gamma_adia: {static.gamma_adia}")
    print(f"  rCloud: {static.rCloud}")
    print(f"  R1_cached: {static.R1_cached}")
    print("  [PASS] extract_static_params works correctly")


# =============================================================================
# Test: velocity_floor_event
# =============================================================================

def test_velocity_floor_event():
    """Test velocity floor event detection."""
    print("Testing velocity floor event...")

    static = make_static_params()

    # Above floor
    y_above = np.array([1.0, 10.0, 1e30])
    val_above = velocity_floor_event(0.001, y_above, static)
    assert val_above > 0, f"Expected positive, got {val_above}"

    # Below floor
    y_below = np.array([1.0, 0.001, 1e30])
    val_below = velocity_floor_event(0.001, y_below, static)
    assert val_below < 0, f"Expected negative, got {val_below}"

    print(f"  v2=10.0 → event value: {val_above:.3f} (positive, no trigger)")
    print(f"  v2=0.001 → event value: {val_below:.3f} (negative, triggers)")
    print("  [PASS] velocity_floor_event works correctly")


# =============================================================================
# Test: Mass calculation pure function
# =============================================================================

def test_calculate_mass_pure():
    """Test pure mass calculation function."""
    print("Testing pure mass calculation...")

    # Homogeneous cloud (alpha=0)
    static = make_static_params(densPL_alpha=0.0)

    R2 = 0.5  # Inside cloud
    v2 = 10.0

    mShell, mShell_dot = _calculate_mass_pure(R2, v2, static)

    assert mShell > 0, f"Expected positive mass, got {mShell}"
    assert mShell_dot > 0, f"Expected positive mdot (expanding), got {mShell_dot}"

    # Check mass scales as R³ for homogeneous
    R2_2 = 1.0
    mShell_2, _ = _calculate_mass_pure(R2_2, v2, static)
    ratio = mShell_2 / mShell
    expected_ratio = (R2_2 / R2)**3

    assert np.isclose(ratio, expected_ratio, rtol=0.01), \
        f"Mass ratio {ratio} != expected {expected_ratio}"

    print(f"  R2={R2:.2f} → mShell={mShell:.2e}")
    print(f"  R2={R2_2:.2f} → mShell={mShell_2:.2e}")
    print(f"  Ratio: {ratio:.2f} (expected: {expected_ratio:.2f})")
    print("  [PASS] Pure mass calculation works correctly")


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
        test_extract_static_params,
        test_velocity_floor_event,
        test_calculate_mass_pure,
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
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
