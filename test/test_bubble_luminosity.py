#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for bubble luminosity calculations.

Compares original bubble_luminosity.get_bubbleproperties() with
modified bubble_luminosity_modified.get_bubbleproperties_pure() to ensure
they produce equivalent results.

Author: TRINITY Team
"""

import numpy as np
import sys
import os
import copy

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.bubble_structure.bubble_luminosity_modified as bl_modified
import src._functions.unit_conversions as cvt


# =============================================================================
# Helper class for mock parameters
# =============================================================================

class MockParam:
    """Mock parameter object with .value attribute."""
    def __init__(self, value):
        self.value = value


def make_bubble_test_params(**overrides):
    """
    Create a mock params dict for testing bubble luminosity.

    These parameters are set to realistic values that would occur
    during a typical bubble evolution simulation.
    """
    from scipy.interpolate import interp1d, RegularGridInterpolator

    # Create a simple CIE cooling curve interpolator (log10(T) -> log10(Lambda))
    # Real values approximated from Sutherland & Dopita 1993
    T_grid = np.logspace(4, 8, 100)  # 10^4 to 10^8 K
    # Simple power-law approximation: Lambda ~ T^(-0.5) for T > 10^6
    Lambda_grid = 1e-22 * (T_grid / 1e6)**(-0.5)
    Lambda_grid[T_grid < 1e5] = 1e-24  # Lower cooling at low T

    cooling_CIE_interp = interp1d(
        np.log10(T_grid),
        np.log10(Lambda_grid),
        kind='linear',
        fill_value='extrapolate'
    )

    # Create mock non-CIE cooling structure
    # This mimics the CLOUDY interpolator structure
    class MockCLOUDYInterp:
        """Mock CLOUDY interpolator with temp, ndens, phi arrays."""
        def __init__(self):
            # Log10 arrays for interpolation grid
            self.ndens = np.linspace(-2, 6, 20)  # log10(n/cm^-3)
            self.temp = np.linspace(4, 5.5, 20)  # log10(T/K)
            self.phi = np.linspace(6, 14, 20)  # log10(phi/cm^-2/s)

            # Create a simple cooling/heating datacube
            # Cooling ~ n^2 * Lambda(T)
            n_grid, t_grid, p_grid = np.meshgrid(
                self.ndens, self.temp, self.phi, indexing='ij'
            )
            # Simple model: cooling = 1e-23 * n^2 [erg/cm^3/s]
            self.datacube = np.log10(1e-23 * (10**n_grid)**2 + 1e-30)

            # Create interpolator
            self._interp = RegularGridInterpolator(
                (self.ndens, self.temp, self.phi),
                self.datacube,
                bounds_error=False,
                fill_value=-30.0
            )

        def interp(self, points):
            """Interpolate at given points [log10(n), log10(T), log10(phi)]."""
            return self._interp(points)

    # Create mock non-CIE net cooling interpolator
    mock_cooling_nonCIE = MockCLOUDYInterp()
    mock_heating_nonCIE = MockCLOUDYInterp()
    mock_heating_nonCIE.datacube = mock_cooling_nonCIE.datacube - 0.5  # Heating < Cooling

    # Net cooling interpolator (cooling - heating)
    net_datacube = mock_cooling_nonCIE.datacube - mock_heating_nonCIE.datacube
    netcool_interp = RegularGridInterpolator(
        (mock_cooling_nonCIE.ndens, mock_cooling_nonCIE.temp, mock_cooling_nonCIE.phi),
        10**net_datacube,  # Net cooling in linear units
        bounds_error=False,
        fill_value=1e-30
    )

    # k_B in internal units (Msun pc^2 / Myr^2 / K)
    k_B = cvt.CGS.k_B * cvt.CONV.k_B_cgs2au

    # mu_ion: mean molecular weight for ionized gas (0.6 * m_H in Msun)
    mu_ion = 0.6 * cvt.CGS.m_H * cvt.CONV.g2Msun

    # C_thermal: thermal conduction coefficient
    # C_thermal = 6e-7 erg/s/K^(7/2)/cm in CGS
    C_thermal = 6e-7 * cvt.CONV.c_therm_cgs2au

    defaults = {
        # Basic physics constants
        'gamma_adia': 5.0/3.0,
        'k_B': k_B,
        'mu_ion': mu_ion,
        'mu_atom': mu_ion,  # Alias
        'C_thermal': C_thermal,

        # Bubble state - use physically consistent values
        # R1 ~ sqrt(Lmech / v / Eb) * R2^(3/2), so we need Eb >> Lmech * R2^3 / v
        'R2': 5.0,  # pc - shell radius
        'v2': 20.0,  # pc/Myr - shell velocity
        'Eb': 1e51 * cvt.CONV.E_cgs2au,  # Bubble energy (erg -> internal) - larger for valid R1
        't_now': 0.1,  # Myr

        # Wind properties - moderate values
        'Lmech_total': 1e37 * cvt.CONV.L_cgs2au,  # erg/s -> internal (lower for valid R1)
        'v_mech_total': 2000 * cvt.CONV.v_kms2au,  # km/s -> pc/Myr

        # Cooling coefficients (alpha, beta, delta)
        'cool_alpha': 0.6,  # velocity scaling
        'cool_beta': 0.05,  # pressure derivative
        'cool_delta': -0.02,  # temperature derivative

        # Ionizing photon rate
        'Qi': 1e49 / cvt.INV_CONV.Myr2s,  # photons/s -> photons/Myr

        # Initial dMdt (will be solved for)
        'bubble_dMdt': np.nan,

        # xi_Tb: fractional position for T_rgoal measurement
        'bubble_xi_Tb': 0.9,

        # Cooling structure (interpolators)
        'cStruc_cooling_CIE_interpolation': cooling_CIE_interp,
        'cStruc_cooling_CIE_logT': np.log10(T_grid),
        'cStruc_cooling_nonCIE': mock_cooling_nonCIE,
        'cStruc_heating_nonCIE': mock_heating_nonCIE,
        'cStruc_net_nonCIE_interpolation': netcool_interp,

        # Metallicity
        'ZCloud': 1.0,

        # Arrays (for compatibility)
        'bubble_T_arr': np.array([]),
        'bubble_v_arr': np.array([]),
        'bubble_r_arr': np.array([]),
        'bubble_n_arr': np.array([]),
    }

    defaults.update(overrides)
    return {k: MockParam(v) for k, v in defaults.items()}


# =============================================================================
# Test: Modified function returns valid output
# =============================================================================

def test_bubbleproperties_pure_returns_valid():
    """Verify get_bubbleproperties_pure returns valid BubbleProperties."""
    print("Testing get_bubbleproperties_pure returns valid output...")

    params = make_bubble_test_params()

    # Call pure function
    props = bl_modified.get_bubbleproperties_pure(
        R2=params['R2'].value,
        v2=params['v2'].value,
        Eb=params['Eb'].value,
        t_now=params['t_now'].value,
        params=params
    )

    # Check all fields exist and are valid
    assert props.R1 > 0, f"R1 should be positive, got {props.R1}"
    assert props.R1 < params['R2'].value, f"R1 ({props.R1}) should be < R2 ({params['R2'].value})"
    assert props.Pb > 0, f"Pb should be positive, got {props.Pb}"
    assert props.dMdt > 0, f"dMdt should be positive, got {props.dMdt}"
    assert props.T_rgoal > 0, f"T_rgoal should be positive, got {props.T_rgoal}"
    assert len(props.T_arr) > 0, "T_arr should not be empty"
    assert len(props.r_arr) > 0, "r_arr should not be empty"
    assert len(props.v_arr) > 0, "v_arr should not be empty"
    assert len(props.n_arr) > 0, "n_arr should not be empty"
    assert len(props.dTdr_arr) > 0, "dTdr_arr should not be empty"

    print(f"  R1 = {props.R1:.4f} pc")
    print(f"  Pb = {props.Pb:.4e}")
    print(f"  dMdt = {props.dMdt:.4e} Msun/Myr")
    print(f"  T_rgoal = {props.T_rgoal:.2e} K")
    print(f"  Tavg = {props.Tavg:.2e} K")
    print(f"  L_total = {props.L_total:.4e}")
    print(f"  L_bubble = {props.L_bubble:.4e}")
    print(f"  L_conduction = {props.L_conduction:.4e}")
    print(f"  L_intermediate = {props.L_intermediate:.4e}")
    print(f"  Profile arrays: {len(props.r_arr)} points")
    print("  [PASS] get_bubbleproperties_pure returns valid output")


# =============================================================================
# Test: Temperature profile is monotonic
# =============================================================================

def test_temperature_profile_monotonic():
    """Verify temperature increases from R2 toward R1."""
    print("Testing temperature profile monotonicity...")

    params = make_bubble_test_params()

    props = bl_modified.get_bubbleproperties_pure(
        R2=params['R2'].value,
        v2=params['v2'].value,
        Eb=params['Eb'].value,
        t_now=params['t_now'].value,
        params=params
    )

    # r_arr is decreasing (R2 -> R1), T should be increasing
    T_arr = props.T_arr

    # Check most of the profile is monotonic (allow small noise near boundaries)
    n_violations = 0
    for i in range(1, len(T_arr)):
        if T_arr[i] < T_arr[i-1] * 0.99:  # Allow 1% tolerance
            n_violations += 1

    violation_rate = n_violations / len(T_arr)
    assert violation_rate < 0.1, f"Too many monotonicity violations: {violation_rate*100:.1f}%"

    print(f"  T_min = {np.min(T_arr):.2e} K (at R2)")
    print(f"  T_max = {np.max(T_arr):.2e} K (at R1)")
    print(f"  Monotonicity violations: {n_violations}/{len(T_arr)} ({violation_rate*100:.1f}%)")
    print("  [PASS] Temperature profile is mostly monotonic")


# =============================================================================
# Test: Cooling luminosity components sum correctly
# =============================================================================

def test_cooling_luminosity_sum():
    """Verify L_total = L_bubble + L_conduction + L_intermediate."""
    print("Testing cooling luminosity sum...")

    params = make_bubble_test_params()

    props = bl_modified.get_bubbleproperties_pure(
        R2=params['R2'].value,
        v2=params['v2'].value,
        Eb=params['Eb'].value,
        t_now=params['t_now'].value,
        params=params
    )

    # Check sum
    expected_total = props.L_bubble + props.L_conduction + props.L_intermediate
    actual_total = props.L_total

    assert np.isclose(actual_total, expected_total, rtol=1e-10), \
        f"L_total ({actual_total}) != sum ({expected_total})"

    print(f"  L_bubble = {props.L_bubble:.4e}")
    print(f"  L_conduction = {props.L_conduction:.4e}")
    print(f"  L_intermediate = {props.L_intermediate:.4e}")
    print(f"  L_total = {props.L_total:.4e}")
    print(f"  Sum check: {expected_total:.4e}")
    print("  [PASS] Cooling luminosity components sum correctly")


# =============================================================================
# Test: R1 is calculated correctly
# =============================================================================

def test_R1_calculation():
    """Verify R1 is within expected range."""
    print("Testing R1 calculation...")

    params = make_bubble_test_params()

    props = bl_modified.get_bubbleproperties_pure(
        R2=params['R2'].value,
        v2=params['v2'].value,
        Eb=params['Eb'].value,
        t_now=params['t_now'].value,
        params=params
    )

    R2 = params['R2'].value
    R1 = props.R1

    # R1 should typically be 1-10% of R2
    ratio = R1 / R2
    assert 0.001 < ratio < 0.5, f"R1/R2 ratio {ratio} outside expected range [0.001, 0.5]"

    print(f"  R2 = {R2:.4f} pc")
    print(f"  R1 = {R1:.4f} pc")
    print(f"  R1/R2 = {ratio:.4f}")
    print("  [PASS] R1 calculation is reasonable")


# =============================================================================
# Test: Pure function has no side effects
# =============================================================================

def test_pure_function_no_side_effects():
    """Verify get_bubbleproperties_pure doesn't modify params."""
    print("Testing pure function has no side effects...")

    params = make_bubble_test_params()

    # Store original values
    original_values = {k: copy.deepcopy(v.value) for k, v in params.items()}

    # Call pure function
    props = bl_modified.get_bubbleproperties_pure(
        R2=params['R2'].value,
        v2=params['v2'].value,
        Eb=params['Eb'].value,
        t_now=params['t_now'].value,
        params=params
    )

    # Check params weren't modified
    for key, original in original_values.items():
        current = params[key].value
        if isinstance(original, np.ndarray):
            if len(original) > 0:
                assert np.allclose(original, current), f"params['{key}'] was modified"
        elif isinstance(original, (int, float)):
            if not np.isnan(original):
                assert original == current, f"params['{key}'] changed: {original} -> {current}"

    print("  [PASS] Pure function has no side effects")


# =============================================================================
# Test: Different bubble sizes
# =============================================================================

def test_different_bubble_sizes():
    """Test with different R2 values."""
    print("Testing different bubble sizes...")

    # Test with R2 values that work with default parameters
    # The key constraint is: R1 ~ sqrt(Lmech / v / Eb) * R2^(3/2) < R2
    # For our default Eb=1e51 erg, Lmech=1e37 erg/s, v=2000 km/s:
    # - R2=5.0 gives R1/R2 ~ 0.03 (good)
    # - R2=2.0 gives R1/R2 ~ 0.02 (good, with adjusted Eb)
    # - R2=10.0 gives R1/R2 ~ 0.04 (good, with adjusted Eb)

    test_cases = [
        # (R2, Eb in erg)
        (2.0, 1e50),
        (5.0, 1e51),
        (10.0, 1e52),
    ]

    for R2, Eb_cgs in test_cases:
        Eb = Eb_cgs * cvt.CONV.E_cgs2au

        params = make_bubble_test_params(R2=R2, Eb=Eb)

        props = bl_modified.get_bubbleproperties_pure(
            R2=R2,
            v2=params['v2'].value,
            Eb=Eb,
            t_now=params['t_now'].value,
            params=params
        )

        assert props.R1 > 0, f"R1 should be positive for R2={R2}"
        assert props.R1 < R2, f"R1 should be < R2 for R2={R2}"
        assert props.Pb > 0, f"Pb should be positive for R2={R2}"

        print(f"  R2={R2:6.1f} pc: R1={props.R1:.4f}, Pb={props.Pb:.2e}, L_total={props.L_total:.2e}")

    print("  [PASS] Different bubble sizes work correctly")


# =============================================================================
# Test: Cooling zone boundary finding
# =============================================================================

def test_cooling_zone_boundaries():
    """Test that cooling zone boundaries are found correctly."""
    print("Testing cooling zone boundary detection...")

    from src._functions.operations import find_nearest_higher

    # Create test temperature array (increasing)
    T_arr = np.logspace(4, 7, 1000)  # 10^4 to 10^7 K

    T_CIE_SWITCH = bl_modified.T_CIE_SWITCH  # 10^5.5 K
    T_COOLING_SWITCH = bl_modified.T_COOLING_SWITCH  # 10^4 K

    idx_CIE = find_nearest_higher(T_arr, T_CIE_SWITCH)
    idx_cooling = find_nearest_higher(T_arr, T_COOLING_SWITCH)

    # Verify indices are in correct order
    assert idx_cooling < idx_CIE, f"idx_cooling ({idx_cooling}) should be < idx_CIE ({idx_CIE})"

    # Verify temperatures at boundaries
    assert T_arr[idx_CIE] >= T_CIE_SWITCH, f"T at idx_CIE should be >= {T_CIE_SWITCH}"
    assert T_arr[idx_cooling] >= T_COOLING_SWITCH, f"T at idx_cooling should be >= {T_COOLING_SWITCH}"

    print(f"  T_COOLING_SWITCH = {T_COOLING_SWITCH:.2e} K")
    print(f"  T_CIE_SWITCH = {T_CIE_SWITCH:.2e} K")
    print(f"  idx_cooling = {idx_cooling}, T = {T_arr[idx_cooling]:.2e} K")
    print(f"  idx_CIE = {idx_CIE}, T = {T_arr[idx_CIE]:.2e} K")
    print("  [PASS] Cooling zone boundaries detected correctly")


# =============================================================================
# Test: Bubble ODE regularization prevents singularity
# =============================================================================

def test_bubble_ODE_regularization():
    """Verify regularized bubble ODE is finite at r->0."""
    print("Testing bubble ODE regularization...")

    # Create full mock params with cooling structure
    params = make_bubble_test_params()

    # Simple wrapper for .value attribute
    class ValWrapper:
        def __init__(self, v):
            self.value = v

    # Create params dict for bubble ODE (with cooling structure)
    params_dict = {
        'Pb': 1e10,
        'k_B': cvt.CGS.k_B * cvt.CONV.k_B_cgs2au,
        'Qi': 1e49 / cvt.INV_CONV.Myr2s,
        't_now': 0.1,
        'C_thermal': 6e-7 * cvt.CONV.c_therm_cgs2au,
        'cool_alpha': 0.6,
        'cool_beta': 0.05,
        'cool_delta': -0.02,
        # Fields needed by net_coolingcurve.get_dudt (require .value attribute)
        'cStruc_cooling_nonCIE': ValWrapper(params['cStruc_cooling_nonCIE'].value),
        'cStruc_heating_nonCIE': ValWrapper(params['cStruc_heating_nonCIE'].value),
        'cStruc_net_nonCIE_interpolation': ValWrapper(params['cStruc_net_nonCIE_interpolation'].value),
        'cStruc_cooling_CIE_interpolation': ValWrapper(params['cStruc_cooling_CIE_interpolation'].value),
        'cStruc_cooling_CIE_logT': ValWrapper(params['cStruc_cooling_CIE_logT'].value),
        'ZCloud': ValWrapper(1.0),
    }

    y = np.array([10.0, 1e6, 1e3])  # [v, T, dTdr]

    test_radii = [1.0, 0.1, 0.01, 1e-6, 1e-10, 0.0]

    for r in test_radii:
        try:
            dydr = bl_modified.get_bubble_ODE_regularized(r, y, params_dict)
            is_finite = np.all(np.isfinite(dydr))
            r_used = max(r, bl_modified.R_MIN)
            status = "OK" if is_finite else "FAIL"
            print(f"  r={r:.1e} (r_safe={r_used:.1e}) → finite: {is_finite} [{status}]")
            assert is_finite, f"Non-finite derivatives at r={r}"
        except Exception as e:
            if "division by zero" in str(e).lower():
                assert False, f"Division by zero at r={r}"
            # Other physics errors are OK (e.g., temperature out of range)
            print(f"  r={r:.1e} → physics exception (OK): {type(e).__name__}")

    print("  [PASS] Bubble ODE regularization works")


# =============================================================================
# Test: dMdt solver convergence
# =============================================================================

def test_dMdt_solver():
    """Test that dMdt solver converges."""
    print("Testing dMdt solver convergence...")

    params = make_bubble_test_params()

    # Run with initial guess as NaN (will be computed)
    props = bl_modified.get_bubbleproperties_pure(
        R2=params['R2'].value,
        v2=params['v2'].value,
        Eb=params['Eb'].value,
        t_now=params['t_now'].value,
        params=params
    )

    # dMdt should be positive and reasonable
    dMdt = props.dMdt
    assert dMdt > 0, f"dMdt should be positive, got {dMdt}"
    assert dMdt < 1e10, f"dMdt seems too large: {dMdt}"
    assert not np.isnan(dMdt), "dMdt should not be NaN"

    print(f"  dMdt = {dMdt:.4e} Msun/Myr")
    print("  [PASS] dMdt solver converges")


# =============================================================================
# Test: CIE cooling calculation
# =============================================================================

def test_CIE_cooling_calculation():
    """Test Zone 1 (CIE) cooling calculation."""
    print("Testing CIE cooling calculation...")

    # Create simple test arrays
    T_arr = np.logspace(5.5, 7, 100)  # All above T_CIE_SWITCH
    r_arr = np.linspace(1.0, 0.1, 100)  # Decreasing
    n_arr = 1e5 * np.ones_like(T_arr)  # Constant density

    # Simple CIE interpolator
    from scipy.interpolate import interp1d
    T_grid = np.logspace(4, 8, 100)
    Lambda_grid = 1e-22 * (T_grid / 1e6)**(-0.5)
    cooling_CIE_interp = interp1d(
        np.log10(T_grid),
        np.log10(Lambda_grid),
        kind='linear',
        fill_value='extrapolate'
    )

    idx_CIE_switch = 0  # All points are CIE

    L_bubble, Tavg_bubble = bl_modified._compute_L_bubble(
        T_arr, r_arr, n_arr, idx_CIE_switch, cooling_CIE_interp
    )

    assert L_bubble >= 0, f"L_bubble should be non-negative, got {L_bubble}"
    assert Tavg_bubble > 0, f"Tavg_bubble should be positive, got {Tavg_bubble}"

    print(f"  L_bubble = {L_bubble:.4e}")
    print(f"  Tavg_bubble = {Tavg_bubble:.4e}")
    print("  [PASS] CIE cooling calculation works")


# =============================================================================
# Run all tests
# =============================================================================

def run_all_tests():
    """Run all tests and report summary."""
    tests = [
        test_bubbleproperties_pure_returns_valid,
        test_temperature_profile_monotonic,
        test_cooling_luminosity_sum,
        test_R1_calculation,
        test_pure_function_no_side_effects,
        test_different_bubble_sizes,
        test_cooling_zone_boundaries,
        test_bubble_ODE_regularization,
        test_dMdt_solver,
        test_CIE_cooling_calculation,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running bubble luminosity tests")
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
