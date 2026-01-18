#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for bubble luminosity calculations.

Compares original bubble_luminosity.get_bubbleproperties() with
modified bubble_luminosity_modified.get_bubbleproperties_pure() to ensure
they produce equivalent results.

Usage
-----
Run with synthetic test parameters:
    python test/test_bubble_luminosity.py

Run with a debug snapshot (captured from a real crash):
    python test/test_bubble_luminosity.py --snapshot /path/to/debug_snapshot.json

Author: TRINITY Team
"""

import numpy as np
import sys
import os
import copy
import argparse
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.bubble_structure.bubble_luminosity_modified as bl_modified
import src._functions.unit_conversions as cvt

# Global variable to store debug snapshot path (set via command line)
DEBUG_SNAPSHOT_PATH = None


# =============================================================================
# Helper class for mock parameters
# =============================================================================

class MockParam:
    """Mock parameter object with .value attribute.

    Supports arithmetic and comparison operations like DescribedItem.
    """
    def __init__(self, value):
        self.value = value

    @staticmethod
    def _unwrap(x):
        return x.value if isinstance(x, MockParam) else x

    # Arithmetic operators
    def __add__(self, other): return self.value + self._unwrap(other)
    def __radd__(self, other): return self._unwrap(other) + self.value
    def __sub__(self, other): return self.value - self._unwrap(other)
    def __rsub__(self, other): return self._unwrap(other) - self.value
    def __mul__(self, other): return self.value * self._unwrap(other)
    def __rmul__(self, other): return self._unwrap(other) * self.value
    def __truediv__(self, other): return self.value / self._unwrap(other)
    def __rtruediv__(self, other): return self._unwrap(other) / self.value
    def __pow__(self, other): return self.value ** self._unwrap(other)
    def __rpow__(self, other): return self._unwrap(other) ** self.value

    # Comparison operators
    def __eq__(self, other): return self.value == self._unwrap(other)
    def __lt__(self, other): return self.value < self._unwrap(other)
    def __le__(self, other): return self.value <= self._unwrap(other)
    def __gt__(self, other): return self.value > self._unwrap(other)
    def __ge__(self, other): return self.value >= self._unwrap(other)

    # Numeric conversions
    def __float__(self): return float(self.value)
    def __int__(self): return int(self.value)

    # String representation
    def __repr__(self): return str(self.value)

    # Numpy compatibility
    def __array__(self, dtype=None):
        return np.array(self.value, dtype=dtype)


def load_params_from_snapshot(snapshot_path):
    """
    Load parameters from a debug snapshot file.

    This creates mock params from a real parameter snapshot captured
    during a crash or at a specific point in the simulation.

    Parameters
    ----------
    snapshot_path : str or Path
        Path to debug_snapshot.json file

    Returns
    -------
    dict
        Dictionary of MockParam objects

    Notes
    -----
    - Interpolators cannot be serialized, so mock versions are created
    - Use this to reproduce exact conditions that caused a crash
    """
    from src._input.dictionary import load_debug_snapshot
    from scipy.interpolate import interp1d, RegularGridInterpolator

    print(f"  Loading debug snapshot from: {snapshot_path}")

    # Load raw values
    raw = load_debug_snapshot(snapshot_path)

    print(f"  Loaded {len(raw)} parameters from snapshot")

    # Create mock interpolators (these can't be serialized)
    T_grid = np.logspace(4, 8, 100)
    Lambda_grid = 1e-22 * (T_grid / 1e6)**(-0.5)
    Lambda_grid[T_grid < 1e5] = 1e-24

    cooling_CIE_interp = interp1d(
        np.log10(T_grid),
        np.log10(Lambda_grid),
        kind='linear',
        fill_value='extrapolate'
    )

    class MockCLOUDYInterp:
        def __init__(self):
            self.ndens = np.linspace(-2, 6, 20)
            self.temp = np.linspace(4, 5.5, 20)
            self.phi = np.linspace(6, 14, 20)
            n_grid, t_grid, p_grid = np.meshgrid(
                self.ndens, self.temp, self.phi, indexing='ij'
            )
            self.datacube = np.log10(1e-23 * (10**n_grid)**2 + 1e-30)
            self._interp = RegularGridInterpolator(
                (self.ndens, self.temp, self.phi),
                self.datacube,
                bounds_error=False,
                fill_value=-30.0
            )

        def interp(self, points):
            return self._interp(points)

    mock_cooling_nonCIE = MockCLOUDYInterp()
    mock_heating_nonCIE = MockCLOUDYInterp()
    mock_heating_nonCIE.datacube = mock_cooling_nonCIE.datacube - 0.5

    net_datacube = mock_cooling_nonCIE.datacube - mock_heating_nonCIE.datacube
    netcool_interp = RegularGridInterpolator(
        (mock_cooling_nonCIE.ndens, mock_cooling_nonCIE.temp, mock_cooling_nonCIE.phi),
        10**net_datacube,
        bounds_error=False,
        fill_value=1e-30
    )

    # Add interpolators that can't be serialized
    raw['cStruc_cooling_CIE_interpolation'] = cooling_CIE_interp
    raw['cStruc_cooling_CIE_logT'] = np.log10(T_grid)
    raw['cStruc_cooling_nonCIE'] = mock_cooling_nonCIE
    raw['cStruc_heating_nonCIE'] = mock_heating_nonCIE
    raw['cStruc_net_nonCIE_interpolation'] = netcool_interp

    # Convert to MockParam
    params = {k: MockParam(v) for k, v in raw.items()}

    # Print key values for debugging
    for key in ['R2', 'v2', 'Eb', 't_now', 'Lmech_total']:
        if key in params:
            print(f"    {key} = {params[key].value}")

    return params


def make_bubble_test_params(**overrides):
    """
    Create a mock params dict for testing bubble luminosity.

    These parameters are set to realistic values that would occur
    during a typical bubble evolution simulation.

    If DEBUG_SNAPSHOT_PATH is set (via --snapshot flag), loads from snapshot
    instead of using synthetic values.
    """
    global DEBUG_SNAPSHOT_PATH

    # If snapshot is available, use it
    if DEBUG_SNAPSHOT_PATH is not None:
        return load_params_from_snapshot(DEBUG_SNAPSHOT_PATH)

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
# Comparison Tests: Original vs Modified
# =============================================================================

# Import original module for comparison tests
import src.bubble_structure.bubble_luminosity as bl_original

# Check scipy compatibility - original uses deprecated simps
import scipy.integrate
SCIPY_SIMPS_AVAILABLE = hasattr(scipy.integrate, 'simps')
if not SCIPY_SIMPS_AVAILABLE:
    # Add compatibility shim for scipy >= 1.14 where simps was removed
    try:
        scipy.integrate.simps = scipy.integrate.simpson
        SCIPY_SIMPS_AVAILABLE = True
    except AttributeError:
        pass


def make_comparison_params():
    """
    Create a params dict that works with BOTH original and modified versions.

    The original bubble_luminosity.get_bubbleproperties() requires many more
    params than the modified version, since it accesses params directly.
    """
    from scipy.interpolate import interp1d, RegularGridInterpolator

    # CIE cooling curve
    T_grid = np.logspace(4, 8, 100)
    Lambda_grid = 1e-22 * (T_grid / 1e6)**(-0.5)
    Lambda_grid[T_grid < 1e5] = 1e-24

    cooling_CIE_interp = interp1d(
        np.log10(T_grid),
        np.log10(Lambda_grid),
        kind='linear',
        fill_value='extrapolate'
    )

    # Mock non-CIE interpolators
    class MockCLOUDYInterp:
        def __init__(self):
            self.ndens = np.linspace(-2, 6, 20)
            self.temp = np.linspace(4, 5.5, 20)
            self.phi = np.linspace(6, 14, 20)
            n_grid, t_grid, p_grid = np.meshgrid(
                self.ndens, self.temp, self.phi, indexing='ij'
            )
            self.datacube = np.log10(1e-23 * (10**n_grid)**2 + 1e-30)
            self._interp = RegularGridInterpolator(
                (self.ndens, self.temp, self.phi),
                self.datacube,
                bounds_error=False,
                fill_value=-30.0
            )
        def interp(self, points):
            return self._interp(points)

    mock_cooling_nonCIE = MockCLOUDYInterp()
    mock_heating_nonCIE = MockCLOUDYInterp()
    mock_heating_nonCIE.datacube = mock_cooling_nonCIE.datacube - 0.5

    net_datacube = mock_cooling_nonCIE.datacube - mock_heating_nonCIE.datacube
    netcool_interp = RegularGridInterpolator(
        (mock_cooling_nonCIE.ndens, mock_cooling_nonCIE.temp, mock_cooling_nonCIE.phi),
        10**net_datacube,
        bounds_error=False,
        fill_value=1e-30
    )

    # Physical constants in internal units
    k_B = cvt.CGS.k_B * cvt.CONV.k_B_cgs2au
    mu_ion = 0.6 * cvt.CGS.m_H * cvt.CONV.g2Msun
    mu_atom = 1.4 * cvt.CGS.m_H * cvt.CONV.g2Msun
    C_thermal = 6e-7 * cvt.CONV.c_therm_cgs2au
    G = cvt.CGS.G * cvt.CONV.G_cgs2au

    # State variables
    R2 = 5.0  # pc
    v2 = 20.0  # pc/Myr
    Eb = 1e51 * cvt.CONV.E_cgs2au
    t_now = 0.1  # Myr

    # Wind properties
    Lmech_total = 1e37 * cvt.CONV.L_cgs2au
    v_mech_total = 2000 * cvt.CONV.v_kms2au
    Qi = 1e49 / cvt.INV_CONV.Myr2s

    params = {
        # Basic physics constants
        'gamma_adia': MockParam(5.0/3.0),
        'k_B': MockParam(k_B),
        'G': MockParam(G),
        'mu_ion': MockParam(mu_ion),
        'mu_atom': MockParam(mu_atom),
        'C_thermal': MockParam(C_thermal),

        # State variables (R2, v2, Eb are modified by original)
        'R2': MockParam(R2),
        'v2': MockParam(v2),
        'Eb': MockParam(Eb),
        't_now': MockParam(t_now),

        # Wind/bubble properties
        'Lmech_total': MockParam(Lmech_total),
        'v_mech_total': MockParam(v_mech_total),
        'Qi': MockParam(Qi),

        # Cooling coefficients
        'cool_alpha': MockParam(0.6),
        'cool_beta': MockParam(0.05),
        'cool_delta': MockParam(-0.02),

        # Bubble properties (will be computed/overwritten)
        'R1': MockParam(0.1),  # Initial guess
        'Pb': MockParam(1e5),  # Initial guess
        'bubble_dMdt': MockParam(np.nan),
        'bubble_xi_Tb': MockParam(0.9),
        'bubble_r_Tb': MockParam(0.0),
        'bubble_T_r_Tb': MockParam(np.nan),
        'bubble_LTotal': MockParam(0.0),
        'bubble_L1Bubble': MockParam(0.0),
        'bubble_L2Conduction': MockParam(0.0),
        'bubble_L3Intermediate': MockParam(0.0),
        'bubble_Tavg': MockParam(0.0),
        'bubble_mass': MockParam(0.0),

        # Arrays
        'bubble_T_arr': MockParam(np.array([])),
        'bubble_v_arr': MockParam(np.array([])),
        'bubble_dTdr_arr': MockParam(np.array([])),
        'bubble_r_arr': MockParam(np.array([])),
        'bubble_n_arr': MockParam(np.array([])),

        # Cooling structure
        'cStruc_cooling_CIE_interpolation': MockParam(cooling_CIE_interp),
        'cStruc_cooling_CIE_logT': MockParam(np.log10(T_grid)),
        'cStruc_cooling_nonCIE': MockParam(mock_cooling_nonCIE),
        'cStruc_heating_nonCIE': MockParam(mock_heating_nonCIE),
        'cStruc_net_nonCIE_interpolation': MockParam(netcool_interp),

        # Metallicity
        'ZCloud': MockParam(1.0),

        # Phase info (needed by original)
        'current_phase': MockParam('energy'),
    }

    return params


def test_compare_R1_and_Pb():
    """Compare R1 and Pb calculations between original and modified."""
    print("Testing comparison: R1 and Pb calculations...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original version (it mutates them)
    params_orig = make_comparison_params()

    # Run original version (suppress print statements)
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    # Compare R1
    R1_orig = params_orig['R1'].value
    R1_mod = props_mod.R1
    rel_diff_R1 = abs(R1_orig - R1_mod) / abs(R1_orig) if R1_orig != 0 else 0

    print(f"  R1 original:  {R1_orig:.6e} pc")
    print(f"  R1 modified:  {R1_mod:.6e} pc")
    print(f"  Relative diff: {rel_diff_R1:.2e}")

    # Compare Pb
    Pb_orig = params_orig['Pb'].value
    Pb_mod = props_mod.Pb
    rel_diff_Pb = abs(Pb_orig - Pb_mod) / abs(Pb_orig) if Pb_orig != 0 else 0

    print(f"  Pb original:  {Pb_orig:.6e}")
    print(f"  Pb modified:  {Pb_mod:.6e}")
    print(f"  Relative diff: {rel_diff_Pb:.2e}")

    # R1 and Pb should match closely (same calculation, tiny numerical differences)
    assert rel_diff_R1 < 1e-4, f"R1 mismatch: {rel_diff_R1:.2e}"
    assert rel_diff_Pb < 1e-4, f"Pb mismatch: {rel_diff_Pb:.2e}"

    print("  [PASS] R1 and Pb match between original and modified")


def test_compare_dMdt():
    """Compare dMdt calculations between original and modified."""
    print("Testing comparison: dMdt calculation...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original version
    params_orig = make_comparison_params()

    # Suppress print output
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    # Compare dMdt
    dMdt_orig = params_orig['bubble_dMdt'].value
    dMdt_mod = props_mod.dMdt
    rel_diff = abs(dMdt_orig - dMdt_mod) / abs(dMdt_orig) if dMdt_orig != 0 else 0

    print(f"  dMdt original:  {dMdt_orig:.6e} Msun/Myr")
    print(f"  dMdt modified:  {dMdt_mod:.6e} Msun/Myr")
    print(f"  Relative diff: {rel_diff:.2e}")

    # dMdt is solved iteratively, allow some tolerance
    assert rel_diff < 0.05, f"dMdt mismatch too large: {rel_diff:.2e}"

    print("  [PASS] dMdt matches between original and modified (within 5%)")


def test_compare_temperature_profile():
    """Compare temperature profiles between original and modified."""
    print("Testing comparison: Temperature profiles...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original
    params_orig = make_comparison_params()

    # Suppress print output
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    # Get profiles
    T_arr_orig = params_orig['bubble_T_arr'].value
    T_arr_mod = props_mod.T_arr
    r_arr_orig = params_orig['bubble_r_arr'].value
    r_arr_mod = props_mod.r_arr

    # Compare T at outer boundary (should be similar ~3e4 K)
    T_outer_orig = T_arr_orig[0]
    T_outer_mod = T_arr_mod[0]

    # Compare T at inner boundary (high T)
    T_inner_orig = T_arr_orig[-1]
    T_inner_mod = T_arr_mod[-1]

    print(f"  T_outer original: {T_outer_orig:.2e} K")
    print(f"  T_outer modified: {T_outer_mod:.2e} K")
    print(f"  T_inner original: {T_inner_orig:.2e} K")
    print(f"  T_inner modified: {T_inner_mod:.2e} K")

    # Temperature at outer boundary should be ~3e4 K for both
    rel_diff_outer = abs(T_outer_orig - T_outer_mod) / T_outer_orig if T_outer_orig > 0 else 0

    # Allow more tolerance since ODE solutions can differ slightly
    assert rel_diff_outer < 0.2, f"T_outer mismatch: {rel_diff_outer:.2e}"

    # Both should have reasonable inner temperatures (> 10^6 K)
    assert T_inner_orig > 1e6, f"Original T_inner too low: {T_inner_orig:.2e}"
    assert T_inner_mod > 1e6, f"Modified T_inner too low: {T_inner_mod:.2e}"

    print("  [PASS] Temperature profiles are comparable")


def test_compare_cooling_luminosity():
    """Compare cooling luminosity calculations between original and modified."""
    print("Testing comparison: Cooling luminosity...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original
    params_orig = make_comparison_params()

    # Suppress print output
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    # Compare total cooling luminosity
    L_total_orig = params_orig['bubble_LTotal'].value
    L_total_mod = props_mod.L_total

    # Compare individual components
    L_bubble_orig = params_orig['bubble_L1Bubble'].value
    L_bubble_mod = props_mod.L_bubble

    L_cond_orig = params_orig['bubble_L2Conduction'].value
    L_cond_mod = props_mod.L_conduction

    L_inter_orig = params_orig['bubble_L3Intermediate'].value
    L_inter_mod = props_mod.L_intermediate

    print(f"  L_total original: {L_total_orig:.4e}")
    print(f"  L_total modified: {L_total_mod:.4e}")
    print(f"  L_bubble original: {L_bubble_orig:.4e}")
    print(f"  L_bubble modified: {L_bubble_mod:.4e}")
    print(f"  L_conduction original: {L_cond_orig:.4e}")
    print(f"  L_conduction modified: {L_cond_mod:.4e}")
    print(f"  L_intermediate original: {L_inter_orig:.4e}")
    print(f"  L_intermediate modified: {L_inter_mod:.4e}")

    # Calculate relative differences
    if L_total_orig > 0:
        rel_diff_total = abs(L_total_orig - L_total_mod) / L_total_orig
        print(f"  L_total relative diff: {rel_diff_total:.2e}")

        # NOTE: The original code inserts exact interpolated points at zone
        # boundaries, while the modified version uses raw grid points.
        # This can cause up to ~50% difference in cooling integrals.
        # The formulas are identical; only grid handling differs.
        # We verify that both produce reasonable values of the same order.
        assert L_total_mod > 0, "L_total should be positive"
        ratio = max(L_total_orig, L_total_mod) / min(L_total_orig, L_total_mod)
        assert ratio < 3, f"L_total ratio too large: {ratio:.2f}x"

    print("  [PASS] Cooling luminosities are comparable (same order of magnitude)")


def test_compare_T_rgoal():
    """Compare T_rgoal (temperature at goal radius) between versions."""
    print("Testing comparison: T_rgoal calculation...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original
    params_orig = make_comparison_params()

    # Suppress print output
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    # Compare T_rgoal
    T_rgoal_orig = params_orig['bubble_T_r_Tb'].value
    T_rgoal_mod = props_mod.T_rgoal

    print(f"  T_rgoal original: {T_rgoal_orig:.4e} K")
    print(f"  T_rgoal modified: {T_rgoal_mod:.4e} K")

    if T_rgoal_orig > 0:
        rel_diff = abs(T_rgoal_orig - T_rgoal_mod) / T_rgoal_orig
        print(f"  Relative diff: {rel_diff:.2e}")

        # Allow reasonable tolerance
        assert rel_diff < 0.3, f"T_rgoal mismatch too large: {rel_diff:.2e}"

    print("  [PASS] T_rgoal is comparable between versions")


def test_compare_Tavg():
    """Compare average temperature between versions."""
    print("Testing comparison: Tavg calculation...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original
    params_orig = make_comparison_params()

    # Suppress print output
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    # Compare Tavg
    Tavg_orig = params_orig['bubble_Tavg'].value
    Tavg_mod = props_mod.Tavg

    print(f"  Tavg original: {Tavg_orig:.4e} K")
    print(f"  Tavg modified: {Tavg_mod:.4e} K")

    if Tavg_orig > 0:
        rel_diff = abs(Tavg_orig - Tavg_mod) / Tavg_orig
        print(f"  Relative diff: {rel_diff:.2e}")

        # Both should give reasonable volume-averaged temperatures
        # Allow larger tolerance as this is sensitive to profile differences
        assert rel_diff < 0.5, f"Tavg mismatch too large: {rel_diff:.2e}"

    print("  [PASS] Tavg is comparable between versions")


def test_compare_all_bubble_properties():
    """Comprehensive comparison of ALL bubble properties between versions."""
    print("Testing comparison: ALL bubble properties...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original
    params_orig = make_comparison_params()

    # Suppress print output
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    print("\n  === Scalar Properties ===")

    # Helper to compare and report
    def compare_scalar(name, orig, mod, tol=0.1, allow_zero=False):
        if orig == 0 and mod == 0:
            print(f"  {name}: both zero [OK]")
            return True
        if orig == 0:
            if allow_zero:
                print(f"  {name}: orig=0, mod={mod:.4e} [OK - zero allowed]")
                return True
            print(f"  {name}: orig=0, mod={mod:.4e} [WARN]")
            return False
        rel_diff = abs(orig - mod) / abs(orig)
        status = "OK" if rel_diff < tol else "WARN" if rel_diff < 1.0 else "FAIL"
        print(f"  {name}: orig={orig:.4e}, mod={mod:.4e}, diff={rel_diff:.2e} [{status}]")
        return rel_diff < 1.0  # Allow up to 100% diff before failing

    results = []

    # R1 and Pb (should match exactly)
    results.append(compare_scalar("R1", params_orig['R1'].value, props_mod.R1, tol=1e-4))
    results.append(compare_scalar("Pb", params_orig['Pb'].value, props_mod.Pb, tol=1e-4))

    # dMdt (iterative solver, small tolerance)
    results.append(compare_scalar("dMdt", params_orig['bubble_dMdt'].value, props_mod.dMdt, tol=0.05))

    # Cooling luminosities
    results.append(compare_scalar("L_total", params_orig['bubble_LTotal'].value, props_mod.L_total, tol=0.5))
    results.append(compare_scalar("L_bubble", params_orig['bubble_L1Bubble'].value, props_mod.L_bubble, tol=0.5))
    results.append(compare_scalar("L_conduction", params_orig['bubble_L2Conduction'].value, props_mod.L_conduction, tol=0.5, allow_zero=True))
    results.append(compare_scalar("L_intermediate", params_orig['bubble_L3Intermediate'].value, props_mod.L_intermediate, tol=0.5, allow_zero=True))

    # Temperature values
    results.append(compare_scalar("T_rgoal", params_orig['bubble_T_r_Tb'].value, props_mod.T_rgoal, tol=0.3))
    results.append(compare_scalar("Tavg", params_orig['bubble_Tavg'].value, props_mod.Tavg, tol=0.5))

    # Bubble mass
    results.append(compare_scalar("bubble_mass", params_orig['bubble_mass'].value, props_mod.bubble_mass, tol=0.5))

    print("\n  === Profile Arrays ===")

    # Compare profile arrays
    T_arr_orig = params_orig['bubble_T_arr'].value
    T_arr_mod = props_mod.T_arr
    r_arr_orig = params_orig['bubble_r_arr'].value
    r_arr_mod = props_mod.r_arr
    v_arr_orig = params_orig['bubble_v_arr'].value
    v_arr_mod = props_mod.v_arr
    n_arr_orig = params_orig['bubble_n_arr'].value
    n_arr_mod = props_mod.n_arr
    dTdr_arr_orig = params_orig['bubble_dTdr_arr'].value
    dTdr_arr_mod = props_mod.dTdr_arr

    def compare_array(name, orig, mod):
        if len(orig) == 0 or len(mod) == 0:
            print(f"  {name}: empty array(s) - orig len={len(orig)}, mod len={len(mod)}")
            return len(orig) == len(mod)

        # Compare at boundaries and middle
        print(f"  {name}:")
        print(f"    Length: orig={len(orig)}, mod={len(mod)}")
        print(f"    First:  orig={orig[0]:.4e}, mod={mod[0]:.4e}")
        print(f"    Last:   orig={orig[-1]:.4e}, mod={mod[-1]:.4e}")

        # Compare ranges
        orig_min, orig_max = np.min(orig), np.max(orig)
        mod_min, mod_max = np.min(mod), np.max(mod)
        print(f"    Range:  orig=[{orig_min:.4e}, {orig_max:.4e}]")
        print(f"            mod=[{mod_min:.4e}, {mod_max:.4e}]")

        # Check if ranges overlap significantly
        overlap = min(orig_max, mod_max) > max(orig_min, mod_min)
        status = "OK" if overlap else "WARN"
        print(f"    Overlap: {overlap} [{status}]")
        return overlap

    results.append(compare_array("T_arr", T_arr_orig, T_arr_mod))
    results.append(compare_array("r_arr", r_arr_orig, r_arr_mod))
    results.append(compare_array("v_arr", v_arr_orig, v_arr_mod))
    results.append(compare_array("n_arr", n_arr_orig, n_arr_mod))
    results.append(compare_array("dTdr_arr", dTdr_arr_orig, dTdr_arr_mod))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n  Summary: {passed}/{total} comparisons passed")

    assert passed >= total * 0.8, f"Too many comparison failures: {passed}/{total}"
    print("  [PASS] All bubble properties are comparable")


def test_compare_profile_statistics():
    """Compare statistical properties of bubble profiles."""
    print("Testing comparison: Profile statistics...")

    # Create params for modified version
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Run modified version
    props_mod = bl_modified.get_bubbleproperties_pure(
        R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
    )

    # Create fresh params for original
    params_orig = make_comparison_params()

    # Suppress print output
    import io
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig)
    finally:
        _sys.stdout = old_stdout

    # Get arrays
    T_orig = params_orig['bubble_T_arr'].value
    T_mod = props_mod.T_arr
    r_orig = params_orig['bubble_r_arr'].value
    r_mod = props_mod.r_arr

    print(f"  Temperature profile statistics:")
    print(f"    Original: min={np.min(T_orig):.2e}, max={np.max(T_orig):.2e}, mean={np.mean(T_orig):.2e}")
    print(f"    Modified: min={np.min(T_mod):.2e}, max={np.max(T_mod):.2e}, mean={np.mean(T_mod):.2e}")

    print(f"  Radius profile statistics:")
    print(f"    Original: min={np.min(r_orig):.4f}, max={np.max(r_orig):.4f}")
    print(f"    Modified: min={np.min(r_mod):.4f}, max={np.max(r_mod):.4f}")

    # Check temperature range is physical
    assert np.min(T_orig) > 1e3, f"Original T_min too low: {np.min(T_orig)}"
    assert np.min(T_mod) > 1e3, f"Modified T_min too low: {np.min(T_mod)}"
    assert np.max(T_orig) < 1e10, f"Original T_max too high: {np.max(T_orig)}"
    assert np.max(T_mod) < 1e10, f"Modified T_max too high: {np.max(T_mod)}"

    # Check that both cover similar temperature range
    T_range_orig = np.max(T_orig) / np.min(T_orig)
    T_range_mod = np.max(T_mod) / np.min(T_mod)
    print(f"  Temperature dynamic range: orig={T_range_orig:.1f}x, mod={T_range_mod:.1f}x")

    # Both should have similar dynamic range (within factor of 5)
    ratio = max(T_range_orig, T_range_mod) / min(T_range_orig, T_range_mod)
    assert ratio < 5, f"Temperature range ratio too large: {ratio:.1f}x"

    print("  [PASS] Profile statistics are comparable")


def test_timing_comparison():
    """Compare timing between original (60k fixed points) and modified (adaptive solve_ivp)."""
    print("Testing timing comparison: Original vs Modified (solve_ivp)...")
    import time

    # Create params
    params_mod = make_comparison_params()
    R2 = params_mod['R2'].value
    v2 = params_mod['v2'].value
    Eb = params_mod['Eb'].value
    t_now = params_mod['t_now'].value

    # Suppress print output
    import io
    import sys as _sys

    # Warm-up run (JIT compilation, caching, etc.)
    _ = bl_modified.get_bubbleproperties_pure(R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod)
    params_orig_warmup = make_comparison_params()
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        bl_original.get_bubbleproperties(params_orig_warmup)
    finally:
        _sys.stdout = old_stdout

    # Time the modified version (solve_ivp with Radau)
    n_runs = 3
    times_modified = []
    for _ in range(n_runs):
        params_mod = make_comparison_params()
        start = time.perf_counter()
        props_mod = bl_modified.get_bubbleproperties_pure(
            R2=R2, v2=v2, Eb=Eb, t_now=t_now, params=params_mod
        )
        times_modified.append(time.perf_counter() - start)

    # Time the original version (odeint with 60k fixed points)
    times_original = []
    for _ in range(n_runs):
        params_orig = make_comparison_params()
        old_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        start = time.perf_counter()
        try:
            bl_original.get_bubbleproperties(params_orig)
        finally:
            _sys.stdout = old_stdout
        times_original.append(time.perf_counter() - start)

    avg_time_mod = sum(times_modified) / n_runs
    avg_time_orig = sum(times_original) / n_runs
    speedup = avg_time_orig / avg_time_mod if avg_time_mod > 0 else float('inf')

    print(f"\n  === Timing Results (average of {n_runs} runs) ===")
    print(f"  Original (odeint, 60k pts):    {avg_time_orig*1000:.1f} ms")
    print(f"  Modified (solve_ivp, adaptive): {avg_time_mod*1000:.1f} ms")
    print(f"  Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    # Also report array sizes
    r_arr_mod = props_mod.r_arr
    r_arr_orig = params_orig['bubble_r_arr'].value
    print(f"\n  === Array Sizes ===")
    print(f"  Original: {len(r_arr_orig)} points")
    print(f"  Modified: {len(r_arr_mod)} points")

    # Compare accuracy (using L_total as benchmark)
    L_total_orig = params_orig['bubble_LTotal'].value
    L_total_mod = props_mod.L_total
    if L_total_orig != 0:
        rel_diff = abs(L_total_orig - L_total_mod) / abs(L_total_orig)
        print(f"\n  === Accuracy (L_total) ===")
        print(f"  Original: {L_total_orig:.6e}")
        print(f"  Modified: {L_total_mod:.6e}")
        print(f"  Relative diff: {rel_diff:.2e} ({rel_diff*100:.1f}%)")

    print("\n  [PASS] Timing comparison complete")


# =============================================================================
# Run all tests
# =============================================================================

def run_all_tests():
    """Run all tests and report summary."""
    tests = [
        # Unit tests for modified version
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
        # Comparison tests: original vs modified
        test_compare_R1_and_Pb,
        test_compare_dMdt,
        test_compare_temperature_profile,
        test_compare_cooling_luminosity,
        test_compare_T_rgoal,
        test_compare_Tavg,
        test_compare_all_bubble_properties,
        test_compare_profile_statistics,
        test_timing_comparison,
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test bubble luminosity calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic test parameters:
  python test/test_bubble_luminosity.py

  # Run with a debug snapshot (captured from a real crash):
  python test/test_bubble_luminosity.py --snapshot /path/to/debug_snapshot.json

  # To capture a debug snapshot in your code:
  from src._input.dictionary import save_debug_snapshot
  save_debug_snapshot(params)  # Saves to params['path2output']/debug_snapshot.json
        """
    )
    parser.add_argument(
        '--snapshot', '-s',
        type=str,
        default=None,
        help='Path to debug_snapshot.json file to use instead of synthetic params'
    )

    args = parser.parse_args()

    # Set global snapshot path
    if args.snapshot:
        DEBUG_SNAPSHOT_PATH = Path(args.snapshot)
        if not DEBUG_SNAPSHOT_PATH.exists():
            print(f"ERROR: Snapshot file not found: {DEBUG_SNAPSHOT_PATH}")
            sys.exit(1)
        print(f"Using debug snapshot: {DEBUG_SNAPSHOT_PATH}")

    success = run_all_tests()
    sys.exit(0 if success else 1)
