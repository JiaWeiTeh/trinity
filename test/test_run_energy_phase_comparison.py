#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification test comparing energy_phase_ODEs and energy_phase_ODEs_modified.

This test loads snapshots from the dictionary.jsonl file and verifies that
both the original get_ODE_Edot() and the new get_ODE_Edot_pure()
return identical results.

Author: TRINITY Team
"""

import numpy as np
import scipy.interpolate
import json
import sys
import os
import random
import time

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_energy.energy_phase_ODEs import get_ODE_Edot
from src.phase1_energy.energy_phase_ODEs_modified import (
    get_ODE_Edot_pure, create_ODE_snapshot, ODESnapshot
)
from src._functions.unit_conversions import CONV, CGS


# =============================================================================
# Project root for finding data files
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Helper class for mock parameters
# =============================================================================

class MockParam:
    """Mock parameter object with .value attribute, mimics DescribedItem."""
    def __init__(self, value):
        self.value = value

    def _unwrap(self, other):
        return other.value if isinstance(other, MockParam) else other

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __lt__(self, other):
        return self.value < self._unwrap(other)

    def __le__(self, other):
        return self.value <= self._unwrap(other)

    def __gt__(self, other):
        return self.value > self._unwrap(other)

    def __ge__(self, other):
        return self.value >= self._unwrap(other)

    def __eq__(self, other):
        return self.value == self._unwrap(other)

    def __ne__(self, other):
        return self.value != self._unwrap(other)

    def __add__(self, other):
        return self.value + self._unwrap(other)

    def __radd__(self, other):
        return self._unwrap(other) + self.value

    def __sub__(self, other):
        return self.value - self._unwrap(other)

    def __rsub__(self, other):
        return self._unwrap(other) - self.value

    def __mul__(self, other):
        return self.value * self._unwrap(other)

    def __rmul__(self, other):
        return self._unwrap(other) * self.value

    def __truediv__(self, other):
        return self.value / self._unwrap(other)

    def __rtruediv__(self, other):
        return self._unwrap(other) / self.value

    def __pow__(self, other):
        return self.value ** self._unwrap(other)

    def __rpow__(self, other):
        return self._unwrap(other) ** self.value

    def __neg__(self):
        return -self.value

    def __abs__(self):
        return abs(self.value)

    def __repr__(self):
        return f"MockParam({self.value})"


def load_snapshot_from_jsonl(filepath: str, line_number: int) -> dict:
    """
    Load a specific snapshot (line) from a JSONL file.

    Parameters
    ----------
    filepath : str
        Path to the JSONL file
    line_number : int
        1-indexed line number to load

    Returns
    -------
    dict
        Parsed JSON data from the specified line
    """
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number - 1:  # Convert to 0-indexed
                return json.loads(line.strip())
    raise ValueError(f"Line {line_number} not found in {filepath}")


def add_physical_constants(params: dict) -> dict:
    """
    Add physical constants that are missing from the dictionary snapshot.

    These constants are normally set during initialization but are not
    saved in the dictionary.jsonl output file.
    """
    # Unit conversion factors
    ndens_cgs2au = CONV.ndens_cgs2au      # cm^-3 -> pc^-3
    m_H_to_Msun = CGS.m_H * CONV.g2Msun   # m_H unit -> Msun
    k_B_cgs2au = CONV.k_B_cgs2au          # erg/K -> AU
    G_cgs2au = CONV.G_cgs2au              # CGS G -> AU
    v_cms2au = CONV.v_cms2au              # cm/s -> pc/Myr

    # Boltzmann constant in AU
    params['k_B'] = CGS.k_B * k_B_cgs2au

    # Gravitational constant in AU
    params['G'] = CGS.G * G_cgs2au

    # Speed of light in AU
    params['c_light'] = CGS.c * v_cms2au

    # Mean molecular weights (in Msun)
    params['mu_atom'] = 2.3 * m_H_to_Msun
    params['mu_ion'] = 1.4 * m_H_to_Msun
    params['mu_convert'] = 1.4 * m_H_to_Msun  # For mass density conversion

    # Shell temperatures (K)
    params['TShell_ion'] = 1e4
    params['TShell_neu'] = 100

    # Case B recombination coefficient
    caseB_alpha_cgs = 2.54e-13
    cm3_to_pc3 = CONV.cm2pc**3
    s_to_Myr = CONV.s2Myr
    params['caseB_alpha'] = caseB_alpha_cgs * cm3_to_pc3 / s_to_Myr

    # Dust cross section
    dust_sigma_cgs = 1e-21
    cm2_to_pc2 = CONV.cm2pc**2
    params['dust_sigma'] = dust_sigma_cgs * cm2_to_pc2

    # Dust IR opacity
    dust_KappaIR_cgs = 4.0
    params['dust_KappaIR'] = dust_KappaIR_cgs * cm2_to_pc2 / CONV.g2Msun

    # Shell dissolution threshold
    params['stop_n_diss'] = 0.1 * ndens_cgs2au

    # ISM number density
    if 'nISM' not in params:
        params['nISM'] = 1.0 * ndens_cgs2au

    # ZCloud for metallicity
    params['ZCloud'] = 1.0

    # Ensure key parameters exist with defaults
    if 'isCollapse' not in params:
        params['isCollapse'] = False
    if 'EarlyPhaseApproximation' not in params:
        params['EarlyPhaseApproximation'] = False
    if 'current_phase' not in params:
        params['current_phase'] = 'energy'
    if 'PISM' not in params:
        params['PISM'] = 1e4  # Typical ISM pressure

    # =========================================================================
    # Energy phase specific parameters
    # =========================================================================

    # Cluster mass (typically derived from cloud parameters)
    # Use a typical value for testing - in reality this comes from star formation
    if 'mCluster' not in params:
        # Assume ~1% star formation efficiency from cloud mass
        mCloud = params.get('mCloud', 1e7)
        params['mCluster'] = 0.01 * mCloud

    # Adiabatic index (gamma) for bubble calculations
    if 'gamma_adia' not in params:
        params['gamma_adia'] = 5.0 / 3.0  # Monatomic gas

    # Mechanical luminosity and wind velocity from SB99
    # These are normally calculated from feedback, but need defaults
    if 'Lmech_total' not in params:
        params['Lmech_total'] = 1e36 * CONV.L_cgs2au  # Typical value in AU

    if 'v_mech_total' not in params:
        params['v_mech_total'] = 1000e5 * v_cms2au  # 1000 km/s in AU

    # Start of star formation time
    if 'tSF' not in params:
        params['tSF'] = 0.0

    # =========================================================================
    # Cloud density profile parameters
    # =========================================================================

    # Core number density (in AU: pc^-3)
    if 'nCore' not in params:
        params['nCore'] = 1e4 * ndens_cgs2au  # 10^4 cm^-3 typical

    # Core radius
    if 'rCore' not in params:
        rCloud = params.get('rCloud', 10.0)
        params['rCore'] = 0.1 * rCloud  # Assume core is 10% of cloud radius

    # Density profile type
    if 'dens_profile' not in params:
        params['dens_profile'] = 'densPL'  # Power-law profile

    # Power-law exponent
    if 'densPL_alpha' not in params:
        params['densPL_alpha'] = -2.0  # Standard power-law

    # =========================================================================
    # Feedback force terms (ensure initialized)
    # =========================================================================

    if 'F_grav' not in params:
        params['F_grav'] = 0.0
    if 'F_ion_in' not in params:
        params['F_ion_in'] = 0.0
    if 'F_ion_out' not in params:
        params['F_ion_out'] = 0.0
    if 'F_ram' not in params:
        params['F_ram'] = 0.0

    return params


class MockSB99Interpolator:
    """Mock interpolator for SB99 data that returns constant values."""
    def __init__(self, value, t_min=0.0, t_max=100.0):
        self.value = value
        self.x = np.array([t_min, t_max])

    def __call__(self, t):
        return np.array(self.value)


def create_mock_sb99f(params: dict) -> dict:
    """
    Create a mock SB99f structure with interpolation functions.

    Uses values from the params dictionary if available, otherwise uses
    reasonable defaults for energy phase testing.

    The values in the snapshot are already in TRINITY's AU units:
    - LWind: Mechanical luminosity [Msun*pc^2/Myr^3]
    - pWindDot: Momentum rate [Msun*pc/Myr^2]
    - vWind: Wind velocity [pc/Myr]
    - Qi: Ionizing photon rate [s^-1] - but converted to AU units
    """
    # Get existing values from params - these are already in AU
    Qi = params.get('Qi', 1e49)  # Ionizing photon rate
    Lbol = params.get('Lbol', 1e10)  # Bolometric luminosity in AU
    Li = params.get('Li', 0.5 * Lbol)  # Ionizing luminosity
    Ln = params.get('Ln', 0.5 * Lbol)  # Non-ionizing luminosity

    # Mechanical luminosity - use LWind from snapshot (already in AU)
    Lmech = params.get('LWind', params.get('Lmech_total', 1e10))

    # Momentum rate from snapshot - use pWindDot (already in AU)
    pdot_W = params.get('pWindDot', 1e7)
    pdot_SN = 0.0  # No SN early in energy phase
    pdot_total = pdot_W + pdot_SN

    # Time range (ensure it covers the snapshot time)
    t_min = 0.0
    t_max = 100.0

    return {
        'fQi': MockSB99Interpolator(Qi, t_min, t_max),
        'fLi': MockSB99Interpolator(Li, t_min, t_max),
        'fLn': MockSB99Interpolator(Ln, t_min, t_max),
        'fLbol': MockSB99Interpolator(Lbol, t_min, t_max),
        'fLmech_W': MockSB99Interpolator(Lmech, t_min, t_max),
        'fLmech_SN': MockSB99Interpolator(0.0, t_min, t_max),  # No SN early
        'fLmech_total': MockSB99Interpolator(Lmech, t_min, t_max),
        'fpdot_W': MockSB99Interpolator(pdot_W, t_min, t_max),
        'fpdot_SN': MockSB99Interpolator(pdot_SN, t_min, t_max),
        'fpdot_total': MockSB99Interpolator(pdot_total, t_min, t_max),
    }


def prime_params(params: dict) -> dict:
    """
    Initialize parameters that require setup beyond simple values.

    This function sets up:
    - Cooling interpolation (CIE and non-CIE)
    - Starburst99 interpolation tables
    - Any other computed/interpolated parameters
    """
    # =========================================================================
    # Cooling interpolation (CIE)
    # =========================================================================
    cie_files = {
        1: 'lib/cooling/CIE/coolingCIE_1_Cloudy.dat',
        2: 'lib/cooling/CIE/coolingCIE_2_Cloudy_grains.dat',
        3: 'lib/cooling/CIE/coolingCIE_3_Gnat-Ferland2012.dat',
        4: 'lib/cooling/CIE/coolingCIE_4_Sutherland-Dopita1993.dat',
    }

    cooling_loaded = False
    for cie_choice in [3, 4, 1, 2]:
        cooling_path = os.path.join(PROJECT_ROOT, cie_files.get(cie_choice, ''))
        if os.path.exists(cooling_path):
            try:
                logT, logLambda = np.loadtxt(cooling_path, unpack=True)
                cooling_CIE_interpolation = scipy.interpolate.interp1d(
                    logT, logLambda, kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                params['path_cooling_CIE'] = cooling_path
                params['cStruc_cooling_CIE_logT'] = logT
                params['cStruc_cooling_CIE_logLambda'] = logLambda
                params['cStruc_cooling_CIE_interpolation'] = cooling_CIE_interpolation
                cooling_loaded = True
                break
            except Exception as e:
                print(f"Warning: Could not load cooling curve from {cooling_path}: {e}")

    if not cooling_loaded:
        logT = np.linspace(4, 9, 100)
        logLambda = -22 + 0.5 * (logT - 6)
        cooling_CIE_interpolation = scipy.interpolate.interp1d(
            logT, logLambda, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        params['path_cooling_CIE'] = 'mock_cooling'
        params['cStruc_cooling_CIE_logT'] = logT
        params['cStruc_cooling_CIE_logLambda'] = logLambda
        params['cStruc_cooling_CIE_interpolation'] = cooling_CIE_interpolation

    # Non-CIE cooling
    if 'cStruc_cooling_nonCIE' not in params:
        params['cStruc_cooling_nonCIE'] = None
    if 'cStruc_heating_nonCIE' not in params:
        params['cStruc_heating_nonCIE'] = None
    if 'cStruc_net_nonCIE_interpolation' not in params:
        params['cStruc_net_nonCIE_interpolation'] = None

    if 't_previousCoolingUpdate' not in params:
        params['t_previousCoolingUpdate'] = 0.0

    # =========================================================================
    # Starburst99 feedback (SB99f) - use mock for testing
    # =========================================================================
    # The SB99f structure contains interpolation functions for stellar feedback
    # For testing, we create a mock that returns values from the snapshot
    if 'SB99f' not in params:
        params['SB99f'] = create_mock_sb99f(params)

    # Ensure Lmech_total in params matches what feedback will return
    # This is needed because original code reads params["Lmech_total"] directly
    # while also calling get_currentSB99feedback() which returns LWind
    # The modified version consistently uses feedback values
    Lmech_from_sb99 = params.get('LWind', params.get('Lmech_total', 1e10))
    params['Lmech_total'] = Lmech_from_sb99

    # Same for v_mech_total - use vWind from snapshot if available
    v_mech_from_sb99 = params.get('vWind', params.get('v_mech_total', 1000))
    params['v_mech_total'] = v_mech_from_sb99

    return params


def make_params_dict(snapshot: dict, include_priming: bool = True) -> dict:
    """
    Create a params dictionary with MockParam wrappers from a snapshot.
    """
    # Add physical constants
    snapshot = add_physical_constants(snapshot)

    # Prime params with interpolation objects
    if include_priming:
        snapshot = prime_params(snapshot)

    # Wrap all values in MockParam
    return {k: MockParam(v) for k, v in snapshot.items()}


def compare_values(name: str, original, modified, rtol: float = 1e-10):
    """
    Compare two values and return (passed, rel_diff, message).
    """
    if isinstance(original, bool):
        passed = (original == modified)
        return passed, 0.0 if passed else 1.0, "bool"

    elif isinstance(original, np.ndarray):
        if not isinstance(modified, np.ndarray):
            return False, 1.0, f"type mismatch: array vs {type(modified)}"
        if original.shape != modified.shape:
            return False, 1.0, f"shape mismatch {original.shape} vs {modified.shape}"
        if np.allclose(original, modified, rtol=rtol, equal_nan=True):
            return True, 0.0, "arrays match"
        else:
            max_diff = np.max(np.abs(original - modified))
            return False, max_diff, f"arrays differ, max_diff={max_diff:.6e}"

    elif isinstance(original, (int, float)):
        if np.isnan(original) and np.isnan(modified):
            return True, 0.0, "both NaN"
        if np.isclose(original, modified, rtol=rtol):
            rel_diff = abs(original - modified) / max(abs(original), 1e-300)
            return True, rel_diff, "match"
        else:
            rel_diff = abs(original - modified) / max(abs(original), abs(modified), 1e-300)
            return False, rel_diff, f"differ: {original:.6e} vs {modified:.6e}"

    else:
        passed = (original == modified)
        return passed, 0.0 if passed else 1.0, "other"


def test_snapshot(snapshot: dict, verbose: bool = False) -> dict:
    """
    Test a single snapshot and return results dictionary.

    Returns dict with:
        - 'passed': bool
        - 't_now': float
        - 'field_results': dict mapping field_name -> (passed, rel_diff, original_val, pct_error)
        - 'time_original': float (seconds)
        - 'time_modified': float (seconds)
        - 'speedup': float
    """
    params_original = make_params_dict(snapshot.copy())
    params_modified = make_params_dict(snapshot.copy())

    # Extract state for ODE evaluation
    t_now = params_original['t_now'].value
    R2 = params_original['R2'].value
    v2 = params_original['v2'].value
    Eb = params_original['Eb'].value
    y = [R2, v2, Eb]

    # Run original version with timing
    start_orig = time.perf_counter()
    result_original = get_ODE_Edot(y.copy(), t_now, params_original)
    time_original = time.perf_counter() - start_orig

    # Create snapshot for modified version
    snapshot_obj = create_ODE_snapshot(params_modified)

    # Run modified version with timing
    start_mod = time.perf_counter()
    result_modified = get_ODE_Edot_pure(t_now, y.copy(), snapshot_obj, params_modified)
    time_modified = time.perf_counter() - start_mod

    # Calculate speedup
    speedup = time_original / time_modified if time_modified > 0 else float('inf')

    # Compare ODE outputs [rd, vd, Ed]
    fields = ['rd', 'vd', 'Ed']
    field_results = {}
    all_passed = True

    for i, field_name in enumerate(fields):
        original_val = result_original[i]
        modified_val = result_modified[i]

        passed, rel_diff, msg = compare_values(field_name, original_val, modified_val)
        # Calculate % error
        if isinstance(original_val, (int, float)) and not np.isnan(original_val) and abs(original_val) > 1e-300:
            pct_error = 100.0 * abs(original_val - modified_val) / abs(original_val)
        else:
            pct_error = 0.0 if passed else 100.0

        field_results[field_name] = (passed, rel_diff, original_val, pct_error)
        if not passed:
            all_passed = False
            if verbose:
                print(f"  ✗ {field_name}: rel_diff={rel_diff:.2e}, %error={pct_error:.2e}%")

    return {
        'passed': all_passed,
        't_now': snapshot.get('t_now', 0),
        'field_results': field_results,
        'time_original': time_original,
        'time_modified': time_modified,
        'speedup': speedup,
    }


def get_total_lines(filepath: str) -> int:
    """Count total lines in a file."""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


def print_comparison_table(results: list, fields: list):
    """Print a formatted comparison table with timing and % error."""
    # Header
    print("\n" + "=" * 130)
    print("ENERGY PHASE ODE COMPARISON TABLE")
    print("=" * 130)

    # Column headers for main table
    header = f"{'Snap':<6} {'t_now':<11} {'Status':<6} {'Orig(ms)':<9} {'Mod(ms)':<9} {'Speedup':<8}"
    for field in fields:
        header += f" {field:<14}"
    print(header)
    print("-" * 130)

    # Data rows
    total_time_orig = 0.0
    total_time_mod = 0.0

    for i, res in enumerate(results):
        line = res.get('line', i)
        t_now = res['t_now']
        status = "PASS" if res['passed'] else "FAIL"
        time_orig_ms = res.get('time_original', 0) * 1000
        time_mod_ms = res.get('time_modified', 0) * 1000
        speedup = res.get('speedup', 1.0)

        total_time_orig += res.get('time_original', 0)
        total_time_mod += res.get('time_modified', 0)

        row = f"{line:<6} {t_now:<11.4e} {status:<6} {time_orig_ms:<9.2f} {time_mod_ms:<9.2f} {speedup:<8.2f}x"
        for field in fields:
            if field in res['field_results']:
                passed, rel_diff, val, pct_error = res['field_results'][field]
                if passed:
                    row += f" {val:<14.4e}"
                else:
                    row += f" {val:<14.4e}*"
            else:
                row += f" {'N/A':<14}"
        print(row)

    print("-" * 130)

    # Timing summary
    avg_speedup = total_time_orig / total_time_mod if total_time_mod > 0 else 1.0
    print(f"\nTiming Summary:")
    print(f"  Total original time: {total_time_orig*1000:.2f} ms")
    print(f"  Total modified time: {total_time_mod*1000:.2f} ms")
    print(f"  Average speedup: {avg_speedup:.2f}x")

    # Error summary table
    print("\n" + "=" * 100)
    print("% ERROR PER FIELD (max across all snapshots)")
    print("=" * 100)

    # Collect max % error per field
    field_max_errors = {}
    for field in fields:
        max_err = 0.0
        for res in results:
            if field in res['field_results']:
                _, _, _, pct_error = res['field_results'][field]
                max_err = max(max_err, pct_error)
        field_max_errors[field] = max_err

    # Print error table
    for field, err in field_max_errors.items():
        print(f"  {field:<25}: {err:>12.2e}%")

    print("-" * 100)

    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    print(f"\nSummary: {passed_count}/{len(results)} snapshots passed all comparisons")
    if passed_count < len(results):
        print("* indicates field with mismatch")


def test_energy_phase_comparison():
    """
    Test energy_phase_ODEs vs energy_phase_ODEs_modified on 10 random snapshots.
    """
    print("=" * 70)
    print("Testing energy_phase_ODEs vs energy_phase_ODEs_modified")
    print("=" * 70)

    jsonl_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'comparison', '1e7_sfe020_n1e4_test_dictionary.jsonl'
    )

    if not os.path.exists(jsonl_path):
        print(f"ERROR: Test dictionary not found at {jsonl_path}")
        return False

    # Get total lines and determine valid range
    total_lines = get_total_lines(jsonl_path)
    max_line = min(300, total_lines)

    # Pick 10 random snapshots in range 10-300
    random.seed(42)  # For reproducibility
    snapshot_lines = sorted(random.sample(range(10, max_line + 1), min(10, max_line - 9)))

    print(f"\nTesting {len(snapshot_lines)} random snapshots from lines {snapshot_lines[0]}-{snapshot_lines[-1]}")
    print(f"Selected lines: {snapshot_lines}")

    results = []
    fields = ['rd', 'vd', 'Ed']

    for line_num in snapshot_lines:
        print(f"\nProcessing snapshot {line_num}...", end=" ")
        try:
            snapshot = load_snapshot_from_jsonl(jsonl_path, line_num)
            result = test_snapshot(snapshot, verbose=False)
            result['line'] = line_num
            results.append(result)
            status = "✓" if result['passed'] else "✗"
            print(f"{status} (t={result['t_now']:.4e})")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'passed': False, 'line': line_num, 't_now': 0, 'field_results': {}})

    # Print comparison table
    print_comparison_table(results, fields)

    all_passed = all(r['passed'] for r in results)
    return all_passed


if __name__ == '__main__':
    print("Energy Phase ODE Comparison Test")
    print("=================================\n")

    passed = test_energy_phase_comparison()

    print("\n" + "=" * 70)
    if passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
