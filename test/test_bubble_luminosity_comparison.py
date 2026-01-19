#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification test comparing bubble_luminosity and bubble_luminosity_modified.

This test loads snapshots from the dictionary.jsonl file and verifies that
both the original get_bubbleproperties() and the new get_bubbleproperties_pure()
return identical results.

Outputs a comparison table for 10 random snapshots in range 10-300.

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

from src.bubble_structure.bubble_luminosity import get_bubbleproperties
from src.bubble_structure.bubble_luminosity_modified import get_bubbleproperties_pure, BubbleProperties
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

    def __repr__(self):
        return f"MockParam({self.value})"


def load_snapshot_from_jsonl(filepath: str, line_number: int) -> dict:
    """Load a specific snapshot (line) from a JSONL file."""
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number - 1:
                return json.loads(line.strip())
    raise ValueError(f"Line {line_number} not found in {filepath}")


def get_total_lines(filepath: str) -> int:
    """Count total lines in a file."""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


def add_physical_constants(params: dict) -> dict:
    """
    Add physical constants that are missing from the dictionary snapshot.
    Includes both shell and bubble-specific constants.
    """
    # Unit conversion factors
    ndens_cgs2au = CONV.ndens_cgs2au
    m_H_to_Msun = CGS.m_H * CONV.g2Msun
    k_B_cgs2au = CONV.k_B_cgs2au
    G_cgs2au = CONV.G_cgs2au
    v_cms2au = CONV.v_cms2au

    # Boltzmann constant in AU
    params['k_B'] = CGS.k_B * k_B_cgs2au

    # Gravitational constant in AU
    params['G'] = CGS.G * G_cgs2au

    # Speed of light in AU
    params['c_light'] = CGS.c * v_cms2au

    # Mean molecular weights (in Msun)
    params['mu_atom'] = 2.3 * m_H_to_Msun
    params['mu_ion'] = 1.4 * m_H_to_Msun

    # Shell temperatures (K)
    params['TShell_ion'] = 1e4
    params['TShell_neu'] = 100

    # Case B recombination coefficient
    caseB_alpha_cgs = 2.54e-13
    cm3_to_pc3 = CONV.cm2pc**3
    s_to_Myr = CONV.s2Myr
    params['caseB_alpha'] = caseB_alpha_cgs * cm3_to_pc3 / s_to_Myr

    # Dust parameters
    dust_sigma_cgs = 1e-21
    cm2_to_pc2 = CONV.cm2pc**2
    params['dust_sigma'] = dust_sigma_cgs * cm2_to_pc2
    dust_KappaIR_cgs = 4.0
    params['dust_KappaIR'] = dust_KappaIR_cgs * cm2_to_pc2 / CONV.g2Msun

    # Shell dissolution threshold
    params['stop_n_diss'] = 0.1 * ndens_cgs2au

    # ISM number density
    if 'nISM' not in params:
        params['nISM'] = 1.0 * ndens_cgs2au

    # =========================================================================
    # Bubble-specific constants
    # =========================================================================

    # Adiabatic index (dimensionless)
    params['gamma_adia'] = 5.0 / 3.0

    # Thermal conductivity coefficient (CGS to AU)
    C_thermal_cgs = 6e-7  # erg/(s·cm·K^(7/2))
    params['C_thermal'] = C_thermal_cgs * CONV.c_therm_cgs2au

    # Bubble temperature measurement radius ratio
    params['bubble_xi_Tb'] = 0.98

    # Cloud metallicity (solar = 1.0)
    if 'ZCloud' not in params:
        params['ZCloud'] = 1.0

    # Mechanical luminosity and velocity from wind data
    if 'LWind' in params and 'pWindDot' in params:
        params['Lmech_total'] = params['LWind']
        if params['pWindDot'] > 0:
            params['v_mech_total'] = 2.0 * params['LWind'] / params['pWindDot']
        else:
            params['v_mech_total'] = 1000.0  # Default fallback

    # Pre-initialize bubble arrays (will be populated by get_bubbleproperties)
    if 'bubble_v_arr' not in params:
        params['bubble_v_arr'] = np.array([])
    if 'bubble_T_arr' not in params:
        params['bubble_T_arr'] = np.array([])
    if 'bubble_dTdr_arr' not in params:
        params['bubble_dTdr_arr'] = np.array([])
    if 'bubble_r_arr' not in params:
        params['bubble_r_arr'] = np.array([])
    if 'bubble_n_arr' not in params:
        params['bubble_n_arr'] = np.array([])
    if 'bubble_r_Tb' not in params:
        params['bubble_r_Tb'] = 0.0

    # Pre-initialize bubble output values
    if 'bubble_LTotal' not in params:
        params['bubble_LTotal'] = 0.0
    if 'bubble_T_r_Tb' not in params:
        params['bubble_T_r_Tb'] = 0.0
    if 'bubble_L1Bubble' not in params:
        params['bubble_L1Bubble'] = 0.0
    if 'bubble_L2Conduction' not in params:
        params['bubble_L2Conduction'] = 0.0
    if 'bubble_L3Intermediate' not in params:
        params['bubble_L3Intermediate'] = 0.0
    if 'bubble_Tavg' not in params:
        params['bubble_Tavg'] = 0.0
    if 'bubble_mass' not in params:
        params['bubble_mass'] = 0.0

    # Pre-initialize R1 and Pb (will be computed)
    if 'R1' not in params:
        params['R1'] = 0.0
    if 'Pb' not in params:
        params['Pb'] = 0.0

    return params


def prime_params(params: dict) -> dict:
    """
    Initialize parameters that require setup beyond simple values.
    Sets up cooling interpolation and other computed parameters.
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
            except Exception:
                pass

    if not cooling_loaded:
        # Mock cooling curve
        logT = np.linspace(4, 9, 100)
        logLambda = -22 + 0.5 * (logT - 6)
        cooling_CIE_interpolation = scipy.interpolate.interp1d(
            logT, logLambda, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        params['path_cooling_CIE'] = 'mock_cooling'
        params['cStruc_cooling_CIE_logT'] = logT
        params['cStruc_cooling_CIE_logLambda'] = logLambda
        params['cStruc_cooling_CIE_interpolation'] = cooling_CIE_interpolation

    # Non-CIE cooling placeholders
    # Note: These need to be objects with .temp, .ndens, .phi, .interp attributes
    class MockCloudyCube:
        """Mock CloudyCube with temperature range for non-CIE cooling."""
        def __init__(self):
            self.temp = np.array([4.0, 4.5, 5.0, 5.5])  # log10(T)
            self.ndens = np.array([0, 2, 4, 6])  # log10(n)
            self.phi = np.array([0, 5, 10, 15])  # log10(phi)
            self.cooling = np.zeros((4, 4, 4))
            self.heating = np.zeros((4, 4, 4))
            # Interpolation function that returns 0 (log10 of small value)
            self.interp = scipy.interpolate.RegularGridInterpolator(
                (self.ndens, self.temp, self.phi),
                np.full((4, 4, 4), -30.0),  # log10(very small) = -30
                bounds_error=False,
                fill_value=-30.0
            )

    if 'cStruc_cooling_nonCIE' not in params:
        params['cStruc_cooling_nonCIE'] = MockCloudyCube()
    if 'cStruc_heating_nonCIE' not in params:
        params['cStruc_heating_nonCIE'] = MockCloudyCube()
    if 'cStruc_net_nonCIE_interpolation' not in params:
        # Mock interpolator that returns 0
        mock_cube = MockCloudyCube()
        params['cStruc_net_nonCIE_interpolation'] = scipy.interpolate.RegularGridInterpolator(
            (mock_cube.ndens, mock_cube.temp, mock_cube.phi),
            np.zeros((4, 4, 4)),
            bounds_error=False,
            fill_value=0.0
        )

    # Cooling timing
    if 't_previousCoolingUpdate' not in params:
        params['t_previousCoolingUpdate'] = 0.0

    return params


def make_params_dict(snapshot: dict, include_priming: bool = True) -> dict:
    """Create a params dictionary with MockParam wrappers from a snapshot."""
    snapshot = add_physical_constants(snapshot)
    if include_priming:
        snapshot = prime_params(snapshot)
    return {k: MockParam(v) for k, v in snapshot.items()}


def compare_values(name: str, original, modified, rtol: float = 1e-8) -> tuple:
    """Compare two values and return (passed, rel_diff, message)."""
    if isinstance(original, bool):
        passed = (original == modified)
        return passed, 0.0 if passed else 1.0, "bool"

    elif isinstance(original, np.ndarray):
        if not isinstance(modified, np.ndarray):
            return False, 1.0, "type_mismatch"
        if original.shape != modified.shape:
            return False, 1.0, "shape_mismatch"
        if np.allclose(original, modified, rtol=rtol, equal_nan=True):
            return True, 0.0, "array_match"
        else:
            max_diff = np.max(np.abs(original - modified) / (np.abs(original) + 1e-300))
            return False, max_diff, "array_diff"

    elif isinstance(original, (int, float)):
        if np.isnan(original) and np.isnan(modified):
            return True, 0.0, "both_nan"
        if np.isclose(original, modified, rtol=rtol):
            return True, 0.0, "match"
        else:
            rel_diff = abs(original - modified) / max(abs(original), abs(modified), 1e-300)
            return False, rel_diff, "diff"

    else:
        passed = (original == modified)
        return passed, 0.0 if passed else 1.0, "other"


def test_snapshot(snapshot: dict, verbose: bool = False) -> dict:
    """
    Test a single snapshot and return results dictionary.

    Returns dict with:
        - 'passed': bool
        - 'line': int
        - 't_now': float
        - 'field_results': dict mapping field_name -> (passed, rel_diff, original_val, pct_error)
        - 'time_original': float (seconds)
        - 'time_modified': float (seconds)
        - 'speedup': float
    """
    params_original = make_params_dict(snapshot.copy())
    params_modified = make_params_dict(snapshot.copy())

    # Run original version with timing
    start_orig = time.perf_counter()
    get_bubbleproperties(params_original)
    time_original = time.perf_counter() - start_orig

    # Run modified version with timing
    start_mod = time.perf_counter()
    result_modified = get_bubbleproperties_pure(params_modified)
    time_modified = time.perf_counter() - start_mod

    # Calculate speedup
    speedup = time_original / time_modified if time_modified > 0 else float('inf')

    # Fields to compare
    fields = [
        ('bubble_LTotal', 'bubble_LTotal'),
        ('bubble_T_r_Tb', 'bubble_T_r_Tb'),
        ('bubble_Tavg', 'bubble_Tavg'),
        ('bubble_mass', 'bubble_mass'),
        ('bubble_L1Bubble', 'bubble_L1Bubble'),
        ('bubble_L2Conduction', 'bubble_L2Conduction'),
        ('bubble_L3Intermediate', 'bubble_L3Intermediate'),
        ('bubble_dMdt', 'bubble_dMdt'),
        ('R1', 'R1'),
        ('Pb', 'Pb'),
        ('bubble_r_Tb', 'bubble_r_Tb'),
    ]

    field_results = {}
    all_passed = True

    for params_key, dataclass_attr in fields:
        original_val = params_original[params_key].value
        modified_val = getattr(result_modified, dataclass_attr)
        passed, rel_diff, msg = compare_values(params_key, original_val, modified_val)
        # Calculate % error
        if isinstance(original_val, (int, float)) and not np.isnan(original_val) and abs(original_val) > 1e-300:
            pct_error = 100.0 * abs(original_val - modified_val) / abs(original_val)
        else:
            pct_error = 0.0 if passed else 100.0
        field_results[params_key] = (passed, rel_diff, original_val, pct_error)
        if not passed:
            all_passed = False
            if verbose:
                print(f"  ✗ {params_key}: rel_diff={rel_diff:.2e}, %error={pct_error:.2e}%")

    return {
        'passed': all_passed,
        't_now': snapshot.get('t_now', 0),
        'field_results': field_results,
        'time_original': time_original,
        'time_modified': time_modified,
        'speedup': speedup,
    }


def print_comparison_table(results: list, fields: list):
    """Print a formatted comparison table with timing and % error."""
    # Header
    print("\n" + "=" * 130)
    print("BUBBLE LUMINOSITY COMPARISON TABLE")
    print("=" * 130)

    # Column headers for main table
    header = f"{'Snap':<6} {'t_now':<11} {'Status':<6} {'Orig(ms)':<9} {'Mod(ms)':<9} {'Speedup':<8}"
    for field in fields[:4]:  # Show first 4 key fields
        header += f" {field[:10]:<12}"
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
        for field in fields[:4]:
            if field in res['field_results']:
                passed, rel_diff, val, pct_error = res['field_results'][field]
                if passed:
                    row += f" {val:<12.4e}"
                else:
                    row += f" {val:<12.4e}*"
            else:
                row += f" {'N/A':<12}"
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

    # Print error table (2 columns)
    field_list = list(field_max_errors.items())
    for i in range(0, len(field_list), 2):
        row = ""
        for j in range(2):
            if i + j < len(field_list):
                field, err = field_list[i + j]
                row += f"  {field:<25}: {err:>12.2e}%"
        print(row)

    print("-" * 100)

    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    print(f"\nSummary: {passed_count}/{len(results)} snapshots passed all comparisons")
    if passed_count < len(results):
        print("* indicates field with mismatch")


def test_bubble_luminosity_comparison():
    """
    Test bubble_luminosity vs bubble_luminosity_modified on 10 random snapshots.
    """
    print("=" * 70)
    print("Testing bubble_luminosity vs bubble_luminosity_modified")
    print("=" * 70)

    jsonl_path = os.path.join(PROJECT_ROOT, 'comparison', '1e7_sfe020_n1e4_test_dictionary.jsonl')

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
    fields = ['bubble_LTotal', 'bubble_T_r_Tb', 'bubble_Tavg', 'bubble_mass',
              'bubble_L1Bubble', 'bubble_dMdt', 'R1', 'Pb']

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
            results.append({'passed': False, 'line': line_num, 't_now': 0, 'field_results': {}})

    # Print comparison table
    print_comparison_table(results, fields)

    all_passed = all(r['passed'] for r in results)
    return all_passed


if __name__ == '__main__':
    print("Bubble Luminosity Comparison Test")
    print("==================================\n")

    passed = test_bubble_luminosity_comparison()

    print("\n" + "=" * 70)
    if passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
