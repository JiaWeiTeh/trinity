#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification test comparing shell_structure and shell_structure_modified.

This test loads a snapshot from the dictionary.jsonl file and verifies that
both the original shell_structure() and the new shell_structure_pure()
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

from src.shell_structure.shell_structure import shell_structure
from src.shell_structure.shell_structure_modified import shell_structure_pure, ShellProperties
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

    # Boltzmann constant in AU: k_B [erg/K] * conversion
    params['k_B'] = CGS.k_B * k_B_cgs2au  # Msun·pc²/Myr²/K

    # Gravitational constant in AU: G [cm³/g/s²] * conversion
    params['G'] = CGS.G * G_cgs2au  # pc³/Msun/Myr²

    # Speed of light in AU: c [cm/s] * conversion
    params['c_light'] = CGS.c * v_cms2au  # pc/Myr

    # Mean molecular weights (in Msun)
    params['mu_atom'] = 2.3 * m_H_to_Msun  # Neutral/atomic gas
    params['mu_ion'] = 1.4 * m_H_to_Msun   # Ionized gas

    # Shell temperatures (K)
    params['TShell_ion'] = 1e4  # Ionized shell temperature
    params['TShell_neu'] = 100  # Neutral shell temperature

    # Case B recombination coefficient: 2.54e-13 cm³/s at 10^4 K
    # Convert to AU: cm³/s -> pc³/Myr
    caseB_alpha_cgs = 2.54e-13  # cm³/s
    cm3_to_pc3 = CONV.cm2pc**3
    s_to_Myr = CONV.s2Myr
    params['caseB_alpha'] = caseB_alpha_cgs * cm3_to_pc3 / s_to_Myr

    # Dust cross section: 1e-21 cm² (scaled with metallicity, assume solar)
    # Convert to AU: cm² -> pc²
    dust_sigma_cgs = 1e-21  # cm²
    cm2_to_pc2 = CONV.cm2pc**2
    params['dust_sigma'] = dust_sigma_cgs * cm2_to_pc2

    # Dust IR opacity: 4.0 cm²/g
    # Convert to AU: cm²/g -> pc²/Msun
    dust_KappaIR_cgs = 4.0  # cm²/g
    params['dust_KappaIR'] = dust_KappaIR_cgs * cm2_to_pc2 / CONV.g2Msun

    # Shell dissolution threshold (number density in AU)
    params['stop_n_diss'] = 0.1 * ndens_cgs2au  # Convert from cm^-3 to pc^-3

    # ISM number density (if not present, use a typical value)
    if 'nISM' not in params:
        params['nISM'] = 1.0 * ndens_cgs2au  # 1 cm^-3 in AU

    return params


def prime_params(params: dict) -> dict:
    """
    Initialize parameters that require setup beyond simple values.

    This function sets up:
    - Cooling interpolation (CIE and non-CIE)
    - Any other computed/interpolated parameters

    Parameters
    ----------
    params : dict
        Dictionary with raw values (already has physical constants added)

    Returns
    -------
    dict
        Dictionary with initialized interpolation objects and computed params
    """
    # =========================================================================
    # Cooling interpolation (CIE - Collisional Ionization Equilibrium)
    # =========================================================================

    # Try to load cooling curve from lib/cooling/CIE/
    cie_files = {
        1: 'lib/cooling/CIE/coolingCIE_1_Cloudy.dat',
        2: 'lib/cooling/CIE/coolingCIE_2_Cloudy_grains.dat',
        3: 'lib/cooling/CIE/coolingCIE_3_Gnat-Ferland2012.dat',
        4: 'lib/cooling/CIE/coolingCIE_4_Sutherland-Dopita1993.dat',
    }

    cooling_loaded = False
    for cie_choice in [3, 4, 1, 2]:  # Try Gnat-Ferland first, then others
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
        # Create mock cooling interpolation for testing
        # This is a simple approximation of the cooling curve
        logT = np.linspace(4, 9, 100)  # log10(T) from 10^4 to 10^9 K
        # Simple cooling function approximation: Lambda ~ T^0.5 for high T
        logLambda = -22 + 0.5 * (logT - 6)  # Rough approximation
        cooling_CIE_interpolation = scipy.interpolate.interp1d(
            logT, logLambda, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        params['path_cooling_CIE'] = 'mock_cooling'
        params['cStruc_cooling_CIE_logT'] = logT
        params['cStruc_cooling_CIE_logLambda'] = logLambda
        params['cStruc_cooling_CIE_interpolation'] = cooling_CIE_interpolation
        print("Note: Using mock cooling curve (real cooling data not found)")

    # =========================================================================
    # Non-CIE cooling (placeholder - set to None if not available)
    # =========================================================================
    if 'cStruc_cooling_nonCIE' not in params:
        params['cStruc_cooling_nonCIE'] = None
    if 'cStruc_heating_nonCIE' not in params:
        params['cStruc_heating_nonCIE'] = None
    if 'cStruc_net_nonCIE_interpolation' not in params:
        params['cStruc_net_nonCIE_interpolation'] = None

    # =========================================================================
    # Cooling timing
    # =========================================================================
    if 't_previousCoolingUpdate' not in params:
        params['t_previousCoolingUpdate'] = 0.0

    return params


def make_params_dict(snapshot: dict, include_priming: bool = True) -> dict:
    """
    Create a params dictionary with MockParam wrappers from a snapshot.

    Parameters
    ----------
    snapshot : dict
        Raw dictionary loaded from JSONL
    include_priming : bool
        If True, also initialize interpolation objects (cooling, etc.)

    Returns
    -------
    dict
        Dictionary with MockParam objects for .value access
    """
    # Add physical constants
    snapshot = add_physical_constants(snapshot)

    # Prime params with interpolation objects (cooling curves, etc.)
    if include_priming:
        snapshot = prime_params(snapshot)

    # Wrap all values in MockParam
    return {k: MockParam(v) for k, v in snapshot.items()}


def compare_values(name: str, original, modified, rtol: float = 1e-10) -> tuple:
    """
    Compare two values and return (passed, message).

    Handles floats, arrays, and booleans.
    """
    if isinstance(original, bool):
        if original == modified:
            return True, f"  ✓ {name}: {original}"
        else:
            return False, f"  ✗ {name}: original={original}, modified={modified}"

    elif isinstance(original, np.ndarray):
        if not isinstance(modified, np.ndarray):
            return False, f"  ✗ {name}: original is array, modified is {type(modified)}"
        if original.shape != modified.shape:
            return False, f"  ✗ {name}: shape mismatch {original.shape} vs {modified.shape}"
        if np.allclose(original, modified, rtol=rtol, equal_nan=True):
            return True, f"  ✓ {name}: arrays match (shape={original.shape})"
        else:
            max_diff = np.max(np.abs(original - modified))
            return False, f"  ✗ {name}: arrays differ, max_diff={max_diff:.6e}"

    elif isinstance(original, (int, float)):
        if np.isnan(original) and np.isnan(modified):
            return True, f"  ✓ {name}: both NaN"
        if np.isclose(original, modified, rtol=rtol):
            return True, f"  ✓ {name}: {original:.6e}"
        else:
            rel_diff = abs(original - modified) / max(abs(original), abs(modified), 1e-300)
            return False, f"  ✗ {name}: original={original:.6e}, modified={modified:.6e}, rel_diff={rel_diff:.6e}"

    else:
        if original == modified:
            return True, f"  ✓ {name}: {original}"
        else:
            return False, f"  ✗ {name}: original={original}, modified={modified}"


def get_total_lines(filepath: str) -> int:
    """Count total lines in a file."""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


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

    # Run original version with timing
    start_orig = time.perf_counter()
    shell_structure(params_original)
    time_original = time.perf_counter() - start_orig

    # Run modified version with timing
    start_mod = time.perf_counter()
    result_modified = shell_structure_pure(params_modified)
    time_modified = time.perf_counter() - start_mod

    # Calculate speedup
    speedup = time_original / time_modified if time_modified > 0 else float('inf')

    # Fields to compare
    fields = [
        ('shell_n0', 'shell_n0'),
        ('rShell', 'rShell'),
        ('shell_thickness', 'shell_thickness'),
        ('shell_fAbsorbedIon', 'shell_fAbsorbedIon'),
        ('shell_fAbsorbedNeu', 'shell_fAbsorbedNeu'),
        ('shell_fAbsorbedWeightedTotal', 'shell_fAbsorbedWeightedTotal'),
        ('shell_fIonisedDust', 'shell_fIonisedDust'),
        ('shell_nMax', 'shell_nMax'),
        ('shell_tauKappaRatio', 'shell_tauKappaRatio'),
        ('shell_F_rad', 'shell_F_rad'),
        ('isDissolved', 'isDissolved'),
    ]

    field_results = {}
    all_passed = True

    for params_key, dataclass_attr in fields:
        original_val = params_original[params_key].value
        modified_val = getattr(result_modified, dataclass_attr)

        if isinstance(original_val, bool):
            passed = (original_val == modified_val)
            rel_diff = 0.0 if passed else 1.0
            pct_error = 0.0 if passed else 100.0
        elif isinstance(original_val, (int, float)):
            if np.isnan(original_val) and np.isnan(modified_val):
                passed, rel_diff, pct_error = True, 0.0, 0.0
            elif np.isclose(original_val, modified_val, rtol=1e-10):
                passed, rel_diff = True, 0.0
                pct_error = 100.0 * abs(original_val - modified_val) / abs(original_val) if abs(original_val) > 1e-300 else 0.0
            else:
                rel_diff = abs(original_val - modified_val) / max(abs(original_val), abs(modified_val), 1e-300)
                pct_error = 100.0 * abs(original_val - modified_val) / abs(original_val) if abs(original_val) > 1e-300 else 100.0
                passed = False
        else:
            passed, rel_diff, pct_error = True, 0.0, 0.0

        field_results[params_key] = (passed, rel_diff, original_val, pct_error)
        if not passed:
            all_passed = False

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
    print("\n" + "=" * 130)
    print("SHELL STRUCTURE COMPARISON TABLE")
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
                if isinstance(val, bool):
                    row += f" {str(val):<12}"
                elif isinstance(val, (int, float)):
                    if passed:
                        row += f" {val:<12.4e}"
                    else:
                        row += f" {val:<12.4e}*"
                else:
                    row += f" {'N/A':<12}"
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


def test_shell_structure_comparison():
    """
    Test shell_structure vs shell_structure_modified on 10 random snapshots.
    """
    print("=" * 70)
    print("Testing shell_structure vs shell_structure_modified")
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
    fields = ['shell_n0', 'rShell', 'shell_thickness', 'shell_fAbsorbedIon',
              'shell_fAbsorbedNeu', 'shell_nMax', 'shell_F_rad', 'isDissolved']

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
    print("Shell Structure Comparison Test")
    print("================================\n")

    passed = test_shell_structure_comparison()

    print("\n" + "=" * 70)
    if passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
