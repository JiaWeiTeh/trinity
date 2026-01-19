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


def load_default_params(param_file: str = None) -> dict:
    """
    Load default parameters from default.param file.

    Parameters
    ----------
    param_file : str, optional
        Path to param file. If None, uses PROJECT_ROOT/param/default.param

    Returns
    -------
    dict
        Dictionary of parameter name -> value (as strings initially)
    """
    if param_file is None:
        param_file = os.path.join(PROJECT_ROOT, 'param', 'default.param')

    defaults = {}
    if not os.path.exists(param_file):
        return defaults

    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#') or line.startswith('INFO'):
                continue
            # Parse "param value" format
            parts = line.split(None, 1)  # Split on first whitespace
            if len(parts) == 2:
                key, value = parts
                # Try to convert to numeric
                try:
                    # Handle fractions like 5/3
                    if '/' in value and not value.startswith('/'):
                        num, denom = value.split('/')
                        defaults[key] = float(num) / float(denom)
                    elif value.lower() == 'true':
                        defaults[key] = True
                    elif value.lower() == 'false':
                        defaults[key] = False
                    else:
                        defaults[key] = float(value)
                except ValueError:
                    defaults[key] = value  # Keep as string

    return defaults


def search_jsonl_for_param(jsonl_path: str, param_name: str) -> any:
    """
    Search through JSONL file for a parameter value.

    Parameters
    ----------
    jsonl_path : str
        Path to the JSONL file
    param_name : str
        Parameter name to search for

    Returns
    -------
    any
        The parameter value if found, None otherwise
    """
    if not os.path.exists(jsonl_path):
        return None

    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if param_name in data:
                    return data[param_name]
            except json.JSONDecodeError:
                continue
    return None


# Cache for default params and jsonl search
_DEFAULT_PARAMS_CACHE = None
_JSONL_PATH_CACHE = None


def get_param_value(params: dict, key: str, defaults: dict, jsonl_path: str):
    """
    Get parameter value from params, defaults, or jsonl (in that order).

    Parameters
    ----------
    params : dict
        Current params dictionary
    key : str
        Parameter name
    defaults : dict
        Default params from default.param
    jsonl_path : str
        Path to JSONL file for fallback search

    Returns
    -------
    any
        Parameter value if found, None otherwise
    """
    # First check params
    if key in params:
        return params[key]
    # Then check defaults
    if key in defaults:
        return defaults[key]
    # Finally search jsonl
    return search_jsonl_for_param(jsonl_path, key)


def add_physical_constants(params: dict, jsonl_path: str = None) -> dict:
    """
    Add physical constants that are missing from the dictionary snapshot.

    Reads defaults from /param/default.param, then searches jsonl if not found.
    """
    global _DEFAULT_PARAMS_CACHE, _JSONL_PATH_CACHE

    # Load defaults once
    if _DEFAULT_PARAMS_CACHE is None:
        _DEFAULT_PARAMS_CACHE = load_default_params()

    defaults = _DEFAULT_PARAMS_CACHE

    if jsonl_path is None:
        jsonl_path = os.path.join(PROJECT_ROOT, 'comparison', '1e7_sfe020_n1e4_test_dictionary.jsonl')
    _JSONL_PATH_CACHE = jsonl_path

    # Unit conversion factors
    ndens_cgs2au = CONV.ndens_cgs2au
    m_H_to_Msun = CGS.m_H * CONV.g2Msun
    k_B_cgs2au = CONV.k_B_cgs2au
    G_cgs2au = CONV.G_cgs2au
    v_cms2au = CONV.v_cms2au
    cm2_to_pc2 = CONV.cm2pc**2
    cm3_to_pc3 = CONV.cm2pc**3
    s_to_Myr = CONV.s2Myr

    # Physical constants with unit conversion (these are fundamental, read from default.param)
    k_B_cgs = get_param_value(params, 'k_B', defaults, jsonl_path) or CGS.k_B
    params['k_B'] = k_B_cgs * k_B_cgs2au

    G_cgs = get_param_value(params, 'G', defaults, jsonl_path) or CGS.G
    params['G'] = G_cgs * G_cgs2au

    c_cgs = get_param_value(params, 'c_light', defaults, jsonl_path) or CGS.c
    params['c_light'] = c_cgs * v_cms2au

    # Mean molecular weights (from default.param, in units of m_H -> convert to Msun)
    mu_atom = get_param_value(params, 'mu_atom', defaults, jsonl_path)
    params['mu_atom'] = mu_atom * m_H_to_Msun if mu_atom else 1.27 * m_H_to_Msun

    mu_ion = get_param_value(params, 'mu_ion', defaults, jsonl_path)
    params['mu_ion'] = mu_ion * m_H_to_Msun if mu_ion else 0.61 * m_H_to_Msun

    # Shell temperatures (K) - direct values
    params['TShell_ion'] = get_param_value(params, 'TShell_ion', defaults, jsonl_path) or 1e4
    params['TShell_neu'] = get_param_value(params, 'TShell_neu', defaults, jsonl_path) or 100

    # Case B recombination coefficient (from default.param in cm³/s -> convert to AU)
    caseB_cgs = get_param_value(params, 'caseB_alpha', defaults, jsonl_path) or 2.59e-13
    params['caseB_alpha'] = caseB_cgs * cm3_to_pc3 / s_to_Myr

    # Dust cross section (from default.param in cm² -> convert to AU)
    dust_sigma_cgs = get_param_value(params, 'dust_sigma', defaults, jsonl_path) or 1.5e-21
    params['dust_sigma'] = dust_sigma_cgs * cm2_to_pc2

    # Dust IR opacity (from default.param in cm²/g -> convert to AU)
    dust_KappaIR_cgs = get_param_value(params, 'dust_KappaIR', defaults, jsonl_path) or 4.0
    params['dust_KappaIR'] = dust_KappaIR_cgs * cm2_to_pc2 / CONV.g2Msun

    # Shell dissolution threshold (from default.param in cm^-3 -> convert to AU)
    stop_n_diss_cgs = get_param_value(params, 'stop_n_diss', defaults, jsonl_path) or 1.0
    params['stop_n_diss'] = stop_n_diss_cgs * ndens_cgs2au

    # ISM number density (from default.param in cm^-3 -> convert to AU)
    if 'nISM' not in params:
        nISM_cgs = get_param_value(params, 'nISM', defaults, jsonl_path) or 1.0
        params['nISM'] = nISM_cgs * ndens_cgs2au

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
    print("\n" + "=" * 80)
    print("SHELL STRUCTURE COMPARISON TEST")
    print("=" * 80)

    # Show tested parameters at top
    print(f"\nParameters tested: {', '.join(fields)}")

    # Compact results table
    print("\n" + "-" * 80)
    header = f"{'Snap':<6} {'t_now':<12} {'Status':<8} {'Orig(ms)':<10} {'Mod(ms)':<10} {'Speedup':<10}"
    print(header)
    print("-" * 80)

    # Data rows
    total_time_orig = 0.0
    total_time_mod = 0.0
    failed_snapshots = []

    for i, res in enumerate(results):
        line = res.get('line', i)
        t_now = res['t_now']
        status = "PASS" if res['passed'] else "FAIL"
        time_orig_ms = res.get('time_original', 0) * 1000
        time_mod_ms = res.get('time_modified', 0) * 1000
        speedup = res.get('speedup', 1.0)

        total_time_orig += res.get('time_original', 0)
        total_time_mod += res.get('time_modified', 0)

        row = f"{line:<6} {t_now:<12.4e} {status:<8} {time_orig_ms:<10.2f} {time_mod_ms:<10.2f} {speedup:<10.2f}x"
        print(row)

        # Track failed snapshots for error detail
        if not res['passed']:
            failed_snapshots.append((line, res))

    print("-" * 80)

    # Timing summary
    avg_speedup = total_time_orig / total_time_mod if total_time_mod > 0 else 1.0
    print(f"\nTiming Summary:")
    print(f"  Total original time: {total_time_orig*1000:.2f} ms")
    print(f"  Total modified time: {total_time_mod*1000:.2f} ms")
    print(f"  Average speedup: {avg_speedup:.2f}x")

    # Only show error details if there are failures
    if failed_snapshots:
        print("\n" + "=" * 80)
        print("ERROR DETAILS (only showing parameters with errors)")
        print("=" * 80)

        for line_num, res in failed_snapshots:
            print(f"\nSnapshot {line_num} (t={res['t_now']:.4e}):")
            for field, (passed, rel_diff, orig_val, pct_error) in res['field_results'].items():
                if not passed:
                    print(f"  {field}: {pct_error:.2e}% error (original={orig_val:.4e})")

        print("-" * 80)

    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    print(f"\nSummary: {passed_count}/{len(results)} snapshots passed all comparisons")


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
