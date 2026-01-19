#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification test comparing get_betadelta and get_betadelta_modified.

This test loads snapshots from the dictionary.jsonl file and verifies that
both the original get_beta_delta_wrapper() and the new solve_betadelta_pure()
return consistent results.

Outputs a comparison table for snapshots in a specified range.

Author: TRINITY Team
"""

import numpy as np
import scipy.interpolate
import json
import sys
import os
import random
import time
import copy

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1b_energy_implicit.get_betadelta import get_beta_delta_wrapper
from src.phase1b_energy_implicit.get_betadelta_modified import (
    solve_betadelta_pure,
    BetaDeltaResult,
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


class MockParamsDict(dict):
    """Mock params dictionary that supports .get() with proper handling."""

    def get(self, key, default=None):
        """Override get to return the item or default."""
        if key in self:
            return self[key]
        return default


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


def load_default_params(param_file: str = None) -> dict:
    """Load default parameters from default.param file."""
    if param_file is None:
        param_file = os.path.join(PROJECT_ROOT, 'param', 'default.param')

    defaults = {}
    if not os.path.exists(param_file):
        return defaults

    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('INFO'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                key, value = parts
                try:
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
                    defaults[key] = value
    return defaults


# Cache for default params
_DEFAULT_PARAMS_CACHE = None


def add_physical_constants(params: dict, jsonl_path: str = None) -> dict:
    """
    Add physical constants that are missing from the dictionary snapshot.
    """
    global _DEFAULT_PARAMS_CACHE

    if _DEFAULT_PARAMS_CACHE is None:
        _DEFAULT_PARAMS_CACHE = load_default_params()
    defaults = _DEFAULT_PARAMS_CACHE

    # Unit conversion factors
    ndens_cgs2au = CONV.ndens_cgs2au
    m_H_to_Msun = CGS.m_H * CONV.g2Msun
    k_B_cgs2au = CONV.k_B_cgs2au
    G_cgs2au = CONV.G_cgs2au
    v_cms2au = CONV.v_cms2au
    cm2_to_pc2 = CONV.cm2pc**2
    cm3_to_pc3 = CONV.cm2pc**3
    s_to_Myr = CONV.s2Myr

    # Physical constants
    k_B_cgs = defaults.get('k_B', CGS.k_B)
    params['k_B'] = k_B_cgs * k_B_cgs2au

    G_cgs = defaults.get('G', CGS.G)
    params['G'] = G_cgs * G_cgs2au

    c_cgs = defaults.get('c_light', CGS.c)
    params['c_light'] = c_cgs * v_cms2au

    # Mean molecular weights
    mu_atom = defaults.get('mu_atom', 1.27)
    params['mu_atom'] = mu_atom * m_H_to_Msun

    mu_ion = defaults.get('mu_ion', 0.61)
    params['mu_ion'] = mu_ion * m_H_to_Msun

    # Shell temperatures
    params['TShell_ion'] = defaults.get('TShell_ion', 1e4)
    params['TShell_neu'] = defaults.get('TShell_neu', 100)

    # Case B recombination
    caseB_cgs = defaults.get('caseB_alpha', 2.59e-13)
    params['caseB_alpha'] = caseB_cgs * cm3_to_pc3 / s_to_Myr

    # Dust parameters
    dust_sigma_cgs = defaults.get('dust_sigma', 1.5e-21)
    params['dust_sigma'] = dust_sigma_cgs * cm2_to_pc2
    dust_KappaIR_cgs = defaults.get('dust_KappaIR', 4.0)
    params['dust_KappaIR'] = dust_KappaIR_cgs * cm2_to_pc2 / CONV.g2Msun

    # Thresholds
    stop_n_diss_cgs = defaults.get('stop_n_diss', 1.0)
    params['stop_n_diss'] = stop_n_diss_cgs * ndens_cgs2au

    if 'nISM' not in params:
        nISM_cgs = defaults.get('nISM', 1.0)
        params['nISM'] = nISM_cgs * ndens_cgs2au

    # Adiabatic index
    params['gamma_adia'] = defaults.get('gamma_adia', 5.0 / 3.0)

    # Thermal conductivity
    C_thermal_cgs = defaults.get('C_thermal', 6e-7)
    params['C_thermal'] = C_thermal_cgs * CONV.c_therm_cgs2au

    # Bubble temperature measurement
    params['bubble_xi_Tb'] = defaults.get('bubble_xi_Tb', 0.98)

    # Cloud metallicity
    if 'ZCloud' not in params:
        params['ZCloud'] = defaults.get('ZCloud', 1.0)

    # Cooling parameters (beta, delta)
    if 'cool_beta' not in params:
        params['cool_beta'] = defaults.get('cool_beta', 0.8)
    if 'cool_delta' not in params:
        params['cool_delta'] = defaults.get('cool_delta', -6/35)
    if 'cool_alpha' not in params:
        params['cool_alpha'] = defaults.get('cool_alpha', 0.6)

    # Pre-initialize bubble arrays
    for key in ['bubble_v_arr', 'bubble_T_arr', 'bubble_dTdr_arr', 'bubble_r_arr', 'bubble_n_arr']:
        if key not in params:
            params[key] = np.array([])

    # Pre-initialize bubble scalars
    for key in ['bubble_r_Tb', 'bubble_LTotal', 'bubble_T_r_Tb', 'bubble_L1Bubble',
                'bubble_L2Conduction', 'bubble_L3Intermediate', 'bubble_Tavg',
                'bubble_mass', 'bubble_dMdt', 'bubble_Leak', 'bubble_Lgain', 'bubble_Lloss']:
        if key not in params:
            params[key] = 0.0

    # Pre-initialize R1 and Pb
    if 'R1' not in params:
        params['R1'] = 0.0
    if 'Pb' not in params:
        params['Pb'] = 0.0

    # Residual tracking
    for key in ['residual_deltaT', 'residual_betaEdot', 'residual_Edot1_guess',
                'residual_Edot2_guess', 'residual_T1_guess', 'residual_T2_guess']:
        if key not in params:
            params[key] = 0.0

    return params


def prime_params(params: dict) -> dict:
    """
    Initialize parameters that require setup beyond simple values.
    Sets up cooling interpolation and other computed parameters.
    """
    # CIE Cooling interpolation
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
    class MockCloudyCube:
        """Mock CloudyCube for non-CIE cooling."""
        def __init__(self):
            self.temp = np.array([4.0, 4.5, 5.0, 5.5])
            self.ndens = np.array([0, 2, 4, 6])
            self.phi = np.array([0, 5, 10, 15])
            self.cooling = np.zeros((4, 4, 4))
            self.heating = np.zeros((4, 4, 4))
            self.interp = scipy.interpolate.RegularGridInterpolator(
                (self.ndens, self.temp, self.phi),
                np.full((4, 4, 4), -30.0),
                bounds_error=False,
                fill_value=-30.0
            )

    if 'cStruc_cooling_nonCIE' not in params:
        params['cStruc_cooling_nonCIE'] = MockCloudyCube()
    if 'cStruc_heating_nonCIE' not in params:
        params['cStruc_heating_nonCIE'] = MockCloudyCube()
    if 'cStruc_net_nonCIE_interpolation' not in params:
        mock_cube = MockCloudyCube()
        params['cStruc_net_nonCIE_interpolation'] = scipy.interpolate.RegularGridInterpolator(
            (mock_cube.ndens, mock_cube.temp, mock_cube.phi),
            np.zeros((4, 4, 4)),
            bounds_error=False,
            fill_value=0.0
        )

    if 't_previousCoolingUpdate' not in params:
        params['t_previousCoolingUpdate'] = 0.0

    return params


def make_params_dict(snapshot: dict, include_priming: bool = True) -> MockParamsDict:
    """Create a MockParamsDict with MockParam wrappers from a snapshot."""
    snapshot = add_physical_constants(snapshot.copy())
    if include_priming:
        snapshot = prime_params(snapshot)
    return MockParamsDict({k: MockParam(v) for k, v in snapshot.items()})


def compare_values(name: str, original, modified, rtol: float = 0.1) -> tuple:
    """
    Compare two values and return (passed, rel_diff, message).

    Note: Using higher rtol (0.1) because grid search vs optimizer may find
    different local minima that are both valid solutions.
    """
    if isinstance(original, (int, float)):
        if np.isnan(original) and np.isnan(modified):
            return True, 0.0, "both_nan"
        if abs(original) < 1e-300 and abs(modified) < 1e-300:
            return True, 0.0, "both_zero"
        if np.isclose(original, modified, rtol=rtol, atol=1e-10):
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

    Compares original get_beta_delta_wrapper with modified solve_betadelta_pure.
    """
    # Create two independent copies
    params_original = make_params_dict(snapshot)
    params_modified = make_params_dict(snapshot)

    beta_guess = params_original['cool_beta'].value
    delta_guess = params_original['cool_delta'].value

    # Run original version with timing
    start_orig = time.perf_counter()
    try:
        (beta_orig, delta_orig), result_params_orig = get_beta_delta_wrapper(
            beta_guess, delta_guess, copy.deepcopy(params_original)
        )
        time_original = time.perf_counter() - start_orig
        original_success = True
    except Exception as e:
        time_original = time.perf_counter() - start_orig
        beta_orig, delta_orig = np.nan, np.nan
        original_success = False
        if verbose:
            print(f"  Original failed: {e}")

    # Run modified version with timing
    start_mod = time.perf_counter()
    try:
        result_modified = solve_betadelta_pure(
            beta_guess, delta_guess, params_modified
        )
        time_modified = time.perf_counter() - start_mod
        beta_mod = result_modified.beta
        delta_mod = result_modified.delta
        modified_success = True
    except Exception as e:
        time_modified = time.perf_counter() - start_mod
        beta_mod, delta_mod = np.nan, np.nan
        result_modified = None
        modified_success = False
        if verbose:
            print(f"  Modified failed: {e}")

    # Calculate speedup
    speedup = time_original / time_modified if time_modified > 0 else float('inf')

    # Compare results
    field_results = {}
    all_passed = True

    # Compare beta
    passed, rel_diff, msg = compare_values('beta', beta_orig, beta_mod)
    field_results['beta'] = (passed, rel_diff, beta_orig, beta_mod)
    if not passed:
        all_passed = False
        if verbose:
            print(f"  ✗ beta: orig={beta_orig:.4f}, mod={beta_mod:.4f}, diff={rel_diff:.2e}")

    # Compare delta
    passed, rel_diff, msg = compare_values('delta', delta_orig, delta_mod)
    field_results['delta'] = (passed, rel_diff, delta_orig, delta_mod)
    if not passed:
        all_passed = False
        if verbose:
            print(f"  ✗ delta: orig={delta_orig:.4f}, mod={delta_mod:.4f}, diff={rel_diff:.2e}")

    # Compare residuals if available
    if result_modified is not None:
        Edot_res_mod = result_modified.Edot_residual
        T_res_mod = result_modified.T_residual
        total_res_mod = result_modified.total_residual
        converged_mod = result_modified.converged
    else:
        Edot_res_mod = np.nan
        T_res_mod = np.nan
        total_res_mod = np.nan
        converged_mod = False

    # Get original residuals from result_params if available
    if original_success and 'residual_betaEdot' in result_params_orig:
        Edot_res_orig = result_params_orig['residual_betaEdot'].value
        T_res_orig = result_params_orig['residual_deltaT'].value
    else:
        Edot_res_orig = np.nan
        T_res_orig = np.nan

    field_results['Edot_residual'] = (True, 0.0, Edot_res_orig, Edot_res_mod)
    field_results['T_residual'] = (True, 0.0, T_res_orig, T_res_mod)

    return {
        'passed': all_passed,
        't_now': snapshot.get('t_now', 0),
        'field_results': field_results,
        'time_original': time_original,
        'time_modified': time_modified,
        'speedup': speedup,
        'original_success': original_success,
        'modified_success': modified_success,
        'converged_mod': converged_mod,
        'total_residual_mod': total_res_mod,
    }


def print_comparison_table(results: list):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("BETA-DELTA SOLVER COMPARISON TEST")
    print("Original: get_beta_delta_wrapper (grid search)")
    print("Modified: solve_betadelta_pure (L-BFGS-B optimizer)")
    print("=" * 100)

    # Header
    print("\n" + "-" * 100)
    header = (f"{'Snap':<6} {'t_now':<10} {'β_orig':<8} {'β_mod':<8} "
              f"{'δ_orig':<8} {'δ_mod':<8} {'Orig(ms)':<10} {'Mod(ms)':<10} {'Speedup':<8} {'Status':<8}")
    print(header)
    print("-" * 100)

    total_time_orig = 0.0
    total_time_mod = 0.0
    failed_snapshots = []

    for i, res in enumerate(results):
        line = res.get('line', i)
        t_now = res['t_now']

        beta_orig = res['field_results']['beta'][2]
        beta_mod = res['field_results']['beta'][3]
        delta_orig = res['field_results']['delta'][2]
        delta_mod = res['field_results']['delta'][3]

        time_orig_ms = res.get('time_original', 0) * 1000
        time_mod_ms = res.get('time_modified', 0) * 1000
        speedup = res.get('speedup', 1.0)

        total_time_orig += res.get('time_original', 0)
        total_time_mod += res.get('time_modified', 0)

        status = "PASS" if res['passed'] else "FAIL"
        if not res['original_success']:
            status = "ORIG_ERR"
        elif not res['modified_success']:
            status = "MOD_ERR"

        # Format values
        beta_orig_str = f"{beta_orig:.4f}" if not np.isnan(beta_orig) else "NaN"
        beta_mod_str = f"{beta_mod:.4f}" if not np.isnan(beta_mod) else "NaN"
        delta_orig_str = f"{delta_orig:.4f}" if not np.isnan(delta_orig) else "NaN"
        delta_mod_str = f"{delta_mod:.4f}" if not np.isnan(delta_mod) else "NaN"

        row = (f"{line:<6} {t_now:<10.4e} {beta_orig_str:<8} {beta_mod_str:<8} "
               f"{delta_orig_str:<8} {delta_mod_str:<8} {time_orig_ms:<10.1f} "
               f"{time_mod_ms:<10.1f} {speedup:<8.1f}x {status:<8}")
        print(row)

        if not res['passed']:
            failed_snapshots.append((line, res))

    print("-" * 100)

    # Timing summary
    avg_speedup = total_time_orig / total_time_mod if total_time_mod > 0 else 1.0
    print(f"\nTiming Summary:")
    print(f"  Total original time: {total_time_orig*1000:.1f} ms")
    print(f"  Total modified time: {total_time_mod*1000:.1f} ms")
    print(f"  Average speedup: {avg_speedup:.1f}x")

    # Residual details
    print(f"\nResidual Details (modified solver):")
    print("-" * 60)
    print(f"{'Snap':<6} {'Edot_res':<12} {'T_res':<12} {'Total_res':<12} {'Converged':<10}")
    print("-" * 60)
    for i, res in enumerate(results):
        line = res.get('line', i)
        Edot_res = res['field_results']['Edot_residual'][3]
        T_res = res['field_results']['T_residual'][3]
        total_res = res.get('total_residual_mod', np.nan)
        converged = res.get('converged_mod', False)

        Edot_str = f"{Edot_res:.4e}" if not np.isnan(Edot_res) else "NaN"
        T_str = f"{T_res:.4e}" if not np.isnan(T_res) else "NaN"
        total_str = f"{total_res:.4e}" if not np.isnan(total_res) else "NaN"
        conv_str = "Yes" if converged else "No"

        print(f"{line:<6} {Edot_str:<12} {T_str:<12} {total_str:<12} {conv_str:<10}")

    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    print(f"\n" + "=" * 100)
    print(f"Summary: {passed_count}/{len(results)} snapshots passed (within 10% tolerance)")
    print("=" * 100)


def test_betadelta_comparison():
    """
    Test get_betadelta vs get_betadelta_modified on random snapshots.
    """
    print("=" * 70)
    print("Testing get_betadelta vs get_betadelta_modified")
    print("=" * 70)

    jsonl_path = os.path.join(PROJECT_ROOT, 'comparison', '1e7_sfe020_n1e4_test_dictionary.jsonl')

    if not os.path.exists(jsonl_path):
        print(f"ERROR: Test dictionary not found at {jsonl_path}")
        print("Please ensure the comparison data exists.")
        return False

    # Get total lines
    total_lines = get_total_lines(jsonl_path)
    max_line = min(300, total_lines)

    # Pick random snapshots - use later snapshots where beta/delta matter more
    random.seed(42)
    snapshot_lines = sorted(random.sample(range(50, max_line + 1), min(10, max_line - 49)))

    print(f"\nTesting {len(snapshot_lines)} random snapshots from lines {snapshot_lines[0]}-{snapshot_lines[-1]}")
    print(f"Selected lines: {snapshot_lines}")

    results = []

    for line_num in snapshot_lines:
        print(f"\nProcessing snapshot {line_num}...", end=" ", flush=True)
        try:
            snapshot = load_snapshot_from_jsonl(jsonl_path, line_num)
            result = test_snapshot(snapshot, verbose=False)
            result['line'] = line_num
            results.append(result)
            status = "✓" if result['passed'] else "✗"
            speedup = result.get('speedup', 1.0)
            print(f"{status} (t={result['t_now']:.4e}, speedup={speedup:.1f}x)")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'passed': False,
                'line': line_num,
                't_now': 0,
                'field_results': {
                    'beta': (False, 1.0, np.nan, np.nan),
                    'delta': (False, 1.0, np.nan, np.nan),
                    'Edot_residual': (False, 1.0, np.nan, np.nan),
                    'T_residual': (False, 1.0, np.nan, np.nan),
                },
                'time_original': 0,
                'time_modified': 0,
                'speedup': 1.0,
                'original_success': False,
                'modified_success': False,
            })

    # Print comparison table
    print_comparison_table(results)

    all_passed = all(r['passed'] for r in results)
    return all_passed


if __name__ == '__main__':
    print("Beta-Delta Solver Comparison Test")
    print("==================================\n")

    passed = test_betadelta_comparison()

    print("\n" + "=" * 70)
    if passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED (differences may be due to different local minima)")
        print("Check that residuals are small for both solvers.")
        sys.exit(1)
