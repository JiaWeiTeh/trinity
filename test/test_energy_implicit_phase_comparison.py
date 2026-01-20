#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification test comparing run_energy_implicit_phase vs run_energy_implicit_phase_modified.

This test loads snapshots from the dictionary.jsonl file and runs both the original
and modified implicit energy phase runners for a short integration period (t_now to t_now + 0.1 Myr).

Outputs a comparison table for speed and parameter output error %.

Usage:
    python test_energy_implicit_phase_comparison.py [-n NUM_TESTS]

Author: TRINITY Team
"""

import numpy as np
import scipy.interpolate
import json
import sys
import os
import time
import copy
import argparse
import logging

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src._functions.unit_conversions import CONV, CGS

# =============================================================================
# Project root for finding data files
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Configuration
# =============================================================================
NUM_TESTS = 3  # Number of test runs
INTEGRATION_DT = 0.1  # Myr - duration of each test run

# Target start times for tests (Myr)
# These should be times where implicit phase snapshots exist
# Using earlier times where the phase is more stable
TARGET_START_TIMES = [0.1]  # t_start values to test

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing


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


class MockParamsDict(dict):
    """Mock params dictionary that supports .get() with proper handling."""

    def get(self, key, default=None):
        """Override get to return the item or default."""
        if key in self:
            return self[key]
        return default

    def save_snapshot(self):
        """Mock save_snapshot (no-op for testing)."""
        pass

    def flush(self):
        """Mock flush (no-op for testing)."""
        pass

    def reset_keys(self, keys, value=np.nan):
        """Mock reset_keys - set multiple keys to a value."""
        for key in keys:
            if key in self:
                self[key].value = value


# =============================================================================
# Snapshot loading utilities
# =============================================================================

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


def find_implicit_phase_lines(filepath: str) -> list:
    """Find all line numbers where current_phase == 'implicit'."""
    implicit_lines = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            if data.get('current_phase') == 'implicit':
                implicit_lines.append(i + 1)  # 1-indexed
    return implicit_lines


def find_snapshot_near_time(filepath: str, target_time: float, phase: str = 'implicit') -> tuple:
    """
    Find the snapshot line number closest to a target time.

    Parameters
    ----------
    filepath : str
        Path to JSONL file
    target_time : float
        Target time in Myr
    phase : str, optional
        Required phase ('implicit', 'energy', etc.). Default 'implicit'.

    Returns
    -------
    tuple
        (line_number, actual_t_now) of the closest snapshot
    """
    best_line = None
    best_time = None
    best_diff = float('inf')

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            if phase and data.get('current_phase') != phase:
                continue
            t_now = data.get('t_now', 0)
            diff = abs(t_now - target_time)
            if diff < best_diff:
                best_diff = diff
                best_line = i + 1  # 1-indexed
                best_time = t_now

    if best_line is None:
        raise ValueError(f"No {phase} phase snapshots found in {filepath}")

    return best_line, best_time


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
    """Add physical constants that are missing from the dictionary snapshot."""
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

    mu_convert = defaults.get('mu_convert', 1.4)
    params['mu_convert'] = mu_convert * m_H_to_Msun

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

    # Mechanical feedback parameters
    if 'Lmech_total' not in params:
        params['Lmech_total'] = params.get('LWind', 0.0)
    if 'v_mech_total' not in params:
        params['v_mech_total'] = params.get('vWind', 0.0)
    if 'pdot_total' not in params:
        params['pdot_total'] = params.get('pWindDot', 0.0)
    if 'pdotdot_total' not in params:
        params['pdotdot_total'] = params.get('pWindDotDot', 0.0)

    # Cloud parameters
    if 'nCore' not in params:
        params['nCore'] = 1e4 * ndens_cgs2au

    if 'rCore' not in params:
        rCloud = params.get('rCloud', 10.0)
        params['rCore'] = 0.1 * rCloud

    if 'dens_profile' not in params:
        params['dens_profile'] = 'densPL'

    if 'densPL_alpha' not in params:
        params['densPL_alpha'] = -2.0

    # Cluster mass
    if 'mCluster' not in params:
        sfe = params.get('sfe', 0.01)
        mCloud = params.get('mCloud', 1e7)
        params['mCluster'] = sfe * mCloud

    # Stop conditions
    if 'stop_t' not in params:
        params['stop_t'] = 10.0  # Myr

    # Feedback force terms
    for key in ['F_grav', 'F_ion_in', 'F_ion_out', 'F_ram', 'F_rad']:
        if key not in params:
            params[key] = 0.0

    # Flags
    if 'isCollapse' not in params:
        params['isCollapse'] = False
    if 'EarlyPhaseApproximation' not in params:
        params['EarlyPhaseApproximation'] = False
    if 'current_phase' not in params:
        params['current_phase'] = 'implicit'
    if 'PISM' not in params:
        params['PISM'] = 5e3

    return params


class MockSB99Interpolator:
    """Mock interpolator for SB99 data that returns constant values."""
    def __init__(self, value, t_min=0.0, t_max=100.0):
        self.value = value
        self.x = np.array([t_min, t_max])

    def __call__(self, t):
        return np.array(self.value)


def create_mock_sb99f(params: dict) -> dict:
    """Create a mock SB99f structure with interpolation functions."""
    Qi = params.get('Qi', 1e49)
    Lbol = params.get('Lbol', 1e10)
    Li = params.get('Li', 0.5 * Lbol)
    Ln = params.get('Ln', 0.5 * Lbol)
    Lmech = params.get('LWind', params.get('Lmech_total', 1e10))
    pdot_W = params.get('pWindDot', 1e7)
    pdot_SN = 0.0
    pdot_total = pdot_W + pdot_SN

    t_min = 0.0
    t_max = 100.0

    return {
        'fQi': MockSB99Interpolator(Qi, t_min, t_max),
        'fLi': MockSB99Interpolator(Li, t_min, t_max),
        'fLn': MockSB99Interpolator(Ln, t_min, t_max),
        'fLbol': MockSB99Interpolator(Lbol, t_min, t_max),
        'fLmech_W': MockSB99Interpolator(Lmech, t_min, t_max),
        'fLmech_SN': MockSB99Interpolator(0.0, t_min, t_max),
        'fLmech_total': MockSB99Interpolator(Lmech, t_min, t_max),
        'fpdot_W': MockSB99Interpolator(pdot_W, t_min, t_max),
        'fpdot_SN': MockSB99Interpolator(pdot_SN, t_min, t_max),
        'fpdot_total': MockSB99Interpolator(pdot_total, t_min, t_max),
    }


def prime_params(params: dict) -> dict:
    """Initialize parameters that require setup beyond simple values."""
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
    class MockCloudyCube:
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

    # SB99 feedback
    if 'SB99f' not in params:
        params['SB99f'] = create_mock_sb99f(params)

    # Mechanical feedback
    Lmech_from_sb99 = params.get('LWind', params.get('Lmech_total', 1e10))
    params['Lmech_total'] = Lmech_from_sb99

    v_mech_from_sb99 = params.get('vWind', params.get('v_mech_total', 1000))
    params['v_mech_total'] = v_mech_from_sb99

    # Output path (required by original)
    if 'path2output' not in params:
        params['path2output'] = '/tmp/trinity_test_output'
        os.makedirs(params['path2output'], exist_ok=True)

    return params


def make_params_dict(snapshot: dict, include_priming: bool = True) -> MockParamsDict:
    """Create a MockParamsDict with MockParam wrappers from a snapshot."""
    snapshot = add_physical_constants(snapshot.copy())
    if include_priming:
        snapshot = prime_params(snapshot)
    return MockParamsDict({k: MockParam(v) for k, v in snapshot.items()})


# =============================================================================
# Test runner
# =============================================================================

def run_original_phase(params, stop_t: float):
    """Run the original energy implicit phase."""
    from src.phase1b_energy_implicit import run_energy_implicit_phase

    params['stop_t'].value = stop_t
    run_energy_implicit_phase.run_phase_energy(params)

    return {
        't_now': params['t_now'].value,
        'R2': params['R2'].value,
        'v2': params['v2'].value,
        'Eb': params['Eb'].value,
        'T0': params['T0'].value,
        'cool_beta': params['cool_beta'].value,
        'cool_delta': params['cool_delta'].value,
    }


def run_modified_phase(params, stop_t: float):
    """Run the modified energy implicit phase."""
    from src.phase1b_energy_implicit import run_energy_implicit_phase_modified

    params['stop_t'].value = stop_t
    results = run_energy_implicit_phase_modified.run_phase_energy(params)

    return {
        't_now': params['t_now'].value,
        'R2': params['R2'].value,
        'v2': params['v2'].value,
        'Eb': params['Eb'].value,
        'T0': params['T0'].value,
        'cool_beta': params['cool_beta'].value,
        'cool_delta': params['cool_delta'].value,
    }


def test_snapshot(snapshot: dict, dt: float = INTEGRATION_DT, verbose: bool = False) -> dict:
    """
    Test a single snapshot by running both original and modified phase runners.

    Parameters
    ----------
    snapshot : dict
        Snapshot data from jsonl
    dt : float
        Duration of integration (Myr)
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Test results including timing and errors
    """
    t_start = snapshot.get('t_now', 0.1)
    t_stop = t_start + dt

    if verbose:
        print(f"  Running from t={t_start:.4e} to t={t_stop:.4e}")

    # Create two independent param sets
    params_original = make_params_dict(snapshot)
    params_modified = make_params_dict(snapshot)

    # Run original version with timing
    start_orig = time.perf_counter()
    try:
        result_original = run_original_phase(params_original, t_stop)
        time_original = time.perf_counter() - start_orig
        original_success = True
    except Exception as e:
        time_original = time.perf_counter() - start_orig
        result_original = None
        original_success = False
        if verbose:
            print(f"  Original failed: {e}")
            import traceback
            traceback.print_exc()

    # Run modified version with timing
    start_mod = time.perf_counter()
    try:
        result_modified = run_modified_phase(params_modified, t_stop)
        time_modified = time.perf_counter() - start_mod
        modified_success = True
    except Exception as e:
        time_modified = time.perf_counter() - start_mod
        result_modified = None
        modified_success = False
        if verbose:
            print(f"  Modified failed: {e}")
            import traceback
            traceback.print_exc()

    # Calculate speedup
    speedup = time_original / time_modified if time_modified > 0 else float('inf')

    # Compare results
    fields = ['R2', 'v2', 'Eb', 'T0', 'cool_beta', 'cool_delta']
    field_results = {}
    all_passed = True

    if original_success and modified_success:
        for field in fields:
            orig_val = result_original[field]
            mod_val = result_modified[field]

            # Calculate relative error
            if abs(orig_val) > 1e-300:
                rel_error = abs(orig_val - mod_val) / abs(orig_val)
                pct_error = 100.0 * rel_error
            else:
                rel_error = abs(mod_val) if abs(mod_val) > 1e-300 else 0.0
                pct_error = 100.0 * rel_error

            # Allow 10% tolerance for slightly different integration paths
            passed = rel_error < 0.10

            field_results[field] = {
                'passed': passed,
                'original': orig_val,
                'modified': mod_val,
                'rel_error': rel_error,
                'pct_error': pct_error,
            }

            if not passed:
                all_passed = False
                if verbose:
                    print(f"  {field}: orig={orig_val:.4e}, mod={mod_val:.4e}, err={pct_error:.2f}%")
    else:
        all_passed = False
        for field in fields:
            field_results[field] = {
                'passed': False,
                'original': result_original[field] if result_original else np.nan,
                'modified': result_modified[field] if result_modified else np.nan,
                'rel_error': np.nan,
                'pct_error': np.nan,
            }

    return {
        'passed': all_passed,
        't_start': t_start,
        't_stop': t_stop,
        'field_results': field_results,
        'time_original': time_original,
        'time_modified': time_modified,
        'speedup': speedup,
        'original_success': original_success,
        'modified_success': modified_success,
    }


def print_comparison_table(results: list):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("ENERGY IMPLICIT PHASE COMPARISON TEST")
    print("Original: run_energy_implicit_phase.run_phase_energy (manual Euler stepping)")
    print("Modified: run_energy_implicit_phase_modified.run_phase_energy (scipy.solve_ivp)")
    print("=" * 100)

    # Header
    print("\n" + "-" * 100)
    header = (f"{'Test':<6} {'t_start':<10} {'t_stop':<10} "
              f"{'Orig(s)':<10} {'Mod(s)':<10} {'Speedup':<10} {'Status':<10}")
    print(header)
    print("-" * 100)

    total_time_orig = 0.0
    total_time_mod = 0.0

    for i, res in enumerate(results):
        t_start = res['t_start']
        t_stop = res['t_stop']
        time_orig = res.get('time_original', 0)
        time_mod = res.get('time_modified', 0)
        speedup = res.get('speedup', 1.0)

        total_time_orig += time_orig
        total_time_mod += time_mod

        status = "PASS" if res['passed'] else "FAIL"
        if not res['original_success']:
            status = "ORIG_ERR"
        elif not res['modified_success']:
            status = "MOD_ERR"

        row = (f"{i+1:<6} {t_start:<10.4e} {t_stop:<10.4e} "
               f"{time_orig:<10.2f} {time_mod:<10.2f} {speedup:<10.2f}x {status:<10}")
        print(row)

    print("-" * 100)

    # Timing summary
    avg_speedup = total_time_orig / total_time_mod if total_time_mod > 0 else 1.0
    print(f"\nTiming Summary:")
    print(f"  Total original time: {total_time_orig:.2f} s")
    print(f"  Total modified time: {total_time_mod:.2f} s")
    print(f"  Average speedup: {avg_speedup:.2f}x")

    # Error details
    print("\n" + "=" * 100)
    print("PARAMETER COMPARISON (% Error)")
    print("=" * 100)
    print("-" * 100)
    header2 = (f"{'Test':<6} {'R2':<12} {'v2':<12} {'Eb':<12} {'T0':<12} "
               f"{'beta':<12} {'delta':<12}")
    print(header2)
    print("-" * 100)

    for i, res in enumerate(results):
        if res['original_success'] and res['modified_success']:
            fr = res['field_results']
            row = (f"{i+1:<6} "
                   f"{fr['R2']['pct_error']:<12.4f} "
                   f"{fr['v2']['pct_error']:<12.4f} "
                   f"{fr['Eb']['pct_error']:<12.4f} "
                   f"{fr['T0']['pct_error']:<12.4f} "
                   f"{fr['cool_beta']['pct_error']:<12.4f} "
                   f"{fr['cool_delta']['pct_error']:<12.4f}")
            print(row)
        else:
            print(f"{i+1:<6} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    print("-" * 100)

    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    print(f"\nSummary: {passed_count}/{len(results)} tests passed (within 10% tolerance)")
    print("=" * 100)


def test_energy_implicit_phase_comparison(num_tests: int = None, target_times: list = None):
    """
    Test run_energy_implicit_phase vs run_energy_implicit_phase_modified.

    Parameters
    ----------
    num_tests : int, optional
        Number of tests to run. Defaults to NUM_TESTS.
    target_times : list, optional
        List of target start times (Myr). Defaults to TARGET_START_TIMES.
    """
    if num_tests is None:
        num_tests = NUM_TESTS
    if target_times is None:
        target_times = TARGET_START_TIMES[:num_tests]

    print("=" * 70)
    print("Testing run_energy_implicit_phase vs run_energy_implicit_phase_modified")
    print("=" * 70)

    jsonl_path = os.path.join(PROJECT_ROOT, 'test', '1e7_sfe001_n1e4_test_dictionary.jsonl')

    if not os.path.exists(jsonl_path):
        print(f"ERROR: Test dictionary not found at {jsonl_path}")
        print("Please ensure the test data exists.")
        return False

    # Find implicit phase lines
    implicit_lines = find_implicit_phase_lines(jsonl_path)
    print(f"\nFound {len(implicit_lines)} implicit phase snapshots")

    # Find snapshots near each target time
    selected_snapshots = []
    print(f"\nFinding snapshots near target times: {target_times}")
    for target_t in target_times:
        try:
            line_num, actual_t = find_snapshot_near_time(jsonl_path, target_t, phase='implicit')
            selected_snapshots.append((line_num, target_t, actual_t))
            print(f"  Target t={target_t:.2f} Myr -> Line {line_num} (actual t={actual_t:.4e} Myr)")
        except ValueError as e:
            print(f"  Target t={target_t:.2f} Myr -> Not found: {e}")

    if not selected_snapshots:
        print("ERROR: No valid snapshots found for target times")
        return False

    print(f"\nSelected {len(selected_snapshots)} snapshots for testing")

    results = []

    for i, (line_num, target_t, actual_t) in enumerate(selected_snapshots):
        print(f"\nTest {i+1}/{len(selected_snapshots)} (line {line_num}, t={actual_t:.4e})...", end=" ", flush=True)
        try:
            snapshot = load_snapshot_from_jsonl(jsonl_path, line_num)
            result = test_snapshot(snapshot, verbose=False)
            result['line'] = line_num
            result['target_t'] = target_t
            results.append(result)
            status = "PASS" if result['passed'] else "FAIL"
            speedup = result.get('speedup', 1.0)
            print(f"{status} (speedup={speedup:.2f}x)")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'passed': False,
                'line': line_num,
                'target_t': target_t,
                't_start': actual_t,
                't_stop': actual_t + INTEGRATION_DT,
                'field_results': {},
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
    parser = argparse.ArgumentParser(description='Energy Implicit Phase Comparison Test')
    parser.add_argument('-n', '--num-tests', type=int, default=NUM_TESTS,
                        help=f'Number of tests to run (default: {NUM_TESTS})')
    args = parser.parse_args()

    print("Energy Implicit Phase Comparison Test")
    print("=====================================\n")

    passed = test_energy_implicit_phase_comparison(num_tests=args.num_tests)

    print("\n" + "=" * 70)
    if passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED (differences may be due to different integration methods)")
        print("Check that final parameter values are reasonably close.")
        sys.exit(1)
