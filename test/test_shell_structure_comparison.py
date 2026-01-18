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
    """Mock parameter object with .value attribute."""
    def __init__(self, value):
        self.value = value


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


def test_shell_structure_comparison():
    """
    Test that shell_structure_modified returns the same values as shell_structure.
    """
    print("=" * 70)
    print("Testing shell_structure vs shell_structure_modified comparison")
    print("=" * 70)

    # Path to test dictionary
    jsonl_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'comparison', '1e7_sfe020_n1e4_test_dictionary.jsonl'
    )

    if not os.path.exists(jsonl_path):
        print(f"ERROR: Test dictionary not found at {jsonl_path}")
        return False

    # Pick a random snapshot between lines 10-15
    line_number = random.randint(10, 15)
    print(f"\nLoading snapshot from line {line_number}...")

    snapshot = load_snapshot_from_jsonl(jsonl_path, line_number)
    print(f"  t_now: {snapshot.get('t_now', 'N/A'):.6e} Myr")
    print(f"  R2: {snapshot.get('R2', 'N/A'):.6e} pc")

    # Create params dict for original (will be mutated)
    params_original = make_params_dict(snapshot.copy())

    # Create params dict for modified (read-only)
    params_modified = make_params_dict(snapshot.copy())

    print("\nRunning original shell_structure()...")
    shell_structure(params_original)

    print("Running modified shell_structure_pure()...")
    result_modified = shell_structure_pure(params_modified)

    # Fields to compare (matching ShellProperties dataclass)
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
        ('shell_grav_r', 'shell_grav_r'),
        ('shell_grav_phi', 'shell_grav_phi'),
        ('shell_grav_force_m', 'shell_grav_force_m'),
        ('isDissolved', 'isDissolved'),
        # Note: is_fullyIonised not stored in original params, skip comparison
    ]

    print("\nComparing outputs:")
    print("-" * 50)

    all_passed = True
    for params_key, dataclass_attr in fields:
        # Get original value from params (mutated by shell_structure)
        original_val = params_original[params_key].value

        # Get modified value from dataclass
        modified_val = getattr(result_modified, dataclass_attr)

        passed, message = compare_values(params_key, original_val, modified_val)
        print(message)
        if not passed:
            all_passed = False

    print("-" * 50)
    if all_passed:
        print("✓ All comparisons PASSED!")
    else:
        print("✗ Some comparisons FAILED!")

    return all_passed


def test_multiple_snapshots():
    """
    Test multiple snapshots to ensure consistency across different states.
    """
    print("\n" + "=" * 70)
    print("Testing multiple snapshots (10-15)")
    print("=" * 70)

    jsonl_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'comparison', '1e7_sfe020_n1e4_test_dictionary.jsonl'
    )

    if not os.path.exists(jsonl_path):
        print(f"ERROR: Test dictionary not found at {jsonl_path}")
        return False

    all_passed = True
    for line_num in range(10, 16):
        print(f"\n--- Snapshot {line_num} ---")

        try:
            snapshot = load_snapshot_from_jsonl(jsonl_path, line_num)
            params_original = make_params_dict(snapshot.copy())
            params_modified = make_params_dict(snapshot.copy())

            shell_structure(params_original)
            result_modified = shell_structure_pure(params_modified)

            # Quick check of key values
            n0_orig = params_original['shell_n0'].value
            n0_mod = result_modified.shell_n0

            if np.isclose(n0_orig, n0_mod, rtol=1e-10):
                print(f"  ✓ shell_n0 matches: {n0_orig:.6e}")
            else:
                print(f"  ✗ shell_n0 mismatch: {n0_orig:.6e} vs {n0_mod:.6e}")
                all_passed = False

        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_passed = False

    return all_passed


if __name__ == '__main__':
    print("Shell Structure Comparison Test")
    print("================================\n")

    # Run single detailed comparison
    passed1 = test_shell_structure_comparison()

    # Run multiple snapshot test
    passed2 = test_multiple_snapshots()

    print("\n" + "=" * 70)
    if passed1 and passed2:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
