#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full comparison test for run_energy_phase vs run_energy_phase_modified.

This test runs the complete energy phase using both implementations from a
single t=0 snapshot and compares their outputs. It generates PDF plots showing
the evolution of parameters over time.

Uses: test/mockParams/1e7_sfe001_n1e4_t0_debug_snapshot.json

Author: TRINITY Team
"""

import numpy as np
import scipy.interpolate
import json
import sys
import os
import copy
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src._functions.unit_conversions import CONV, CGS

# =============================================================================
# Project root for finding data files
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_SNAPSHOT_PATH = os.path.join(TEST_DIR, 'mockParams', '1e7_sfe001_n1e4_t0_debug_snapshot.json')


# =============================================================================
# MockParam class that mimics DescribedItem
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


# =============================================================================
# MockParamsDict that tracks history
# =============================================================================

class MockParamsDict(dict):
    """
    Mock params dictionary that tracks history via save_snapshot().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self._snapshot_count = 0

    def save_snapshot(self):
        """Record current state to history."""
        snapshot = {}
        for key, param in self.items():
            if hasattr(param, 'value'):
                val = param.value
                # Only store scalars for history tracking
                if isinstance(val, (int, float, bool, np.integer, np.floating)):
                    snapshot[key] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val
                elif isinstance(val, np.ndarray) and val.size == 1:
                    snapshot[key] = float(val.item())
        self.history.append(snapshot)
        self._snapshot_count += 1

    def get_history_array(self, key):
        """Get array of values for a key across all snapshots."""
        values = []
        for snapshot in self.history:
            if key in snapshot:
                values.append(snapshot[key])
            else:
                values.append(np.nan)
        return np.array(values)

    def get_time_array(self):
        """Get array of t_now values from history."""
        return self.get_history_array('t_now')


# =============================================================================
# Helper functions
# =============================================================================

def load_debug_snapshot(filepath: str) -> dict:
    """Load the debug snapshot JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class MockSB99Interpolator:
    """Mock interpolator for SB99 data that interpolates from data arrays."""
    def __init__(self, times, values):
        self.x = np.array(times)
        self._values = np.array(values)
        self._interp = scipy.interpolate.interp1d(
            self.x, self._values,
            kind='linear',
            bounds_error=False,
            fill_value=(self._values[0], self._values[-1])
        )

    def __call__(self, t):
        return self._interp(t)


def create_sb99f_from_snapshot(snapshot: dict) -> dict:
    """Create SB99f structure from snapshot's SB99_data."""
    sb99_data = snapshot.get('SB99_data', [])
    if not sb99_data or len(sb99_data) < 11:
        raise ValueError("SB99_data not found or incomplete in snapshot")

    # SB99_data structure: [times, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]
    times = sb99_data[0]
    Qi = sb99_data[1]
    Li = sb99_data[2]
    Ln = sb99_data[3]
    Lbol = sb99_data[4]
    Lmech_W = sb99_data[5]
    Lmech_SN = sb99_data[6]
    Lmech_total = sb99_data[7]
    pdot_W = sb99_data[8]
    pdot_SN = sb99_data[9]
    pdot_total = sb99_data[10]

    return {
        'fQi': MockSB99Interpolator(times, Qi),
        'fLi': MockSB99Interpolator(times, Li),
        'fLn': MockSB99Interpolator(times, Ln),
        'fLbol': MockSB99Interpolator(times, Lbol),
        'fLmech_W': MockSB99Interpolator(times, Lmech_W),
        'fLmech_SN': MockSB99Interpolator(times, Lmech_SN),
        'fLmech_total': MockSB99Interpolator(times, Lmech_total),
        'fpdot_W': MockSB99Interpolator(times, pdot_W),
        'fpdot_SN': MockSB99Interpolator(times, pdot_SN),
        'fpdot_total': MockSB99Interpolator(times, pdot_total),
    }


def setup_cooling_interpolation(snapshot: dict, params: dict):
    """Set up cooling interpolation from snapshot data."""
    # CIE cooling
    logT = snapshot.get('cStruc_cooling_CIE_logT', [])
    logLambda = snapshot.get('cStruc_cooling_CIE_logLambda', [])

    if logT and logLambda:
        logT = np.array(logT)
        logLambda = np.array(logLambda)
        cooling_CIE_interpolation = scipy.interpolate.interp1d(
            logT, logLambda, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        params['cStruc_cooling_CIE_logT'] = MockParam(logT)
        params['cStruc_cooling_CIE_logLambda'] = MockParam(logLambda)
        params['cStruc_cooling_CIE_interpolation'] = MockParam(cooling_CIE_interpolation)
    else:
        # Mock cooling
        logT = np.linspace(4, 9, 100)
        logLambda = -22 + 0.5 * (logT - 6)
        cooling_CIE_interpolation = scipy.interpolate.interp1d(
            logT, logLambda, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        params['cStruc_cooling_CIE_logT'] = MockParam(logT)
        params['cStruc_cooling_CIE_logLambda'] = MockParam(logLambda)
        params['cStruc_cooling_CIE_interpolation'] = MockParam(cooling_CIE_interpolation)

    # Non-CIE cooling - create mock structures
    class MockCoolingCube:
        """Mock non-CIE cooling structure with required attributes."""
        def __init__(self):
            self.temp = np.linspace(4.0, 5.5, 20)
            self.ndens = np.linspace(-2, 6, 20)
            self.phi = np.linspace(8, 14, 20)
            self.datacube = np.ones((20, 20, 20)) * (-23.0)
            self.interp = scipy.interpolate.RegularGridInterpolator(
                (self.ndens, self.temp, self.phi),
                self.datacube,
                method='linear',
                bounds_error=False,
                fill_value=-23.0
            )

    params['cStruc_cooling_nonCIE'] = MockParam(MockCoolingCube())
    params['cStruc_heating_nonCIE'] = MockParam(MockCoolingCube())

    mock_cube = params['cStruc_cooling_nonCIE'].value
    net_cooling = np.ones((20, 20, 20)) * (-23.0)
    params['cStruc_net_nonCIE_interpolation'] = MockParam(
        scipy.interpolate.RegularGridInterpolator(
            (mock_cube.ndens, mock_cube.temp, mock_cube.phi),
            net_cooling,
            method='linear',
            bounds_error=False,
            fill_value=-23.0
        )
    )


def make_mock_params(snapshot: dict) -> MockParamsDict:
    """
    Create a MockParamsDict from the debug snapshot.

    The snapshot already has values in AU units, so we just need to wrap them.
    """
    params = MockParamsDict()

    # Skip meta and special keys
    skip_keys = {'_meta', 'SB99_data', 'cStruc_cooling_CIE_logT', 'cStruc_cooling_CIE_logLambda'}

    for key, value in snapshot.items():
        if key in skip_keys:
            continue
        # Handle arrays
        if isinstance(value, list):
            params[key] = MockParam(np.array(value))
        else:
            params[key] = MockParam(value)

    # Set up SB99 feedback interpolation
    params['SB99f'] = MockParam(create_sb99f_from_snapshot(snapshot))

    # Set up cooling interpolation
    setup_cooling_interpolation(snapshot, params)

    # Ensure required parameters exist
    # Note: t_previousCoolingUpdate should stay at 1e30 to trigger cooling update
    # The path_cooling_nonCIE needs to point to the correct location
    if 'path_cooling_nonCIE' in params:
        # Fix path to be relative to PROJECT_ROOT if it's an absolute path from another machine
        orig_path = params['path_cooling_nonCIE'].value
        if isinstance(orig_path, str) and '/Users/' in orig_path:
            # Extract relative path and make it relative to PROJECT_ROOT
            params['path_cooling_nonCIE'] = MockParam(os.path.join(PROJECT_ROOT, 'lib', 'cooling', 'opiate/'))

    if 't_next' not in params:
        params['t_next'] = MockParam(params['t_now'].value + 1e-6)
    if 'isDissolved' not in params:
        params['isDissolved'] = MockParam(False)
    if 'isCollapse' not in params:
        params['isCollapse'] = MockParam(False)

    # Initialize bubble arrays if not present
    for key in ['bubble_v_arr', 'bubble_T_arr', 'bubble_dTdr_arr', 'bubble_r_arr', 'bubble_n_arr']:
        if key not in params:
            params[key] = MockParam(np.array([]))

    # Initialize bubble scalars
    for key in ['bubble_r_Tb', 'bubble_LTotal', 'bubble_T_r_Tb', 'bubble_L1Bubble',
                'bubble_L2Conduction', 'bubble_L3Intermediate', 'bubble_Tavg',
                'bubble_mass', 'bubble_dMdt']:
        if key not in params:
            params[key] = MockParam(0.0)

    # Initialize shell parameters
    for key in ['shell_fAbsorbedIon', 'shell_fAbsorbedNeu', 'shell_fAbsorbedWeightedTotal',
                'shell_fIonisedDust', 'shell_F_rad', 'shell_thickness', 'shell_nMax',
                'shell_n0', 'shell_grav_force_m', 'shell_mass', 'shell_massDot']:
        if key not in params:
            params[key] = MockParam(0.0)

    if 'shell_tauKappaRatio' not in params:
        params['shell_tauKappaRatio'] = MockParam(1.0)

    for key in ['shell_grav_r', 'shell_grav_phi']:
        if key not in params:
            params[key] = MockParam(np.array([]))

    # Force parameters
    for key in ['F_grav', 'F_ion_in', 'F_ion_out', 'F_ram', 'F_rad', 'F_ISM', 'F_wind', 'F_SN']:
        if key not in params:
            params[key] = MockParam(0.0)

    return params


# =============================================================================
# Plotting functions
# =============================================================================

def plot_parameter_grid(params_orig, params_mod, keys, title, pdf_path):
    """
    Plot a grid of parameter evolution comparisons.
    """
    # Filter to only scalar parameters that exist in history
    valid_keys = []
    for key in keys:
        orig_arr = params_orig.get_history_array(key)
        if len(orig_arr) > 0 and not np.all(np.isnan(orig_arr)):
            valid_keys.append(key)

    if not valid_keys:
        print(f"  No valid keys to plot for {title}")
        return

    # Calculate grid dimensions
    n_params = len(valid_keys)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_params == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    t_orig = params_orig.get_time_array()
    t_mod = params_mod.get_time_array()

    for idx, key in enumerate(valid_keys):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        orig_arr = params_orig.get_history_array(key)
        mod_arr = params_mod.get_history_array(key)

        # Plot both
        ax.plot(t_orig * 1e3, orig_arr, 'b-', label='Original', linewidth=1.5, alpha=0.8)
        ax.plot(t_mod * 1e3, mod_arr, 'r--', label='Modified', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time [kyr]')
        ax.set_ylabel(key)
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 3))

    # Hide empty subplots
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")


def generate_comparison_plots(params_orig, params_mod, output_dir):
    """Generate all comparison PDF plots."""

    print("\nGenerating comparison plots...")

    # A) Shell parameters
    shell_keys = [k for k in params_orig.keys()
                  if k.startswith('shell_') and hasattr(params_orig[k], 'value')
                  and isinstance(params_orig[k].value, (int, float, np.integer, np.floating))]
    if shell_keys:
        plot_parameter_grid(
            params_orig, params_mod, shell_keys,
            "Shell Parameters Evolution",
            os.path.join(output_dir, "comparison_shell_parameters.pdf")
        )

    # B) Bubble parameters
    bubble_keys = [k for k in params_orig.keys()
                   if k.startswith('bubble_') and hasattr(params_orig[k], 'value')
                   and isinstance(params_orig[k].value, (int, float, np.integer, np.floating))]
    if bubble_keys:
        plot_parameter_grid(
            params_orig, params_mod, bubble_keys,
            "Bubble Parameters Evolution",
            os.path.join(output_dir, "comparison_bubble_parameters.pdf")
        )

    # C) TRINITY essentials
    essential_keys = ['R1', 'R2', 'rShell', 'Pb', 'Eb', 'T0']
    essential_keys = [k for k in essential_keys if k in params_orig]
    if essential_keys:
        plot_parameter_grid(
            params_orig, params_mod, essential_keys,
            "TRINITY Essential Parameters Evolution",
            os.path.join(output_dir, "comparison_essential_parameters.pdf")
        )

    # D) Force parameters
    force_keys = [k for k in params_orig.keys()
                  if 'F_' in k and hasattr(params_orig[k], 'value')
                  and isinstance(params_orig[k].value, (int, float, np.integer, np.floating))]
    if force_keys:
        plot_parameter_grid(
            params_orig, params_mod, force_keys,
            "Force Parameters Evolution",
            os.path.join(output_dir, "comparison_force_parameters.pdf")
        )


# =============================================================================
# Main test function
# =============================================================================

def run_full_comparison():
    """Run full energy phase comparison test."""

    print("=" * 70)
    print("Full Energy Phase Comparison Test")
    print("run_energy_phase.py vs run_energy_phase_modified.py")
    print("=" * 70)

    # Load snapshot
    if not os.path.exists(MOCK_SNAPSHOT_PATH):
        print(f"ERROR: Snapshot not found at {MOCK_SNAPSHOT_PATH}")
        return False

    print(f"\nLoading snapshot from: {MOCK_SNAPSHOT_PATH}")
    snapshot = load_debug_snapshot(MOCK_SNAPSHOT_PATH)
    print(f"  Model: {snapshot.get('model_name', 'N/A')}")
    print(f"  Initial t_now: {snapshot.get('t_now', 'N/A'):.6e} Myr")
    print(f"  Initial R2: {snapshot.get('R2', 'N/A'):.6e} pc")
    print(f"  Initial Eb: {snapshot.get('Eb', 'N/A'):.6e}")

    # Create two independent params dictionaries
    print("\nCreating mock params for both implementations...")
    params_orig = make_mock_params(snapshot)
    params_mod = make_mock_params(snapshot)

    # Import run_energy functions
    print("\nImporting run_energy functions...")
    try:
        from src.phase1_energy.run_energy_phase import run_energy as run_energy_original
        from src.phase1_energy.run_energy_phase_modified import run_energy as run_energy_modified
    except ImportError as e:
        print(f"ERROR: Failed to import run_energy functions: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run original implementation
    print("\n" + "-" * 70)
    print("Running ORIGINAL implementation...")
    print("-" * 70)
    start_orig = time.perf_counter()
    try:
        run_energy_original(params_orig)
        time_orig = time.perf_counter() - start_orig
        print(f"\nOriginal completed in {time_orig:.2f} seconds")
        print(f"  Snapshots recorded: {len(params_orig.history)}")
        print(f"  Final t_now: {params_orig['t_now'].value:.6e} Myr")
        print(f"  Final R2: {params_orig['R2'].value:.6e} pc")
    except Exception as e:
        print(f"ERROR: Original implementation failed: {e}")
        import traceback
        traceback.print_exc()
        time_orig = None

    # Run modified implementation
    print("\n" + "-" * 70)
    print("Running MODIFIED implementation...")
    print("-" * 70)
    start_mod = time.perf_counter()
    try:
        run_energy_modified(params_mod)
        time_mod = time.perf_counter() - start_mod
        print(f"\nModified completed in {time_mod:.2f} seconds")
        print(f"  Snapshots recorded: {len(params_mod.history)}")
        print(f"  Final t_now: {params_mod['t_now'].value:.6e} Myr")
        print(f"  Final R2: {params_mod['R2'].value:.6e} pc")
    except Exception as e:
        print(f"ERROR: Modified implementation failed: {e}")
        import traceback
        traceback.print_exc()
        time_mod = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if time_orig and time_mod:
        speedup = time_orig / time_mod if time_mod > 0 else float('inf')
        print(f"\nTiming:")
        print(f"  Original: {time_orig:.2f} s")
        print(f"  Modified: {time_mod:.2f} s")
        print(f"  Speedup:  {speedup:.2f}x")

        # Compare final states
        print(f"\nFinal State Comparison:")
        essentials = ['R2', 'v2', 'Eb', 't_now', 'R1', 'Pb', 'T0']
        all_close = True
        for key in essentials:
            if key in params_orig and key in params_mod:
                orig_val = params_orig[key].value
                mod_val = params_mod[key].value
                if isinstance(orig_val, (int, float)) and isinstance(mod_val, (int, float)):
                    rel_diff = abs(orig_val - mod_val) / max(abs(orig_val), 1e-300)
                    status = "OK" if rel_diff < 0.01 else "DIFF"
                    if rel_diff >= 0.01:
                        all_close = False
                    print(f"  {key:12s}: orig={orig_val:.6e}, mod={mod_val:.6e}, rel_diff={rel_diff:.2e} [{status}]")

        # Generate plots
        generate_comparison_plots(params_orig, params_mod, TEST_DIR)

        print("\n" + "=" * 70)
        if all_close:
            print("TEST COMPLETED - Results match within tolerance")
        else:
            print("TEST COMPLETED - Some differences detected")
        print("=" * 70)
        return True
    else:
        print("\nOne or both implementations failed. Cannot compare.")
        return False


if __name__ == '__main__':
    success = run_full_comparison()
    sys.exit(0 if success else 1)
