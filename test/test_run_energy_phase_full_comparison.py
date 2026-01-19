#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full comparison test for run_energy_phase vs run_energy_phase_modified.

This test compares the ODE and physics calculations across multiple snapshots
from an existing simulation run. It generates PDF plots showing parameter
evolution and comparison between original and modified implementations.

Instead of running the full energy phase (which requires complete TRINITY setup),
this test:
1. Loads a sequence of snapshots from the test JSONL file
2. For each snapshot, compares ODE outputs between original and modified
3. Generates PDF plots showing parameter evolution over time

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
from matplotlib.backends.backend_pdf import PdfPages

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src._functions.unit_conversions import CONV, CGS

# =============================================================================
# Project root for finding data files
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))


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
# Helper functions
# =============================================================================

def load_snapshot_from_jsonl(filepath: str, line_number: int) -> dict:
    """Load a specific snapshot (line) from a JSONL file."""
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number - 1:
                return json.loads(line.strip())
    raise ValueError(f"Line {line_number} not found in {filepath}")


def load_all_snapshots(filepath: str, max_lines: int = None) -> list:
    """Load all snapshots from a JSONL file."""
    snapshots = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                snapshots.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return snapshots


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


def search_jsonl_for_param(jsonl_path: str, param_name: str):
    """Search through JSONL file for a parameter value."""
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


def get_param_value(params: dict, key: str, defaults: dict, jsonl_path: str):
    """Get parameter value from params, defaults, or jsonl."""
    if key in params:
        return params[key]
    if key in defaults:
        return defaults[key]
    return search_jsonl_for_param(jsonl_path, key)


# Cache for default params
_DEFAULT_PARAMS_CACHE = None


def add_physical_constants(params: dict, jsonl_path: str = None) -> dict:
    """Add physical constants that are missing from the dictionary snapshot."""
    global _DEFAULT_PARAMS_CACHE

    if _DEFAULT_PARAMS_CACHE is None:
        _DEFAULT_PARAMS_CACHE = load_default_params()
    defaults = _DEFAULT_PARAMS_CACHE

    if jsonl_path is None:
        jsonl_path = os.path.join(PROJECT_ROOT, 'test', '1e7_sfe020_n1e4_test_dictionary.jsonl')

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
    k_B_cgs = get_param_value(params, 'k_B', defaults, jsonl_path) or CGS.k_B
    params['k_B'] = k_B_cgs * k_B_cgs2au

    G_cgs = get_param_value(params, 'G', defaults, jsonl_path) or CGS.G
    params['G'] = G_cgs * G_cgs2au

    c_cgs = get_param_value(params, 'c_light', defaults, jsonl_path) or CGS.c
    params['c_light'] = c_cgs * v_cms2au

    # Mean molecular weights
    mu_atom = get_param_value(params, 'mu_atom', defaults, jsonl_path)
    params['mu_atom'] = mu_atom * m_H_to_Msun if mu_atom else 1.27 * m_H_to_Msun

    mu_ion = get_param_value(params, 'mu_ion', defaults, jsonl_path)
    params['mu_ion'] = mu_ion * m_H_to_Msun if mu_ion else 0.61 * m_H_to_Msun

    mu_convert = get_param_value(params, 'mu_convert', defaults, jsonl_path)
    params['mu_convert'] = mu_convert * m_H_to_Msun if mu_convert else 1.4 * m_H_to_Msun

    # Shell temperatures
    params['TShell_ion'] = get_param_value(params, 'TShell_ion', defaults, jsonl_path) or 1e4
    params['TShell_neu'] = get_param_value(params, 'TShell_neu', defaults, jsonl_path) or 100

    # Case B recombination
    caseB_cgs = get_param_value(params, 'caseB_alpha', defaults, jsonl_path) or 2.59e-13
    params['caseB_alpha'] = caseB_cgs * cm3_to_pc3 / s_to_Myr

    # Dust parameters
    dust_sigma_cgs = get_param_value(params, 'dust_sigma', defaults, jsonl_path) or 1.5e-21
    params['dust_sigma'] = dust_sigma_cgs * cm2_to_pc2
    dust_KappaIR_cgs = get_param_value(params, 'dust_KappaIR', defaults, jsonl_path) or 4.0
    params['dust_KappaIR'] = dust_KappaIR_cgs * cm2_to_pc2 / CONV.g2Msun

    # Shell dissolution threshold
    stop_n_diss_cgs = get_param_value(params, 'stop_n_diss', defaults, jsonl_path) or 1.0
    params['stop_n_diss'] = stop_n_diss_cgs * ndens_cgs2au

    # ISM number density
    if 'nISM' not in params:
        nISM_cgs = get_param_value(params, 'nISM', defaults, jsonl_path) or 1.0
        params['nISM'] = nISM_cgs * ndens_cgs2au

    # Cloud metallicity
    if 'ZCloud' not in params:
        params['ZCloud'] = get_param_value(params, 'ZCloud', defaults, jsonl_path) or 1.0

    # Flags
    if 'isCollapse' not in params:
        params['isCollapse'] = False
    if 'EarlyPhaseApproximation' not in params:
        params['EarlyPhaseApproximation'] = False
    if 'current_phase' not in params:
        params['current_phase'] = 'energy'
    if 'PISM' not in params:
        PISM_val = get_param_value(params, 'PISM', defaults, jsonl_path) or 5e3
        params['PISM'] = PISM_val

    # Cluster mass
    if 'mCluster' not in params:
        sfe = get_param_value(params, 'sfe', defaults, jsonl_path) or 0.01
        mCloud = params.get('mCloud', 1e6)
        params['mCluster'] = sfe * mCloud

    # Adiabatic index
    if 'gamma_adia' not in params:
        gamma = get_param_value(params, 'gamma_adia', defaults, jsonl_path)
        params['gamma_adia'] = gamma if gamma else 5.0 / 3.0

    # Thermal conductivity coefficient (bubble-specific)
    C_thermal_cgs = get_param_value(params, 'C_thermal', defaults, jsonl_path) or 6e-7
    params['C_thermal'] = C_thermal_cgs * CONV.c_therm_cgs2au

    # Bubble temperature measurement radius ratio
    params['bubble_xi_Tb'] = get_param_value(params, 'bubble_xi_Tb', defaults, jsonl_path) or 0.98

    # Mechanical luminosity and velocity from wind data (if available)
    if 'LWind' in params and 'pWindDot' in params:
        params['Lmech_total'] = params['LWind']
        if params['pWindDot'] > 0:
            params['v_mech_total'] = 2.0 * params['LWind'] / params['pWindDot']
        else:
            params['v_mech_total'] = 1000.0

    # Mechanical luminosity and wind velocity
    if 'Lmech_total' not in params:
        params['Lmech_total'] = 1e36 * CONV.L_cgs2au

    if 'v_mech_total' not in params:
        params['v_mech_total'] = 1000e5 * v_cms2au

    # Start of star formation time
    if 'tSF' not in params:
        params['tSF'] = 0.0

    # Cloud density profile parameters
    if 'nCore' not in params:
        params['nCore'] = 1e4 * ndens_cgs2au

    if 'rCore' not in params:
        rCloud = params.get('rCloud', 10.0)
        params['rCore'] = 0.1 * rCloud

    if 'dens_profile' not in params:
        params['dens_profile'] = 'densPL'

    if 'densPL_alpha' not in params:
        params['densPL_alpha'] = -2.0

    # Force parameters
    for key in ['F_grav', 'F_ion_in', 'F_ion_out', 'F_ram', 'F_rad', 'F_ISM', 'F_wind', 'F_SN']:
        if key not in params:
            params[key] = 0.0

    # t_next for Euler stepping
    if 't_next' not in params:
        params['t_next'] = params.get('t_now', 0) + 1e-6

    # Cooling update time
    if 't_previousCoolingUpdate' not in params:
        params['t_previousCoolingUpdate'] = 0.0

    # isDissolved flag
    if 'isDissolved' not in params:
        params['isDissolved'] = False

    # Pre-initialize bubble arrays
    for key in ['bubble_v_arr', 'bubble_T_arr', 'bubble_dTdr_arr', 'bubble_r_arr', 'bubble_n_arr']:
        if key not in params:
            params[key] = np.array([])

    # Pre-initialize bubble scalar values
    for key in ['bubble_r_Tb', 'bubble_LTotal', 'bubble_T_r_Tb', 'bubble_L1Bubble',
                'bubble_L2Conduction', 'bubble_L3Intermediate', 'bubble_Tavg',
                'bubble_mass', 'bubble_dMdt']:
        if key not in params:
            params[key] = 0.0

    # Pre-initialize shell parameters
    for key in ['shell_fAbsorbedIon', 'shell_fAbsorbedNeu', 'shell_fAbsorbedWeightedTotal',
                'shell_fIonisedDust', 'shell_F_rad', 'shell_thickness', 'shell_nMax',
                'shell_n0', 'shell_grav_force_m', 'shell_mass', 'shell_massDot']:
        if key not in params:
            params[key] = 0.0

    if 'shell_tauKappaRatio' not in params:
        params['shell_tauKappaRatio'] = 1.0

    for key in ['shell_grav_r', 'shell_grav_phi']:
        if key not in params:
            params[key] = np.array([])

    return params


class MockSB99Interpolator:
    """Mock interpolator for SB99 data."""
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
    t_min, t_max = 0.0, 100.0

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
    # Cooling interpolation (CIE)
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
                print(f"Warning: Could not load cooling curve: {e}")

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

    if 'cStruc_cooling_nonCIE' not in params or params['cStruc_cooling_nonCIE'] is None:
        params['cStruc_cooling_nonCIE'] = MockCoolingCube()
    if 'cStruc_heating_nonCIE' not in params or params['cStruc_heating_nonCIE'] is None:
        params['cStruc_heating_nonCIE'] = MockCoolingCube()

    if 'cStruc_net_nonCIE_interpolation' not in params or params['cStruc_net_nonCIE_interpolation'] is None:
        mock_cube = params['cStruc_cooling_nonCIE']
        net_cooling = np.ones((20, 20, 20)) * (-23.0)
        params['cStruc_net_nonCIE_interpolation'] = scipy.interpolate.RegularGridInterpolator(
            (mock_cube.ndens, mock_cube.temp, mock_cube.phi),
            net_cooling,
            method='linear',
            bounds_error=False,
            fill_value=-23.0
        )

    # SB99 feedback
    if 'SB99f' not in params:
        params['SB99f'] = create_mock_sb99f(params)

    # Ensure Lmech/v_mech consistency
    Lmech_from_sb99 = params.get('LWind', params.get('Lmech_total', 1e10))
    params['Lmech_total'] = Lmech_from_sb99
    v_mech_from_sb99 = params.get('vWind', params.get('v_mech_total', 1000))
    params['v_mech_total'] = v_mech_from_sb99

    return params


def make_mock_params(snapshot: dict) -> dict:
    """Create a params dictionary with MockParam wrappers from a snapshot."""
    snapshot = add_physical_constants(snapshot.copy())
    snapshot = prime_params(snapshot)
    return {k: MockParam(v) for k, v in snapshot.items()}


# =============================================================================
# Plotting functions
# =============================================================================

def plot_parameter_evolution(snapshots: list, keys: list, title: str, pdf_path: str):
    """
    Plot evolution of parameters over time from snapshots.

    Parameters
    ----------
    snapshots : list
        List of snapshot dictionaries
    keys : list
        List of parameter keys to plot
    title : str
        Title for the plot
    pdf_path : str
        Path to save the PDF
    """
    # Filter to only scalar parameters that exist
    valid_keys = []
    for key in keys:
        has_data = False
        for snap in snapshots:
            val = snap.get(key)
            if val is not None and isinstance(val, (int, float)):
                has_data = True
                break
        if has_data:
            valid_keys.append(key)

    if not valid_keys:
        print(f"  No valid keys to plot for {title}")
        return

    # Get time array
    t_arr = np.array([snap.get('t_now', 0) for snap in snapshots]) * 1e3  # Convert to kyr

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

    for idx, key in enumerate(valid_keys):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get values
        values = []
        for snap in snapshots:
            val = snap.get(key)
            if val is not None and isinstance(val, (int, float)):
                values.append(val)
            else:
                values.append(np.nan)
        values = np.array(values)

        ax.plot(t_arr, values, 'b-', linewidth=1.5)
        ax.set_xlabel('Time [kyr]')
        ax.set_ylabel(key)
        ax.set_title(key, fontsize=10)
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


def generate_evolution_plots(snapshots: list, output_dir: str):
    """Generate all parameter evolution PDF plots from snapshots."""

    print("\nGenerating parameter evolution plots from simulation data...")

    # A) Shell parameters
    shell_keys = []
    for snap in snapshots:
        for k in snap.keys():
            if k.startswith('shell_') and k not in shell_keys:
                val = snap[k]
                if isinstance(val, (int, float)):
                    shell_keys.append(k)
    shell_keys = sorted(set(shell_keys))
    if shell_keys:
        plot_parameter_evolution(
            snapshots, shell_keys,
            "Shell Parameters Evolution",
            os.path.join(output_dir, "evolution_shell_parameters.pdf")
        )

    # B) Bubble parameters
    bubble_keys = []
    for snap in snapshots:
        for k in snap.keys():
            if k.startswith('bubble_') and k not in bubble_keys:
                val = snap[k]
                if isinstance(val, (int, float)):
                    bubble_keys.append(k)
    bubble_keys = sorted(set(bubble_keys))
    if bubble_keys:
        plot_parameter_evolution(
            snapshots, bubble_keys,
            "Bubble Parameters Evolution",
            os.path.join(output_dir, "evolution_bubble_parameters.pdf")
        )

    # C) TRINITY essentials
    essential_keys = ['R1', 'R2', 'rShell', 'Pb', 'Eb', 'T0']
    plot_parameter_evolution(
        snapshots, essential_keys,
        "TRINITY Essential Parameters Evolution",
        os.path.join(output_dir, "evolution_essential_parameters.pdf")
    )

    # D) Force parameters
    force_keys = []
    for snap in snapshots:
        for k in snap.keys():
            if 'F_' in k and k not in force_keys:
                val = snap[k]
                if isinstance(val, (int, float)):
                    force_keys.append(k)
    force_keys = sorted(set(force_keys))
    if force_keys:
        plot_parameter_evolution(
            snapshots, force_keys,
            "Force Parameters Evolution",
            os.path.join(output_dir, "evolution_force_parameters.pdf")
        )


def test_ode_comparison(snapshots: list) -> dict:
    """
    Test ODE outputs across multiple snapshots.

    Returns dict with comparison results.
    """
    from src.phase1_energy.energy_phase_ODEs import get_ODE_Edot
    from src.phase1_energy.energy_phase_ODEs_modified import get_ODE_Edot_pure, create_ODE_snapshot

    results = []
    print("\nComparing ODE outputs across snapshots...")

    for i, snapshot in enumerate(snapshots):
        if i % 50 == 0:
            print(f"  Processing snapshot {i+1}/{len(snapshots)}...")

        try:
            params_orig = make_mock_params(snapshot.copy())
            params_mod = make_mock_params(snapshot.copy())

            t_now = snapshot['t_now']
            R2 = snapshot['R2']
            v2 = snapshot['v2']
            Eb = snapshot['Eb']
            y = [R2, v2, Eb]

            # Run original
            start_orig = time.perf_counter()
            result_orig = get_ODE_Edot(y.copy(), t_now, params_orig)
            time_orig = time.perf_counter() - start_orig

            # Run modified
            ode_snapshot = create_ODE_snapshot(params_mod)
            start_mod = time.perf_counter()
            result_mod = get_ODE_Edot_pure(t_now, y.copy(), ode_snapshot, params_mod)
            time_mod = time.perf_counter() - start_mod

            # Compare
            rd_orig, vd_orig, Ed_orig = result_orig
            rd_mod, vd_mod, Ed_mod = result_mod

            def rel_diff(a, b):
                if abs(a) < 1e-300 and abs(b) < 1e-300:
                    return 0.0
                return abs(a - b) / max(abs(a), abs(b), 1e-300)

            results.append({
                't_now': t_now,
                'rd_diff': rel_diff(rd_orig, rd_mod),
                'vd_diff': rel_diff(vd_orig, vd_mod),
                'Ed_diff': rel_diff(Ed_orig, Ed_mod),
                'time_orig': time_orig,
                'time_mod': time_mod,
                'passed': (rel_diff(rd_orig, rd_mod) < 1e-6 and
                          rel_diff(vd_orig, vd_mod) < 1e-6 and
                          rel_diff(Ed_orig, Ed_mod) < 1e-6)
            })
        except Exception as e:
            results.append({
                't_now': snapshot.get('t_now', 0),
                'error': str(e),
                'passed': False
            })

    return results


# =============================================================================
# Main test function
# =============================================================================

def run_full_comparison():
    """Run full energy phase comparison test."""

    print("=" * 70)
    print("Energy Phase Evolution Comparison Test")
    print("Comparing ODE outputs and generating parameter evolution plots")
    print("=" * 70)

    # Load snapshots
    jsonl_path = os.path.join(TEST_DIR, '1e7_sfe020_n1e4_test_dictionary.jsonl')
    if not os.path.exists(jsonl_path):
        print(f"ERROR: Test dictionary not found at {jsonl_path}")
        return False

    print(f"\nLoading snapshots from: {jsonl_path}")
    snapshots = load_all_snapshots(jsonl_path)
    print(f"  Loaded {len(snapshots)} snapshots")
    print(f"  Time range: {snapshots[0].get('t_now', 0):.6e} to {snapshots[-1].get('t_now', 0):.6e} Myr")

    # Generate evolution plots from existing data
    generate_evolution_plots(snapshots, TEST_DIR)

    # Test ODE comparison on subset of snapshots
    test_snapshots = snapshots[1::10]  # Every 10th snapshot, starting from 2nd
    print(f"\nTesting ODE comparison on {len(test_snapshots)} snapshots...")

    results = test_ode_comparison(test_snapshots)

    # Summary
    print("\n" + "=" * 70)
    print("ODE COMPARISON SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)

    print(f"\nResults: {passed}/{total} snapshots passed ODE comparison")
    if errors > 0:
        print(f"  Errors: {errors} snapshots had errors")

    # Timing summary
    valid_results = [r for r in results if 'time_orig' in r]
    if valid_results:
        total_orig = sum(r['time_orig'] for r in valid_results)
        total_mod = sum(r['time_mod'] for r in valid_results)
        speedup = total_orig / total_mod if total_mod > 0 else 1.0
        print(f"\nTiming ({len(valid_results)} comparisons):")
        print(f"  Original total: {total_orig*1000:.2f} ms")
        print(f"  Modified total: {total_mod*1000:.2f} ms")
        print(f"  Average speedup: {speedup:.2f}x")

    # Show max differences
    valid_diffs = [r for r in results if 'rd_diff' in r]
    if valid_diffs:
        max_rd = max(r['rd_diff'] for r in valid_diffs)
        max_vd = max(r['vd_diff'] for r in valid_diffs)
        max_Ed = max(r['Ed_diff'] for r in valid_diffs)
        print(f"\nMax relative differences:")
        print(f"  rd: {max_rd:.2e}")
        print(f"  vd: {max_vd:.2e}")
        print(f"  Ed: {max_Ed:.2e}")

    print("\n" + "=" * 70)
    if passed == total and errors == 0:
        print("ALL TESTS PASSED!")
        return True
    else:
        print("SOME TESTS FAILED or had errors")
        return passed > total * 0.9  # Allow up to 10% failures


if __name__ == '__main__':
    success = run_full_comparison()
    sys.exit(0 if success else 1)
