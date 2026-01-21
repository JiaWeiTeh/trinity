#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRINITY Output Reader
=====================

A helper class for reading and processing TRINITY simulation output files (.jsonl).
Similar to astropy.io.fits, provides easy access to simulation data with a clean,
Pythonic API.

This module is the recommended way to access TRINITY output data, replacing direct
JSON parsing in plotting and analysis scripts.

Installation
------------
This module is part of the TRINITY package. Import it as:

    from src._output.trinity_reader import TrinityOutput, find_data_file, find_data_path

Basic Usage
-----------
    from src._output.trinity_reader import TrinityOutput

    # Open a TRINITY output file
    output = TrinityOutput.open('path/to/output.jsonl')

    # Get information about the output
    output.info()              # Summary
    output.info(verbose=True)  # Detailed parameter documentation

    # Access time series data as numpy arrays
    times = output.get('t_now')
    radii = output.get('R2')
    velocity = output.get('v2')

    # For non-numeric data, disable array conversion
    phases = output.get('current_phase', as_array=False)

    # Filter by phase or time range
    implicit_data = output.filter(phase='implicit')
    early_data = output.filter(t_max=1.0)

    # Get a specific snapshot by index
    snapshot = output[100]
    print(snapshot['R2'], snapshot['v2'])

    # Get snapshot closest to a specific time
    snap_at_1myr = output.get_at_time(1.0)

    # Get snapshot at a specific time (interpolated by default)
    snap = output.get_at_time(0.5)  # Returns interpolated snapshot
    snap = output.get_at_time(0.5, mode='closest')  # Returns closest actual snapshot

    # Iterate over snapshots
    for snap in output:
        print(snap.t_now, snap['R2'])

    # Convert to pandas DataFrame (scalar values only)
    df = output.to_dataframe()

Key Parameters
--------------
The most commonly used output parameters are:

**Dynamical Variables:**
- t_now: Current simulation time [Myr]
- R2: Outer bubble/shell radius [pc]
- v2: Shell expansion velocity [pc/Myr]
- Eb: Bubble thermal energy [erg]
- T0: Characteristic bubble temperature [K]
- R1: Inner bubble radius (wind termination shock) [pc]
- Pb: Bubble pressure [dyn/cm^2]

**Cooling Parameters (from beta-delta solver):**
- cool_beta: Pressure evolution parameter β = -(t/Pb)(dPb/dt)
- cool_delta: Temperature evolution parameter δ

**Forces:**
- F_grav: Gravitational force
- F_ram: Ram pressure force (total)
- F_ion_out: Ionization force (outward)
- F_rad: Radiation pressure force

**Residual Diagnostics (beta-delta solver):**
- residual_Edot1_guess: Edot from beta [erg/Myr]
- residual_Edot2_guess: Edot from energy balance [erg/Myr]
- residual_T1_guess: Bubble temperature T_bubble [K]
- residual_T2_guess: Target temperature T0 [K]

Use output.info(verbose=True) for a complete list of available parameters
with documentation.

Snapshot Consistency
--------------------
As of January 2026, TRINITY snapshots are saved BEFORE ODE integration,
ensuring all values in a snapshot correspond to the same timestamp (t_now).
This includes: t_now, R2, v2, Eb, T0, feedback properties, shell properties,
bubble properties, forces, and beta-delta residuals.

See Also
--------
- example_scripts/example_reader_overview.py: Comprehensive usage examples
- example_scripts/example_plot_radius_vs_time.py: Plotting examples

@author: TRINITY Team
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Literal
from dataclasses import dataclass, field
import pandas as pd
from scipy import interpolate as scipy_interp


# =============================================================================
# Parameter Documentation
# =============================================================================

PARAM_DOCS = {
    # Model info
    'model_name': 'Model identifier/name',
    'mCloud': 'Initial cloud mass [Msun]',
    'rCloud': 'Initial cloud radius [pc]',

    # Simulation state
    'current_phase': 'Current simulation phase (energy/implicit)',
    't_now': 'Current simulation time [Myr]',
    't_next': 'Next timestep target [Myr]',
    'tSF': 'Star formation timescale [Myr]',

    # Termination flags
    'EndSimulationDirectly': 'Flag to end simulation immediately',
    'SimulationEndReason': 'Reason for simulation termination',
    'EarlyPhaseApproximation': 'Using early phase approximation',
    'isCollapse': 'Shell has collapsed',
    'isDissolved': 'Cloud has dissolved',

    # Main dynamical variables
    'R2': 'Outer bubble/shell radius [pc]',
    'v2': 'Shell expansion velocity [pc/Myr]',
    'Eb': 'Bubble thermal energy [erg]',
    'T0': 'Characteristic bubble temperature [K]',
    'R1': 'Inner bubble radius (wind termination shock) [pc]',
    'Pb': 'Bubble pressure [dyn/cm^2]',
    'c_sound': 'Sound speed in bubble [pc/Myr]',

    # Cooling parameters
    'cool_alpha': 'Cooling power-law index α',
    'cool_beta': 'Pressure evolution parameter β = -(t/Pb)(dPb/dt)',
    'cool_delta': 'Temperature evolution parameter δ',

    # Feedback luminosities
    'Lmech_W': 'Mechanical luminosity from winds [erg/Myr]',
    'Lmech_SN': 'Mechanical luminosity from supernovae [erg/Myr]',
    'Lmech_total': 'Total mechanical luminosity [erg/Myr]',
    'Lbol': 'Bolometric luminosity [erg/s]',
    'Li': 'Ionizing luminosity [erg/s]',
    'Ln': 'Non-ionizing luminosity [erg/s]',
    'Qi': 'Ionizing photon rate [photons/s]',

    # Momentum injection
    'pdot_W': 'Momentum injection rate from winds',
    'pdot_SN': 'Momentum injection rate from supernovae',
    'pdot_total': 'Total momentum injection rate',
    'pdotdot_total': 'Second derivative of momentum injection',
    'v_mech_total': 'Mechanical velocity [km/s]',

    # Forces
    'F_grav': 'Gravitational force',
    'F_ram': 'Ram pressure force',
    'F_ion_in': 'Ionization force (inward)',
    'F_ion_out': 'Ionization force (outward)',
    'F_rad': 'Radiation pressure force',
    'F_ISM': 'ISM pressure force',
    'F_SN': 'Supernova force',
    'F_wind': 'Wind force',

    # Shell properties
    'shell_mass': 'Shell mass [Msun]',
    'shell_massDot': 'Shell mass accretion rate [Msun/Myr]',
    'shell_thickness': 'Shell thickness [pc]',
    'shell_n0': 'Shell number density at inner edge [cm^-3]',
    'shell_nMax': 'Maximum shell number density [cm^-3]',
    'nEdge': 'Number density at shell edge [cm^-3]',
    'rShell': 'Shell radius [pc]',

    # Shell absorption
    'shell_fAbsorbedIon': 'Fraction of ionizing radiation absorbed',
    'shell_fAbsorbedNeu': 'Fraction of non-ionizing radiation absorbed',
    'shell_fAbsorbedWeightedTotal': 'Weighted total absorbed fraction',
    'shell_fIonisedDust': 'Ionized dust fraction',
    'shell_F_rad': 'Shell radiative force',

    # Bubble luminosities
    'bubble_LTotal': 'Total bubble cooling luminosity [erg/Myr]',
    'bubble_L1Bubble': 'Bubble cooling component 1',
    'bubble_L2Conduction': 'Conductive cooling component',
    'bubble_L3Intermediate': 'Intermediate cooling component',
    'bubble_Leak': 'Energy leak from bubble [erg/Myr]',
    'bubble_Lgain': 'Energy gain in bubble [erg/Myr]',
    'bubble_Lloss': 'Energy loss from bubble [erg/Myr]',

    # Bubble structure
    'bubble_mass': 'Bubble mass [Msun]',
    'bubble_Tavg': 'Average bubble temperature [K]',
    'bubble_T_r_Tb': 'Bubble temperature at measurement radius [K]',
    'bubble_r_Tb': 'Radius for temperature measurement [pc]',
    'bubble_dMdt': 'Bubble mass flow rate [Msun/Myr]',
    'bubble_dMdtGuess': 'Initial guess for bubble mass flow rate',

    # Bubble arrays (radial profiles)
    'bubble_v_arr': 'Bubble velocity profile array',
    'bubble_T_arr_r_arr': 'Bubble temperature profile (T, r)',
    'bubble_n_arr_r_arr': 'Bubble density profile (n, r)',
    'bubble_dTdr_arr_r_arr': 'Bubble temperature gradient profile',
    'bubble_v_arr_r_arr': 'Bubble velocity profile (v, r)',
    'log_bubble_T_arr': 'Log bubble temperature array',
    'log_bubble_n_arr': 'Log bubble density array',
    'log_bubble_dTdr_arr': 'Log bubble temperature gradient array',

    # Residuals (beta-delta solver diagnostics)
    'residual_deltaT': 'Temperature residual (normalized)',
    'residual_betaEdot': 'Energy derivative residual (normalized)',
    'residual_Edot1_guess': 'Edot from beta [erg/Myr]',
    'residual_Edot2_guess': 'Edot from energy balance [erg/Myr]',
    'residual_T1_guess': 'Bubble temperature T_bubble [K]',
    'residual_T2_guess': 'Target temperature T0 [K]',

    # Gravitational potential arrays
    'shell_grav_r': 'Radii for gravitational potential [pc]',
    'shell_grav_phi': 'Gravitational potential values',
    'shell_grav_force_m': 'Gravitational force per unit mass',

    # Cooling update
    't_previousCoolingUpdate': 'Time of previous cooling structure update [Myr]',
    'shell_interpolate_massDot': 'Using interpolated mass dot',
    'shell_tauKappaRatio': 'Optical depth / opacity ratio',
}


# =============================================================================
# Snapshot Class
# =============================================================================

@dataclass
class Snapshot:
    """A single simulation snapshot."""
    data: Dict[str, Any]
    index: int
    is_interpolated: bool = False
    interpolation_time: Optional[float] = None

    def __getitem__(self, key: str) -> Any:
        """Access snapshot data by key."""
        return self.data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get snapshot data with default."""
        return self.data.get(key, default)

    def keys(self) -> List[str]:
        """Get all available keys."""
        return list(self.data.keys())

    @property
    def t_now(self) -> float:
        """Current time."""
        return self.data.get('t_now', 0.0)

    @property
    def phase(self) -> str:
        """Current phase."""
        return self.data.get('current_phase', 'unknown')

    def __repr__(self) -> str:
        if self.is_interpolated:
            return f"Snapshot(INTERPOLATED, t={self.t_now:.4e}, phase='{self.phase}')"
        return f"Snapshot(index={self.index}, t={self.t_now:.4e}, phase='{self.phase}')"


# =============================================================================
# Main Reader Class
# =============================================================================

class TrinityOutput:
    """
    Reader for TRINITY simulation output files (.jsonl).

    Examples
    --------
    >>> output = TrinityOutput.open('simulation.jsonl')
    >>> output.info()
    >>> times = output.get('t_now')
    >>> radii = output.get('R2')
    >>> implicit_data = output.filter(phase='implicit')
    """

    def __init__(self, filepath: Union[str, Path], snapshots: List[Dict[str, Any]]):
        """
        Initialize TrinityOutput.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSONL file
        snapshots : list
            List of snapshot dictionaries
        """
        self.filepath = Path(filepath)
        self._snapshots = snapshots
        self._keys = set()
        for snap in snapshots:
            self._keys.update(snap.keys())
        self._keys = sorted(self._keys)

    @classmethod
    def open(cls, filepath: Union[str, Path]) -> 'TrinityOutput':
        """
        Open a TRINITY output file.

        Parameters
        ----------
        filepath : str or Path
            Path to the .jsonl output file

        Returns
        -------
        TrinityOutput
            Reader object for the output file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        snapshots = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    snapshots.append(json.loads(line))

        return cls(filepath, snapshots)

    def __len__(self) -> int:
        """Number of snapshots."""
        return len(self._snapshots)

    def __getitem__(self, index: int) -> Snapshot:
        """Get a specific snapshot by index."""
        if isinstance(index, slice):
            return [Snapshot(self._snapshots[i], i) for i in range(*index.indices(len(self)))]
        return Snapshot(self._snapshots[index], index)

    def __iter__(self) -> Iterator[Snapshot]:
        """Iterate over snapshots."""
        for i, snap in enumerate(self._snapshots):
            yield Snapshot(snap, i)

    @property
    def model_name(self) -> str:
        """Model name from first snapshot."""
        return self._snapshots[0].get('model_name', 'unknown') if self._snapshots else 'unknown'

    @property
    def keys(self) -> List[str]:
        """All available parameter keys."""
        return self._keys

    @property
    def phases(self) -> List[str]:
        """List of unique phases in the output."""
        return list(set(s.get('current_phase', 'unknown') for s in self._snapshots))

    @property
    def t_min(self) -> float:
        """Minimum time in output."""
        return min(s.get('t_now', 0) for s in self._snapshots)

    @property
    def t_max(self) -> float:
        """Maximum time in output."""
        return max(s.get('t_now', 0) for s in self._snapshots)

    def get(self, key: str, as_array: bool = True) -> Union[np.ndarray, List[Any]]:
        """
        Get a parameter across all snapshots.

        Parameters
        ----------
        key : str
            Parameter name
        as_array : bool
            If True, return numpy array (for numeric data)

        Returns
        -------
        array or list
            Values across all snapshots
        """
        values = [s.get(key) for s in self._snapshots]
        if as_array:
            try:
                return np.array(values)
            except (ValueError, TypeError):
                return values
        return values

    def get_at_time(self, t: float, key: Optional[str] = None,
                    mode: Literal['interpolate', 'closest'] = 'interpolate',
                    n_neighbors: int = 5, quiet: bool = False) -> Union[Snapshot, Any]:
        """
        Get snapshot at a specific time.

        Parameters
        ----------
        t : float
            Target time [Myr]
        key : str, optional
            If provided, return just this parameter value
        mode : str
            - 'interpolate' (default): Interpolate values from neighboring snapshots.
              Returns an interpolated snapshot with a warning message.
            - 'closest': Return the actual snapshot closest to requested time.
        n_neighbors : int
            Number of neighbors to use for interpolation (default 5, uses 2-3 on each side)
        quiet : bool
            If True, suppress the interpolation warning message

        Returns
        -------
        Snapshot or value
            Snapshot at time t (interpolated or closest), or specific value if key provided
        """
        times = self.get('t_now')

        # Check if exact time exists
        exact_idx = np.where(np.isclose(times, t, rtol=1e-10))[0]
        if len(exact_idx) > 0:
            snap = self[exact_idx[0]]
            if key is not None:
                return snap[key]
            return snap

        # Time not exact - use requested mode
        if mode == 'closest':
            idx = np.argmin(np.abs(times - t))
            snap = self[idx]
            if not quiet:
                print(f"[TrinityOutput] Time t={t:.6e} Myr not found in snapshots. "
                      f"Returning closest snapshot at t={times[idx]:.6e} Myr.")
            if key is not None:
                return snap[key]
            return snap
        else:
            # Interpolate
            snap = self._interpolate_snapshot(t, n_neighbors, quiet)
            if key is not None:
                return snap[key]
            return snap

    def _interpolate_snapshot(self, t: float, n_neighbors: int = 5, quiet: bool = False) -> Snapshot:
        """
        Create an interpolated snapshot at time t using neighboring snapshots.

        Parameters
        ----------
        t : float
            Target time [Myr]
        n_neighbors : int
            Total number of neighbors to use (split between before/after)
        quiet : bool
            If True, suppress warning message

        Returns
        -------
        Snapshot
            Interpolated snapshot with is_interpolated=True
        """
        times = np.array(self.get('t_now'))

        # Check bounds
        if t < times.min() or t > times.max():
            raise ValueError(
                f"Requested time t={t:.6e} is outside data range "
                f"[{times.min():.6e}, {times.max():.6e}] Myr"
            )

        # Find indices of neighbors
        sorted_indices = np.argsort(times)
        sorted_times = times[sorted_indices]

        # Find insertion point
        insert_idx = np.searchsorted(sorted_times, t)

        # Get neighbors on each side (n_neighbors total, roughly split)
        n_before = n_neighbors // 2
        n_after = n_neighbors - n_before

        start_idx = max(0, insert_idx - n_before)
        end_idx = min(len(sorted_times), insert_idx + n_after)

        # Adjust if we hit boundaries
        if start_idx == 0:
            end_idx = min(len(sorted_times), n_neighbors)
        if end_idx == len(sorted_times):
            start_idx = max(0, len(sorted_times) - n_neighbors)

        neighbor_indices = sorted_indices[start_idx:end_idx]
        neighbor_times = times[neighbor_indices]

        if not quiet:
            print(f"[TrinityOutput] Time t={t:.6e} Myr not found in snapshots. "
                  f"Interpolating from {len(neighbor_indices)} neighbors "
                  f"(t=[{neighbor_times.min():.6e}, {neighbor_times.max():.6e}] Myr). "
                  f"NOTE: These are interpolated values, not actual simulation output. "
                  f"Use mode='closest' for the actual snapshot closest to the requested time.")

        # Build interpolated data dictionary
        interpolated_data = {}

        # Get all keys from first snapshot
        all_keys = self._snapshots[neighbor_indices[0]].keys()

        for key in all_keys:
            try:
                # Get values from neighbor snapshots
                values = [self._snapshots[idx].get(key) for idx in neighbor_indices]

                # Determine if this is interpolatable
                first_val = values[0]

                if first_val is None:
                    interpolated_data[key] = None
                    continue

                # Handle strings/phases - use closest
                if isinstance(first_val, str):
                    closest_idx = neighbor_indices[np.argmin(np.abs(neighbor_times - t))]
                    interpolated_data[key] = self._snapshots[closest_idx].get(key)
                    continue

                # Handle booleans - use closest
                if isinstance(first_val, bool):
                    closest_idx = neighbor_indices[np.argmin(np.abs(neighbor_times - t))]
                    interpolated_data[key] = self._snapshots[closest_idx].get(key)
                    continue

                # Handle numeric scalars
                if isinstance(first_val, (int, float)):
                    y_vals = np.array(values, dtype=float)

                    # Handle NaN values
                    valid_mask = np.isfinite(y_vals)
                    if not np.any(valid_mask):
                        interpolated_data[key] = np.nan
                        continue

                    if np.sum(valid_mask) < 2:
                        # Not enough points to interpolate
                        interpolated_data[key] = y_vals[valid_mask][0] if np.any(valid_mask) else np.nan
                        continue

                    # Use linear interpolation
                    valid_times = neighbor_times[valid_mask]
                    valid_vals = y_vals[valid_mask]

                    interp_func = scipy_interp.interp1d(
                        valid_times, valid_vals,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    interpolated_data[key] = float(interp_func(t))
                    continue

                # Handle arrays/lists
                if isinstance(first_val, (list, np.ndarray)):
                    # For arrays, interpolate element-wise if same length
                    arr_lengths = [len(v) if v is not None else 0 for v in values]

                    if len(set(arr_lengths)) == 1 and arr_lengths[0] > 0:
                        # All same length - interpolate each element
                        arr_len = arr_lengths[0]
                        result = []

                        for elem_idx in range(arr_len):
                            elem_values = np.array([v[elem_idx] for v in values], dtype=float)
                            valid_mask = np.isfinite(elem_values)

                            if np.sum(valid_mask) < 2:
                                result.append(elem_values[0] if len(elem_values) > 0 else np.nan)
                            else:
                                valid_times = neighbor_times[valid_mask]
                                valid_vals = elem_values[valid_mask]
                                interp_func = scipy_interp.interp1d(
                                    valid_times, valid_vals,
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value='extrapolate'
                                )
                                result.append(float(interp_func(t)))

                        interpolated_data[key] = result
                    else:
                        # Different lengths or empty - use closest
                        closest_idx = neighbor_indices[np.argmin(np.abs(neighbor_times - t))]
                        interpolated_data[key] = self._snapshots[closest_idx].get(key)
                    continue

                # Default: use closest value
                closest_idx = neighbor_indices[np.argmin(np.abs(neighbor_times - t))]
                interpolated_data[key] = self._snapshots[closest_idx].get(key)

            except Exception:
                # If interpolation fails for any reason, use closest value
                closest_idx = neighbor_indices[np.argmin(np.abs(neighbor_times - t))]
                interpolated_data[key] = self._snapshots[closest_idx].get(key)

        # Set the interpolated time
        interpolated_data['t_now'] = t

        return Snapshot(
            data=interpolated_data,
            index=-1,  # -1 indicates interpolated
            is_interpolated=True,
            interpolation_time=t
        )

    def filter(self, phase: Optional[str] = None,
               t_min: Optional[float] = None,
               t_max: Optional[float] = None) -> 'TrinityOutput':
        """
        Filter snapshots by criteria.

        Parameters
        ----------
        phase : str, optional
            Filter by simulation phase
        t_min : float, optional
            Minimum time
        t_max : float, optional
            Maximum time

        Returns
        -------
        TrinityOutput
            New TrinityOutput with filtered snapshots
        """
        filtered = []
        for snap in self._snapshots:
            t = snap.get('t_now', 0)
            p = snap.get('current_phase', '')

            if phase is not None and p != phase:
                continue
            if t_min is not None and t < t_min:
                continue
            if t_max is not None and t > t_max:
                continue

            filtered.append(snap)

        return TrinityOutput(self.filepath, filtered)

    def info(self, verbose: bool = False) -> None:
        """
        Print information about the output file.

        Parameters
        ----------
        verbose : bool
            If True, show all parameters with documentation
        """
        print("=" * 70)
        print(f"TRINITY Output: {self.filepath.name}")
        print("=" * 70)
        print()
        print(f"  Model name:    {self.model_name}")
        print(f"  Snapshots:     {len(self)}")
        print(f"  Time range:    [{self.t_min:.4e}, {self.t_max:.4e}] Myr")
        print(f"  Parameters:    {len(self.keys)}")
        print()

        # Phase breakdown
        print("  Phases:")
        for phase in sorted(self.phases):
            phase_snaps = [s for s in self._snapshots if s.get('current_phase') == phase]
            t_vals = [s.get('t_now', 0) for s in phase_snaps]
            print(f"    {phase:12s}: {len(phase_snaps):4d} snapshots, "
                  f"t=[{min(t_vals):.4e}, {max(t_vals):.4e}]")
        print()

        if verbose:
            self._print_parameters()

    def _print_parameters(self) -> None:
        """Print all parameters with documentation."""
        print("  Parameters:")
        print("  " + "-" * 66)

        # Group parameters
        groups = {
            'Model': ['model_name', 'mCloud', 'rCloud'],
            'Time': ['t_now', 't_next', 'tSF', 't_previousCoolingUpdate'],
            'State': ['current_phase', 'EndSimulationDirectly', 'SimulationEndReason',
                     'EarlyPhaseApproximation', 'isCollapse', 'isDissolved'],
            'Dynamics': ['R2', 'v2', 'Eb', 'T0', 'R1', 'Pb', 'c_sound'],
            'Cooling': ['cool_alpha', 'cool_beta', 'cool_delta'],
            'Feedback': ['Lmech_W', 'Lmech_SN', 'Lmech_total', 'Lbol', 'Li', 'Ln', 'Qi',
                        'pdot_W', 'pdot_SN', 'pdot_total', 'pdotdot_total', 'v_mech_total'],
            'Forces': ['F_grav', 'F_ram', 'F_ion_in', 'F_ion_out', 'F_rad', 'F_ISM'],
            'Shell': ['shell_mass', 'shell_massDot', 'shell_thickness', 'nEdge', 'rShell',
                     'shell_n0', 'shell_nMax', 'shell_fAbsorbedIon', 'shell_fAbsorbedNeu'],
            'Bubble': ['bubble_mass', 'bubble_Tavg', 'bubble_T_r_Tb', 'bubble_LTotal',
                      'bubble_Leak', 'bubble_Lgain', 'bubble_Lloss', 'bubble_dMdt'],
            'Residuals': ['residual_deltaT', 'residual_betaEdot',
                         'residual_Edot1_guess', 'residual_Edot2_guess',
                         'residual_T1_guess', 'residual_T2_guess'],
        }

        documented_keys = set()
        for group_name, keys in groups.items():
            group_keys = [k for k in keys if k in self._keys]
            if not group_keys:
                continue

            print(f"\n  [{group_name}]")
            for key in group_keys:
                doc = PARAM_DOCS.get(key, '')
                sample = self._snapshots[0].get(key, None)
                stype = type(sample).__name__ if sample is not None else '?'
                if isinstance(sample, list):
                    stype = f'array[{len(sample)}]'
                elif isinstance(sample, float) and not np.isnan(sample):
                    stype = f'float'
                print(f"    {key:30s} ({stype:10s}) : {doc}")
                documented_keys.add(key)

        # Show undocumented keys
        other_keys = [k for k in self._keys if k not in documented_keys]
        if other_keys:
            print(f"\n  [Other]")
            for key in other_keys:
                sample = self._snapshots[0].get(key, None)
                stype = type(sample).__name__ if sample is not None else '?'
                if isinstance(sample, list):
                    stype = f'array[{len(sample)}]'
                doc = PARAM_DOCS.get(key, '')
                print(f"    {key:30s} ({stype:10s}) : {doc}")

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert to pandas DataFrame (scalar values only).

        Returns
        -------
        DataFrame
            Pandas DataFrame with one row per snapshot
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        # Only include scalar values
        scalar_keys = []
        for key in self._keys:
            sample = self._snapshots[0].get(key)
            if isinstance(sample, (int, float, str, bool, type(None))):
                scalar_keys.append(key)

        data = {key: self.get(key, as_array=False) for key in scalar_keys}
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return (f"TrinityOutput('{self.filepath.name}', "
                f"snapshots={len(self)}, t=[{self.t_min:.4e}, {self.t_max:.4e}])")


# =============================================================================
# Convenience function
# =============================================================================

def read(filepath: Union[str, Path]) -> TrinityOutput:
    """
    Open a TRINITY output file (convenience function).

    Parameters
    ----------
    filepath : str or Path
        Path to the .jsonl output file

    Returns
    -------
    TrinityOutput
        Reader object for the output file

    Examples
    --------
    >>> import src._output.trinity_reader as trinity
    >>> output = trinity.read('simulation.jsonl')
    >>> output.info()

    # Or use the class method directly:
    >>> output = TrinityOutput.open('simulation.jsonl')
    """
    return TrinityOutput.open(filepath)


# Alias for backwards compatibility
load_output = read


# =============================================================================
# Path Finding Utilities
# =============================================================================

def find_data_file(base_dir: Union[str, Path], run_name: str) -> Optional[Path]:
    """
    Find the data file for a run, preferring JSONL over JSON.

    Searches for:
    1. {run_name}_dictionary.jsonl
    2. dictionary.jsonl
    3. dictionary.json

    Parameters
    ----------
    base_dir : Path or str
        Base directory containing run folders
    run_name : str
        Name of the run (e.g., "1e7_sfe020_n1e4")

    Returns
    -------
    Optional[Path]
        Path to data file, or None if not found
    """
    base_dir = Path(base_dir)
    run_dir = base_dir / run_name

    # Check various file locations/names
    candidates = [
        run_dir / f"{run_name}_dictionary.jsonl",
        run_dir / "dictionary.jsonl",
        run_dir / "dictionary.json",
        # Also check if the file is directly in base_dir with run_name prefix
        base_dir / f"{run_name}_dictionary.jsonl",
        base_dir / f"{run_name}_dictionary.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_data_path(base_path: Union[str, Path]) -> Path:
    """
    Find the data file, preferring JSONL over JSON.

    Given a base path (with or without extension), searches for:
    1. {base_path}.jsonl (if base_path has no extension)
    2. {base_path}.json (if base_path has no extension)
    3. {base_path} as-is (if it has an extension and exists)
    4. {base_path with .json replaced by .jsonl} (if base_path ends with .json)

    Parameters
    ----------
    base_path : str or Path
        Base path to the data file. Can be:
        - Path without extension: will try .jsonl then .json
        - Path with .json: will try .jsonl first, then .json
        - Path with .jsonl: will use as-is if exists

    Returns
    -------
    Path
        Path to the found data file

    Raises
    ------
    FileNotFoundError
        If no data file is found
    """
    base_path = Path(base_path)

    # If the path exists as-is, check if we should prefer .jsonl
    if base_path.suffix == '.json':
        # Try .jsonl first
        jsonl_path = base_path.with_suffix('.jsonl')
        if jsonl_path.exists():
            return jsonl_path
        if base_path.exists():
            return base_path
    elif base_path.suffix == '.jsonl':
        if base_path.exists():
            return base_path
        # Fall back to .json
        json_path = base_path.with_suffix('.json')
        if json_path.exists():
            return json_path
    else:
        # No extension - try adding .jsonl then .json
        jsonl_path = Path(str(base_path) + '.jsonl')
        if jsonl_path.exists():
            return jsonl_path
        json_path = Path(str(base_path) + '.json')
        if json_path.exists():
            return json_path
        # Also try as directory with dictionary files
        if base_path.is_dir():
            for suffix in ['.jsonl', '.json']:
                dict_path = base_path / f'dictionary{suffix}'
                if dict_path.exists():
                    return dict_path

    raise FileNotFoundError(
        f"No data file found for: {base_path}\n"
        f"Tried: .jsonl and .json variants"
    )


def resolve_data_input(data_input: Union[str, Path],
                       output_dir: Union[str, Path] = None) -> Path:
    """
    Resolve various data input formats to a data file path.

    Accepts:
    1. Output folder name (e.g., "1e7_sfe020_n1e4") - searches in output_dir
    2. Folder path (e.g., "/path/to/outputs/1e7_sfe020_n1e4") - looks for dictionary inside
    3. File path (e.g., "/path/to/dictionary.jsonl") - uses directly

    Parameters
    ----------
    data_input : str or Path
        The input to resolve. Can be a folder name, folder path, or file path.
    output_dir : str or Path, optional
        Base directory for output folders. Defaults to 'outputs' or TRINITY_OUTPUT_DIR env var.

    Returns
    -------
    Path
        Resolved path to the data file

    Raises
    ------
    FileNotFoundError
        If no data file can be found
    """
    import os

    data_input = Path(data_input)

    # Default output directory
    if output_dir is None:
        output_dir = Path(os.environ.get('TRINITY_OUTPUT_DIR', 'outputs'))
    else:
        output_dir = Path(output_dir)

    # Case 1: It's a file that exists
    if data_input.is_file():
        return data_input

    # Case 2: It's a directory - look for dictionary files inside
    if data_input.is_dir():
        for suffix in ['.jsonl', '.json']:
            dict_path = data_input / f'dictionary{suffix}'
            if dict_path.exists():
                return dict_path
        raise FileNotFoundError(
            f"No dictionary.jsonl or dictionary.json found in: {data_input}"
        )

    # Case 3: Check if it's a path with extension that doesn't exist yet
    if data_input.suffix in ['.json', '.jsonl']:
        # Try find_data_path which handles .jsonl/.json priority
        try:
            return find_data_path(data_input)
        except FileNotFoundError:
            pass

    # Case 4: It might be a folder name - check in output_dir
    folder_path = output_dir / data_input
    if folder_path.is_dir():
        for suffix in ['.jsonl', '.json']:
            dict_path = folder_path / f'dictionary{suffix}'
            if dict_path.exists():
                return dict_path

    # Case 5: Try as a base path (no extension) with find_data_path
    try:
        return find_data_path(data_input)
    except FileNotFoundError:
        pass

    # Case 6: Try in output_dir as base path
    try:
        return find_data_path(output_dir / data_input / 'dictionary')
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        f"Could not resolve data input: {data_input}\n"
        f"Tried as: file, directory, folder name in {output_dir}"
    )
