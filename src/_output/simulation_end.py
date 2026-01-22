#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation End Reason Logger

Writes simulation termination details to simulationEnd.txt in output directory.
Provides structured exit codes for batch processing and post-run analysis.

Also provides write_termination_debug_report() to dump the last two snapshots
with comparison tables for debugging termination issues.

Author: Claude Code
Date: 2026-01-15
"""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Import unit conversions for display
from src._functions.unit_conversions import INV_CONV


class SimulationEndCode(Enum):
    """
    Enumeration of simulation end reasons with exit codes.

    Exit code ranges:
    - 0-9: Success states (simulation completed normally)
    - 10-19: Parameter/configuration errors
    - 20-29: Numerical/runtime errors
    - 99: Unknown/unhandled termination
    """
    # Success states (0-9)
    SUCCESS_DISSOLVED = (0, "Shell dissolved into ISM")
    SUCCESS_MAX_TIME = (1, "Maximum simulation time reached")
    SUCCESS_MAX_RADIUS = (2, "Maximum radius reached (shell exceeded rCloud)")
    SUCCESS_COMPLETE = (3, "Simulation completed successfully")

    # Parameter errors (10-19)
    ERROR_INVALID_PARAMS = (10, "Invalid cloud parameters")
    ERROR_MASS_INCONSISTENCY = (11, "Mass inconsistency > 0.1%")
    ERROR_EDGE_DENSITY = (12, "Edge density below ISM")
    ERROR_RADIUS_TOO_LARGE = (13, "Cloud radius exceeds physical limit")

    # Numerical/runtime errors (20-29)
    ERROR_NUMERICAL = (20, "Numerical instability")
    ERROR_VELOCITY = (21, "Velocity below threshold")
    ERROR_SOLVER = (22, "ODE solver failed")
    ERROR_NEGATIVE_VALUES = (23, "Negative physical values encountered")
    ERROR_SMALL_RADIUS = (24, "Shell radius became too small")

    # Unknown
    UNKNOWN = (99, "Unknown termination reason")

    def __init__(self, code: int, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> int:
        """Numeric exit code."""
        return self._code

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self._description

    def is_success(self) -> bool:
        """True if this is a success state (code 0-9)."""
        return 0 <= self._code <= 9

    def is_error(self) -> bool:
        """True if this is an error state (code >= 10)."""
        return self._code >= 10


def get_end_code_from_reason(reason_str: str) -> SimulationEndCode:
    """
    Map SimulationEndReason string to SimulationEndCode enum.

    Parameters
    ----------
    reason_str : str
        The reason string from params['SimulationEndReason'].value

    Returns
    -------
    SimulationEndCode
        Matching enum value, or UNKNOWN if no match found
    """
    if reason_str is None:
        return SimulationEndCode.UNKNOWN

    reason_lower = reason_str.lower()

    # Map common reason strings to codes
    reason_map = {
        # Success states
        'shell dissolved': SimulationEndCode.SUCCESS_DISSOLVED,
        'dissolved': SimulationEndCode.SUCCESS_DISSOLVED,
        'stopping time reached': SimulationEndCode.SUCCESS_MAX_TIME,
        'max time': SimulationEndCode.SUCCESS_MAX_TIME,
        'large radius reached': SimulationEndCode.SUCCESS_MAX_RADIUS,
        'max radius': SimulationEndCode.SUCCESS_MAX_RADIUS,
        'exceeded rcloud': SimulationEndCode.SUCCESS_MAX_RADIUS,
        'complete': SimulationEndCode.SUCCESS_COMPLETE,
        # Parameter errors
        'invalid cloud parameters': SimulationEndCode.ERROR_INVALID_PARAMS,
        'invalid param': SimulationEndCode.ERROR_INVALID_PARAMS,
        'mass inconsistency': SimulationEndCode.ERROR_MASS_INCONSISTENCY,
        'mass error': SimulationEndCode.ERROR_MASS_INCONSISTENCY,
        'edge density': SimulationEndCode.ERROR_EDGE_DENSITY,
        'nedge < nism': SimulationEndCode.ERROR_EDGE_DENSITY,
        # Numerical errors
        'numerical instability': SimulationEndCode.ERROR_NUMERICAL,
        'numerical error': SimulationEndCode.ERROR_NUMERICAL,
        'velocity threshold': SimulationEndCode.ERROR_VELOCITY,
        'velocity below': SimulationEndCode.ERROR_VELOCITY,
        'solver failed': SimulationEndCode.ERROR_SOLVER,
        'ode error': SimulationEndCode.ERROR_SOLVER,
        'negative': SimulationEndCode.ERROR_NEGATIVE_VALUES,
        # Radius-related terminations
        'small radius': SimulationEndCode.ERROR_SMALL_RADIUS,
        'radius too small': SimulationEndCode.ERROR_SMALL_RADIUS,
        'shell collapsed': SimulationEndCode.ERROR_SMALL_RADIUS,
    }

    for key, code in reason_map.items():
        if key in reason_lower:
            return code

    return SimulationEndCode.UNKNOWN


def write_simulation_end(params: Dict[str, Any], output_dir: Optional[str] = None) -> int:
    """
    Write simulation end summary to simulationEnd.txt.

    This function should be called at the end of every simulation run
    to create a structured record of why and how the simulation ended.

    Parameters
    ----------
    params : dict
        TRINITY parameter dictionary containing simulation state.
        Expected keys: SimulationEndReason, model_name, path2output,
        t_now, R2, shell_nMax, v_R2, mCloud, nCore, rCloud, densPL_alpha
    output_dir : str, optional
        Output directory. If None, uses params['path2output'].value

    Returns
    -------
    int
        Exit code from SimulationEndCode enum

    Creates
    -------
    simulationEnd.txt in output_dir with structured format including:
    - Timestamp
    - Model name
    - End reason and exit code
    - Final simulation state
    - Initial cloud parameters
    """
    # Determine output directory
    if output_dir is None:
        if 'path2output' in params:
            output_dir = params['path2output'].value
        else:
            output_dir = '.'

    # Get end reason
    if 'SimulationEndReason' in params:
        reason_str = params['SimulationEndReason'].value
    else:
        reason_str = 'Unknown'

    end_code = get_end_code_from_reason(reason_str)

    # Get model name
    if 'model_name' in params:
        model_name = params['model_name'].value
    else:
        model_name = 'unknown'

    # Build report lines
    lines = [
        "=" * 50,
        "TRINITY Simulation End Report",
        "=" * 50,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_name}",
        "",
        "-" * 50,
        "TERMINATION",
        "-" * 50,
        f"End Reason: {end_code.description}",
        f"Exit Code: {end_code.code}",
        f"Status: {'SUCCESS' if end_code.is_success() else 'ERROR'}",
        f"Raw Reason: {reason_str}",
    ]

    # Add final state section
    lines.extend([
        "",
        "-" * 50,
        "FINAL STATE",
        "-" * 50,
    ])

    # Helper to safely get param values with optional unit conversion
    def get_param(key, fmt=".4e", default="N/A", conversion=1.0):
        if key in params:
            val = params[key].value
            if val is not None:
                try:
                    return f"{val * conversion:{fmt}}"
                except:
                    return str(val)
        return default

    # Unit conversion factors:
    # - Velocity: pc/Myr -> km/s (INV_CONV.v_au2kms)
    # - Number density: pc^-3 -> cm^-3 (INV_CONV.ndens_au2cgs)

    lines.append(f"  Time:           {get_param('t_now', '.3f')} Myr")
    lines.append(f"  Radius (R2):    {get_param('R2', '.2f')} pc")
    lines.append(f"  Shell nMax:     {get_param('shell_nMax', '.2e', conversion=INV_CONV.ndens_au2cgs)} cm^-3")
    lines.append(f"  Shell Velocity: {get_param('v2', '.2f', conversion=INV_CONV.v_au2kms)} km/s")

    # Add initial parameters section
    lines.extend([
        "",
        "-" * 50,
        "INITIAL CLOUD PARAMETERS",
        "-" * 50,
        f"  mCloud:  {get_param('mCloud', '.2e')} Msun",
        f"  nCore:   {get_param('nCore', '.2e', conversion=INV_CONV.ndens_au2cgs)} cm^-3",
        f"  rCloud:  {get_param('rCloud', '.2f')} pc",
        f"  rCore:   {get_param('rCore', '.2f')} pc",
        f"  alpha:   {get_param('densPL_alpha', '.1f')}",
        f"  nISM:    {get_param('nISM', '.2e', conversion=INV_CONV.ndens_au2cgs)} cm^-3",
    ])

    # Add validation info if available
    if 'validation_mass_error' in params:
        lines.extend([
            "",
            "-" * 50,
            "VALIDATION",
            "-" * 50,
            f"  Mass Error: {get_param('validation_mass_error', '.4f')}%",
        ])

    lines.extend([
        "",
        "=" * 50,
    ])

    # Write to file
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'simulationEnd.txt')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Simulation end report written to: {filepath}")

    return end_code.code


def read_simulation_end(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Read and parse a simulationEnd.txt file.

    Parameters
    ----------
    output_dir : str
        Directory containing simulationEnd.txt

    Returns
    -------
    dict or None
        Parsed content with keys: exit_code, reason, status, timestamp, model
        Returns None if file doesn't exist
    """
    filepath = os.path.join(output_dir, 'simulationEnd.txt')

    if not os.path.exists(filepath):
        return None

    result = {
        'exit_code': None,
        'reason': None,
        'status': None,
        'timestamp': None,
        'model': None,
    }

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Exit Code:'):
                try:
                    result['exit_code'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('End Reason:'):
                result['reason'] = line.split(':', 1)[1].strip()
            elif line.startswith('Status:'):
                result['status'] = line.split(':')[1].strip()
            elif line.startswith('Timestamp:'):
                result['timestamp'] = line.split(':', 1)[1].strip()
            elif line.startswith('Model:'):
                result['model'] = line.split(':')[1].strip()

    return result


# =============================================================================
# Termination Debug Report
# =============================================================================

# Key parameters to track in comparison (most likely to indicate problems)
CRITICAL_PARAMS = [
    # Time and radii
    ('t_now', 'Time', 'Myr', 1.0),
    ('R1', 'Inner radius', 'pc', 1.0),
    ('R2', 'Outer radius', 'pc', 1.0),
    ('rShell', 'Shell radius', 'pc', 1.0),
    # Velocities
    ('v2', 'Shell velocity', 'km/s', 0.9778),  # pc/Myr -> km/s approx
    # Energies
    ('Eb', 'Bubble energy', 'erg', 1.0),
    ('Pb', 'Bubble pressure', 'erg/cm³', 1.0),
    # Shell properties
    ('shell_mass', 'Shell mass', 'Msun', 1.0),
    ('shell_nMax', 'Shell peak density', 'cm⁻³', 1.0),
    # Forces
    ('F_grav', 'Gravity force', 'code', 1.0),
    ('F_ram', 'Ram force', 'code', 1.0),
    ('F_rad', 'Radiation force', 'code', 1.0),
    ('F_ion', 'Ion force', 'code', 1.0),
    # Temperatures
    ('T0', 'Inner temp', 'K', 1.0),
    ('bubble_Tavg', 'Avg bubble temp', 'K', 1.0),
    # Phase
    ('current_phase', 'Phase', '', 1.0),
    ('isCollapse', 'Collapsing', '', 1.0),
]

# Thresholds for flagging large changes
CHANGE_THRESHOLDS = {
    'default': 0.5,       # 50% change flagged by default
    't_now': 10.0,        # Time can jump (don't flag)
    'current_phase': 0,   # Phase changes are always flagged if different
    'isCollapse': 0,      # Collapse status changes always flagged
    'v2': 1.0,            # Velocity can change sign, be more lenient
    'Eb': 1.0,            # Energy can change rapidly
    'Pb': 1.0,            # Pressure can change rapidly
}


def _load_last_snapshots(output_dir: str, n: int = 2) -> List[Dict[str, Any]]:
    """
    Load the last N snapshots from dictionary.jsonl.

    Parameters
    ----------
    output_dir : str
        Directory containing dictionary.jsonl
    n : int
        Number of snapshots to load (from end)

    Returns
    -------
    list
        List of snapshot dictionaries, oldest first
    """
    jsonl_path = Path(output_dir) / "dictionary.jsonl"

    if not jsonl_path.exists():
        return []

    # Read all lines to get last N
    snapshots = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Get last N non-empty lines
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            snap = json.loads(line)
            snapshots.append(snap)
        except json.JSONDecodeError:
            continue

    return snapshots


def _format_value(val: Any, precision: int = 6) -> str:
    """Format a value for display."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val)
        if arr.size == 0:
            return "[]"
        if arr.size <= 5:
            return str(arr.tolist())
        return f"[{arr[0]:.4g}, {arr[1]:.4g}, ... {arr[-1]:.4g}] (len={arr.size})"
    if isinstance(val, float):
        if abs(val) < 1e-3 or abs(val) > 1e4:
            return f"{val:.{precision}e}"
        return f"{val:.{precision}f}"
    if isinstance(val, int):
        return str(val)
    return str(val)


def _compute_change(old_val: Any, new_val: Any) -> Tuple[str, float, bool]:
    """
    Compute change between two values.

    Returns
    -------
    tuple
        (change_str, relative_change, is_significant)
    """
    # Handle None
    if old_val is None and new_val is None:
        return "—", 0.0, False
    if old_val is None:
        return "NEW", float('inf'), True
    if new_val is None:
        return "GONE", float('inf'), True

    # Handle strings/booleans (categorical)
    if isinstance(old_val, (str, bool)) or isinstance(new_val, (str, bool)):
        if old_val != new_val:
            return f"{old_val} → {new_val}", float('inf'), True
        return "—", 0.0, False

    # Handle arrays
    if isinstance(old_val, (list, np.ndarray)) or isinstance(new_val, (list, np.ndarray)):
        old_arr = np.asarray(old_val)
        new_arr = np.asarray(new_val)
        if old_arr.shape != new_arr.shape:
            return f"shape {old_arr.shape}→{new_arr.shape}", float('inf'), True
        if old_arr.size == 0:
            return "—", 0.0, False
        # Compare max relative change
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.maximum(np.abs(old_arr), 1e-30)
            rel = np.abs(new_arr - old_arr) / denom
            max_rel = np.nanmax(rel)
        if not np.isfinite(max_rel):
            max_rel = 0.0
        return f"max Δ={max_rel:.1%}", max_rel, max_rel > 0.5

    # Handle numeric
    try:
        old_f = float(old_val)
        new_f = float(new_val)
    except (TypeError, ValueError):
        return "?", 0.0, False

    if old_f == new_f:
        return "—", 0.0, False

    diff = new_f - old_f
    if old_f == 0:
        rel_change = float('inf') if diff != 0 else 0.0
    else:
        rel_change = abs(diff / old_f)

    # Format change string
    if abs(diff) < 1e-10:
        change_str = "~0"
    elif rel_change > 10:
        change_str = f"{diff:+.2e} (×{new_f/old_f if old_f != 0 else 'inf':.1f})"
    else:
        change_str = f"{diff:+.3g} ({rel_change:+.1%})"

    return change_str, rel_change, False  # significance determined by threshold


def write_termination_debug_report(output_dir: str, reason: str = "Unknown") -> Optional[str]:
    """
    Write a debug report with the last two snapshots and comparison.

    This creates termination_debug.txt with:
    1. Last two snapshots printed in readable format
    2. Comparison table highlighting large changes
    3. Rate of change for key variables
    4. Warnings for potentially problematic values

    Parameters
    ----------
    output_dir : str
        Directory containing dictionary.jsonl and where to write report
    reason : str
        Termination reason string

    Returns
    -------
    str or None
        Path to written file, or None if failed
    """
    output_path = Path(output_dir)
    report_path = output_path / "termination_debug.txt"

    # Load last two snapshots
    snapshots = _load_last_snapshots(output_dir, n=2)

    if not snapshots:
        # No snapshots - write minimal report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TERMINATION DEBUG REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Reason: {reason}\n\n")
            f.write("No snapshots found in dictionary.jsonl\n")
        return str(report_path)

    lines = []
    lines.append("=" * 80)
    lines.append("TERMINATION DEBUG REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Termination Reason: {reason}")
    lines.append(f"Snapshots loaded: {len(snapshots)}")
    lines.append("")

    # ==========================================================================
    # Section 1: Comparison Table (if 2 snapshots)
    # ==========================================================================
    if len(snapshots) >= 2:
        snap_old = snapshots[-2]
        snap_new = snapshots[-1]

        lines.append("-" * 80)
        lines.append("COMPARISON TABLE: Second-to-last vs Last Snapshot")
        lines.append("-" * 80)

        # Time info
        t_old = snap_old.get('t_now', 'N/A')
        t_new = snap_new.get('t_now', 'N/A')
        lines.append(f"Time: {_format_value(t_old)} → {_format_value(t_new)} Myr")
        if isinstance(t_old, (int, float)) and isinstance(t_new, (int, float)):
            dt = t_new - t_old
            lines.append(f"Time step (dt): {dt:.6e} Myr")
        lines.append("")

        # Table header
        lines.append(f"{'Parameter':<25} {'Old Value':<20} {'New Value':<20} {'Change':<25} {'Flag'}")
        lines.append("-" * 95)

        warnings = []

        for key, label, unit, conv in CRITICAL_PARAMS:
            old_val = snap_old.get(key)
            new_val = snap_new.get(key)

            # Apply conversion
            if old_val is not None and isinstance(old_val, (int, float)):
                old_val = old_val * conv
            if new_val is not None and isinstance(new_val, (int, float)):
                new_val = new_val * conv

            old_str = _format_value(old_val, precision=4)
            new_str = _format_value(new_val, precision=4)
            change_str, rel_change, is_sig = _compute_change(old_val, new_val)

            # Determine if flagged
            threshold = CHANGE_THRESHOLDS.get(key, CHANGE_THRESHOLDS['default'])
            if threshold == 0:
                # Categorical - flag any change
                flagged = is_sig or (old_val != new_val)
            else:
                flagged = rel_change > threshold

            flag = "⚠️ LARGE" if flagged else ""

            # Truncate for display
            if len(old_str) > 18:
                old_str = old_str[:15] + "..."
            if len(new_str) > 18:
                new_str = new_str[:15] + "..."
            if len(change_str) > 23:
                change_str = change_str[:20] + "..."

            display_label = f"{label} ({key})"[:25]
            lines.append(f"{display_label:<25} {old_str:<20} {new_str:<20} {change_str:<25} {flag}")

            if flagged:
                warnings.append(f"  - {label}: {change_str}")

        lines.append("")

        # Warnings summary
        if warnings:
            lines.append("-" * 80)
            lines.append("⚠️  WARNINGS: Large changes detected")
            lines.append("-" * 80)
            for w in warnings:
                lines.append(w)
            lines.append("")

        # Check for NaN/Inf values
        nan_keys = []
        inf_keys = []
        for key, val in snap_new.items():
            if isinstance(val, (int, float)):
                if np.isnan(val):
                    nan_keys.append(key)
                elif np.isinf(val):
                    inf_keys.append(key)
            elif isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val)
                if np.any(np.isnan(arr)):
                    nan_keys.append(f"{key} (array)")
                if np.any(np.isinf(arr)):
                    inf_keys.append(f"{key} (array)")

        if nan_keys or inf_keys:
            lines.append("-" * 80)
            lines.append("⚠️  INVALID VALUES IN LAST SNAPSHOT")
            lines.append("-" * 80)
            if nan_keys:
                lines.append(f"NaN values: {', '.join(nan_keys[:10])}" +
                           (f" (+{len(nan_keys)-10} more)" if len(nan_keys) > 10 else ""))
            if inf_keys:
                lines.append(f"Inf values: {', '.join(inf_keys[:10])}" +
                           (f" (+{len(inf_keys)-10} more)" if len(inf_keys) > 10 else ""))
            lines.append("")

    # ==========================================================================
    # Section 2: Full Snapshot Dumps
    # ==========================================================================
    for i, snap in enumerate(snapshots):
        snap_idx = "SECOND-TO-LAST" if i == len(snapshots) - 2 else "LAST"
        if len(snapshots) == 1:
            snap_idx = "ONLY"

        lines.append("=" * 80)
        lines.append(f"SNAPSHOT: {snap_idx} (t = {_format_value(snap.get('t_now'))} Myr)")
        lines.append("=" * 80)

        # Group keys by category
        time_keys = ['t_now', 'snap_id', 'current_phase', 'isCollapse']
        radii_keys = ['R1', 'R2', 'rShell', 'r_Tb', 'bubble_r_Tb', 'rCloud', 'rCore']
        velocity_keys = ['v2', 'v_R1', 'v_R2']
        energy_keys = ['Eb', 'Pb', 'T0', 'bubble_Tavg']
        force_keys = [k for k in snap.keys() if k.startswith('F_')]
        shell_keys = [k for k in snap.keys() if k.startswith('shell_') and not isinstance(snap[k], list)]
        bubble_keys = [k for k in snap.keys() if k.startswith('bubble_') and not isinstance(snap[k], list)]
        array_keys = [k for k in snap.keys() if isinstance(snap[k], list)]
        other_keys = set(snap.keys()) - set(time_keys) - set(radii_keys) - set(velocity_keys) - \
                     set(energy_keys) - set(force_keys) - set(shell_keys) - set(bubble_keys) - set(array_keys)

        def print_section(title, keys):
            if not keys:
                return
            lines.append(f"\n--- {title} ---")
            for key in sorted(keys):
                if key in snap:
                    lines.append(f"  {key:<30} = {_format_value(snap[key])}")

        print_section("Time & Phase", time_keys)
        print_section("Radii", radii_keys)
        print_section("Velocities", velocity_keys)
        print_section("Energy & Temperature", energy_keys)
        print_section("Forces", force_keys)
        print_section("Shell Properties", shell_keys)
        print_section("Bubble Properties", bubble_keys)
        print_section("Other Scalars", sorted(other_keys)[:30])  # Limit to 30

        if array_keys:
            lines.append(f"\n--- Arrays ({len(array_keys)} total) ---")
            for key in sorted(array_keys)[:15]:  # Limit display
                arr = np.asarray(snap[key])
                lines.append(f"  {key:<30} : shape={arr.shape}, range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")
            if len(array_keys) > 15:
                lines.append(f"  ... and {len(array_keys) - 15} more arrays")

        lines.append("")

    # ==========================================================================
    # Section 3: Diagnostic Summary
    # ==========================================================================
    if len(snapshots) >= 1:
        snap = snapshots[-1]
        lines.append("=" * 80)
        lines.append("DIAGNOSTIC SUMMARY")
        lines.append("=" * 80)

        # Physics sanity checks
        checks = []

        # Check R1 < R2
        R1 = snap.get('R1')
        R2 = snap.get('R2')
        if R1 is not None and R2 is not None:
            if R1 >= R2:
                checks.append(f"❌ R1 >= R2: {R1:.4g} >= {R2:.4g}")
            else:
                checks.append(f"✓ R1 < R2: {R1:.4g} < {R2:.4g}")

        # Check positive energy
        Eb = snap.get('Eb')
        if Eb is not None:
            if Eb <= 0:
                checks.append(f"❌ Eb <= 0: {Eb:.4e}")
            else:
                checks.append(f"✓ Eb > 0: {Eb:.4e}")

        # Check positive pressure
        Pb = snap.get('Pb')
        if Pb is not None:
            if Pb <= 0:
                checks.append(f"❌ Pb <= 0: {Pb:.4e}")
            else:
                checks.append(f"✓ Pb > 0: {Pb:.4e}")

        # Check shell mass
        shell_mass = snap.get('shell_mass')
        if shell_mass is not None:
            if shell_mass <= 0:
                checks.append(f"❌ shell_mass <= 0: {shell_mass:.4e}")
            else:
                checks.append(f"✓ shell_mass > 0: {shell_mass:.4e}")

        # Check collapse status
        isCollapse = snap.get('isCollapse')
        if isCollapse is not None:
            if isCollapse:
                checks.append(f"⚠️ isCollapse = True (shell collapsing)")
            else:
                checks.append(f"✓ isCollapse = False")

        for check in checks:
            lines.append(f"  {check}")

        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF DEBUG REPORT")
    lines.append("=" * 80)

    # Write to file
    output_path.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Termination debug report written to: {report_path}")

    return str(report_path)
