#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation End Reason Logger

Writes simulation termination details to simulationEnd.txt in output directory.
Provides structured exit codes for batch processing and post-run analysis.

Author: Claude Code
Date: 2026-01-15
"""

import os
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

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
