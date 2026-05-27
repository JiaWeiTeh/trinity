#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation End Reason Logger
============================

End-of-run reporting for TRINITY simulations.  Produces two
complementary artefacts from a single in-memory dict:

* ``simulationEnd.txt`` — human-readable summary (timestamp,
  outcome, exit code, final state, initial cloud parameters).
  Pretty-printed with section headers; units converted for display
  (km/s, cm⁻³).
* ``metadata.json[termination]`` and ``metadata.json[final_state]``
  — structured machine-readable blocks merged into ``metadata.json``
  via :mod:`src._output._metadata_io`.  Internal units (pc/Myr,
  pc⁻³).  Phase 2 (v3+ schema) of the metadata-source-of-truth
  migration.

Both writes derive from the same in-memory data so they cannot drift.
A failure to update ``metadata.json`` is logged but does not abort
the text-file write.

Public functions
~~~~~~~~~~~~~~~~
* :func:`write_simulation_end` — called at run end by ``main.py``.
* :func:`read_simulation_end` — reads from ``metadata.json[termination]``
  first (clean v3+ path); falls back to text-parsing
  ``simulationEnd.txt`` for legacy runs.
* :func:`write_termination_debug_report` — separate debug dump
  (last-two-snapshot comparison) written to ``termination_debug.txt``.
  Phase 5 of the migration will merge it into ``simulationEnd.txt``.
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
    - 0-9:   Clean physical or intentional terminations (auto-trust)
    - 10-19: Parameter/configuration errors
    - 20-29: Numerical/runtime errors
    - 50-59: Inspection required (completed, but warrants a human look)
    - 99:    Unknown/unhandled termination (fallback safety net)

    Each member carries (code, outcome_token). The outcome token is the
    short categorical label written to simulationEnd.txt as 'Outcome:'.
    """
    # Clean (0-9)
    SHELL_DISSOLVED = (0, "shell_dissolved")
    STOPPING_TIME = (1, "stopping_time")
    LARGE_RADIUS = (2, "large_radius")
    RCLOUD_BOUNDARY = (3, "rcloud_boundary")
    SHELL_COLLAPSED = (4, "shell_collapsed")

    # Parameter errors (10-19)
    ERROR_INVALID_PARAMS = (10, "error_invalid_params")
    ERROR_MASS_INCONSISTENCY = (11, "error_mass_inconsistency")
    ERROR_EDGE_DENSITY = (12, "error_edge_density")
    ERROR_RADIUS_TOO_LARGE = (13, "error_radius_too_large")

    # Numerical errors (20-29)
    ERROR_NUMERICAL = (20, "error_numerical")
    ERROR_VELOCITY = (21, "error_velocity")
    ERROR_SOLVER = (22, "error_solver")
    ERROR_NEGATIVE_VALUES = (23, "error_negative_values")

    # Inspection required (50-59)
    VELOCITY_RUNAWAY = (50, "velocity_runaway")

    # Unknown — also treated as inspection-required
    UNKNOWN = (99, "unknown")

    def __init__(self, code: int, outcome: str):
        self._code = code
        self._outcome = outcome

    @property
    def code(self) -> int:
        """Numeric exit code."""
        return self._code

    @property
    def outcome(self) -> str:
        """Short categorical label written to simulationEnd.txt."""
        return self._outcome

    def is_clean(self) -> bool:
        """True if the run finished with a clean physical/intentional outcome (0-9)."""
        return 0 <= self._code <= 9

    def is_error(self) -> bool:
        """True if the run failed with a parameter or numerical error (10-29)."""
        return 10 <= self._code <= 29

    def is_inspection_required(self) -> bool:
        """True if the run completed but warrants a human look (50-59 or 99)."""
        return (50 <= self._code <= 59) or self._code == 99

    @classmethod
    def from_code(cls, code: int) -> "SimulationEndCode":
        """Look up the enum member by numeric code, or UNKNOWN if no match."""
        for member in cls:
            if member._code == code:
                return member
        return cls.UNKNOWN


def write_simulation_end(params: Dict[str, Any], output_dir: Optional[str] = None) -> int:
    """
    Write simulation end summary to ``simulationEnd.txt`` and mirror the
    structured form into ``metadata.json``.

    Called at the end of every run.  The exit code and outcome category
    are read directly from ``params['SimulationEndCode']`` (set at the
    source by the site that decided to terminate); the verbatim
    ``SimulationEndReason`` message becomes ``Detail:``.

    Two outputs from one in-memory dict:

    * ``simulationEnd.txt`` — pretty-printed for humans; section
      headers and unit-converted final-state values (km/s, cm⁻³).
    * ``metadata.json``     — adds two top-level blocks (v3+ schema):

        * ``termination``   : ``{exit_code, outcome, detail, timestamp,
                                 model_name}`` — mirrors
                                 ``read_simulation_end()``'s return
                                 shape so consumer migrations are
                                 one-line.
        * ``final_state``   : every non-array non-run-constant scalar
                                 from ``params`` at run end, in
                                 INTERNAL units (pc/Myr, pc⁻³, …) —
                                 same convention as
                                 ``Snapshot.get(key)``.

    Both writes go through the same atomic helper so they cannot drift.
    A failure to update ``metadata.json`` is logged and swallowed —
    the text-file write succeeds independently.

    Parameters
    ----------
    params : dict
        TRINITY parameter dictionary. Expected keys: ``SimulationEndCode``,
        ``SimulationEndReason``, ``model_name``, ``path2output``,
        ``t_now``, ``R2``, ``shell_nMax``, ``v2``, ``mCloud``, ``nCore``,
        ``rCloud``, ``rCore``, ``densPL_alpha``, ``nISM`` (plus everything
        else; ``final_state`` collects all scalars).
    output_dir : str, optional
        Output directory.  If ``None``, uses ``params['path2output'].value``.

    Returns
    -------
    int
        Numeric exit code from ``SimulationEndCode``.
    """
    # Determine output directory
    if output_dir is None:
        if 'path2output' in params:
            output_dir = params['path2output'].value
        else:
            output_dir = '.'

    # Verbatim source-side message
    if 'SimulationEndReason' in params:
        reason_str = params['SimulationEndReason'].value or 'unknown'
    else:
        reason_str = 'unknown'

    # End code is set at the source (phase runners, main.py, phase_events)
    # as the integer .code so it survives JSON serialization. If a site
    # forgot to set it, fall back to UNKNOWN (an inspection-required state).
    end_code = SimulationEndCode.UNKNOWN
    if 'SimulationEndCode' in params:
        raw = params['SimulationEndCode'].value
        if isinstance(raw, int):
            end_code = SimulationEndCode.from_code(raw)

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
        f"Outcome: {end_code.outcome}",
        f"Detail: {reason_str}",
        f"Exit Code: {end_code.code}",
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

    # --- Phase 2: also mirror structured termination + final_state
    # into metadata.json so plotters / analysis tools can read it
    # without text-parsing.  The text file stays as the human-readable
    # view; both are derived from the same in-memory data, so they
    # cannot drift.  Imports are local to keep this module's import
    # graph independent of dictionary.py.
    try:
        from src._output._metadata_io import update_metadata_atomic

        termination_block = {
            "exit_code": int(end_code.code),
            "outcome": str(end_code.outcome),
            "detail": str(reason_str),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": str(model_name),
        }

        final_state_block = _build_final_state_block(params)

        update_metadata_atomic(
            Path(output_dir),
            termination=termination_block,
            final_state=final_state_block,
        )
    except Exception as e:
        # The text file write succeeded; a metadata.json failure
        # should not bring down the run.  Log loud-but-non-fatal.
        import logging
        logging.getLogger(__name__).warning(
            "Failed to mirror termination/final_state into metadata.json: %s",
            e,
        )

    return end_code.code


def _build_final_state_block(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the ``final_state`` block from the runtime params.

    Includes every scalar/string/bool key on ``params`` EXCEPT:
      * run-constants (already in metadata.json's top-level scalars);
      * keys in ``METADATA_EXCLUDE`` (paths, function tables, …);
      * long arrays listed in ``FINAL_STATE_EXCLUDE_ARRAYS`` (their
        last-snapshot values are still available in the dictionary.jsonl
        stream's final line);
      * the ``SimulationEndCode`` proxy (already reflected by
        ``termination.exit_code``).

    Values are stored in internal units (pc/Myr, pc⁻³ …) to match
    ``Snapshot.get(key)``.  The text file in ``simulationEnd.txt``
    still applies km/s and cm⁻³ conversions for human readability.

    NaN / non-finite values are kept as-is — ``json.dump`` emits them
    as ``NaN``, which Python's ``json.load`` reads back faithfully
    (technically non-standard JSON; this is the same compromise the
    snapshot writer makes for fields like ``cool_beta`` in the
    momentum phase).
    """
    from src._input.dictionary import DescribedItem
    from src._output.run_constants import (
        RUN_CONST_KEYS, METADATA_EXCLUDE, FINAL_STATE_EXCLUDE_ARRAYS,
    )
    skip = (set(RUN_CONST_KEYS) | set(METADATA_EXCLUDE)
            | set(FINAL_STATE_EXCLUDE_ARRAYS)
            # ``SimulationEndCode`` is already reflected by
            # ``termination.exit_code``; ``SimulationEndReason`` is the
            # source string for ``termination.detail``.  Including either
            # here would leak duplicated (and possibly inconsistent —
            # the per-snapshot value is set AFTER save_snapshot ran) info
            # into final_state.  ``path2output`` is the absolute path of
            # the run dir itself; redundant and a privacy concern.
            | {"SimulationEndCode", "SimulationEndReason", "path2output"})

    block: Dict[str, Any] = {}
    for key, item in params.items():
        if key in skip:
            continue
        if not isinstance(item, DescribedItem):
            continue
        val = item.value
        # Skip arrays/lists with any length — final_state is scalars only.
        if isinstance(val, (list, tuple)) and len(val) > 0:
            continue
        if isinstance(val, np.ndarray) and val.size > 0:
            continue
        # Coerce numpy scalars / bools / NaN to plain types where possible.
        try:
            if isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            elif isinstance(val, np.bool_):
                val = bool(val)
            # Defensive: only include keys whose final value is JSON-friendly
            # (json.dumps tolerates None/str/int/float/bool/NaN).
            json.dumps(val, allow_nan=True)
        except (TypeError, ValueError):
            continue
        block[key] = val
    return block


def read_simulation_end(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Read the termination summary for a run.

    Prefers the ``termination`` block in ``metadata.json`` (v3+ schema,
    written by :func:`write_simulation_end`); falls back to text-parsing
    ``simulationEnd.txt`` for legacy runs (v1/v2 metadata or any run
    that pre-dates the metadata-source-of-truth migration).

    Parameters
    ----------
    output_dir : str
        Directory containing ``metadata.json`` and/or ``simulationEnd.txt``.

    Returns
    -------
    dict or None
        Keys: ``exit_code``, ``outcome``, ``detail``, ``timestamp``,
        ``model``.  Returns ``None`` if neither source is present.
    """
    # --- Preferred path: metadata.json[termination] (Phase 2+) -----
    try:
        from src._output._metadata_io import read_metadata
        metadata = read_metadata(Path(output_dir))
        block = metadata.get("termination") if metadata else None
        if isinstance(block, dict) and "exit_code" in block:
            return {
                "exit_code": block.get("exit_code"),
                "outcome": block.get("outcome"),
                "detail": block.get("detail"),
                "timestamp": block.get("timestamp"),
                # Legacy callers expect 'model' (not 'model_name')
                "model": block.get("model_name") or metadata.get("model_name"),
            }
    except Exception:
        # If the JSON path is broken in any way, fall through to text.
        pass

    # --- Legacy path: text-parse simulationEnd.txt -----------------
    filepath = os.path.join(output_dir, 'simulationEnd.txt')

    if not os.path.exists(filepath):
        return None

    result = {
        'exit_code': None,
        'outcome': None,
        'detail': None,
        'timestamp': None,
        'model': None,
    }

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Exit Code:'):
                try:
                    result['exit_code'] = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('Outcome:'):
                result['outcome'] = line.split(':', 1)[1].strip()
            elif line.startswith('Detail:'):
                result['detail'] = line.split(':', 1)[1].strip()
            elif line.startswith('Timestamp:'):
                result['timestamp'] = line.split(':', 1)[1].strip()
            elif line.startswith('Model:'):
                result['model'] = line.split(':', 1)[1].strip()
            # Legacy keys (pre-fix runs) — only used as fallback if the
            # corresponding new key wasn't present.
            elif line.startswith('Raw Reason:') and result['detail'] is None:
                result['detail'] = line.split(':', 1)[1].strip()

    # Fill outcome from exit_code if the file lacked it (legacy)
    if result['outcome'] is None and result['exit_code'] is not None:
        result['outcome'] = SimulationEndCode.from_code(result['exit_code']).outcome

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
    ('v2', 'Shell velocity', 'km/s', INV_CONV.v_au2kms),  # pc/Myr -> km/s
    # Energies
    ('Eb', 'Bubble energy', 'erg', 1.0),
    ('Pb', 'Bubble pressure', 'erg/cm³', 1.0),
    # Shell properties
    ('shell_mass', 'Shell mass', 'Msun', 1.0),
    ('shell_nMax', 'Shell peak density', 'cm⁻³', INV_CONV.ndens_au2cgs),
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
        if old_f != 0:
            change_str = f"{diff:+.2e} (×{new_f/old_f:.1f})"
        else:
            change_str = f"{diff:+.2e} (from 0)"
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
                if arr.size == 0:
                    lines.append(f"  {key:<30} : shape={arr.shape}, range=[empty]")
                else:
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
