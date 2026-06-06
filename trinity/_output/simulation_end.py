#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation End Reason Logger
============================

End-of-run reporting for TRINITY simulations.  All run-end data
lands in ``metadata.json`` (v4+ schema) as three structured blocks:

* ``metadata.json[termination]`` — ``{exit_code, outcome, detail,
  timestamp, model_name}``.  Written by :func:`write_simulation_end`.
* ``metadata.json[final_state]`` — every scalar/bool/string on
  ``params`` at run end, in INTERNAL units (pc/Myr, pc⁻³, …).  Same
  convention as ``Snapshot.get(key)``; arrays excluded (their last
  values live in the dictionary.jsonl tail).
* ``metadata.json[termination_debug]`` — last-2-snapshot diff,
  NaN/Inf inventory, and physics sanity checks.  Written by
  :func:`write_termination_debug_report` at emergency-flush time.

All writes go through the shared atomic helper in
:mod:`trinity._output._metadata_io`, so a partial write can never leave
a corrupt file.

Phase 5 of the metadata-source-of-truth migration removed three
text artefacts: ``simulationEnd.txt``, ``termination_debug.txt``,
and ``<run>_summary.txt``.  Human-readable views come from
``python -m trinity._output.show_run <run_dir>`` instead.  Readers
keep a back-compat path for legacy runs (emit ``DeprecationWarning``
and parse the old text files); the back-compat path will be
removed in Phase 6.

Public functions
~~~~~~~~~~~~~~~~
* :func:`write_simulation_end` — called at run end by ``main.py``.
* :func:`read_simulation_end` — reads ``metadata.json[termination]``
  first (v3+); falls back to text-parsing ``simulationEnd.txt`` (with
  ``DeprecationWarning``) for legacy runs.
* :func:`write_termination_debug_report` — last-2-snapshot debug
  dump, merged into ``metadata.json[termination_debug]``.
"""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Import unit conversions for display
from trinity._functions.unit_conversions import INV_CONV, Pb_au2_KcmInv


class SimulationEndCode(Enum):
    """
    Enumeration of simulation end reasons with exit codes.

    Exit code ranges:
    - 0-9:   Clean physical or intentional terminations (auto-trust)
    - 10-19: Parameter/configuration errors
    - 20-29: Numerical/runtime errors
    - 50-59: Inspection required (completed, but warrants a human look)
    - 99:    Unknown/unhandled termination (fallback safety net)

    Each member carries (code, outcome_token). The outcome token is
    mirrored into ``metadata.json[termination].outcome``.
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
        """Short categorical label mirrored into ``metadata.json[termination].outcome``."""
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
    Mirror the end-of-run termination + final-state data into
    ``metadata.json``.

    Called at the end of every run.  The exit code and outcome category
    are read directly from ``params['SimulationEndCode']`` (set at the
    source by the site that decided to terminate); the verbatim
    ``SimulationEndReason`` message becomes ``termination.detail``.

    What gets written (v4+ schema):

    * ``metadata.json[termination]`` — ``{exit_code, outcome, detail,
      timestamp, model_name}`` — mirrors ``read_simulation_end()``'s
      return shape so consumer migrations are one-line.
    * ``metadata.json[final_state]`` — every non-array non-run-constant
      scalar from ``params`` at run end, in INTERNAL units (pc/Myr,
      pc⁻³, …) — same convention as ``Snapshot.get(key)``.

    Both blocks land via the shared atomic helper in
    :mod:`trinity._output._metadata_io` so a partial write can never leave
    a corrupt file.

    Phase 5 change (this commit): the legacy ``simulationEnd.txt``
    text file is no longer written.  Human consumers should use
    ``python -m trinity._output.show_run <run_dir>`` for the formatted
    view.  ``read_simulation_end()`` keeps a text-parse fallback
    (with ``DeprecationWarning``) for runs produced before this
    phase.

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

    # Ensure the output dir exists (the metadata writer would create
    # it too, but having it here keeps an empty-run dir from
    # producing surprising errors).
    os.makedirs(output_dir, exist_ok=True)

    # Build the structured blocks and write them atomically into
    # metadata.json.  Phase 5 drop: no longer writes simulationEnd.txt.
    # Phase 6 will remove read_simulation_end's text-parse fallback;
    # until then, the migration grace period covers any caller still
    # reaching for the text file.
    from trinity._output._metadata_io import update_metadata_atomic

    termination_block = {
        "exit_code": int(end_code.code),
        "outcome": str(end_code.outcome),
        "detail": str(reason_str),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": str(model_name),
    }
    final_state_block = _build_final_state_block(params)

    try:
        update_metadata_atomic(
            Path(output_dir),
            termination=termination_block,
            final_state=final_state_block,
        )
    except Exception as e:
        # The exit code is the contract of this function; the
        # metadata write failing should not bring the run down.
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
    ``Snapshot.get(key)``.  ``python -m trinity._output.show_run`` re-
    applies km/s and cm⁻³ conversions for human readability.

    NaN / non-finite values are kept as-is — ``json.dump`` emits them
    as ``NaN``, which Python's ``json.load`` reads back faithfully
    (technically non-standard JSON; this is the same compromise the
    snapshot writer makes for fields like ``cool_beta`` in the
    momentum phase).
    """
    from trinity._input.dictionary import DescribedItem
    from trinity._output.run_constants import (
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
        Directory containing ``metadata.json`` (and optionally a legacy
        ``simulationEnd.txt`` from a pre-Phase-5 run).

    Returns
    -------
    dict or None
        Keys: ``exit_code``, ``outcome``, ``detail``, ``timestamp``,
        ``model``.  Returns ``None`` if neither source is present.
    """
    # --- Preferred path: metadata.json[termination] (v3+ schema) -----
    try:
        from trinity._output._metadata_io import read_metadata
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

    # Phase 5+: this path runs only for pre-Phase-5 runs.  Emit a
    # warning so consumers know they're on the back-compat branch
    # (removed in Phase 6).
    import warnings
    warnings.warn(
        "Reading simulationEnd.txt — run pre-dates Phase 5 of the "
        "metadata migration.  Re-run the simulation to populate "
        "metadata.json[termination]; the text-parse fallback will be "
        "removed in Phase 6.",
        DeprecationWarning,
        stacklevel=2,
    )

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
    ('Pb', 'Bubble pressure', 'K cm⁻³', Pb_au2_KcmInv),  # P/k_B
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


def write_termination_debug_report(output_dir: str, reason: str = "Unknown") -> None:
    """
    Mirror last-2-snapshot debug data into ``metadata.json[termination_debug]``.

    Builds a structured dict from the dictionary.jsonl tail (comparison
    rows, NaN/Inf inventory, physics sanity checks) and merges it into
    ``metadata.json``.

    Phase 5 change (this commit): the legacy ``termination_debug.txt``
    text file is no longer written.  Human consumers can format the
    block with ``python -m trinity._output.show_run <run_dir>`` (or read
    ``metadata.json[termination_debug]`` directly).  The output dict
    has stable keys (``comparison``, ``warnings``, ``invalid_values``,
    ``sanity_checks``) so downstream automation no longer has to parse
    free-form text.

    Parameters
    ----------
    output_dir : str
        Directory containing ``dictionary.jsonl`` and ``metadata.json``.
    reason : str
        Termination reason string (verbatim, surfaced as
        ``termination_debug.reason``).

    Returns
    -------
    None
        Returns ``None`` for backwards compatibility with the old
        callers (they only logged the return; the path was never
        consumed programmatically).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snapshots = _load_last_snapshots(output_dir, n=2)
    debug_block: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": str(reason),
        "snapshot_count": len(snapshots),
    }

    if not snapshots:
        debug_block["note"] = "No snapshots found in dictionary.jsonl"
        _merge_termination_debug(output_path, debug_block)
        return None

    snap_new = snapshots[-1]
    snap_old = snapshots[-2] if len(snapshots) >= 2 else None

    # --- Time + dt -------------------------------------------------
    t_old = snap_old.get('t_now') if snap_old is not None else None
    t_new = snap_new.get('t_now')
    time_block: Dict[str, Any] = {"new": _jsonable(t_new)}
    if snap_old is not None:
        time_block["old"] = _jsonable(t_old)
        if isinstance(t_old, (int, float)) and isinstance(t_new, (int, float)):
            time_block["dt"] = float(t_new - t_old)
    debug_block["time"] = time_block

    # --- Comparison table + warnings ------------------------------
    comparison: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    if snap_old is not None:
        for key, label, unit, conv in CRITICAL_PARAMS:
            old_val = snap_old.get(key)
            new_val = snap_new.get(key)
            if old_val is not None and isinstance(old_val, (int, float)) and not isinstance(old_val, bool):
                old_val = old_val * conv
            if new_val is not None and isinstance(new_val, (int, float)) and not isinstance(new_val, bool):
                new_val = new_val * conv

            change_str, rel_change, is_sig = _compute_change(old_val, new_val)
            threshold = CHANGE_THRESHOLDS.get(key, CHANGE_THRESHOLDS['default'])
            if threshold == 0:
                flagged = is_sig or (old_val != new_val)
            else:
                flagged = rel_change > threshold

            row: Dict[str, Any] = {
                "key": key,
                "label": label,
                "unit": unit,
                "old": _jsonable(old_val),
                "new": _jsonable(new_val),
                "change": change_str,
                "rel_change": _jsonable(rel_change),
                "flagged": bool(flagged),
            }
            comparison.append(row)
            if flagged:
                warnings.append({"key": key, "label": label, "change": change_str})
    debug_block["comparison"] = comparison
    debug_block["warnings"] = warnings

    # --- NaN / Inf inventory --------------------------------------
    nan_keys: List[str] = []
    inf_keys: List[str] = []
    for key, val in snap_new.items():
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            if np.isnan(val):
                nan_keys.append(key)
            elif np.isinf(val):
                inf_keys.append(key)
        elif isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val)
            if arr.size and np.issubdtype(arr.dtype, np.number):
                if np.any(np.isnan(arr)):
                    nan_keys.append(f"{key} (array)")
                if np.any(np.isinf(arr)):
                    inf_keys.append(f"{key} (array)")
    debug_block["invalid_values"] = {"nan": nan_keys, "inf": inf_keys}

    # --- Physics sanity checks ------------------------------------
    debug_block["sanity_checks"] = _build_sanity_checks(snap_new)

    _merge_termination_debug(output_path, debug_block)
    return None


def _jsonable(val: Any) -> Any:
    """
    Coerce a numeric value to a JSON-friendly type.

    ``json.dump(..., allow_nan=True)`` accepts NaN/Inf — same compromise
    the snapshot writer makes.  numpy scalars are unboxed so they don't
    leak into the metadata file as ``{"__numpy__": ...}``-style escapes.
    """
    if val is None or isinstance(val, (bool, str)):
        return val
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, (int, float)):
        return val
    return str(val)


def _build_sanity_checks(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run the small set of last-snapshot physics sanity checks."""
    checks: List[Dict[str, Any]] = []

    R1 = snap.get('R1')
    R2 = snap.get('R2')
    if R1 is not None and R2 is not None:
        checks.append({
            "check": "R1 < R2",
            "passed": bool(R1 < R2),
            "detail": f"R1={R1:.4g}, R2={R2:.4g}",
        })

    for key, label in (("Eb", "Eb > 0"), ("Pb", "Pb > 0"),
                       ("shell_mass", "shell_mass > 0")):
        val = snap.get(key)
        if val is not None and isinstance(val, (int, float)):
            checks.append({
                "check": label,
                "passed": bool(val > 0),
                "detail": f"{key}={val:.4e}",
            })

    isCollapse = snap.get('isCollapse')
    if isCollapse is not None:
        checks.append({
            "check": "isCollapse",
            "passed": not bool(isCollapse),
            "detail": f"isCollapse={bool(isCollapse)}",
        })

    return checks


def _merge_termination_debug(output_path: Path, block: Dict[str, Any]) -> None:
    """Merge ``termination_debug`` into metadata.json; never raise."""
    try:
        from trinity._output._metadata_io import update_metadata_atomic
        update_metadata_atomic(output_path, termination_debug=block)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to write termination_debug into metadata.json: %s", e,
        )
