#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRINITY Snapshot Loading Utilities
===================================

Utility module for loading TRINITY simulation snapshots from JSON or JSONL files.
This module provides both low-level loading functions and a high-level TrinityOutput
reader interface.

Supports both formats:
- JSON: Single file with nested dictionary {"0": {...}, "1": {...}, ...}
- JSONL: One JSON object per line (each line is a snapshot) - RECOMMENDED

Recommended Usage (TrinityOutput Reader)
----------------------------------------
The TrinityOutput reader provides the cleanest API for accessing simulation data:

    from load_snapshots import load_output, find_data_file

    # Find and load data file
    data_path = find_data_file(BASE_DIR, run_name)
    output = load_output(data_path)

    # Access time series as numpy arrays
    t = output.get('t_now')
    R2 = output.get('R2')
    v2 = output.get('v2')

    # For non-numeric data
    phase = np.array(output.get('current_phase', as_array=False))

    # Get scalar from first snapshot
    rcloud = float(output[0].get('rCloud', np.nan))

    # Print summary
    output.info()

Legacy Usage (Direct Snapshot Loading)
--------------------------------------
For backward compatibility, the original load_snapshots function is still available:

    from load_snapshots import load_snapshots, find_data_file

    snaps = load_snapshots(data_path)
    t = np.array([s["t_now"] for s in snaps], dtype=float)
    R2 = np.array([s["R2"] for s in snaps], dtype=float)

Key Functions
-------------
- load_output(path): Load as TrinityOutput object (RECOMMENDED)
- load_snapshots(path): Load as list of dictionaries (legacy)
- find_data_file(base_dir, run_name): Locate data file for a simulation run

All paper_* plotting scripts in this directory have been updated (January 2026)
to use load_output() for cleaner, more maintainable code.

See Also
--------
- src/_output/trinity_reader.py: Full TrinityOutput reader implementation
- example_scripts/: Usage examples for the reader

@author: TRINITY Team
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import TrinityOutput reader
import sys
_src_path = str(Path(__file__).parent.parent.parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from src._output.trinity_reader import TrinityOutput


def _ensure_path(file_path: Union[str, Path]) -> Path:
    """Convert string path to Path object if needed."""
    return Path(file_path) if isinstance(file_path, str) else file_path


def load_snapshots(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load simulation snapshots from either JSON or JSONL file.

    Parameters
    ----------
    file_path : Path or str
        Path to either a .json or .jsonl file

    Returns
    -------
    List[Dict[str, Any]]
        List of snapshot dictionaries, sorted by snapshot index/order
    """
    file_path = _ensure_path(file_path)

    if file_path.suffix == '.jsonl':
        return load_jsonl(file_path)
    else:
        return load_json(file_path)


def load_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load snapshots from traditional dictionary.json format.

    The JSON file has structure: {"0": {snapshot_data}, "1": {snapshot_data}, ...}
    """
    with json_path.open("r") as f:
        data = json.load(f)

    # Get snapshot keys (numeric indices)
    snap_keys = sorted(
        (k for k in data.keys() if str(k).isdigit()),
        key=lambda k: int(k)
    )

    return [data[k] for k in snap_keys]


def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Load snapshots from JSONL format (one snapshot per line).
    """
    snapshots = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    return snapshots


def find_data_file(base_dir: Path, run_name: str) -> Optional[Path]:
    """
    Find the data file for a run, preferring JSONL over JSON.

    Searches for:
    1. {run_name}_dictionary.jsonl
    2. dictionary.jsonl
    3. dictionary.json

    Parameters
    ----------
    base_dir : Path
        Base directory containing run folders
    run_name : str
        Name of the run (e.g., "1e7_sfe020_n1e4")

    Returns
    -------
    Optional[Path]
        Path to data file, or None if not found
    """
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


def load_output(file_path: Union[str, Path]) -> TrinityOutput:
    """
    Load simulation data as a TrinityOutput object for clean data access.

    This is the preferred way to load TRINITY output files.

    Parameters
    ----------
    file_path : Path or str
        Path to either a .json or .jsonl file

    Returns
    -------
    TrinityOutput
        Reader object with convenient methods:
        - output.get('key') -> numpy array of values across all snapshots
        - output.filter(phase='implicit') -> filtered TrinityOutput
        - output.info() -> print summary
        - output[i] -> access snapshot i
        - output.t_min, output.t_max -> time range

    Examples
    --------
    >>> output = load_output('simulation.jsonl')
    >>> t = output.get('t_now')
    >>> R2 = output.get('R2')
    >>> output.info()
    """
    return TrinityOutput.open(file_path)
