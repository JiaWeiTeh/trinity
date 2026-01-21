#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility module for loading TRINITY simulation snapshots from JSON or JSONL files.

Supports both formats:
- JSON: Single file with nested dictionary {"0": {...}, "1": {...}, ...}
- JSONL: One JSON object per line (each line is a snapshot)

Usage:
    # Import in scripts located in the same directory:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from load_snapshots import load_snapshots, find_data_file

    # NEW: Use TrinityOutput reader for cleaner access
    from load_snapshots import load_output
    output = load_output(data_path)
    t = output.get('t_now')
    R2 = output.get('R2')
    output.info()

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
    base_path = _ensure_path(base_path)

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
