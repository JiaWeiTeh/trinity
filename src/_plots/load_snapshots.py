#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility module for loading TRINITY simulation snapshots from JSON or JSONL files.

Supports both formats:
- JSON: Single file with nested dictionary {"0": {...}, "1": {...}, ...}
- JSONL: One JSON object per line (each line is a snapshot)

@author: TRINITY Team
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_snapshots(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load simulation snapshots from either JSON or JSONL file.

    Parameters
    ----------
    file_path : Path
        Path to either a .json or .jsonl file

    Returns
    -------
    List[Dict[str, Any]]
        List of snapshot dictionaries, sorted by snapshot index/order
    """
    file_path = Path(file_path)

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
