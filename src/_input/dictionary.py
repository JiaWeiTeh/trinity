#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Wed Jul 26 15:21:52 2023
Modified: January 2026 - Rewritten for JSONL format with O(n) performance

@author: Jia Wei Teh

Purpose
-------
Provide a "params" container that behaves like a dictionary of objects:
    params["R2"].value
    params["R2"].value = 10.0

...and supports saving snapshots efficiently using line-delimited JSON (JSONL):
- All data (scalars + arrays) stored inline in JSON
- Each snapshot is one line in dictionary.jsonl
- Append-only writes for O(1) flush performance (vs O(n²) before)
- No HDF5 dependency - pure JSON architecture

Files written to params["path2output"].value
-------------------------------------------
dictionary.jsonl : One JSON object per line, each line = one snapshot
                   Line 0 = snapshot "0", Line 1 = snapshot "1", etc.

Loading
-------
params = DescribedDict.load_snapshot(path2output, snap_id)
arr = params["initial_cloud_n_arr"].value   # returns numpy array
"""

import collections.abc
import json
import sys
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# JSON helper: encode numpy types
# =============================================================================
class NpEncoder(json.JSONEncoder):
    """
    JSON encoder that converts numpy types to plain Python types.
    Handles both scalars and arrays.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# DescribedItem: the stored object at params[key]
# =============================================================================
class DescribedItem:
    """
    Container for a value (scalar/array) + light metadata.

    Key behavior
    ------------
    - Implements numeric conversions so you can do:
        '%E' % params['SB99_mass']          # works without .value
        f"{params['SB99_mass']:.2e}"        # also works

    Metadata fields
    ---------------
    info : str or None
        Human-readable description (optional).
    ori_units : str or None
        Units label (optional).
    exclude_from_snapshot : bool
        If True, key won't be saved into snapshots.
    """

    __slots__ = ("_value", "info", "ori_units", "exclude_from_snapshot")

    def __init__(
        self,
        value: Any = None,
        info: Optional[str] = None,
        ori_units: Optional[str] = None,
        exclude_from_snapshot: bool = False,
    ):
        self._value = value
        self.info = info
        self.ori_units = ori_units
        self.exclude_from_snapshot = exclude_from_snapshot

    @property
    def value(self) -> Any:
        """Return the stored value."""
        return self._value

    @value.setter
    def value(self, v: Any) -> None:
        """Set the underlying value (scalar or array)."""
        self._value = v

    # ----- numeric formatting/conversion helpers (quality-of-life) -----
    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __complex__(self) -> complex:
        return complex(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)

    def __index__(self) -> int:
        # Allows use in contexts requiring an integer index
        return int(self.value)
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Human-friendly string showing value and info."""
        return f"{self.value}\t({self.info})"

    def __repr__(self) -> str:
        """Short representation used in lists/debug prints."""
        return str(self.value)

    @staticmethod
    def _unwrap(x: Any) -> Any:
        """Extract numeric value if x is a DescribedItem, else return x."""
        return x.value if isinstance(x, DescribedItem) else x

    # arithmetic operators allow mixing DescribedItem with numbers
    def __add__(self, other): return self.value + self._unwrap(other)
    def __radd__(self, other): return self._unwrap(other) + self.value
    def __sub__(self, other): return self.value - self._unwrap(other)
    def __rsub__(self, other): return self._unwrap(other) - self.value
    def __mul__(self, other): return self.value * self._unwrap(other)
    def __rmul__(self, other): return self._unwrap(other) * self.value
    def __truediv__(self, other): return self.value / self._unwrap(other)
    def __rtruediv__(self, other): return self._unwrap(other) / self.value
    def __pow__(self, other): return self.value ** self._unwrap(other)
    def __rpow__(self, other): return self._unwrap(other) ** self.value

    # comparisons
    def __eq__(self, other): return self.value == self._unwrap(other)
    def __lt__(self, other): return self.value < self._unwrap(other)
    def __le__(self, other): return self.value <= self._unwrap(other)
    def __gt__(self, other): return self.value > self._unwrap(other)
    def __ge__(self, other): return self.value >= self._unwrap(other)

    # numpy compatibility
    def __array__(self, dtype=None):
        return np.array(self.value, dtype=dtype)


# =============================================================================
# DescribedDict: your main "params" container + snapshot machinery
# =============================================================================
class DescribedDict(dict):
    """
    A dictionary mapping string keys -> DescribedItem.

    Snapshot storage policy
    -----------------------
    - All data (scalars, arrays) stored inline in JSON
    - Each snapshot is one line in dictionary.jsonl
    - Append-only writes ensure O(1) flush performance (vs O(n²) in old version)

    Required key before saving
    --------------------------
    params["path2output"].value must exist and point to the output directory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Snapshot counters
        self.save_count: int = 0                  # how many snapshots have been saved in-memory
        self.snapshot_interval: int = 10          # flush every N snapshots
        self.previous_snapshot: Dict[str, Dict[str, Any]] = {}  # pending snapshots not yet flushed
        self.flush_count: int = 0                 # number of flush() calls (used for "fresh run" logic)

        # Key flags
        self._excluded_keys: set[str] = set()     # keys to omit from snapshots

    def __setitem__(self, key: str, value: DescribedItem) -> None:
        """
        Enforce that all stored values are DescribedItem.
        This keeps a consistent interface: params[key].value always exists.
        """
        if not isinstance(value, DescribedItem):
            raise TypeError(
                f"Value assigned to '{key}' must be a DescribedItem. "
                f"Did you mean: params['{key}'].value = <val> ?"
            )

        # Track exclusions based on item flags
        if value.exclude_from_snapshot:
            self._excluded_keys.add(key)

        super().__setitem__(key, value)

    # -------------------------------------------------------------------------
    # Display helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def shorten_display(arr, nshow: int = 3):
        """
        Shorten an array for display purposes to avoid clogging output.
        
        Parameters
        ----------
        arr : array-like
            Array or sequence to shorten.
        nshow : int, optional
            Number of elements to show at beginning and end (default: 3).
        
        Returns
        -------
        list
            Shortened representation of array.
        """
        if len(arr) > 10:
            arr = list(arr[:nshow]) + ['...'] + list(arr[-nshow:])
        return arr

    def __str__(self) -> str:
        """
        Customize the printed string for the dictionary.
        
        Features:
        - Alphabetically sorted by key
        - Long arrays (>10 elements) are shortened for display
        - Shows snapshot count
        - Only displays DescribedItem objects (not internal data)
        """
        custom_str = "\n" + "=" * 80 + "\n"
        custom_str += "DescribedDict Contents\n"
        custom_str += "=" * 80 + "\n\n"
        
        # Sort items alphabetically
        sorted_items = sorted(self.items())
        
        for key, val in sorted_items:
            # Only display DescribedItem objects (skip internal snapshot data)
            if not isinstance(val, DescribedItem):
                continue
            
            # Handle arrays/sequences separately for shortening
            if isinstance(val.value, (collections.abc.Sequence, np.ndarray)):
                # Check if it has length but is not a string
                if hasattr(val.value, "__len__") and not isinstance(val.value, str):
                    shortened_val = self.shorten_display(val.value)
                    custom_str += f"{key:<35} : {shortened_val}\n"
                else:
                    custom_str += f"{key:<35} : {val}\n"
            else:
                custom_str += f"{key:<35} : {val}\n"
        
        custom_str += "\n" + "-" * 80 + "\n"
        custom_str += f"Saved snapshot(s): {self.save_count}\n"
        custom_str += "=" * 80 + "\n"
        
        return custom_str

    # -------------------------------------------------------------------------
    # (Optional) curve simplification for very long profile arrays
    # -------------------------------------------------------------------------
    @staticmethod
    def simplify(
        x_arr: Union[np.ndarray, Sequence[float]],
        y_arr: Union[np.ndarray, Sequence[float]],
        nmin: int = 100,
        grad_inc: float = 1.0,
        keyname: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Heuristic downsampling of a curve y(x), preserving:
        - endpoints
        - large gradient-change points
        - sign-change points
        - points chosen by cumulative distance in y

        Intended use: reduce output size for long profile arrays before snapshotting.
        """
        x = np.asarray(x_arr)
        y = np.asarray(y_arr)

        if x.size == 0 or y.size == 0:
            return x, y
        if x.size != y.size:
            raise ValueError(f"simplify(): x and y must have same length for {keyname}.")
        if nmin >= x.size:
            return x, y
        nmin = max(int(nmin), 100)

        # Gradient-based features
        grad = np.gradient(y)
        eps = 1e-30
        denom = np.where(np.abs(grad[:-1]) < eps, np.sign(grad[:-1]) * eps + eps, grad[:-1])
        pct = np.diff(grad) / denom
        important_percent = np.where(np.abs(pct) > grad_inc)[0]
        important_sign = np.where(np.diff(np.sign(grad)) != 0)[0]

        # Distance-based sampling in y
        yrng = float(np.nanmax(y) - np.nanmin(y))
        if not np.isfinite(yrng) or yrng == 0:
            # flat curve: uniform indices
            idx = np.unique(np.linspace(0, x.size - 1, nmin).astype(int))
            return x[idx], y[idx]

        maxdist = yrng / nmin
        y_cum = np.cumsum(np.abs(np.diff(y)))
        bins = (y_cum / maxdist).astype(int)
        idx_diff = np.where(bins[:-1] != bins[1:])[0]

        # Combine all candidate indices
        merged = reduce(
            np.union1d,
            [
                np.array([0], dtype=int),
                important_percent.astype(int),
                important_sign.astype(int),
                idx_diff.astype(int),
                np.array([x.size - 1], dtype=int),
            ],
        )
        merged = np.unique(np.clip(merged, 0, x.size - 1))
        return x[merged], y[merged]

    # -------------------------------------------------------------------------
    # Internal helpers for snapshot serialization
    # -------------------------------------------------------------------------
    def _get_output_dir(self) -> Path:
        """
        Return output directory from params["path2output"].value.
        Raises a helpful error if missing.
        """
        try:
            return Path(self["path2output"].value)
        except KeyError as e:
            raise KeyError("save_snapshot()/flush() require params['path2output'].value") from e

    def _to_json_ready_value(self, val: Any) -> Any:
        """
        Convert an arbitrary value to something JSON-storable.
        All arrays are inlined as lists (no HDF5).
        """
        # None, primitives
        if val is None or isinstance(val, (str, float, int, bool)):
            return val

        # Numpy types
        if isinstance(val, (np.integer, np.floating, np.bool_)):
            return NpEncoder().default(val)

        # Arrays (numpy or sequences)
        if isinstance(val, np.ndarray):
            return val.tolist()

        if isinstance(val, (list, tuple)):
            # Already a list/tuple, but might contain numpy types
            return [self._to_json_ready_value(item) for item in val]

        # Fallback
        return val

    def _clean_for_snapshot(self, snap_id: int) -> Dict[str, Any]:
        """
        Build a JSON-ready snapshot dict of the current params.

        Includes special handling for certain long profile arrays (bubble_*, shell_grav_*)
        where we store a simplified representation (and sometimes log-space).
        """
        # Refresh excluded sets in case flags changed after insertion
        for k, item in self.items():
            if isinstance(item, DescribedItem):
                if item.exclude_from_snapshot:
                    self._excluded_keys.add(k)

        new_dict: Dict[str, Any] = {}
        eps = 1e-300  # used for safe log10()

        for key, item in self.items():
            # Skip excluded keys and non-DescribedItem values (shouldn't happen)
            if key in self._excluded_keys:
                continue
            if not isinstance(item, DescribedItem):
                continue

            val = item.value

            # -----------------------------------------------------------------
            # Special-case: bubble arrays (mirrors your previous behavior)
            # -----------------------------------------------------------------
            if key == "bubble_r_arr":
                # bubble_r_arr is stored alongside each derived bubble array as <key>_r_arr
                continue

            if key in ("bubble_T_arr", "bubble_n_arr"):
                x_arr = np.asarray(self["bubble_r_arr"].value)
                y_arr = np.log10(np.maximum(np.asarray(val), eps))
                new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)

                new_dict["log_" + key] = self._to_json_ready_value(np.asarray(new_y))
                new_dict[key + "_r_arr"] = self._to_json_ready_value(np.asarray(new_r))
                continue

            if key == "bubble_dTdr_arr":
                x_arr = np.asarray(self["bubble_r_arr"].value)
                v = np.asarray(val)
                y_arr = np.log10(np.maximum(np.abs(v), eps))
                new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)

                new_dict["log_" + key] = self._to_json_ready_value(np.asarray(new_y))
                new_dict[key + "_r_arr"] = self._to_json_ready_value(np.asarray(new_r))
                continue

            if key == "bubble_v_arr":
                x_arr = np.asarray(self["bubble_r_arr"].value)
                y_arr = np.asarray(val)
                new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)

                new_dict[key] = self._to_json_ready_value(np.asarray(new_y))
                new_dict[key + "_r_arr"] = self._to_json_ready_value(np.asarray(new_r))
                continue

            # -----------------------------------------------------------------
            # Special-case: shell gravity arrays (mirrors your previous behavior)
            # -----------------------------------------------------------------
            if key == "shell_grav_r":
                # saved together with simplified force arrays as "shell_grav_r"
                continue

            if key == "shell_grav_force_m":
                x_arr = np.asarray(self["shell_grav_r"].value)
                y_arr = np.log10(np.maximum(np.abs(np.asarray(val)), eps))
                new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)

                new_dict[key] = self._to_json_ready_value(np.asarray(new_y))
                new_dict["shell_grav_r"] = self._to_json_ready_value(np.asarray(new_r))
                continue

            # Default: store everything as JSON-ready values
            new_dict[key] = self._to_json_ready_value(val)

        return new_dict

    # -------------------------------------------------------------------------
    # Public API: snapshot saving and flushing to disk
    # -------------------------------------------------------------------------
    def save_snapshot(self) -> None:
        """
        Save the current state into self.previous_snapshot.

        Duplicate guard:
        - If the last saved snapshot has the same t_now or R2, it will not save again.
        """
        import logging
        logger = logging.getLogger(__name__)

        if self.save_count >= 1 and self.previous_snapshot:
            last = self.previous_snapshot.get(str(self.save_count - 1), {})
            try:
                t_now = self["t_now"].value
                r2 = self["R2"].value
                if ("t_now" in last and t_now == last["t_now"]) or ("R2" in last and r2 == last["R2"]):
                    logger.debug(f"Duplicate detected in save_snapshot at t = {t_now}. Snapshot not saved.")
                    return
            except KeyError:
                # If t_now/R2 not present, skip duplicate detection
                pass

        # Snapshot index is current save_count
        snap_id = self.save_count

        # Convert to JSON-friendly dict
        clean_dict = self._clean_for_snapshot(snap_id=snap_id)

        # Store in the "pending" snapshot buffer
        self.previous_snapshot[str(snap_id)] = clean_dict
        self.save_count += 1

        # Calculate progress toward next flush
        pending_count = len(self.previous_snapshot)
        until_flush = self.snapshot_interval - (self.save_count % self.snapshot_interval)
        if until_flush == self.snapshot_interval:
            until_flush = 0  # We're about to flush

        # Flush periodically
        if self.save_count % self.snapshot_interval == 0:
            logger.info(f"Snapshot #{self.save_count} saved. Flushing {pending_count} snapshots to disk...")
            self.flush()
            try:
                logger.info(f"All snapshots flushed to JSON at t = {self['t_now'].value:.6e} Myr")
            except KeyError:
                logger.info("All snapshots flushed to JSON.")
        else:
            try:
                logger.info(f"Snapshot #{self.save_count} saved at t = {self['t_now'].value:.6e} Myr "
                           f"({pending_count} pending, {until_flush} until flush)")
            except KeyError:
                logger.info(f"Snapshot #{self.save_count} saved ({pending_count} pending, {until_flush} until flush)")

    def flush(self) -> None:
        """
        Append pending snapshots to dictionary.jsonl (line-delimited JSON).

        Performance: O(pending_snapshots) - only writes new data, never reads existing file.
        This is a MASSIVE improvement over the old O(n²) behavior.

        Format
        ------
        Each line in dictionary.jsonl is one snapshot:
            Line 0: snapshot "0" as JSON object
            Line 1: snapshot "1" as JSON object
            ...

        Behavior
        --------
        - If flush_count == 0 and file exists: overwrite (fresh run)
        - Else: append new snapshots
        """
        import logging
        logger = logging.getLogger(__name__)

        path2output = self._get_output_dir()
        path2output.mkdir(parents=True, exist_ok=True)
        path2jsonl = path2output / "dictionary.jsonl"

        # Fresh run: delete existing file
        if self.flush_count == 0 and path2jsonl.exists():
            path2jsonl.unlink()
            logger.debug("Starting fresh run: deleted existing dictionary.jsonl")

        # Sort snapshot IDs to write in order
        snap_ids = sorted([int(k) for k in self.previous_snapshot.keys()])

        # Append each snapshot as one line
        mode = "a" if path2jsonl.exists() else "w"
        with open(path2jsonl, mode, encoding="utf-8") as f:
            for snap_id in snap_ids:
                snap_data = self.previous_snapshot[str(snap_id)]
                json_line = json.dumps(snap_data, cls=NpEncoder)
                f.write(json_line + "\n")

        logger.debug(f"Flushed {len(snap_ids)} snapshot(s) to dictionary.jsonl")

        # Update counters and clear pending buffer
        self.flush_count += 1
        self.previous_snapshot = {}

    # -------------------------------------------------------------------------
    # Public API: loading snapshots from disk
    # -------------------------------------------------------------------------
    @classmethod
    def load_snapshots(cls, path2output: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """
        Load dictionary.jsonl and return all snapshots.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Maps snapshot id (as str) -> snapshot content dict.
            Line N in file = snapshot str(N).
        """
        path2output = Path(path2output)
        path2jsonl = path2output / "dictionary.jsonl"

        if not path2jsonl.exists():
            raise FileNotFoundError(f"No dictionary.jsonl found in {path2output}")

        snapshots = {}
        with open(path2jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    snap_data = json.loads(line)
                    snapshots[str(idx)] = snap_data
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {idx}: {e}")
                    continue

        return snapshots

    @classmethod
    def load_snapshot(
        cls,
        path2output: Union[str, Path],
        snap_id: Union[int, str],
    ) -> "DescribedDict":
        """
        Load a single snapshot into a DescribedDict.

        This reconstructs:
        - scalars directly into DescribedItem(value)
        - list values back into numpy arrays

        Parameters
        ----------
        path2output : str or Path
            Directory containing dictionary.jsonl.
        snap_id : int or str
            Snapshot id to load.
        """
        path2output = Path(path2output)
        snapshots = cls.load_snapshots(path2output)

        sid = str(snap_id)
        if sid not in snapshots:
            raise KeyError(f"Snapshot {sid} not found. Available: {list(snapshots.keys())[:10]}...")

        snap = snapshots[sid]
        params = cls()

        # Put path2output back into the dictionary for downstream code that expects it
        params["path2output"] = DescribedItem(str(path2output), info="Output directory")

        # Reconstruct each key/value
        for key, val in snap.items():
            # Lists are converted back to numpy arrays
            if isinstance(val, list):
                params[key] = DescribedItem(np.asarray(val))
            else:
                params[key] = DescribedItem(val)

        return params

    @classmethod
    def load_latest_snapshot(cls, path2output: Union[str, Path]) -> "DescribedDict":
        """
        Convenience helper: load the snapshot with the largest integer id.
        """
        snapshots = cls.load_snapshots(path2output)
        if not snapshots:
            raise ValueError("No snapshots found in dictionary.jsonl")

        last_id = max(int(k) for k in snapshots.keys())
        return cls.load_snapshot(path2output, last_id)


# =============================================================================
# Debug snapshot: save raw params for crash debugging
# =============================================================================

DEBUG_SNAPSHOT_FILE = "debug_snapshot.json"

def save_debug_snapshot(params: DescribedDict, output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Save a RAW snapshot of all params for debugging.

    Unlike regular snapshots, this:
    - Saves ALL keys without any cleaning/simplification
    - Skips non-serializable objects (interpolators, etc.) gracefully
    - Always OVERWRITES the file (captures latest state before crash)
    - Can be called from anywhere without params['path2output']

    Parameters
    ----------
    params : DescribedDict or dict
        Parameter dictionary to snapshot
    output_path : str or Path, optional
        Directory to save to. If None, uses params['path2output'] or current dir.

    Returns
    -------
    Path
        Path to the saved snapshot file

    Usage
    -----
    # In your code, call periodically or before risky operations:
    from src._input.dictionary import save_debug_snapshot
    save_debug_snapshot(params)

    # Or with explicit path:
    save_debug_snapshot(params, "/tmp/debug")
    """
    import logging
    logger = logging.getLogger(__name__)

    # Determine output directory
    if output_path is not None:
        out_dir = Path(output_path)
    elif hasattr(params, '__getitem__') and 'path2output' in params:
        try:
            out_dir = Path(params['path2output'].value if hasattr(params['path2output'], 'value')
                          else params['path2output'])
        except:
            out_dir = Path(".")
    else:
        out_dir = Path(".")

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / DEBUG_SNAPSHOT_FILE

    # Build raw snapshot dict
    snapshot = {
        "_meta": {
            "type": "debug_snapshot",
            "description": "Raw parameter snapshot for debugging",
        }
    }

    skipped_keys = []

    for key, item in params.items():
        try:
            # Get the actual value
            if hasattr(item, 'value'):
                val = item.value
            else:
                val = item

            # Try to serialize
            if val is None or isinstance(val, (str, int, float, bool)):
                snapshot[key] = val
            elif isinstance(val, (np.integer, np.floating, np.bool_)):
                snapshot[key] = NpEncoder().default(val)
            elif isinstance(val, np.ndarray):
                snapshot[key] = val.tolist()
            elif isinstance(val, (list, tuple)):
                # Try to convert, may contain numpy types
                snapshot[key] = json.loads(json.dumps(val, cls=NpEncoder))
            elif callable(val):
                # Skip functions/interpolators
                skipped_keys.append(f"{key} (callable)")
                continue
            else:
                # Try generic serialization
                try:
                    snapshot[key] = json.loads(json.dumps(val, cls=NpEncoder))
                except (TypeError, ValueError):
                    skipped_keys.append(f"{key} ({type(val).__name__})")
                    continue

        except Exception as e:
            skipped_keys.append(f"{key} (error: {e})")
            continue

    # Add metadata about skipped keys
    if skipped_keys:
        snapshot["_meta"]["skipped_keys"] = skipped_keys

    # Write snapshot (always overwrite)
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, cls=NpEncoder, indent=2)

    logger.info(f"Debug snapshot saved to {snapshot_path} ({len(snapshot)-1} keys, {len(skipped_keys)} skipped)")

    return snapshot_path


def load_debug_snapshot(snapshot_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a debug snapshot for use in tests.

    Parameters
    ----------
    snapshot_path : str or Path
        Path to debug_snapshot.json file

    Returns
    -------
    dict
        Raw dictionary with values (not DescribedItem wrapped)
        Arrays are converted back to numpy arrays.

    Usage in tests
    --------------
    from src._input.dictionary import load_debug_snapshot

    # Load snapshot
    raw_params = load_debug_snapshot("/path/to/debug_snapshot.json")

    # Convert to MockParam format for tests
    params = {k: MockParam(v) for k, v in raw_params.items() if not k.startswith('_')}
    """
    snapshot_path = Path(snapshot_path)

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Debug snapshot not found: {snapshot_path}")

    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    # Convert lists back to numpy arrays
    result = {}
    for key, val in snapshot.items():
        if key.startswith('_'):
            # Skip metadata
            continue
        if isinstance(val, list):
            result[key] = np.asarray(val)
        else:
            result[key] = val

    return result


# =============================================================================
# Convenience helper: bulk updates
# =============================================================================
def updateDict(dictionary: DescribedDict, keys: Sequence[str], values: Sequence[Any]) -> None:
    """
    Bulk update helper:
        updateDict(params, ["R2", "t_now"], [R2, t])

    Expects keys to exist already in dictionary.
    """
    if len(keys) != len(values):
        raise ValueError("Length of keys must match length of values.")
    for key, val in zip(keys, values):
        dictionary[key].value = val


# =============================================================================
# Quick test example
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing DescribedDict")
    print("=" * 80)
    
    # Create params container
    params = DescribedDict()
    
    # Required for snapshotting
    params["path2output"] = DescribedItem("./_example_output", info="Output directory")
    
    # Typical scalar parameters
    params["t_now"] = DescribedItem(0.0, info="Current time", ori_units="Myr")
    params["R2"] = DescribedItem(1.0, info="Bubble radius", ori_units="pc")
    params["SB99_mass"] = DescribedItem(1e6, info="SB99 cluster mass", ori_units="Msun")
    
    # Example array parameters: one small, one large
    params["small_arr"] = DescribedItem(
        np.linspace(0, 1, 5), 
        info="Small array (will show all elements)",
        ori_units="dimensionless"
    )
    params["large_arr"] = DescribedItem(
        np.linspace(0, 100, 50), 
        info="Large array (will be shortened in display)",
        ori_units="pc"
    )
    
    # Test alphabetical sorting and array shortening
    print("\n--- Testing print(params) ---")
    print(params)
    
    # Test that actual array is not modified
    print("\n--- Verifying actual array length is unchanged ---")
    print(f"large_arr length: {len(params['large_arr'].value)}")
    print(f"First 5 elements: {params['large_arr'].value[:5]}")
    
    # Test numeric formatting without .value
    print("\n--- Testing numeric operations without .value ---")
    def format_e(n):
        a = "%E" % n
        return a.split("E")[0].rstrip("0").rstrip(".") + "e" + a.split("E")[1].strip("+").strip("0")
    
    SBmass_str = format_e(params["SB99_mass"])  # works because DescribedItem implements __float__
    print(f"SB99 mass formatted: {SBmass_str}")
    
    # Test saving snapshots
    print("\n--- Testing snapshot saving ---")
    for i in range(3):
        params["t_now"].value = i * 0.1
        params["R2"].value = 1.0 + 0.5 * i
        params.save_snapshot()
    
    # Flush pending snapshots
    params.flush()
    
    # Test loading
    print("\n--- Testing snapshot loading ---")
    loaded = DescribedDict.load_snapshot("./_example_output", 1)
    print(f"Loaded t_now: {loaded['t_now'].value}")
    print(f"Loaded R2: {loaded['R2'].value}")
    
    # Test array loading
    large = loaded["large_arr"].value
    print(f"Loaded large_arr shape: {large.shape}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
