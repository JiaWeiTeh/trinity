#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Wed Jul 26 15:21:52 2023

@author: Jia Wei Teh

Purpose
-------
Provide a "params" container that behaves like a dictionary of objects:
    params["R2"].value
    params["R2"].value = 10.0

…and supports saving snapshots efficiently:
- Scalars + small arrays: stored inline in JSON (dictionary.json)
- Large arrays: stored in HDF5 (arrays.h5) and JSON contains a small reference pointer

This design keeps simulation code clean (dictionary access) while keeping output sizes
and write-time manageable for large profiles.

Files written to params["path2output"].value
-------------------------------------------
dictionary.json : snapshot index -> { key: scalar / list / {"__h5__": {...}} }
arrays.h5       : actual large array data at /snapshots/<snap_id>/<key>

Loading
-------
params = DescribedDict.load_snapshot(path2output, snap_id)
arr = params["initial_cloud_n_arr"].value   # returns numpy array (loads lazily from HDF5)
"""

import collections.abc
import json
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

# h5py is required for HDF5 storage of large arrays
try:
    import h5py
except Exception as e:  # pragma: no cover
    raise ImportError("This script requires h5py for HDF5 array storage.") from e


# =============================================================================
# JSON helper: encode numpy scalar types (not arrays)
# =============================================================================
class NpEncoder(json.JSONEncoder):
    """
    JSON encoder that converts numpy scalar types to plain Python scalars.
    We intentionally do NOT handle numpy arrays here because we either:
      - inline small arrays as lists ourselves, or
      - store large arrays in HDF5 and only store a reference dict in JSON.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# HDF5 reference format + lightweight proxy
# =============================================================================

# Key used in JSON to indicate "this value is an HDF5 reference"
H5_REF_KEY = "__h5__"


def is_h5_ref(x: Any) -> bool:
    """
    Return True if x looks like an HDF5 reference dict of the form:
        {"__h5__": {"file": "arrays.h5", "dataset": "/snapshots/0/some_key"}}
    """
    return isinstance(x, dict) and (H5_REF_KEY in x) and isinstance(x[H5_REF_KEY], dict)


def make_h5_ref(file: str, dataset: str) -> dict:
    """
    Make a JSON-storable reference dict that points to an array stored in HDF5.

    Parameters
    ----------
    file : str
        HDF5 filename (usually "arrays.h5") relative to output directory.
    dataset : str
        HDF5 dataset path inside the file (e.g. "/snapshots/12/initial_cloud_n_arr").
    """
    return {H5_REF_KEY: {"file": file, "dataset": dataset}}


@dataclass
class H5ArrayProxy:
    """
    Lazy-loading proxy for a numpy array stored in HDF5.

    Why a proxy?
    ------------
    - When loading snapshots, you don't always want to load all arrays immediately.
    - This object stores (file_path, dataset) and loads the array only when needed.
    - If cache=True, it stores the loaded array in memory after the first read.

    Notes
    -----
    DescribedItem.value will "materialize" this proxy automatically.
    """
    file_path: Path                 # absolute path to arrays.h5
    dataset: str                    # dataset path within the HDF5 file
    cache: bool = True              # whether to keep the array in memory once loaded
    _cached: Optional[np.ndarray] = None  # internal cache

    def asarray(self) -> np.ndarray:
        """Load and return the referenced array (possibly from cache)."""
        if self.cache and self._cached is not None:
            return self._cached

        # Open file read-only and load dataset into memory
        with h5py.File(self.file_path, "r") as f:
            arr = f[self.dataset][...]

        if self.cache:
            self._cached = arr
        return arr

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Allows numpy to treat this object as an array if passed into np.asarray().
        """
        arr = self.asarray()
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

    def __repr__(self) -> str:
        return f"H5ArrayProxy(file='{self.file_path.name}', dataset='{self.dataset}')"


class H5ArrayStore:
    """
    Minimal helper class to write arrays to a single HDF5 file.

    Dataset layout
    --------------
    Arrays are written at:
        /snapshots/<snap_id>/<key>

    Example:
        snap_id = 12
        key     = "initial_cloud_n_arr"
        dataset = "/snapshots/12/initial_cloud_n_arr"
    """
    def __init__(self, path2output: Path, arrays_filename: str = "arrays.h5"):
        self.path2output = Path(path2output)
        self.arrays_filename = arrays_filename

    @property
    def file_path(self) -> Path:
        """Absolute path to the HDF5 file."""
        return self.path2output / self.arrays_filename

    def ensure_file(self, reset: bool = False) -> None:
        """
        Ensure output directory exists and HDF5 file exists.

        reset=True deletes the existing arrays file before creating a new one.
        Useful when starting a "fresh" run that should overwrite previous array outputs.
        """
        self.path2output.mkdir(parents=True, exist_ok=True)

        # If requested, delete old file first
        if reset and self.file_path.exists():
            self.file_path.unlink()

        # Create empty file if missing
        if not self.file_path.exists():
            with h5py.File(self.file_path, "w"):
                pass

    def write_array(
        self,
        snap_id: int,
        key: str,
        array: np.ndarray,
        overwrite: bool = True,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
    ) -> str:
        """
        Write array to HDF5 and return its dataset path.

        Parameters
        ----------
        snap_id : int
            Snapshot index (0,1,2,...).
        key : str
            Dictionary key (used as dataset name).
        array : np.ndarray
            Array to store.
        overwrite : bool
            If True, delete existing dataset and rewrite it.
        compression, compression_opts
            Optional HDF5 compression settings (e.g. compression="gzip", compression_opts=4).

        Returns
        -------
        str
            HDF5 dataset path (e.g. "/snapshots/3/bubble_T_arr").
        """
        self.ensure_file(reset=False)

        # Dataset path within HDF5
        ds_path = f"/snapshots/{snap_id}/{key}"

        with h5py.File(self.file_path, "a") as f:
            # Make sure the snapshot group exists
            f.require_group(f"/snapshots/{snap_id}")

            # If dataset exists, delete or reuse depending on overwrite
            if ds_path in f:
                if overwrite:
                    del f[ds_path]
                else:
                    return ds_path

            # Write dataset
            f.create_dataset(
                ds_path,
                data=np.asarray(array),
                compression=compression,
                compression_opts=compression_opts,
            )

        return ds_path


# =============================================================================
# DescribedItem: the stored object at params[key]
# =============================================================================
class DescribedItem:
    """
    Container for a value (scalar/array/proxy) + light metadata.

    Key behavior
    ------------
    - If internal value is H5ArrayProxy, .value returns a numpy array (materialized).
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
    isPersistent : bool
        Reserved flag you can use for special treatment later.
    """

    __slots__ = ("_value", "info", "ori_units", "exclude_from_snapshot", "isPersistent")

    def __init__(
        self,
        value: Any = None,
        info: Optional[str] = None,
        ori_units: Optional[str] = None,
        exclude_from_snapshot: bool = False,
        isPersistent: bool = False,
    ):
        self._value = value
        self.info = info
        self.ori_units = ori_units
        self.exclude_from_snapshot = exclude_from_snapshot
        self.isPersistent = isPersistent

    @property
    def value(self) -> Any:
        """
        Return the stored value.

        If value is an H5ArrayProxy, it is loaded from disk (and possibly cached).
        """
        if isinstance(self._value, H5ArrayProxy):
            return self._value.asarray()
        return self._value

    @value.setter
    def value(self, v: Any) -> None:
        """Set the underlying value (scalar, array, or proxy)."""
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
    - Scalars: always stored in JSON
    - Arrays:
        * if array.size <= json_array_max_elems => stored inline in JSON as list
        * else => stored in arrays.h5, and JSON stores {"__h5__": {...}} reference

    Required key before saving
    --------------------------
    params["path2output"].value must exist and point to the output directory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Snapshot counters
        self.save_count: int = 0                  # how many snapshots have been saved in-memory
        self.snapshot_interval: int = 25          # flush every N snapshots
        self.previous_snapshot: Dict[str, Dict[str, Any]] = {}  # pending snapshots not yet flushed
        self.flush_count: int = 0                 # number of flush() calls (used for "fresh run" logic)

        # Storage policy knobs
        self.json_array_max_elems: int = 200      # threshold for inlining arrays into JSON
        self.arrays_filename: str = "arrays.h5"   # HDF5 file for large arrays
        self.h5_compression: Optional[str] = None       # e.g. "gzip"
        self.h5_compression_opts: Optional[int] = None  # e.g. 4..9

        # Key flags
        self._excluded_keys: set[str] = set()     # keys to omit from snapshots
        self._persistent_keys: set[str] = set()   # reserved for future logic (e.g. always store)

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

        # Track exclusions/persistence based on item flags
        if value.exclude_from_snapshot:
            self._excluded_keys.add(key)
        if value.isPersistent:
            self._persistent_keys.add(key)

        super().__setitem__(key, value)

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

    def _array_store(self) -> H5ArrayStore:
        """Create an H5ArrayStore bound to the current output directory."""
        return H5ArrayStore(self._get_output_dir(), arrays_filename=self.arrays_filename)

    def _should_inline_array_in_json(self, arr: np.ndarray) -> bool:
        """Return True if this array is small enough to inline into JSON as a list."""
        return arr.size <= int(self.json_array_max_elems)

    def _to_json_ready_value(self, snap_id: int, key: str, val: Any) -> Any:
        """
        Convert an arbitrary value to something JSON-storable:
        - scalars: return as-is (or via NpEncoder for numpy scalars)
        - arrays:
            * small arrays => list in JSON
            * large arrays => store in HDF5 and return {"__h5__": {...}} reference
        """
        # Primitive scalar types are safe to put directly in JSON
        if isinstance(val, (str, float, int, bool)) or val is None:
            return val

        # Numpy scalar -> plain Python scalar
        if isinstance(val, (np.integer, np.floating, np.bool_)):
            return NpEncoder().default(val)

        # Array-like values
        if isinstance(val, np.ndarray) or (
            isinstance(val, collections.abc.Sequence) and not isinstance(val, (str, bytes))
        ):
            arr = np.asarray(val)

            # Small arrays are human-readable if stored inline as JSON lists
            if self._should_inline_array_in_json(arr):
                return arr.tolist()

            # Large arrays: store in HDF5, JSON gets only a pointer
            store = self._array_store()

            # Reset arrays.h5 only for the first snapshot of a new run
            store.ensure_file(reset=(self.flush_count == 0 and snap_id == 0))

            ds_path = store.write_array(
                snap_id=snap_id,
                key=key,
                array=arr,
                overwrite=True,
                compression=self.h5_compression,
                compression_opts=self.h5_compression_opts,
            )
            return make_h5_ref(file=store.file_path.name, dataset=ds_path)

        # Fallback: attempt JSON encoding (may fail for complex objects)
        return val

    def _clean_for_snapshot(self, snap_id: int) -> Dict[str, Any]:
        """
        Build a JSON-ready snapshot dict of the current params.

        Also includes special handling for certain long profile arrays (bubble_* etc.)
        where you store a simplified representation (and sometimes log-space).
        """
        # Refresh excluded/persistent sets in case flags changed after insertion
        for k, item in self.items():
            if isinstance(item, DescribedItem):
                if item.exclude_from_snapshot:
                    self._excluded_keys.add(k)
                if item.isPersistent:
                    self._persistent_keys.add(k)

        new_dict: Dict[str, Any] = {}
        eps = 1e-300  # used for safe log10()

        for key, item in self.items():
            # Skip excluded keys and non-DescribedItem values (shouldn't happen)
            if key in self._excluded_keys:
                continue
            if not isinstance(item, DescribedItem):
                continue

            # Always snapshot the actual stored value (materializes proxies)
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

                new_dict["log_" + key] = self._to_json_ready_value(snap_id, "log_" + key, np.asarray(new_y))
                new_dict[key + "_r_arr"] = self._to_json_ready_value(snap_id, key + "_r_arr", np.asarray(new_r))
                continue

            if key == "bubble_dTdr_arr":
                x_arr = np.asarray(self["bubble_r_arr"].value)
                v = np.asarray(val)
                y_arr = np.log10(np.maximum(np.abs(v), eps))
                new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)

                new_dict["log_" + key] = self._to_json_ready_value(snap_id, "log_" + key, np.asarray(new_y))
                new_dict[key + "_r_arr"] = self._to_json_ready_value(snap_id, key + "_r_arr", np.asarray(new_r))
                continue

            if key == "bubble_v_arr":
                x_arr = np.asarray(self["bubble_r_arr"].value)
                y_arr = np.asarray(val)
                new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)

                new_dict[key] = self._to_json_ready_value(snap_id, key, np.asarray(new_y))
                new_dict[key + "_r_arr"] = self._to_json_ready_value(snap_id, key + "_r_arr", np.asarray(new_r))
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

                new_dict[key] = self._to_json_ready_value(snap_id, key, np.asarray(new_y))
                new_dict["shell_grav_r"] = self._to_json_ready_value(snap_id, "shell_grav_r", np.asarray(new_r))
                continue

            # Default: store scalars inline; arrays based on size policy (JSON or HDF5)
            new_dict[key] = self._to_json_ready_value(snap_id, key, val)

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
        if self.save_count >= 1 and self.previous_snapshot:
            last = self.previous_snapshot.get(str(self.save_count - 1), {})
            try:
                t_now = self["t_now"].value
                r2 = self["R2"].value
                if ("t_now" in last and t_now == last["t_now"]) or ("R2" in last and r2 == last["R2"]):
                    print(f"duplicate detected in save_snapshot at t = {t_now}. Snapshot not saved.")
                    return
            except KeyError:
                # If t_now/R2 not present, skip duplicate detection
                pass

        # Snapshot index is current save_count
        snap_id = self.save_count

        # Convert to JSON/HDF5-friendly dict
        clean_dict = self._clean_for_snapshot(snap_id=snap_id)

        # Store in the "pending" snapshot buffer
        self.previous_snapshot[str(snap_id)] = clean_dict
        self.save_count += 1

        # Flush periodically
        if self.save_count % self.snapshot_interval == 0:
            print("flushing dictionary...")
            self.flush()
            try:
                print("All snapshots flushed to JSON at t = ", self["t_now"].value)
            except KeyError:
                print("All snapshots flushed to JSON.")
        else:
            try:
                print("Current snapshot saved at t = ", self["t_now"].value)
            except KeyError:
                print("Current snapshot saved.")

    def flush(self) -> None:
        """
        Write pending snapshots in self.previous_snapshot into dictionary.json.

        Behavior
        --------
        - If dictionary.json doesn't exist OR flush_count == 0:
            initialise/overwrite the file (fresh run behavior)
        - Else:
            merge existing JSON with pending snapshots

        After a successful flush, previous_snapshot is cleared.
        """
        path2output = self._get_output_dir()
        path2output.mkdir(parents=True, exist_ok=True)
        path2json = path2output / "dictionary.json"

        # Fresh run: blank the JSON file
        if (not path2json.exists()) or (path2json.exists() and self.flush_count == 0):
            path2json.write_text("", encoding="utf-8")
            print("Initialising JSON file for saving purpose...")

        # Load existing content (if any)
        load_dict: Dict[str, Any] = {}
        try:
            raw = path2json.read_text(encoding="utf-8").strip()
            if raw:
                load_dict = json.loads(raw)
        except json.decoder.JSONDecodeError as e:
            # If file is empty/corrupt, start fresh
            print(f"Exception: {e} caught; JSON empty/corrupt. Overwriting with fresh content.")
            load_dict = {}
        except Exception as e:
            print(f"Something else went wrong in .flush(): {e}")
            sys.exit(1)

        # Merge old + new snapshots
        combined = {**load_dict, **self.previous_snapshot}

        print("Updating dictionary in .flush()")
        path2json.write_text(json.dumps(combined, cls=NpEncoder, indent=2), encoding="utf-8")

        # Update counters and clear pending buffer
        self.flush_count += 1
        self.previous_snapshot = {}

    # -------------------------------------------------------------------------
    # Public API: loading snapshots from disk
    # -------------------------------------------------------------------------
    @classmethod
    def load_snapshots(cls, path2output: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """
        Load dictionary.json and return the raw snapshot dictionary.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Outer dict maps snapshot id (as str) -> snapshot content dict.
            Arrays may be stored as:
              - list (if inlined), or
              - {"__h5__": {...}} reference
        """
        path2output = Path(path2output)
        path2json = path2output / "dictionary.json"
        if not path2json.exists():
            raise FileNotFoundError(f"No dictionary.json found in {path2output}")

        raw = path2json.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)

    @classmethod
    def load_snapshot(
        cls,
        path2output: Union[str, Path],
        snap_id: Union[int, str],
        *,
        cache_arrays: bool = True,
    ) -> "DescribedDict":
        """
        Load a single snapshot into a DescribedDict.

        This reconstructs:
        - scalars directly into DescribedItem(value)
        - list values back into numpy arrays
        - HDF5 references into H5ArrayProxy (lazy loaded)

        Parameters
        ----------
        path2output : str or Path
            Directory containing dictionary.json and arrays.h5.
        snap_id : int or str
            Snapshot id to load.
        cache_arrays : bool
            If True, arrays loaded from HDF5 are cached in memory after first load.
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
            if is_h5_ref(val):
                # Convert reference dict -> lazy proxy
                ref = val[H5_REF_KEY]
                proxy = H5ArrayProxy(
                    file_path=path2output / ref["file"],
                    dataset=ref["dataset"],
                    cache=cache_arrays,
                )
                params[key] = DescribedItem(proxy)
            else:
                # Inlined arrays were stored as lists; convert back to numpy arrays
                if isinstance(val, list):
                    params[key] = DescribedItem(np.asarray(val))
                else:
                    params[key] = DescribedItem(val)

        return params

    @classmethod
    def load_latest_snapshot(cls, path2output: Union[str, Path], *, cache_arrays: bool = True) -> "DescribedDict":
        """
        Convenience helper: load the snapshot with the largest integer id.
        """
        snapshots = cls.load_snapshots(path2output)
        if not snapshots:
            raise ValueError("No snapshots found in dictionary.json")

        last_id = max(int(k) for k in snapshots.keys())
        return cls.load_snapshot(path2output, last_id, cache_arrays=cache_arrays)


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
# Quickstart example (commented out)
# =============================================================================
# if __name__ == "__main__":
#     # --- Create params container ---
#     params = DescribedDict()
#
#     # Required for snapshotting
#     params["path2output"] = DescribedItem("./_example_output", info="Output directory")
#
#     # Typical scalar parameters
#     params["t_now"] = DescribedItem(0.0, info="Current time", ori_units="Myr")
#     params["R2"] = DescribedItem(1.0, info="Bubble radius", ori_units="pc")
#     params["SB99_mass"] = DescribedItem(1e6, info="SB99 cluster mass", ori_units="Msun")
#
#     # Example array parameters: one small, one big
#     params["small_arr"] = DescribedItem(np.linspace(0, 1, 20), info="Small array (inlined to JSON)")
#     params["initial_cloud_n_arr"] = DescribedItem(np.ones(5000), info="Big array (stored in HDF5)")
#
#     # Optional tuning: force almost all arrays to HDF5
#     # params.json_array_max_elems = 0
#
#     # Optional tuning: HDF5 compression
#     # params.h5_compression = "gzip"
#     # params.h5_compression_opts = 4
#
#     # --- Use numeric formatting without .value ---
#     def format_e(n):
#         a = "%E" % n
#         return a.split("E")[0].rstrip("0").rstrip(".") + "e" + a.split("E")[1].strip("+").strip("0")
#
#     SBmass_str = format_e(params["SB99_mass"])  # works because DescribedItem implements __float__
#     print("SB99 mass string:", SBmass_str)
#
#     # --- Save a couple of snapshots ---
#     for i in range(3):
#         params["t_now"].value = i * 0.1
#         params["R2"].value = 1.0 + 0.5 * i
#         params.save_snapshot()
#
#     # Flush any pending snapshots to dictionary.json
#     params.flush()
#
#     # --- Load a snapshot back ---
#     loaded = DescribedDict.load_snapshot("./_example_output", 1)
#     print("Loaded t_now:", loaded["t_now"].value)
#     print("Loaded R2:", loaded["R2"].value)
#
#     # Big arrays load lazily from arrays.h5 when .value is accessed
#     cloud = loaded["initial_cloud_n_arr"].value
#     print("Loaded initial_cloud_n_arr shape:", cloud.shape)
