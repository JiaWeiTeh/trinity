#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atomic read/write helpers for ``metadata.json``.

Two writers touch ``metadata.json`` over the lifetime of a run:

* ``DescribedDict.flush()`` writes the run-constants on the first flush
  (typically at run start, when the first batch of snapshots is saved).
* ``write_simulation_end()`` updates the file at run end to add the
  ``termination`` and ``final_state`` blocks.

This module exposes the small helpers both writers share so the write
path is uniform — atomic temp-file + rename, identical JSON formatting
(pretty-printed, key order preserved), defensive serialization, and
a read-modify-write helper for end-of-run merges.

The helpers stay dependency-light: only ``json``, ``pathlib``, ``os``,
and ``numpy`` (the latter only for ``NpEncoder``).  Importing this
module must not pull in TRINITY's runtime container.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from trinity._output.run_constants import METADATA_FILENAME, METADATA_VERSION

logger = logging.getLogger(__name__)


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that coerces numpy scalars / arrays to plain Python.

    Duplicates ``trinity._input.dictionary.NpEncoder`` to avoid a circular
    import (this module is imported by ``simulation_end``, which is
    imported by ``dictionary``).  The two encoders MUST stay in sync.
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


def read_metadata(run_dir: Path) -> Dict[str, Any]:
    """
    Parse ``<run_dir>/metadata.json`` and return the dict.

    Returns ``{}`` if the file is absent or malformed (the latter is
    logged at WARNING level).  Callers that need to distinguish
    absent-vs-corrupt should check existence themselves.
    """
    path = Path(run_dir) / METADATA_FILENAME
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}


def write_metadata_atomic(run_dir: Path, payload: Dict[str, Any]) -> None:
    """
    Write ``payload`` to ``<run_dir>/metadata.json`` atomically.

    The file is first written to a sibling ``.tmp`` and renamed in
    place; if the process dies mid-write, the existing file (if any)
    survives.  Output is pretty-printed (``indent=2``) for human
    readability and keys are emitted in insertion order.

    Caller is responsible for ensuring ``run_dir`` exists.
    """
    path = Path(run_dir) / METADATA_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, cls=_NpEncoder, indent=2, sort_keys=False)
    os.replace(tmp, path)


def update_metadata_atomic(run_dir: Path, **block_updates: Any) -> None:
    """
    Read ``metadata.json``, merge ``block_updates`` at the top level,
    and atomically rewrite.

    Used by end-of-run writers (``write_simulation_end``) to add the
    ``termination`` and ``final_state`` blocks without disturbing the
    run-constants written earlier by ``flush()``.  Each keyword
    argument is treated as a single top-level key whose value (dict,
    list, scalar) replaces any existing entry under that name.

    If ``metadata.json`` does not exist (the run terminated before
    any flush wrote it), a minimal file is created containing only
    ``_metadata_version`` and the supplied blocks — readers will then
    return ``None`` for run-constants that never made it to disk,
    which is the correct semantics for an aborted run.

    Defensive serialization: any value that fails ``json.dumps`` is
    logged at WARNING and silently dropped from the merged payload
    rather than crashing the write.
    """
    existing = read_metadata(run_dir)
    if not existing:
        existing = {"_metadata_version": METADATA_VERSION}

    for key, value in block_updates.items():
        try:
            json.dumps(value, cls=_NpEncoder)
        except (TypeError, ValueError) as e:
            logger.warning(
                "metadata.json: skipping non-serializable block %r (%s)",
                key, e,
            )
            continue
        existing[key] = value

    write_metadata_atomic(run_dir, existing)
