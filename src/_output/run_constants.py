#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-constants schema: which keys live in ``metadata.json``.

A small, dependency-free module imported by both the writer
(``src._input.dictionary``) and the readers
(``src._output.trinity_reader``, ``src._input.dictionary.load_snapshots``).
Defining the list in one place ensures the two sides agree on what
gets stripped from per-snapshot dicts and what gets rehydrated on
load.

Contract
--------
The keys listed below are *run-constants*: input parameters or
set-once derived values that do not change after phase 0 of a
simulation.  They are written exactly once per run, in
``<run_dir>/metadata.json``, and stripped from every per-snapshot
dictionary in ``dictionary.jsonl``.  The reader rehydrates them
into every snapshot's ``data`` dict via ``setdefault`` (so any
per-snapshot value, when present, takes precedence).

State-machine flags that *happen* to be constant in a particular
run (``EndSimulationDirectly``, ``isCollapse``, ``isDissolved``,
``is_phiDepleted``, ``bubble_dMdtGuess``, ``t_next``,
``shell_interpolate_massDot``, ``F_ISM``, …) are deliberately NOT
listed here — they represent runtime state that varies across
runs even when constant within one run.

Forward compatibility
---------------------
``METADATA_VERSION`` is written into ``metadata.json`` so the reader
can detect (and adapt to) future schema changes.  The version field
is consumed and discarded by the reader before rehydrate.
"""

from __future__ import annotations

# All keys that get factored out of per-snapshot dicts and into
# ``metadata.json``.  Ordering mirrors the conceptual grouping:
# identifiers, scalar inputs, set-once derived scalars, then arrays.
RUN_CONST_KEYS: tuple[str, ...] = (
    # identifiers / inputs
    "model_name",
    "mCloud",
    "dens_profile",
    "densPL_alpha",
    "nCore",
    "nISM",
    "rCore",
    # set-once derived
    "rCloud",
    "nEdge",
    "tSF",
    # initial cloud profile arrays (the dominant size win)
    "initial_cloud_r_arr",
    "initial_cloud_n_arr",
    "initial_cloud_m_arr",
)

# Filename of the per-run metadata sidecar, sibling to
# ``dictionary.jsonl`` in each run output directory.
METADATA_FILENAME: str = "metadata.json"

# Schema version of ``metadata.json``.  Increment whenever the
# layout changes in a backwards-incompatible way.
METADATA_VERSION: int = 1

# Reserved key names inside ``metadata.json`` that are NOT
# rehydrated into snapshots (they describe the metadata file
# itself, not the simulation).
_RESERVED_KEYS: frozenset[str] = frozenset({"_metadata_version"})


def metadata_keys_to_rehydrate(metadata: dict) -> dict:
    """
    Return ``metadata`` with the reserved internal keys removed.

    Used by the reader to take a freshly-loaded ``metadata.json``
    and produce the dict whose entries should be merged into every
    snapshot.
    """
    return {k: v for k, v in metadata.items() if k not in _RESERVED_KEYS}
