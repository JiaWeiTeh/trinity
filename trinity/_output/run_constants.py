#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-constants schema: which keys live in ``metadata.json``.

``RUN_CONST_KEYS`` and ``METADATA_EXCLUDE`` are DERIVED from the
ParamSpec registry (``trinity._input.registry``) — the single source of
truth — via ``run_const_keys()`` / ``metadata_exclude_keys()``.  This
module re-exports them under their historical names plus the
output-schema constants (``DROPPED_IN_V2``, ``METADATA_VERSION``,
``RESERVED_TOP_LEVEL_KEYS``, ``FINAL_STATE_EXCLUDE_ARRAYS``) that are
genuinely output concerns, not input-spec data.  It is imported by both
the writer (``trinity._input.dictionary``) and the readers
(``trinity._output.trinity_reader``, the cloudy run-loader) so the two
sides agree on what gets stripped from per-snapshot dicts and what
gets rehydrated on load.

Contract
--------
The derived keys are *run-constants*: input parameters or set-once
derived values that do not change after phase 0 of a simulation.  They
are written exactly once per run, in ``<run_dir>/metadata.json``, and
stripped from every per-snapshot dictionary in ``dictionary.jsonl``.
The reader rehydrates them into every snapshot's ``data`` dict via
``setdefault`` (so any per-snapshot value, when present, takes
precedence).

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

Version history
~~~~~~~~~~~~~~~
* v1 — PR2: 13 keys (10 scalars + 3 ``initial_cloud_*_arr`` arrays).
* v2 — Phase 1: ~57 scalars/strings/bools covering every constant-
  through-run parameter.  ``initial_cloud_*_arr`` dropped; readers
  reconstruct on demand via ``TrinityOutput.initial_cloud_profile()``.
* v3 — Phase 2: adds top-level ``termination`` and ``final_state``
  blocks written at run end by ``write_simulation_end()``.  These
  blocks are NOT rehydrated into snapshots (see
  ``RESERVED_TOP_LEVEL_KEYS`` below) — they surface via the
  ``TrinityOutput.termination`` / ``.final_state`` properties.
* v4 — Phase 5: adds top-level ``termination_debug`` block (the
  last-2-snapshot comparison data formerly written to
  ``termination_debug.txt``).  ``write_simulation_end`` and
  ``write_termination_debug_report`` no longer write text files;
  ``read_param.write_summary`` likewise no longer writes
  ``<run>_summary.txt``.  Output directory shrinks from 7 files to 4
  (``.param`` + ``trinity_*.log`` + ``dictionary.jsonl`` +
  ``metadata.json``).  Legacy text-parse readers stay (with
  ``DeprecationWarning``) for one cycle, then are removed in Phase 6.
"""

from __future__ import annotations

from trinity._input.registry import metadata_exclude_keys, run_const_keys

# Run-const / metadata-exclude membership is DERIVED from the ParamSpec
# registry (the single source of truth), not hand-curated here.  Each
# spec declares ``run_const`` / ``metadata_exclude``; these helpers
# project those flags.  This is why the four legacy stale entries
# (``expansionBeyondCloud`` in run-consts; ``SB99_data`` / ``SB99f`` /
# ``path_sps`` in the exclude set) are gone — they have no spec, so the
# derivation cannot emit them.
#
# RUN_CONST_KEYS: keys written once to metadata.json (constant after
# phase 0).  Ordering follows registry (SPECS) order.
RUN_CONST_KEYS: tuple[str, ...] = run_const_keys()

# METADATA_EXCLUDE: keys that look constant but must NOT land in
# metadata.json — absolute paths, loaded function tables/interpolators,
# and empty-array placeholders whose real data lives in the per-snapshot
# stream.  The writer also skips them defensively.
METADATA_EXCLUDE: frozenset[str] = metadata_exclude_keys()

# Keys dropped from v1 → v2 because they are reconstructible on
# demand from other run-constants.  Readers can fall back to inline
# arrays in legacy v1 files via ``TrinityOutput.initial_cloud_profile()``.
DROPPED_IN_V2: frozenset[str] = frozenset({
    "initial_cloud_r_arr",
    "initial_cloud_n_arr",
    "initial_cloud_m_arr",
})

# Filename of the per-run metadata sidecar, sibling to
# ``dictionary.jsonl`` in each run output directory.
METADATA_FILENAME: str = "metadata.json"

# Schema version of ``metadata.json``.  Increment whenever the
# layout changes in a backwards-incompatible way.
METADATA_VERSION: int = 4

# Top-level keys in ``metadata.json`` that are NOT rehydrated into
# every snapshot's data dict.  Three reasons a key lives up here:
#
#   * ``_metadata_version`` — describes the metadata file itself.
#   * ``termination`` / ``final_state`` — Phase-2 blocks surfaced via
#     ``TrinityOutput.termination`` / ``.final_state`` properties.
#     Rehydrating them into each snapshot would smear the run-end
#     state into every timestep, which is misleading.
#   * ``termination_debug`` — Phase-5 block; last-2-snapshot
#     comparison written by ``write_termination_debug_report``.
#     Replaces the legacy ``termination_debug.txt`` file.
RESERVED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "_metadata_version",
    "termination",
    "final_state",
    "termination_debug",
})

# Keys excluded from the ``final_state`` block.  Long per-snapshot
# arrays already live in ``dictionary.jsonl`` (the last line is the
# full final-state profile) — duplicating them in metadata.json would
# bloat the file by ~10-50 KB with no information gain.  Anything
# scalar/string/bool flows through to ``final_state``.
FINAL_STATE_EXCLUDE_ARRAYS: frozenset[str] = frozenset({
    "bubble_T_arr_r_arr", "log_bubble_T_arr",
    "bubble_n_arr_r_arr", "log_bubble_n_arr",
    "bubble_dTdr_arr_r_arr", "log_bubble_dTdr_arr",
    "bubble_v_arr", "bubble_v_arr_r_arr",
    "shell_r_arr", "log_shell_n_arr",
    "shell_grav_r", "shell_grav_force_m",
})


def metadata_keys_to_rehydrate(metadata: dict) -> dict:
    """
    Return ``metadata`` with the reserved top-level keys removed.

    Used by the reader to take a freshly-loaded ``metadata.json``
    and produce the dict whose entries should be merged into every
    snapshot.  Reserved entries (``_metadata_version``, ``termination``,
    ``final_state``) are surfaced via dedicated ``TrinityOutput``
    properties instead.
    """
    return {k: v for k, v in metadata.items()
            if k not in RESERVED_TOP_LEVEL_KEYS}
