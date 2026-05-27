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
"""

from __future__ import annotations

# All keys that get factored out of per-snapshot dicts and into
# ``metadata.json``.  Ordering mirrors the conceptual grouping:
# identifiers → physical inputs → solver/control inputs → SB99 inputs
# → feedback inputs → logging inputs → derived (set-once at init).
RUN_CONST_KEYS: tuple[str, ...] = (
    # --- Identifiers / cloud inputs ---
    "model_name",
    "mCloud",
    "sfe",
    "ZCloud",
    "include_PHII",
    "dens_profile",
    "densPL_alpha",
    "nCore",
    "nISM",
    "rCore",

    # --- Run-control inputs ---
    "allowShellDissolution",
    "stop_t_diss",
    "stop_r",
    "stop_v",
    "stop_t",
    "coll_r",
    "expansionBeyondCloud",
    "use_adaptive_solver",
    "adiabaticOnlyInCore",
    "immediate_leak",

    # --- SB99 inputs ---
    "SB99_BHCUT",
    "SB99_mass",
    "SB99_rotation",

    # --- Feedback inputs ---
    "FB_mColdSNFrac",
    "FB_mColdWindFrac",
    "FB_thermCoeffSN",
    "FB_thermCoeffWind",
    "FB_vSN",

    # --- Solver/physics tuning ---
    "phaseSwitch_LlossLgain",
    "bubble_xi_Tb",

    # --- Logging inputs ---
    "output_format",
    "log_level",
    "log_colors",
    "log_console",
    "log_file",

    # --- BE-specific inputs (only populated for dens_profile="densBE") ---
    "densBE_Omega",

    # --- Set-once derived scalars ---
    "rCloud",
    "nEdge",
    "tSF",
    "mCluster",
    "mu_atom",
    "mu_ion",
    "mu_mol",
    "mu_convert",
    "TShell_ion",
    "TShell_neu",
    "caseB_alpha",
    "C_thermal",
    "dust_KappaIR",
    "dust_noZ",
    "dust_sigma",
    "gamma_adia",

    # --- BE-specific derived (only populated for densBE) ---
    "densBE_Teff",

    # --- Physical/numerical constants ---
    "G",
    "c_light",
    "k_B",
    "PISM",
)

# Keys that look constant-through-run but are NOT JSON-serializable
# (loaded function tables, interpolators, file paths used at runtime
# only).  The writer skips them defensively so any future addition
# can't poison the metadata write.  ``path*`` keys are absolute file
# paths from the input ``.param`` — preserved there, not duplicated
# in metadata.
METADATA_EXCLUDE: frozenset[str] = frozenset({
    # File paths — already in <run>.param
    "path2output",
    "path_cooling_CIE",
    "path_cooling_nonCIE",
    "path_sps",
    # Loaded SB99 tables / function objects
    "SB99_data",
    "SB99f",
    # Cooling-table interpolation function objects
    "cStruc_cooling_CIE_interpolation",
    "cStruc_cooling_CIE_logLambda",
    "cStruc_cooling_CIE_logT",
    "cStruc_cooling_nonCIE",
    "cStruc_heating_nonCIE",
    "cStruc_net_nonCIE_interpolation",
    # BE Lane-Emden function references
    "densBE_f_rho_rhoc",
    "densBE_f_m",
    "densBE_xi_out",
    # Empty-array placeholders (real data lives in per-snapshot stream)
    "bubble_T_arr",
    "bubble_dTdr_arr",
    "bubble_n_arr",
    "bubble_r_arr",
    "shell_n_arr",
})

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
METADATA_VERSION: int = 3

# Top-level keys in ``metadata.json`` that are NOT rehydrated into
# every snapshot's data dict.  Two reasons a key lives up here:
#
#   * ``_metadata_version`` — describes the metadata file itself.
#   * ``termination`` / ``final_state`` — Phase-2 blocks surfaced via
#     ``TrinityOutput.termination`` / ``.final_state`` properties.
#     Rehydrating them into each snapshot would smear the run-end
#     state into every timestep, which is misleading.
RESERVED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "_metadata_version",
    "termination",
    "final_state",
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
