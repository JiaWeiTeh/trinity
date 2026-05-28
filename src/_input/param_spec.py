"""ParamSpec — single source of truth for TRINITY parameter definitions.

Sibling to ``src.sps.sps_columns.CanonicalSpec`` (same declarative,
frozen-dataclass + module-level tuple pattern).  Consumed by:

* ``src._input.registry`` derivation helpers (``run_const_keys``,
  ``metadata_exclude_keys`` — Phase 5 drop-in replacements for the
  hand-curated lists in ``src._output.run_constants``).
* ``src._input.read_param`` (Phase 6+ wires ``validator`` /
  ``resolver`` / ``active_when`` into Steps 5/7/8).
* ``tools/gen_default_param.py`` (Phase 3+ regenerates
  ``default.param`` from the registry).

Phase 1 lands this file plus an empty registry; no production code
imports it yet.  See ``test/test_registry.py`` for the guardrails
that pin the construction-time invariants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

# Categories drive Phase-5's derivation of RUN_CONST_KEYS /
# METADATA_EXCLUDE.  Adding a new category is a deliberate act —
# it must also be added to ``registry._INPUT_LIKE_CATEGORIES`` if
# it represents a constant-through-run input.
Category = Literal[
    # ---- Input-side (declared in default.param) ----
    "input_admin",          # model_name, path2output, output_format, log_*, simplify_npoints
    "input_physical",       # mCloud, sfe, ZCloud, nCore, nISM, rCore, include_PHII
    "input_profile",        # dens_profile, densBE_Omega, densPL_alpha
    "input_termination",    # stop_*, allowShellDissolution, coll_r
    "input_sps",            # SB99_rotation, sps_refmass, sps_path, sps_col_*, FB_*
    "input_cooling",        # path_cooling_CIE, path_cooling_nonCIE
    "input_constants",      # mu_*, gamma_adia, G, k_B, c_light, dust_*, caseB_alpha
    "input_solver",         # phaseSwitch_LlossLgain, bubble_xi_Tb, cool_alpha/beta/delta
    # ---- Set-once derived (read_param.py Step 6) ----
    "derived_init",         # rCloud, nEdge, tSF, mCluster, mCloud_input, densBE_Teff
    # ---- Runtime-only (read_param.py Step 10) ----
    "runtime_state",        # v2, R2, T0, Eb, shell_*, bubble_*, F_*, P_*, residual_*
    "runtime_loaded",       # sps_data, sps_f, sps_column_map, cStruc_*, densBE_f_*
    # ---- Parsed for back-compat only, never consumed ----
    "deprecated",           # stop_v, adiabaticOnlyInCore, immediate_leak, use_adaptive_solver
]

# Prefix for sentinel string defaults: ``def_dir``, ``def_path``,
# ``def_value``, ``def_unset``.  Any default that matches this prefix
# must carry a resolver (enforced in ``ParamSpec.__post_init__``).
SENTINEL_PREFIX = "def_"


@dataclass(frozen=True)
class ParamSpec:
    """Declarative spec for one TRINITY parameter.

    Mirrors ``DescribedItem`` at the registry layer: ``info`` and
    ``unit`` carry the same metadata that today's ``default.param``
    annotates via ``# INFO:`` / ``# UNIT:`` comments.
    """

    name: str
    default: Any
    info: str
    category: Category

    unit: Optional[str] = None

    # Snapshot/metadata routing — these reproduce the hand-curated lists
    # in ``src._output.run_constants`` (Phase 5 swaps them to derive from
    # here).  The three axes are independent:
    #   * run_const          → written once to metadata.json (RUN_CONST_KEYS)
    #   * metadata_exclude    → blocked from metadata.json (METADATA_EXCLUDE):
    #                           paths, loaded tables, empty array placeholders
    #   * exclude_from_snapshot → the live DescribedItem flag; omit from the
    #                           per-snapshot jsonl stream
    # run_const ∩ metadata_exclude is always empty (a key is written to
    # metadata or blocked from it, never both).
    run_const: bool = False
    metadata_exclude: bool = False
    exclude_from_snapshot: bool = False

    validator: Optional[Callable[[Any, dict], None]] = None
    # ``resolver`` resolves a sentinel default (``def_*``) against the full
    # params dict.  It is OPTIONAL here: Phase 2 declares the 17 sentinel
    # specs with ``resolver=None``; Phase 7 wires the resolvers and flips
    # the (currently xfail) ``test_every_sentinel_default_has_resolver``
    # guard green.  See SENTINEL_PREFIX.
    resolver: Optional[Callable[[Any, dict], Any]] = None
    active_when: Optional[Callable[[dict], bool]] = None

    deprecated_note: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(
                f"ParamSpec.name must be a valid Python identifier; got {self.name!r}"
            )
        if self.category == "deprecated" and not self.deprecated_note:
            raise ValueError(
                f"{self.name}: category='deprecated' requires deprecated_note"
            )
        if self.run_const and self.metadata_exclude:
            raise ValueError(
                f"{self.name}: run_const and metadata_exclude are mutually "
                f"exclusive (a key is either written to metadata.json or "
                f"blocked from it, never both)"
            )
