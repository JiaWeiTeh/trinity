"""ParamSpec — single source of truth for TRINITY parameter definitions.

Sibling to ``src.sps.sps_columns.CanonicalSpec`` (same declarative,
frozen-dataclass + module-level tuple pattern).  Consumed by:

* ``src._input.registry`` derivation helpers (``run_const_keys``,
  ``metadata_exclude_keys`` — Phase 5 drop-in replacements for the
  hand-curated lists in ``src._output.run_constants``).
* ``src._input.read_param`` (``validator`` wired in Step 5 / Phase 6;
  ``resolver`` in Step 7 / Phase 7; ``active_when`` in Step 8 / Phase 8;
  runtime/derived-init builder in Step 10 / Phase 9).
* ``tools/gen_default_param.py`` (Phase 3+ regenerates
  ``default.param`` from the registry).

The registry is fully populated (187 specs) and live for run-const /
metadata derivation (Phase 5), validation (Phase 6), sentinel
resolution (Phase 7), conditional schema (Phase 8), and runtime/
derived-init materialization (Phase 9); Step 6's derived-init
resolvers (Phase 10) are still pending.  See ``test/test_registry.py``
for the guardrails that pin the spec set against a live ``read_param``
run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

# Categories are a descriptive grouping used for documentation and the
# eventual Phase-11 file split — they do NOT drive run-const /
# metadata-exclude membership (the explicit ``run_const`` /
# ``metadata_exclude`` booleans do, because membership does not follow
# category boundaries; see the registry module docstring).
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
    # ---- Runtime, grouped by physical role (mirrors the trinity_reader
    #      Snapshot table; runtime_control / _residuals / _cloud_profile
    #      are TRINITY-specific buckets not present in that table). ----
    "runtime_time",         # t_now, tSF (set-once), t_next, t_previousCoolingUpdate
    "runtime_radii",        # R1, R2, rShell, R_IF
    "runtime_bubble",       # T0, Eb, Pb, bubble_r/v/T/dTdr/n_arr, bubble_Tavg/mass/r_Tb/T_r_Tb, bubble_dMdt(Guess)
    "runtime_bubble_cooling",  # bubble_LTotal/L1Bubble/L2Conduction/L3Intermediate/Lgain/Lloss/Leak
    "runtime_pressure",     # P_HII, P_drive, P_ram, press_HII_in
    "runtime_force",        # F_grav, F_ram(_wind/_SN), F_ion_in, F_HII, F_rad, F_ISM
    "runtime_shell",        # shell_*, c_sound, n_IF/n_IF_ODE/n_IF_Str (densities at IF)
    "runtime_feedback",     # Qi, Lbol, Ln, Li, Lmech_*, pdot_*, v_mech_total, pdotdot_total
    "runtime_control",      # phase/end flags, EarlyPhaseApproximation, isCollapse/Dissolved, ...
    "runtime_residuals",    # residual_* (diagnostics not in the reader table)
    "runtime_cloud_profile",# initial_cloud_r/n/m_arr (set in phase0, constant thereafter)
    "runtime_loaded",       # sps_data, sps_f, sps_column_map, cStruc_*, densBE_f_*
    # ---- Parsed for back-compat only, never consumed ----
    "deprecated",           # back-compat retired specs (currently none; kept for future use)
]

# Prefix for sentinel string defaults: ``def_dir``, ``def_path``,
# ``def_value``, ``def_unset``.  A sentinel default is resolved either by
#
#   * its own ``resolver`` (the standalone case — ``path2output``,
#     ``path_cooling_nonCIE``, ``sps_path``), or
#   * another spec's resolver, declared via ``consumed_by`` (the
#     bulk-consumed case — ``sps_refmass`` plus the 13 ``sps_col_*`` specs
#     are owned by ``sps_path``'s bundle resolver, which reads the columns
#     en bloc via ``sps_columns.build_user_column_map`` and fills
#     ``sps_refmass``; when ``sps_path`` is the default the columns stay
#     at ``def_unset`` and ``sps_refmass`` falls back to 1e6).
#
# Both axes are wired (Phase 7): ``read_param`` Step 7 drives the three
# standalone resolvers via ``registry.resolve_all``.  The
# ``test_every_sentinel_default_has_resolver_or_pointer`` guard enforces
# that every sentinel spec carries one of the two.
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

    # Snapshot/metadata routing — ``src._output.run_constants`` derives
    # ``RUN_CONST_KEYS`` / ``METADATA_EXCLUDE`` from these flags via
    # ``registry.run_const_keys()`` / ``metadata_exclude_keys()`` (Phase 5
    # made the registry the single source of truth).  The three axes
    # are independent:
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
    # params dict, returning the resolved value (``read_param`` Step 7
    # drives these via ``registry.resolve_all``).  Mutually exclusive with
    # ``consumed_by``: a spec either carries its own resolver OR delegates
    # to another spec's, never both.  Three specs carry resolvers today
    # (path2output, path_cooling_nonCIE, sps_path).  See SENTINEL_PREFIX.
    resolver: Optional[Callable[[Any, dict], Any]] = None
    # ``consumed_by`` names another spec whose resolver bulk-consumes this
    # one's raw value.  Used by ``sps_refmass`` and the 13 ``sps_col_*``
    # specs, all pointing at ``sps_path`` — its bundle resolver assembles
    # the columns via ``sps_columns.build_user_column_map`` and fills
    # ``sps_refmass``.  Target existence is checked by
    # ``test_consumed_by_targets_exist``.
    consumed_by: Optional[str] = None
    # ``active_when`` is a predicate ``params -> bool`` that decides
    # *presence*: ``registry.apply_active_when`` (driven by ``read_param``
    # Step 8) restores the invariant *"this spec is in ``params`` iff
    # ``active_when(params)`` is True"* — adding a fresh DescribedItem
    # (with a deep-copied default) when active and absent, popping when
    # present and inactive.  Today carried only by the densBE / densPL
    # profile-conditional family; the guard
    # ``test_active_when_only_on_conditional_specs`` pins the set.
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
        if self.resolver is not None and self.consumed_by is not None:
            raise ValueError(
                f"{self.name}: resolver and consumed_by are mutually "
                f"exclusive (a sentinel default is resolved either by its "
                f"own resolver or by another spec's, never both)"
            )
