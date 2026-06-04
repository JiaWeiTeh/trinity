"""Module-level registry of ParamSpec entries — the single source of truth.

``SPECS`` holds one ``ParamSpec`` per parameter that TRINITY produces
(186 total: 72 declared in ``default.param`` + 114 runtime/derived
created in ``read_param`` Steps 6/8/10).  Production wiring:
``trinity._output.run_constants`` derives its lists from the registry
(Phase 5); ``read_param`` Step 5 calls ``validate_all`` (Phase 6),
Step 7 calls ``resolve_all`` (Phase 7), Step 8 calls
``apply_active_when`` (Phase 8), and Step 10 calls
``materialize_runtime`` (Phase 9).  Phase 10 will wire Step 6's
derived-init resolvers.

Runtime specs are split into physical buckets that mirror
``trinity_reader``'s ``Snapshot`` grouping (``runtime_time`` /
``runtime_radii`` / ``runtime_bubble`` / ``runtime_bubble_cooling`` /
``runtime_pressure`` / ``runtime_force`` / ``runtime_shell`` /
``runtime_feedback``), plus three TRINITY-specific buckets not in
that table: ``runtime_control`` (phase/end flags), ``runtime_residuals``
(solver diagnostics), and ``runtime_cloud_profile`` (set-once profile
tables built in phase 0).

``default`` field convention
----------------------------
* **Input specs** (``category`` starts with ``input_``): ``default`` is
  the *raw source string* exactly as it would appear in
  ``default.param`` (e.g. ``"1e7"``, ``"5/3"``, ``"None"``, ``"True"``).
  Phase 10's builder parses it via the same ``parse_value`` path
  ``read_param`` uses for file content.  Fraction-encoded constants
  (``mu_atom`` etc.) are stored as ``"14/11"`` strings per the agreed
  representation; they round-trip to the identical float.
* **Runtime / derived specs**: ``default`` is the literal initial
  *value* assigned in Step 6/8/10 (``0``, ``np.nan``, ``np.array([])``,
  ``False``, ``None`` …) — used directly, no parsing.

Derivation helpers
------------------
``run_const_keys()`` and ``metadata_exclude_keys()`` project the
per-spec ``run_const`` / ``metadata_exclude`` booleans.  As of Phase 5
``trinity._output.run_constants`` derives its ``RUN_CONST_KEYS`` /
``METADATA_EXCLUDE`` directly from these helpers, so the registry is
the single source of truth for run-const / metadata-exclude membership.
"""
from __future__ import annotations

import copy
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import trinity.sps.sps_columns as sps_columns
from trinity._input.dictionary import DescribedItem
from trinity._input.errors import ParameterFileError
from trinity._input.param_spec import Category, ParamSpec

logger = logging.getLogger(__name__)

# Anchor bundled-asset lookups to the repo root, not the CWD (mirrors the
# constant of the same name in ``read_param``): resolvers below build
# paths under ``lib/default/...`` that must resolve regardless of where
# ``run.py`` was launched from.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Conditional-schema predicates (Phase 8: consumed by ``apply_active_when``
# from ``read_param`` Step 8).  The predicate decides presence: a spec
# with ``active_when`` is in ``params`` iff its predicate returns True.
# densBE/densPL profiles are mutually exclusive, so each predicate
# matches exactly one of the two profile families.
# ---------------------------------------------------------------------------
def _profile_value(params) -> object:
    item = params.get("dens_profile")
    return getattr(item, "value", item)


def _active_densBE(params) -> bool:
    return _profile_value(params) == "densBE"


def _active_densPL(params) -> bool:
    return _profile_value(params) == "densPL"


# ---------------------------------------------------------------------------
# Validators (consumed by Phase 6; ``validate_all`` invoked from
# ``read_param`` Step 5).  A validator receives the parameter's current
# value plus the full params dict and may either raise
# ``ParameterFileError`` or normalize the value in place (e.g. coerce a
# whole-number float to ``int``).  Error messages are verbatim from the
# pre-Phase-6 Step-5 block so existing user diagnostics are preserved.
# ---------------------------------------------------------------------------
def _validate_ZCloud(value, params) -> None:
    from trinity._input.errors import ParameterFileError
    if value != 1:
        raise ParameterFileError(
            f"Metallicity Z={value} not implemented. "
            f"Currently only Z=1 (solar) is supported."
        )


def _validate_dens_profile(value, params) -> None:
    from trinity._input.errors import ParameterFileError
    if value not in ('densBE', 'densPL'):
        raise ParameterFileError(
            f"Invalid dens_profile '{value}'. "
            f"Must be 'densBE' or 'densPL'."
        )


def _validate_stop_at_rCloud_nSnap(value, params) -> None:
    """Validate AND coerce: whole-number floats (e.g. 5.0 from '5')
    become ints; fractional floats / negatives / non-numerics raise."""
    from trinity._input.errors import ParameterFileError
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterFileError(
            f"Invalid stop_at_rCloud_nSnap '{value}'. "
            f"Must be None or a non-negative integer."
        )
    if isinstance(value, float) and not value.is_integer():
        raise ParameterFileError(
            f"Invalid stop_at_rCloud_nSnap '{value}'. "
            f"Must be a whole-number integer (got fractional value)."
        )
    coerced = int(value)
    if coerced < 0:
        raise ParameterFileError(
            f"Invalid stop_at_rCloud_nSnap '{value}'. "
            f"Must be None or a non-negative integer."
        )
    params['stop_at_rCloud_nSnap'].value = coerced


def _validate_coverFraction(value, params) -> None:
    """Covering fraction Cf must be a number in (0, 1]. Cf=1 is a sealed
    bubble (no leak); Cf=0 is unphysical here (would vent the whole wall)."""
    from trinity._input.errors import ParameterFileError
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterFileError(
            f"Invalid coverFraction '{value}'. Must be a number in (0, 1]."
        )
    if not (0.0 < value <= 1.0):
        raise ParameterFileError(
            f"Invalid coverFraction '{value}'. Must satisfy 0 < Cf <= 1 "
            f"(Cf=1 recovers the sealed bubble)."
        )


def _validate_rCloud_max(value, params) -> None:
    """Maximum plausible cloud radius (rCloud_max) must be a positive number
    [pc].  It caps the pre-run GMC plausibility check in
    ``trinity.cloud_properties.validate_gmc``."""
    from trinity._input.errors import ParameterFileError
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterFileError(
            f"Invalid rCloud_max '{value}'. Must be a positive number in pc."
        )
    if value <= 0:
        raise ParameterFileError(
            f"Invalid rCloud_max '{value}'. Must be > 0 (pc)."
        )


# ---------------------------------------------------------------------------
# Resolvers (consumed by Phase 7; ``resolve_all`` invoked from
# ``read_param`` Step 7).  A resolver receives the parameter's current
# value plus the full params dict and returns the resolved value, which
# the driver assigns back to ``params[name].value``.  Resolvers run
# unconditionally: each handles BOTH the sentinel (``def_*``) branch and
# the user-supplied branch (e.g. a user path2output is still mkdir'd).
# Logic and error messages are lifted verbatim from the pre-Phase-7
# inline Step-7 block so behavior is byte-identical.
# ---------------------------------------------------------------------------
def _resolve_path2output(value, params) -> str:
    """Output directory.  Sentinel 'def_dir' resolves to
    ``<cwd>/outputs/<model_name>``; a user path is taken as-is.  Either
    way the directory is created."""
    if value == 'def_dir':
        path2output = os.path.join(os.getcwd(), 'outputs', params['model_name'].value)
    else:
        path2output = str(value)
    Path(path2output).mkdir(parents=True, exist_ok=True)
    return path2output


def _resolve_path_cooling_nonCIE(value, params) -> str:
    """Non-CIE cooling directory.  Sentinel 'def_dir' resolves to the
    shipped OPIATE cube folder under ``lib/default/opiate/``; a user
    path is taken as-is (and created)."""
    if value == 'def_dir':
        return str(_REPO_ROOT / 'lib' / 'default' / 'opiate') + os.sep
    path_cooling = str(value)
    Path(path_cooling).mkdir(parents=True, exist_ok=True)
    return path_cooling


def _resolve_sps_bundle(value, params) -> str:
    """SPS bundle resolver (sps_path + sps_refmass + sps_column_map).

    These three are physically coupled, so a single resolver owns the
    whole bundle; ``sps_refmass`` and the 13 ``sps_col_*`` specs declare
    ``consumed_by='sps_path'`` rather than carrying their own resolvers.

    Sentinel 'def_path' resolves to the bundled
    ``lib/default/sps/starburst99/1e6cluster_default.csv`` — an SB99 grid
    at rotation=1, ZCloud=1, mass=1e6 Msun in CSV form with the canonical
    7-column SB99 layout (DEFAULT_SPS_COLUMN_MAP). The default rejects
    combinations the bundled cooling tables can't fulfill; users who need
    a different metallicity or rotation must set sps_path explicitly.
    See analysis/sb99-refactor-audit.md §9.

    Side effects (the coupled members of the bundle):
      * ``params['sps_refmass'].value`` — 'def_value' resolves to 1e6 only
        for the bundled file; a user sps_path requires an explicit value
        (silent 1e6 would mis-scale f_mass = mCluster / sps_refmass).
      * ``params['sps_column_map']`` — new DescribedItem holding the
        canonical->ColumnSpec map (default preset, or built from the
        user's sps_col_* declarations).
    """
    sps_path_is_default = value == 'def_path'
    if sps_path_is_default:
        if params['ZCloud'].value != 1.0:
            raise ValueError(
                f"ZCloud={params['ZCloud'].value} is not supported with the "
                "default SPS fallback (only ZCloud=1.0 is bundled). Set "
                "sps_path explicitly to use a non-solar metallicity SPS file."
            )
        if not params['SB99_rotation'].value:
            raise ValueError(
                "SB99_rotation=0 is not supported with the default SPS "
                "fallback (only rot cooling tables are bundled). Set "
                "sps_path explicitly and supply matching cooling tables "
                "for the norot case."
            )
        sps_path = str(
            _REPO_ROOT / 'lib' / 'default' / 'sps' / 'starburst99' / '1e6cluster_default.csv'
        )
        column_map = sps_columns.DEFAULT_SPS_COLUMN_MAP
        logger.info(f"sps_path unset → using default SPS file: {sps_path}")
    else:
        sps_path = str(value)
        try:
            column_map = sps_columns.build_user_column_map(params)
            sps_columns.validate_user_column_map(column_map, sps_path)
        except ValueError as err:
            logger.error(f"SPS column map error:\n{err}")
            raise
        logger.info(f"Using user-defined sps_path = {sps_path}")

    # sps_refmass: only meaningful for the bundled file (1e6 Msun). When
    # the user supplies sps_path, require an explicit value.
    if params['sps_refmass'].value == 'def_value':
        if sps_path_is_default:
            params['sps_refmass'].value = 1e6
        else:
            raise ParameterFileError(
                f"sps_refmass is required when sps_path is user-set "
                f"(got sps_path={sps_path!r}). The default sps_refmass=1e6 "
                f"only matches the bundled SPS file; supplying your own "
                f"sps_path means you must declare the reference cluster "
                f"mass it was normalized to."
            )

    params['sps_column_map'] = DescribedItem(
        column_map,
        info="SPS column mapping (canonical -> ColumnSpec)",
        ori_units="N/A",
        exclude_from_snapshot=True,
    )
    return sps_path


SPECS: tuple[ParamSpec, ...] = (
    ParamSpec(name='model_name', default='default', info='Specifies the model name, which serves as the prefix for all output filenames.', category='input_admin', unit=None, run_const=True),
    ParamSpec(name='path2output', default='def_dir', info='Defines the output directory where all generated files will be stored.', category='input_admin', unit=None, exclude_from_snapshot=True, metadata_exclude=True, resolver=_resolve_path2output),
    ParamSpec(name='output_format', default='JSON', info='Specifies the output format.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='simplify_npoints', default='100', info='Target number of points retained for simplified profile arrays in saved snapshots (bubble_T_arr, bubble_n_arr, bubble_dTdr_arr, bubble_v_arr, shell_grav_force_m, shell_n_arr). Default 100. Larger values give higher-fidelity snapshots at the cost of larger output files. Clamped to >= 20 (matches the coverage-skeleton chunk count). The first two simplify calls per implicit-phase snapshot log their reconstruction R² at INFO level so you can verify the chosen budget is faithful.', category='input_admin', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='log_level', default='DEBUG', info='Logging level for terminal and file output. Controls how much detail is logged.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='log_console', default='False', info='Enable console (terminal) output for logging. If True, log messages are printed to terminal.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='log_file', default='True', info='Enable file output for logging. If True, log messages are written to a .log file in the output directory.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='log_colors', default='True', info='Use colored output in terminal. If True, log messages are color-coded by severity (DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=magenta).', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mCloud', default='1e7', info='The mass of the molecular cloud.', category='input_physical', unit='Msun', run_const=True),
    ParamSpec(name='sfe', default='0.01', info='Star formation efficiency.', category='input_physical', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='ZCloud', default='1', info='Cloud metallicity', category='input_physical', unit='Zsun', exclude_from_snapshot=True, run_const=True, validator=_validate_ZCloud),
    ParamSpec(name='include_PHII', default='True', info='Include HII pressure (from Strömgren ionization balance in shell) in P_drive. When False, P_HII is set to zero.', category='input_physical', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='coverFraction', default='1.0', info='Closed fraction of the bubble wall (covering fraction Cf). Geometry-set energy/mass leak: hot gas vents through the open area (1-Cf)*4*pi*R2^2 at the interior sound speed, draining bubble energy (and, when the mass sink is enabled, mass). Cf=1 recovers the sealed (Weaver) bubble exactly; not fragmentation-triggered. Usable range ~0.9-0.99; Cf near 0 drains the bubble within a step and stresses the integrator.', category='input_physical', unit=None, exclude_from_snapshot=True, run_const=True, validator=_validate_coverFraction),
    ParamSpec(name='dens_profile', default='densPL', info='Specifies how the cloud density scales with radius.', category='input_profile', unit=None, run_const=True, validator=_validate_dens_profile),
    ParamSpec(name='densBE_Omega', default='14.1', info='if `densBE` is selected, then the ratio `Omega = nCore/nCloudEdge` must be specified.', category='input_profile', unit=None, exclude_from_snapshot=True, run_const=True, active_when=_active_densBE),
    ParamSpec(name='densPL_alpha', default='0', info='if `densPL` is selected, then the power-law coefficient `nCore*(r/rCore)^alpha` (0 = homogeneous, -2 = isothermal) must be specified.', category='input_profile', unit=None, run_const=True, active_when=_active_densPL),
    ParamSpec(name='nCore', default='1e5', info='Hydrogen nuclei number density of cloud core (n_H). Standard GMC/ISM convention. Mass density: rho = nCore * mu_convert * m_H. If `densPL` AND densPL_alpha = 0, this is the average cloud density.', category='input_physical', unit='cm**-3', run_const=True),
    ParamSpec(name='nISM', default='1', info='Hydrogen nuclei number density of the ambient ISM (n_H). Mass density: rho = nISM * mu_convert * m_H.', category='input_physical', unit='cm**-3', run_const=True),
    ParamSpec(name='rCore', default='0.01', info='Core radius of the molecular cloud.', category='input_physical', unit='pc', run_const=True),
    ParamSpec(name='rCloud_max', default='200', info='Maximum plausible cloud radius used by the pre-run GMC parameter validation. If the computed rCloud exceeds this, the run is rejected as implausibly diffuse for the given mass. Increase to allow larger, more diffuse clouds.', category='input_physical', unit='pc', exclude_from_snapshot=True, run_const=True, validator=_validate_rCloud_max),
    ParamSpec(name='allowShellDissolution', default='True', info='Allow shell dissolution to terminate simulation. If False, shell dissolution check is disabled.', category='input_termination', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_t_diss', default='1', info='Duration (in Myr) that shell_nMax must remain below nISM before dissolution is triggered.', category='input_termination', unit='Myr', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_r', default='500', info='Maximum radial extent permitted for shell expansion. Set to None to disable this termination condition.', category='input_termination', unit='pc', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_t', default='15', info='Maximum duration of the simulation. Set to None to disable this termination condition.', category='input_termination', unit='Myr', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_at_rCloud_nSnap', default='None', info='Terminate simulation after the shell crosses the cloud edge (R2 > rCloud). Value is the number of post-crossing segment-loop snapshots to record before terminating. Set to None to disable. 0 stops at the edge (1a reconciliation snapshot only). N>0 lets the implicit phase advance for N more segments past the crossing — note the implicit phase\'s end-of-phase reconciliation snapshot adds one extra past-rCloud sample, so the total snapshots with R2 >= rCloud is roughly N + 2 (1 at-edge + N in-loop + 1 recon).', category='input_termination', unit=None, exclude_from_snapshot=True, validator=_validate_stop_at_rCloud_nSnap),
    ParamSpec(name='coll_r', default='1', info='Radius below which the cloud is considered completely collapsed.', category='input_termination', unit='pc', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='SB99_rotation', default='1', info='Stellar-rotation flag. Selects rot vs norot non-CIE cooling tables (trinity/cooling/non_CIE/read_cloudy.py). Only rot tables ship in lib/default/opiate/, so 0 (norot) requires the user to supply matching cooling tables and an sps_path pointing at a norot SPS file; the default SPS fallback rejects SB99_rotation=0. NOTE: name retained for stability. May rename to sps_rotation in a future PR once the cooling subsystem stops being SB99-flavored.', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='sps_refmass', default='def_value', info='Reference cluster mass used in f_mass = mCluster / sps_refmass.', category='input_sps', unit='Msun', exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='FB_mColdWindFrac', default='0', info='Fraction of cold mass entrained in stellar winds (increases Mdot_wind, reduces velocity).', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_mColdSNFrac', default='0', info='Fraction of cold mass entrained in supernova ejecta (increases Mdot_SN, reduces velocity).', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_thermCoeffWind', default='1', info='Defines the thermalization efficiency for colliding stellar winds.', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_thermCoeffSN', default='1', info='Defines the thermalization efficiency for supernova ejecta.', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_vSN', default='1e4', info='Specifies the velocity of supernova ejecta.', category='input_sps', unit='km * s**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_atom', default='14/11', info='Mean molecular weight for neutral atomic gas (HI + He). Dimensionless, in units of m_H. Composition: He:H = 1:10 by number. Formula: (10×1 + 1×4) / 11 = 14/11.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_ion', default='14/23', info='Mean molecular weight for fully ionized gas (H+ + He++). Dimensionless, in units of m_H. Composition: He:H = 1:10 by number. Formula: (10×1 + 1×4) / 23 = 14/23.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_mol', default='14/6', info='Mean molecular weight for molecular gas (H2 + He). Dimensionless, in units of m_H. Composition: He:H = 1:10 by number. Formula: (10×1 + 1×4) / 6 = 14/6.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_convert', default='1.4', info='Conversion factor for mass density calculation (mass integration). Use this for n→rho conversion. Independent of ionization state - ionization changes particle counts but NOT total mass.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='x_He', default='0.1', info='Helium-to-hydrogen number ratio n_He/n_H. Composition source of truth: mu_atom, mu_ion, mu_mol, mu_convert and chi_e are all derived from x_He and Z_He at load (read_param Step 6).', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='Z_He', default='2', info='Helium ionisation state in the ionised gas (2 = doubly ionised). Sets the electron-per-hydrogen factor chi_e = 1 + Z_He*x_He.', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='TShell_neu', default='1e2', info='Temperature of the neutral shell region.', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='TShell_ion', default='1e4', info='Temperature of the ionised shell region.', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='dust_sigma', default='1.5e-21', info='Dust cross-section at solar metallicity.', category='input_constants', unit='cm**2', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='dust_noZ', default='0.05', info='Metallicity below which there is effectively no dust', category='input_constants', unit='Zsun', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='dust_KappaIR', default='4', info='The Rosseland mean dust opacity kappa_IR.', category='input_constants', unit='cm**2 * g**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='caseB_alpha', default='2.59e-13', info='The case B recombination coefficient', category='input_constants', unit='cm**3 * s**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='gamma_adia', default='5/3', info='the adiabatic index', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='C_thermal', default='6e-7', info='The thermal conduction coefficient C', category='input_constants', unit='erg * s**-1 * cm**-1 * K**(-7/2)', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='c_light', default='29979245800', info='speed of light', category='input_constants', unit='cm * s**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='G', default='6.6743e-08', info='Gravitational constant', category='input_constants', unit='cm**3 * g**-1 * s**-2', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='k_B', default='1.380649e-16', info='Boltzmann constant', category='input_constants', unit='erg * K**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='PISM', default='0', info='ISM Pressure, P/k', category='input_constants', unit='K * cm**-3', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='phaseSwitch_LlossLgain', default='0.05', info='When (Lgain-Lloss)/Lgain approaches this value, begin momentum-driving phase.', category='input_solver', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='cool_alpha', default='0.6', info='Cooling related values. alpha = v2*t_now/R2', category='input_solver', unit=None),
    ParamSpec(name='cool_beta', default='0.8', info='Cooling related values. beta = - dPb/dt.', category='input_solver', unit=None),
    ParamSpec(name='cool_delta', default='-6/35', info='Cooling related values. delta = dT/dt.', category='input_solver', unit=None),
    ParamSpec(name='path_cooling_CIE', default='3', info='Selects the CIE (T > 10^5.5 K) cooling curve.', category='input_cooling', unit=None, exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='path_cooling_nonCIE', default='def_dir', info='Folder containing the non-CIE (T < 10^5.5 K) OPIATE/CLOUDY cubes.', category='input_cooling', unit=None, exclude_from_snapshot=True, metadata_exclude=True, resolver=_resolve_path_cooling_nonCIE),
    ParamSpec(name='sps_path', default='def_path', info='Full path to an SPS data file. If def_path (default), TRINITY uses', category='input_sps', unit=None, exclude_from_snapshot=True, resolver=_resolve_sps_bundle),
    ParamSpec(name='sps_col_t', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Qi', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_fi', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Lbol', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Lmech_W', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_pdot_W', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Lmech_total', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Lmech_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_pdot_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Mdot_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_v_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Li', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='sps_col_Ln', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True, consumed_by='sps_path'),
    ParamSpec(name='bubble_xi_Tb', default='0.98', info='The relative radius xi = r/R2, at which we measure the bubble temperature.', category='input_solver', unit=None, exclude_from_snapshot=True, run_const=True),
    # mCloud_input / mCluster have no static default: read_param Step 6
    # computes them from the input mCloud and sfe (mCloud_input = input
    # mCloud; mCluster = mCloud_input * sfe). 0.0 is a placeholder; the
    # value is materialised by the derived-init resolver in Phase 7/10.
    ParamSpec(name='mCloud_input', default=0.0, info='Pre-SFE input cloud mass (= mCloud + mCluster). Matches the .param file and the sweep folder-name tag.', category='derived_init', unit='Msun', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mCluster', default=0.0, info='Cluster mass (mCloud_input * sfe)', category='derived_init', unit='Msun', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='chi_e', default=1.2, info='Electron-per-hydrogen-nucleus factor n_e/n_H = 1 + Z_He*x_He (derived at load from x_He, Z_He). Multiplies n_H^2 in recombination, Stroemgren balance, and CIE cooling.', category='derived_init', unit='dimensionless', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='sps_column_map', default=None, info='SPS column mapping (canonical -> ColumnSpec)', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='current_phase', default='', info='Current simulation phase: energy/implicit/transition/momentum', category='runtime_control', unit='N/A'),
    ParamSpec(name='EndSimulationDirectly', default=False, info='Flag to immediately end simulation', category='runtime_control', unit='N/A'),
    ParamSpec(name='SimulationEndReason', default='', info='Reason for simulation completion', category='runtime_control', unit='N/A'),
    ParamSpec(name='SimulationEndCode', default=None, info='Exit code (SimulationEndCode enum) for simulation completion', category='runtime_control', unit='N/A'),
    ParamSpec(name='EarlyPhaseApproximation', default=True, info='Using approximations for early phase?', category='runtime_control', unit='N/A'),
    ParamSpec(name='_snapshots_after_rCloud', default=0, info='Snapshots saved with R2 > rCloud (used by stop_at_rCloud_nSnap)', category='runtime_control', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='tSF', default=0, info='Time of star formation', category='derived_init', unit='Myr', run_const=True),
    ParamSpec(name='t_now', default=0, info='Current simulation time', category='runtime_time', unit='Myr'),
    ParamSpec(name='v2', default=0, info='Velocity at R2 (outer bubble radius = inner shell edge)', category='runtime_bubble', unit='pc/Myr'),
    ParamSpec(name='R2', default=0, info='Outer bubble radius (= inner shell edge)', category='runtime_radii', unit='pc'),
    ParamSpec(name='T0', default=0, info='Characteristic bubble temperature (at xi_Tb fraction of bubble thickness)', category='runtime_bubble', unit='K'),
    ParamSpec(name='Eb', default=0, info='Bubble energy', category='runtime_bubble', unit='Msun*pc**2/Myr**2'),
    ParamSpec(name='R1', default=0, info='Inner bubble radius', category='runtime_radii', unit='pc'),
    ParamSpec(name='Pb', default=0, info='Bubble pressure', category='runtime_bubble', unit='Msun/Myr**2/pc'),
    ParamSpec(name='c_sound', default=0, info='Sound speed', category='runtime_shell', unit='pc/Myr'),
    ParamSpec(name='t_next', default=0, info='Next time for mShell interpolation', category='runtime_time', unit='Myr'),
    ParamSpec(name='rCloud', default=0, info='Cloud radius', category='derived_init', unit='pc', run_const=True),
    ParamSpec(name='rShell', default=0, info='Shell outer radius', category='runtime_radii', unit='pc'),
    ParamSpec(name='nEdge', default=0, info='Number density at cloud edge', category='derived_init', unit='1/pc**3', run_const=True),
    ParamSpec(name='initial_cloud_r_arr', default=np.array([]), info='Initial cloud radius array', category='runtime_cloud_profile', unit='pc'),
    ParamSpec(name='initial_cloud_n_arr', default=np.array([]), info='Initial cloud density array', category='runtime_cloud_profile', unit='1/cm**3'),
    ParamSpec(name='initial_cloud_m_arr', default=np.array([]), info='Initial cloud enclosed mass array', category='runtime_cloud_profile', unit='Msun'),
    ParamSpec(name='sps_data', default=0, info='SPS raw 11-array datacube', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='sps_f', default=0, info='SPS interpolators (dict of scipy interp1d)', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='Lmech_W', default=0, info='Wind mechanical luminosity', category='runtime_feedback', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Lmech_SN', default=0, info='SN mechanical luminosity', category='runtime_feedback', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Lmech_total', default=0, info='Total mechanical luminosity', category='runtime_feedback', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='v_mech_total', default=0, info='mechanical velocity (winds+SN)', category='runtime_feedback', unit='pc/Myr'),
    ParamSpec(name='pdot_W', default=0, info='Wind momentum rate', category='runtime_feedback', unit='Msun*pc/Myr**2'),
    ParamSpec(name='pdot_SN', default=0, info='Supernova momentum rate', category='runtime_feedback', unit='Msun*pc/Myr**2'),
    ParamSpec(name='pdot_total', default=0, info='Total momentum rate', category='runtime_feedback', unit='Msun*pc/Myr**2'),
    ParamSpec(name='pdotdot_total', default=0, info='Rate of wind momentum rate', category='runtime_feedback', unit='Msun*pc/Myr**3'),
    ParamSpec(name='Qi', default=0, info='Ionizing photon rate', category='runtime_feedback', unit='1/Myr'),
    ParamSpec(name='Lbol', default=0, info='Bolometric luminosity', category='runtime_feedback', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Ln', default=0, info='Non-ionizing luminosity', category='runtime_feedback', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Li', default=0, info='Ionizing luminosity', category='runtime_feedback', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='t_previousCoolingUpdate', default=1e+30, info='Time of previous cooling update', category='runtime_time', unit='Myr'),
    ParamSpec(name='cStruc_cooling_nonCIE', default=0, info='Non-CIE cooling cube', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_heating_nonCIE', default=0, info='Non-CIE heating cube', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_net_nonCIE_interpolation', default=0, info='Non-CIE net cooling interpolation', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_cooling_CIE_logT', default=0, info='CIE log temperature array', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_cooling_CIE_logLambda', default=0, info='CIE log lambda array', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_cooling_CIE_interpolation', default=0, info='CIE cooling interpolation', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='shell_fAbsorbedIon', default=1, info='Fraction of absorbed ionizing radiation', category='runtime_shell', unit='dimensionless'),
    ParamSpec(name='shell_fAbsorbedNeu', default=0, info='Fraction of absorbed non-ionizing radiation', category='runtime_shell', unit='dimensionless'),
    ParamSpec(name='shell_fAbsorbedWeightedTotal', default=0, info='Total absorption fraction (luminosity weighted)', category='runtime_shell', unit='dimensionless'),
    ParamSpec(name='shell_fIonisedDust', default=0, info='Ionized dust fraction', category='runtime_shell', unit='dimensionless'),
    ParamSpec(name='shell_thickness', default=0, info='Shell thickness', category='runtime_shell', unit='pc'),
    ParamSpec(name='shell_nMax', default=0, info='Maximum shell density', category='runtime_shell', unit='1/pc**3'),
    ParamSpec(name='shell_tauKappaRatio', default=0, info='tau_IR / kappa_IR ratio', category='runtime_shell', unit='Msun/pc**2'),
    ParamSpec(name='shell_grav_r', default=np.array([]), info='Radius array for gravitational calculations', category='runtime_shell', unit='pc'),
    ParamSpec(name='shell_grav_phi', default=np.array([]), info='Gravitational potential', category='runtime_shell', unit='pc**2/Myr**2'),
    ParamSpec(name='shell_grav_force_m', default=np.array([]), info='Gravitational force per unit mass', category='runtime_shell', unit='pc/Myr**2'),
    ParamSpec(name='shell_r_arr', default=np.array([]), info='Radial grid through ionized+neutral shell', category='runtime_shell', unit='pc'),
    ParamSpec(name='shell_n_arr', default=np.array([]), info='Number density through ionized+neutral shell', category='runtime_shell', unit='1/pc**3', metadata_exclude=True),
    ParamSpec(name='shell_ion_idx', default=-1, info='Last index of ionized region in shell_r/n_arr (-1 if empty)', category='runtime_shell', unit='N/A'),
    ParamSpec(name='shell_mass', default=0, info='Shell mass', category='runtime_shell', unit='Msun'),
    ParamSpec(name='shell_massDot', default=0, info='Shell mass accretion rate', category='runtime_shell', unit='Msun/Myr'),
    ParamSpec(name='shell_interpolate_massDot', default=False, info='Use shell mass interpolation?', category='runtime_control', unit='N/A'),
    ParamSpec(name='shell_n0', default=0, info='Shell inner density (pressure balance)', category='runtime_shell', unit='1/pc**3'),
    ParamSpec(name='F_grav', default=0, info='Gravitational force', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ram', default=0, info='Ram pressure force (from Pb-Eb relation)', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ram_wind', default=0, info='Wind ram pressure force (from SPS interpolators)', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ram_SN', default=0, info='SN ram pressure force (from SPS interpolators)', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ion_in', default=0, info='Inward photoionization pressure', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_HII', default=0, info='Outward HII pressure force (= P_HII * 4piR2^2)', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_rad', default=0, info='Radiation pressure', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ISM', default=0, info='ISM pressure force (placeholder, never computed — always 0)', category='runtime_force', unit='Msun*pc/Myr**2'),
    ParamSpec(name='n_IF', default=0.0, info='Density at ionization front from shell ODE', category='runtime_shell', unit='1/pc**3'),
    ParamSpec(name='n_IF_ODE', default=0.0, info='Raw ODE-derived n_IF (same as n_IF, kept for diagnostics)', category='runtime_shell', unit='1/pc**3'),
    ParamSpec(name='R_IF', default=0.0, info='Radius of ionization front', category='runtime_radii', unit='pc'),
    ParamSpec(name='n_IF_Str', default=0.0, info='Stroemgren ionization balance density (Lancaster+2025), sole source of P_HII', category='runtime_shell', unit='1/pc**3'),
    ParamSpec(name='P_HII', default=0.0, info='HII pressure from Stroemgren ionization balance in shell (n_IF_Str)', category='runtime_pressure', unit='Msun/Myr**2/pc'),
    ParamSpec(name='P_drive', default=0.0, info='Total driving pressure', category='runtime_pressure', unit='Msun/Myr**2/pc'),
    ParamSpec(name='P_ram', default=0.0, info='Ram pressure from freely-streaming wind', category='runtime_pressure', unit='Msun/Myr**2/pc'),
    ParamSpec(name='press_HII_in', default=0.0, info='Inward HII pressure at shell (confining)', category='runtime_pressure', unit='Msun/Myr**2/pc'),
    ParamSpec(name='bubble_LTotal', default=0, info='Total luminosity lost to cooling', category='runtime_bubble_cooling', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_L1Bubble', default=0, info='Cooling in bubble zone', category='runtime_bubble_cooling', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_L2Conduction', default=0, info='Cooling in conduction zone', category='runtime_bubble_cooling', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_L3Intermediate', default=0, info='Cooling in intermediate zone', category='runtime_bubble_cooling', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_Tavg', default=0, info='Average bubble temperature', category='runtime_bubble', unit='K'),
    ParamSpec(name='bubble_mass', default=0, info='Bubble mass', category='runtime_bubble', unit='Msun'),
    ParamSpec(name='bubble_r_Tb', default=0, info='Radius at bubble_xi_Tb * R2', category='runtime_bubble', unit='pc'),
    ParamSpec(name='bubble_T_r_Tb', default=0, info='Temperature at r_Tb', category='runtime_bubble', unit='K'),
    ParamSpec(name='bubble_r_arr', default=np.array([]), info='Bubble radius structure', category='runtime_bubble', unit='pc', metadata_exclude=True),
    ParamSpec(name='bubble_v_arr', default=np.array([]), info='Bubble velocity structure', category='runtime_bubble', unit='pc/Myr'),
    ParamSpec(name='bubble_T_arr', default=np.array([]), info='Bubble temperature structure', category='runtime_bubble', unit='K', metadata_exclude=True),
    ParamSpec(name='bubble_dTdr_arr', default=np.array([]), info='Bubble temperature gradient', category='runtime_bubble', unit='K/pc', metadata_exclude=True),
    ParamSpec(name='bubble_n_arr', default=np.array([]), info='Bubble density structure', category='runtime_bubble', unit='1/pc**3', metadata_exclude=True),
    ParamSpec(name='bubble_dMdtGuess', default=0, info='Bubble dM/dt guess', category='runtime_bubble', unit='Msun/Myr'),
    ParamSpec(name='bubble_dMdt', default=np.nan, info='Bubble mass loss rate (thermal conduction)', category='runtime_bubble', unit='Msun/Myr'),
    ParamSpec(name='bubble_Lgain', default=np.nan, info='Luminosity gain from winds', category='runtime_bubble_cooling', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_Lloss', default=np.nan, info='Luminosity loss from cooling/leaking', category='runtime_bubble_cooling', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_Leak', default=0, info='Leaking luminosity', category='runtime_bubble_cooling', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='isCollapse', default=False, info='Is cloud collapsing?', category='runtime_control', unit='N/A'),
    ParamSpec(name='isDissolved', default=False, info='Has shell dissolved?', category='runtime_control', unit='N/A'),
    ParamSpec(name='is_phiDepleted', default=False, info='Are ionising photons exhausted inside shell (phi->0)?', category='runtime_control', unit='N/A'),
    ParamSpec(name='residual_deltaT', default=0, info='Temperature residual (T1-T2)/T2', category='runtime_residuals', unit='dimensionless'),
    ParamSpec(name='residual_betaEdot', default=0, info='Energy rate residual', category='runtime_residuals', unit='dimensionless'),
    ParamSpec(name='residual_Edot1_guess', default=np.nan, info='Edot from beta', category='runtime_residuals', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='residual_Edot2_guess', default=np.nan, info='Edot from energy balance', category='runtime_residuals', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='residual_T1_guess', default=np.nan, info='T from bubble_Trgoal', category='runtime_residuals', unit='K'),
    ParamSpec(name='residual_T2_guess', default=np.nan, info='T from T0', category='runtime_residuals', unit='K'),
    ParamSpec(name='densBE_Teff', default=0, info='Effective temperature of BE sphere', category='derived_init', unit='K', exclude_from_snapshot=True, run_const=True, active_when=_active_densBE),
    ParamSpec(name='densBE_xi_arr', default=[], info='Lane-Emden xi array', category='runtime_cloud_profile', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_u_arr', default=[], info='Lane-Emden u array', category='runtime_cloud_profile', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_dudxi_arr', default=[], info='Lane-Emden du/dxi array', category='runtime_cloud_profile', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_rho_rhoc_arr', default=[], info='Density contrast array', category='runtime_cloud_profile', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_f_rho_rhoc', default=0, info='Interpolation function for density contrast', category='runtime_loaded', unit='dimensionless', exclude_from_snapshot=True, metadata_exclude=True, active_when=_active_densBE),
    ParamSpec(name='densBE_f_m', default=None, info='Lane-Emden mass interpolation function', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True, active_when=_active_densBE),
    ParamSpec(name='densBE_xi_out', default=0, info='Dimensionless outer radius at cloud edge', category='runtime_loaded', unit='dimensionless', exclude_from_snapshot=True, metadata_exclude=True, active_when=_active_densBE),
)

REGISTRY: "OrderedDict[str, ParamSpec]" = OrderedDict(
    (s.name, s) for s in SPECS
)


def specs_by_category(*categories: Category) -> Iterable[ParamSpec]:
    cat_set = set(categories)
    return (s for s in SPECS if s.category in cat_set)


def validate_all(params) -> None:
    """Run every spec's ``validator`` callable against ``params``.

    Phase-6 entry point for ``read_param`` Step 5.  A validator receives
    ``(value, params)`` and may either raise
    ``trinity._input.errors.ParameterFileError`` on bad input or normalize
    the value in place (e.g. coerce a whole-number float to ``int``).
    Order follows ``SPECS``; specs missing from ``params`` are skipped
    so densBE-only / densPL-only keys don't trigger on the other path.
    """
    for spec in SPECS:
        if spec.validator is None:
            continue
        if spec.name not in params:
            continue
        spec.validator(params[spec.name].value, params)


def resolve_all(params) -> None:
    """Run every spec's ``resolver`` callable against ``params``.

    Phase-7 entry point for ``read_param`` Step 7.  A resolver receives
    ``(value, params)`` and returns the resolved value, which is assigned
    back to ``params[spec.name].value``; it may also raise
    ``ParameterFileError`` / ``ValueError`` on bad input and mutate
    ``params`` for documented coupled side effects (see
    ``_resolve_sps_bundle``).  Order follows ``SPECS``; the three current
    resolvers (path2output, path_cooling_nonCIE, sps_path) carry no
    inter-dependencies in that order — sps_refmass and the sps_col_*
    family are owned by sps_path's resolver via ``consumed_by``, so the
    one cross-key ordering edge (refmass-after-path) lives inside a single
    resolver rather than across iterations.  Specs missing from ``params``
    are skipped.
    """
    for spec in SPECS:
        if spec.resolver is None:
            continue
        if spec.name not in params:
            continue
        params[spec.name].value = spec.resolver(params[spec.name].value, params)


def apply_active_when(params) -> None:
    """Enforce ``active_when`` presence semantics against ``params``.

    Phase-8 entry point for ``read_param`` Step 8.  For every spec
    carrying an ``active_when`` predicate, the invariant *"the spec is
    in ``params`` iff ``active_when(params)`` returns True"* is
    restored:

      * active and absent → add a fresh ``DescribedItem`` (default
        deep-copied so mutable defaults like ``[]`` aren't shared across
        runs);
      * present and inactive → ``pop`` it;
      * matching presence and activity → no-op.

    Order follows ``SPECS``.  Specs without ``active_when`` are skipped.
    Must run after Step 5 (validators ensure the gating values — today
    ``dens_profile`` ∈ {``densBE``, ``densPL``} — are well-formed) and
    before Step 9 (the snapshot-exclusion sweep, which expects the
    final key set).
    """
    for spec in SPECS:
        if spec.active_when is None:
            continue
        active = spec.active_when(params)
        present = spec.name in params
        if active and not present:
            params[spec.name] = DescribedItem(
                copy.deepcopy(spec.default),
                info=spec.info,
                ori_units=spec.unit if spec.unit is not None else "N/A",
                exclude_from_snapshot=spec.exclude_from_snapshot,
            )
        elif present and not active:
            params.pop(spec.name)


def materialize_runtime(params) -> None:
    """Phase-8/9 entry point for ``read_param`` Step 10.

    Add a fresh ``DescribedItem`` to ``params`` for every spec that
    isn't already present, isn't gated by ``active_when`` (Phase 8
    owns those), and isn't ``consumed_by`` another resolver (Phase 7's
    sps_path bundle owns sps_refmass + sps_col_*).  Each item is
    constructed from spec metadata: ``copy.deepcopy(spec.default)``
    (so mutable defaults like ``[]`` / ``np.array([])`` aren't shared
    across runs), ``info=spec.info``, ``ori_units=spec.unit`` (or
    ``"N/A"`` when unitless), ``exclude_from_snapshot=spec.exclude_from_snapshot``.

    Order follows ``SPECS``.  Must run AFTER Step 9 — the
    time-varying-keys sweep treats only items present *at that point*
    (input/active_when/derived from Step 6/7).  Newly materialized
    runtime items get their ``exclude_from_snapshot`` straight from
    the spec, which matches today's behavior since the original Step
    10 also constructed them after Step 9.

    Today this adds 103 items: 9 with ``exclude_from_snapshot=True``
    (the cooling cubes, sps_data/sps_f, the rcloud counter) and 94
    with False (time-varying simulation state — bubble_*, shell_*,
    forces, residuals, etc.).  mCloud_input / mCluster (Step 6) and
    sps_column_map (Step 7) are pre-added and skipped here.
    """
    for spec in SPECS:
        if spec.active_when is not None:
            continue
        if spec.consumed_by is not None:
            continue
        if spec.name in params:
            continue
        params[spec.name] = DescribedItem(
            copy.deepcopy(spec.default),
            info=spec.info,
            ori_units=spec.unit if spec.unit is not None else "N/A",
            exclude_from_snapshot=spec.exclude_from_snapshot,
        )


def run_const_keys() -> tuple[str, ...]:
    """Keys written once to ``metadata.json`` (constant after phase 0).

    Phase-5 drop-in replacement for
    ``trinity._output.run_constants.RUN_CONST_KEYS``.
    """
    return tuple(s.name for s in SPECS if s.run_const)


def metadata_exclude_keys() -> frozenset[str]:
    """Keys explicitly blocked from ``metadata.json`` (paths / loaded
    tables / empty array placeholders) even though they look constant.

    Phase-5 drop-in replacement for
    ``trinity._output.run_constants.METADATA_EXCLUDE``.
    """
    return frozenset(s.name for s in SPECS if s.metadata_exclude)


# ---------------------------------------------------------------------------
# Companion-key rules
# ---------------------------------------------------------------------------
# Some parameters are silent traps in isolation: setting ``dens_profile
# densPL`` in a .param without ``densPL_alpha`` silently yields the
# default alpha=0 (homogeneous) -- almost never what a user who bothered
# to declare the profile actually wanted.  CompanionRule lets such
# trigger-companion bundles be declared declaratively; ``read_param``
# Step 3 calls ``validate_companions`` on the raw user dict (before
# merging with defaults) so the check fires only when the user
# explicitly set the trigger, not when the trigger came from
# default.param.
@dataclass(frozen=True)
class CompanionRule:
    """If the user .param sets ``trigger`` to a value present as a key
    in ``requires``, every name in ``requires[value]`` must also appear
    in the same .param file."""
    trigger: str
    requires: Mapping[Any, tuple[str, ...]]


COMPANION_RULES: tuple[CompanionRule, ...] = (
    CompanionRule(
        trigger='dens_profile',
        requires={
            'densPL': ('densPL_alpha',),
            'densBE': ('densBE_Omega',),
        },
    ),
)


def validate_companions(user_dict: Mapping[str, Any]) -> None:
    """Enforce every ``CompanionRule`` against the raw user .param dict.

    Called from ``read_param`` Step 3 with the freshly-parsed user
    dictionary (post-Step-2, pre-merge).  Raises ``ParameterFileError``
    on the first violation, listing the missing companion keys.
    """
    for rule in COMPANION_RULES:
        if rule.trigger not in user_dict:
            continue
        trigger_value = user_dict[rule.trigger]
        required = rule.requires.get(trigger_value)
        if not required:
            continue
        missing = [k for k in required if k not in user_dict]
        if missing:
            raise ParameterFileError(
                f"setting {rule.trigger}={trigger_value!r} requires "
                f"explicit values for: {', '.join(missing)}. "
                f"The defaults are only safe when {rule.trigger} is "
                f"left at its default too -- declaring the trigger "
                f"without its companion silently picks a value the "
                f"user almost certainly didn't intend."
            )
