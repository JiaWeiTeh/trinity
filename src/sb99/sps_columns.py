#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical column registry, unit dispatch, and column-map parsing for the
SPS feedback loader.

This module is the single source of truth for:
  - which canonical column names exist (t, Lbol, Qi, ...)
  - which are required vs optional vs conditionally-required
  - which are mass-scaled (f_mass = mCluster / sps_refmass)
  - the canonical AU unit each column ends up in
  - the conversion factor from a declared unit string to canonical AU
  - the legacy SB99 7-column positional preset (the permanent fallback)
  - parsing of sps_col_<canonical> declarations from .param

Used by:
  - src/_input/read_param.py   (builds and validates params['sps_column_map'])
  - src/sb99/read_SB99.py      (loads the file via the column map)

Background: see analysis/sb99-refactor-audit.md, sections 9 (legacy as
permanent fallback) and 10 PR-2 (column-mapping design).
"""

from dataclasses import dataclass
from typing import Union, Dict, Tuple, Optional

import numpy as np

import src._functions.unit_conversions as cvt


# Solar luminosity in erg/s (no L_sun constant currently in cvt).
_L_SUN_ERG_S = 3.828e33


@dataclass(frozen=True)
class ColumnSpec:
    """Description of one column in an SPS file.

    file_column : str or int
        Column identifier. A string (header-row name) for user-defined
        sps_path. An int (positional index) for the legacy SB99 preset.
    units : str
        Declared unit string. Must be a key in UNIT_CONVERSIONS[canonical].
    log : bool
        True if file values are in log10 space; False if linear.
    """
    file_column: Union[str, int]
    units: str
    log: bool


@dataclass(frozen=True)
class CanonicalSpec:
    """Metadata describing a canonical column."""
    canonical_au_unit: str
    mass_scaled: bool
    required: bool
    derivation: Optional[str] = None  # human-readable note; runtime derivation
                                      # lives in read_SB99.read_SB99 itself.


# Canonical AU units the loader produces, regardless of what the file declares.
# Required = no derivation; the loader cannot produce this canonical without it.
CANONICALS: Dict[str, CanonicalSpec] = {
    't':           CanonicalSpec('Myr',              mass_scaled=False, required=True),
    'Qi':          CanonicalSpec('1/Myr',            mass_scaled=True,  required=True),
    'fi':          CanonicalSpec('dimensionless',    mass_scaled=False, required=True),
    'Lbol':        CanonicalSpec('Msun*pc^2/Myr^3',  mass_scaled=True,  required=True),
    'Lmech_W':     CanonicalSpec('Msun*pc^2/Myr^3',  mass_scaled=True,  required=True),
    'pdot_W':      CanonicalSpec('Msun*pc/Myr^2',    mass_scaled=True,  required=True),

    'Lmech_total': CanonicalSpec('Msun*pc^2/Myr^3',  mass_scaled=True,  required=False,
                                 derivation='Lmech_W + Lmech_SN (post-correction)'),
    'Lmech_SN':    CanonicalSpec('Msun*pc^2/Myr^3',  mass_scaled=True,  required=False,
                                 derivation='Lmech_total - Lmech_W'),
    'pdot_SN':     CanonicalSpec('Msun*pc/Myr^2',    mass_scaled=True,  required=False,
                                 derivation='Mdot_SN * v_SN (post-correction)'),
    'Mdot_SN':     CanonicalSpec('Msun/Myr',         mass_scaled=True,  required=False,
                                 derivation='2 * Lmech_SN / v_SN^2'),
    'v_SN':        CanonicalSpec('pc/Myr',           mass_scaled=False, required=False,
                                 derivation='FB_vSN constant'),
    'Li':          CanonicalSpec('Msun*pc^2/Myr^3',  mass_scaled=True,  required=False,
                                 derivation='Lbol * fi'),
    'Ln':          CanonicalSpec('Msun*pc^2/Myr^3',  mass_scaled=True,  required=False,
                                 derivation='Lbol * (1 - fi)'),
}

# Public list of canonical names (preserves declaration order).
CANONICAL_NAMES: Tuple[str, ...] = tuple(CANONICALS.keys())

# Strictly-required canonicals (loader cannot run without them).
# fi is "conditionally required" — see validate_user_column_map.
_REQUIRED_ALWAYS = ('t', 'Qi', 'Lbol', 'Lmech_W', 'pdot_W')


# Per-canonical map: declared unit string -> multiplicative factor that takes
# a LINEAR (already-exponentiated) value into the canonical AU unit.
#
#   Time:      target = Myr
#   Photon:    target = 1/Myr
#   Luminosity:target = Msun*pc^2/Myr^3   (Lbol, Lmech_*, Li, Ln)
#   Momentum:  target = Msun*pc/Myr^2     (pdot_*)
#   Mass-loss: target = Msun/Myr          (Mdot_SN)
#   Velocity:  target = pc/Myr            (v_SN)
#   Fraction:  target = dimensionless     (fi)
UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    't': {
        'yr':  1.0e-6,
        'Myr': 1.0,
        's':   cvt.s2Myr,
    },
    'Qi': {
        '1/s':   1.0 / cvt.s2Myr,
        '1/Myr': 1.0,
    },
    'fi': {
        'dimensionless': 1.0,
    },
    'Lbol':        {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au},
    'Lmech_W':     {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au},
    'Lmech_total': {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au},
    'Lmech_SN':    {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au},
    'Li':          {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au},
    'Ln':          {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au},
    'pdot_W':      {'g*cm/s^2': cvt.pdot_cgs2au},
    'pdot_SN':     {'g*cm/s^2': cvt.pdot_cgs2au},
    'Mdot_SN': {
        'g/s':       cvt.g2Msun / cvt.s2Myr,   # g/s -> Msun/Myr
        'Msun/Myr':  1.0,
    },
    'v_SN': {
        'cm/s':   cvt.cm2pc / cvt.s2Myr,       # cm/s -> pc/Myr
        'km/s':   cvt.v_kms2au,                # km/s -> pc/Myr
        'pc/Myr': 1.0,
    },
}


# Legacy SB99 7-column positional preset. The permanent fallback used when
# sps_path = def_path (see analysis/sb99-refactor-audit.md §9).
#
# Column order matches the existing SB99 files in lib/sps/starburst99/:
#   col 0: time [yr]              (linear)
#   col 1: log10 Qi [1/s]         (log)
#   col 2: log10 fi [-]           (log)        ← yes, log-space; the legacy
#                                                 loader does 10**file[:,2]
#   col 3: log10 Lbol [erg/s]     (log)
#   col 4: log10 Lmech_total[erg/s](log)
#   col 5: log10 pdot_W [g*cm/s^2](log)
#   col 6: log10 Lmech_W [erg/s]  (log)
LEGACY_SB99_COLUMN_MAP: Dict[str, ColumnSpec] = {
    't':           ColumnSpec(file_column=0, units='yr',            log=False),
    'Qi':          ColumnSpec(file_column=1, units='1/s',           log=True),
    'fi':          ColumnSpec(file_column=2, units='dimensionless', log=True),
    'Lbol':        ColumnSpec(file_column=3, units='erg/s',         log=True),
    'Lmech_total': ColumnSpec(file_column=4, units='erg/s',         log=True),
    'pdot_W':      ColumnSpec(file_column=5, units='g*cm/s^2',      log=True),
    'Lmech_W':     ColumnSpec(file_column=6, units='erg/s',         log=True),
}


# --- public helpers -------------------------------------------------------


def convert_to_canonical_au(arr, canonical: str, declared_units: str, log: bool):
    """Convert a raw column to canonical AU units.

    Steps applied in order:
      1. If log is True, exponentiate (10**arr).
      2. Multiply by UNIT_CONVERSIONS[canonical][declared_units].

    Mass scaling (multiply by f_mass) is applied separately by the caller,
    using CANONICALS[canonical].mass_scaled.

    Raises
    ------
    KeyError
        If `canonical` is not a recognized name.
    ValueError
        If `declared_units` is not a recognized unit for this canonical.
    """
    if canonical not in UNIT_CONVERSIONS:
        raise KeyError(
            f"Unknown canonical column '{canonical}'. "
            f"Recognized: {sorted(CANONICALS.keys())}"
        )
    table = UNIT_CONVERSIONS[canonical]
    if declared_units not in table:
        raise ValueError(
            f"Unsupported units '{declared_units}' for canonical '{canonical}'. "
            f"Recognized: {sorted(table.keys())}"
        )
    if log:
        arr = 10.0 ** arr
    return arr * table[declared_units]


def parse_sps_col_value(canonical: str, raw_value: str) -> ColumnSpec:
    """Parse a single sps_col_<canonical> .param value into a ColumnSpec.

    Expected raw_value: "<file_column>  <units>  <log|linear>"  (whitespace-
    separated, exactly 3 fields). Used for user-defined sps_path mode.

    Raises ValueError on malformed input or unrecognized units.
    """
    parts = str(raw_value).split()
    if len(parts) != 3:
        raise ValueError(
            f"sps_col_{canonical} expects 3 fields "
            f"(<file_column> <units> <log|linear>), got: {raw_value!r}"
        )
    file_column, units, log_or_linear = parts
    if log_or_linear not in ('log', 'linear'):
        raise ValueError(
            f"sps_col_{canonical}: third field must be 'log' or 'linear', "
            f"got: {log_or_linear!r}"
        )
    if canonical in UNIT_CONVERSIONS and units not in UNIT_CONVERSIONS[canonical]:
        raise ValueError(
            f"sps_col_{canonical}: unsupported units {units!r}. "
            f"Recognized for {canonical}: "
            f"{sorted(UNIT_CONVERSIONS[canonical].keys())}"
        )
    return ColumnSpec(file_column=file_column, units=units, log=(log_or_linear == 'log'))


def build_user_column_map(params) -> Dict[str, ColumnSpec]:
    """Walk all sps_col_<canonical> entries in params and assemble the user's
    column map. Entries still holding the def_unset sentinel are skipped.

    Validation against the required set happens in validate_user_column_map.

    Returns
    -------
    dict[canonical_name -> ColumnSpec]
    """
    column_map: Dict[str, ColumnSpec] = {}
    for canonical in CANONICAL_NAMES:
        key = f"sps_col_{canonical}"
        if key not in params:
            # Should never happen if default.param declares them all,
            # but be defensive: missing entries are treated as unset.
            continue
        raw_value = params[key].value
        if raw_value == 'def_unset' or raw_value is None:
            continue
        column_map[canonical] = parse_sps_col_value(canonical, raw_value)
    return column_map


def validate_user_column_map(column_map: Dict[str, ColumnSpec], sps_path: str) -> None:
    """Strict validation for user-mode column maps. Raises ValueError with a
    fillable template on failure -- see analysis/sb99-refactor-audit.md §10
    PR-2.

    Rules:
      - Every canonical in _REQUIRED_ALWAYS must be present.
      - Either fi present, OR both Li AND Ln present.
      - Lmech_total OR Lmech_SN must be present (loader needs at least one
        to drive the SN pipeline; Mdot_SN alone is not yet a supported entry
        point here).
      - Li XOR Ln is forbidden (must be both or neither).
    """
    missing_required = [c for c in _REQUIRED_ALWAYS if c not in column_map]

    have_fi = 'fi' in column_map
    have_Li = 'Li' in column_map
    have_Ln = 'Ln' in column_map
    fi_ok = have_fi or (have_Li and have_Ln)
    Li_Ln_partial = have_Li != have_Ln  # XOR

    have_Lmech_total = 'Lmech_total' in column_map
    have_Lmech_SN = 'Lmech_SN' in column_map
    sn_input_ok = have_Lmech_total or have_Lmech_SN

    if missing_required or not fi_ok or Li_Ln_partial or not sn_input_ok:
        raise ValueError(_format_missing_template(
            sps_path=sps_path,
            missing_required=missing_required,
            fi_ok=fi_ok,
            Li_Ln_partial=Li_Ln_partial,
            sn_input_ok=sn_input_ok,
            given=column_map,
        ))


def _format_missing_template(*, sps_path: str,
                             missing_required, fi_ok, Li_Ln_partial,
                             sn_input_ok, given) -> str:
    """Build the user-facing error message with a fillable sps_col_* template."""
    declared = sorted(given.keys()) or '(none)'
    diagnosis = []
    if missing_required:
        diagnosis.append(f"  Missing required canonicals: {missing_required}")
    if not fi_ok:
        diagnosis.append(
            "  Need either sps_col_fi, OR both sps_col_Li AND sps_col_Ln "
            "(to bypass the SB99 13.6 eV threshold)."
        )
    if Li_Ln_partial:
        diagnosis.append(
            "  sps_col_Li and sps_col_Ln must both be set, or neither."
        )
    if not sn_input_ok:
        diagnosis.append(
            "  Need either sps_col_Lmech_total or sps_col_Lmech_SN "
            "(to drive the SN derivation pipeline)."
        )
    diag_block = '\n'.join(diagnosis)

    template = (
        f"sps_path is set to {sps_path!r} but the column mapping is incomplete.\n"
        f"{diag_block}\n\n"
        "Add the following lines to your .param file, filling in the file's\n"
        "actual column names. Each line is:\n"
        "    sps_col_<canonical>    <file_column>    <units>    <log|linear>\n\n"
        "Required (no derivation fallback):\n"
        "    sps_col_t            <file_column>     yr                  linear\n"
        "    sps_col_Lbol         <file_column>     erg/s               log\n"
        "    sps_col_Lmech_W      <file_column>     erg/s               log\n"
        "    sps_col_Qi           <file_column>     1/s                 log\n"
        "    sps_col_pdot_W       <file_column>     g*cm/s^2            log\n"
        "    sps_col_fi           <file_column>     dimensionless       linear\n"
        "        (OR supply BOTH sps_col_Li AND sps_col_Ln instead of sps_col_fi)\n\n"
        "And EITHER (to drive the SN pipeline):\n"
        "    sps_col_Lmech_total  <file_column>     erg/s               log\n"
        "OR:\n"
        "    sps_col_Lmech_SN     <file_column>     erg/s               log\n\n"
        "Optional (skip derivation if provided):\n"
        "    sps_col_pdot_SN, sps_col_Mdot_SN, sps_col_v_SN,\n"
        "    sps_col_Li, sps_col_Ln, sps_col_Lmech_total\n\n"
        f"Recognized units (per canonical): see UNIT_CONVERSIONS in\n"
        f"src/sb99/sps_columns.py.\n\n"
        f"Currently declared in your .param: {declared}\n"
    )
    return template


def load_named_columns(filepath: str, column_map: Dict[str, ColumnSpec]
                       ) -> Dict[str, np.ndarray]:
    """Load a header-equipped CSV via numpy and return a dict keyed by
    canonical name (raw values, no unit conversion yet).

    Used for user-mode sps_path. The file MUST contain a header row whose
    names cover every file_column referenced by column_map. Header-less
    files trigger a clear error.
    """
    try:
        data = np.genfromtxt(filepath, names=True, dtype=None, encoding='utf-8')
    except Exception as e:
        raise IOError(f"Error reading SPS file {filepath}: {e}") from e

    available = data.dtype.names
    if available is None:
        raise ValueError(
            f"User-defined sps_path {filepath!r} appears to have no header row.\n"
            "Column matching by name requires a header. Add a comma- or\n"
            "whitespace-separated header line at the top of the file, then\n"
            "ensure your sps_col_* declarations reference the names you used."
        )

    missing_file_cols = [
        spec.file_column for spec in column_map.values()
        if spec.file_column not in available
    ]
    if missing_file_cols:
        raise ValueError(
            f"User-defined sps_path {filepath!r} is missing columns referenced "
            f"by your sps_col_* declarations: {missing_file_cols}.\n"
            f"Header columns actually present in the file: {list(available)}."
        )

    raw: Dict[str, np.ndarray] = {}
    for canonical, spec in column_map.items():
        # Cast to float so downstream math works regardless of file dtype.
        raw[canonical] = np.asarray(data[spec.file_column], dtype=float)
    return raw
