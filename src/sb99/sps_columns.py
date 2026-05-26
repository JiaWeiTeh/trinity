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
#
# Each per-canonical sub-dict also includes a convenience alias 'cgs' which
# maps to the canonical's default cgs unit (e.g. erg/s for luminosities,
# g*cm/s^2 for momentum rates, etc.). For dimensionless quantities 'cgs'
# is a synonym for 'dimensionless'. This lets users write
#     sps_col_Qi   0   cgs   log
# instead of having to remember that Qi's cgs unit is 1/s.
UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    't': {
        'yr':  1.0e-6,
        'Myr': 1.0,
        's':   cvt.s2Myr,
        'cgs': cvt.s2Myr,                          # alias for 's'
    },
    'Qi': {
        '1/s':   1.0 / cvt.s2Myr,
        '1/Myr': 1.0,
        'cgs':   1.0 / cvt.s2Myr,                  # alias for '1/s'
    },
    'fi': {
        'dimensionless': 1.0,
        'cgs':           1.0,                      # alias for 'dimensionless'
    },
    'Lbol':        {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au, 'cgs': cvt.L_cgs2au},
    'Lmech_W':     {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au, 'cgs': cvt.L_cgs2au},
    'Lmech_total': {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au, 'cgs': cvt.L_cgs2au},
    'Lmech_SN':    {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au, 'cgs': cvt.L_cgs2au},
    'Li':          {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au, 'cgs': cvt.L_cgs2au},
    'Ln':          {'erg/s': cvt.L_cgs2au, 'L_sun': _L_SUN_ERG_S * cvt.L_cgs2au, 'cgs': cvt.L_cgs2au},
    'pdot_W':      {'g*cm/s^2': cvt.pdot_cgs2au, 'cgs': cvt.pdot_cgs2au},
    'pdot_SN':     {'g*cm/s^2': cvt.pdot_cgs2au, 'cgs': cvt.pdot_cgs2au},
    'Mdot_SN': {
        'g/s':       cvt.g2Msun / cvt.s2Myr,       # g/s -> Msun/Myr
        'Msun/Myr':  1.0,
        'cgs':       cvt.g2Msun / cvt.s2Myr,       # alias for 'g/s'
    },
    'v_SN': {
        'cm/s':   cvt.cm2pc / cvt.s2Myr,           # cm/s -> pc/Myr
        'km/s':   cvt.v_kms2au,                    # km/s -> pc/Myr
        'pc/Myr': 1.0,
        'cgs':    cvt.cm2pc / cvt.s2Myr,           # alias for 'cm/s'
    },
}


# Legacy SB99 7-column positional preset. Injected as the column map for
# the bundled default file (sps_path = def_path → lib/default/sps/
# 1e6cluster_default.csv) so users do not need to declare sps_col_* lines.
# See analysis/sb99-refactor-audit.md §9 for background.
#
# Column order matches the canonical SB99 export layout:
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
    separated, exactly 3 fields). The first field is either:

      - a non-negative integer (0-based column index; works on any file,
        with or without a header), OR
      - a string column name matching the file's header row (the file
        must have a header for this to resolve).

    Raises ValueError on malformed input or unrecognized units.
    """
    parts = str(raw_value).split()
    if len(parts) != 3:
        raise ValueError(
            f"sps_col_{canonical} expects 3 fields "
            f"(<file_column> <units> <log|linear>), got: {raw_value!r}"
        )
    file_column_raw, units, log_or_linear = parts

    # First field: 0-based integer index (any file) OR header-row name
    # (file must have a header). Auto-detect: all-digits -> int.
    file_column = (int(file_column_raw)
                   if file_column_raw.isdigit() else file_column_raw)

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
    return ColumnSpec(file_column=file_column, units=units,
                      log=(log_or_linear == 'log'))


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
    """One-line error: what's expected, what's missing, what was declared."""
    missing = list(missing_required)
    if not fi_ok:
        missing.append('fi (or Li+Ln)')
    if Li_Ln_partial:
        missing.append('Li/Ln (both or neither)')
    if not sn_input_ok:
        missing.append('Lmech_total (or Lmech_SN)')
    declared = sorted(given.keys()) or []
    return (
        f"sps_path={sps_path!r}: missing sps_col_* for {missing}. "
        f"Required canonicals: t, Qi, Lbol, Lmech_W, pdot_W, "
        f"fi (or Li+Ln), Lmech_total (or Lmech_SN). "
        f"Declared: {declared}."
    )


def validate_t_monotonic(t: np.ndarray, filepath: str) -> None:
    """Validate that the time array is strictly increasing.

    scipy.interpolate.interp1d (used by read_SB99.get_interpolation)
    requires strict monotonicity; its native error message — "Expect x
    to not have duplicates" — is cryptic and fires deep in scipy.
    This check raises a clearer ValueError at load time, pointing at
    the file and the first offending row.

    Common cause of failure: the file's time column was written with
    too few significant figures (e.g. '%.2e' format collapses
    1.001e7, 1.002e7, 1.003e7 all to the same string "1.00e+07"),
    producing duplicates that scipy refuses to interpolate over.

    Applied by both _read_sb99_legacy and _read_sb99_user in
    src/sb99/read_SB99.py.
    """
    if len(t) < 2:
        return  # trivially fine.
    diffs = np.diff(t)
    if not np.any(diffs <= 0):
        return  # strictly increasing.

    bad = np.where(diffs <= 0)[0]
    first = int(bad[0])
    around = t[max(0, first - 1):first + 3]
    violation_pairs = [(int(i), int(i + 1)) for i in bad[:5]]
    suffix = ' ...' if len(bad) > 5 else ''
    raise ValueError(
        f"Non-monotonic or duplicate `t` values in {filepath}.\n"
        f"  First offending row pairs (0-based array indices, "
        f"post-unit-conversion): {violation_pairs}{suffix}\n"
        f"  t values around the first offender (Myr): "
        f"{around.tolist()}\n"
        f"  scipy.interpolate.interp1d requires strictly-increasing t.\n"
        f"  Common cause: the time column was written with too few "
        f"significant figures (e.g. '%.2e' collapses 1.001e7, 1.002e7,\n"
        f"  1.003e7 all to '1.00e+07'). Regenerate the file with "
        f"'%.5e' or finer precision on column 0."
    )


def _can_parse_float(s: str) -> bool:
    """True iff s parses as a float (covers integers, decimals, scientific
    notation, inf/nan). Used to distinguish data rows from header rows."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _scan_layout(filepath: str):
    """Scan filepath to determine (data_start, header_names, delimiter).

    Returns
    -------
    data_start : int
        Row index where numeric data begins (rows above are skipped).
    header_names : list[str]
        Captured header tokens, or [] if no header was detected. A header
        is "the non-blank non-# row immediately above data_start that has
        the same token count as the data row and contains at least one
        non-numeric token".
    delimiter : str or None
        ',' if the first data line contains a comma; else None
        (whitespace, the np.loadtxt default).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Pass 1: find the first all-numeric data line and sniff its delimiter.
    data_start = None
    delimiter = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        line_delim = ',' if ',' in line else None
        tokens = ([t.strip() for t in line.split(',')] if line_delim
                  else line.split())
        if tokens and all(_can_parse_float(t) for t in tokens):
            data_start = i
            delimiter = line_delim
            break

    if data_start is None:
        raise ValueError(
            f"No numeric data rows found in {filepath}. "
            "The file appears to contain only headers/comments/blank lines."
        )

    # Pass 2: header is the *immediately-preceding* non-blank non-# row,
    # IF it has matching token count and at least one non-numeric token.
    header_names = []
    data_line = lines[data_start]
    data_tokens = ([t.strip() for t in data_line.split(',')] if delimiter == ','
                   else data_line.split())
    for j in range(data_start - 1, -1, -1):
        s = lines[j].strip()
        if not s or s.startswith('#'):
            continue
        tokens = ([t.strip() for t in lines[j].split(',')] if delimiter == ','
                  else lines[j].split())
        if (len(tokens) == len(data_tokens)
                and any(not _can_parse_float(t) for t in tokens)):
            header_names = tokens
        break  # only the immediate predecessor counts

    return data_start, header_names, delimiter


def load_user_columns(filepath: str, column_map: Dict[str, ColumnSpec]
                      ) -> Dict[str, np.ndarray]:
    """Load a user-defined SPS file (any layout) and return a dict keyed by
    canonical name (raw values, no unit conversion yet).

    Supports:
      - .txt (whitespace) and .csv (comma) — delimiter auto-sniffed from
        the first data line.
      - Headered and headerless files — header auto-detected.
      - '#'-prefixed comment lines and blank lines anywhere above the data.

    Each ColumnSpec.file_column resolves as:
      - int (>= 0)  -> data[:, file_column]  (0-based positional)
      - str         -> data[:, header_names.index(file_column)]
                       (raises if no header detected or name not in header)
    """
    data_start, header_names, delimiter = _scan_layout(filepath)

    try:
        data = np.loadtxt(filepath, skiprows=data_start, delimiter=delimiter,
                          comments='#', ndmin=2)
    except Exception as e:
        raise IOError(f"Error reading SPS file {filepath}: {e}") from e

    n_cols = data.shape[1]

    raw: Dict[str, np.ndarray] = {}
    for canonical, spec in column_map.items():
        if isinstance(spec.file_column, int):
            if not 0 <= spec.file_column < n_cols:
                raise ValueError(
                    f"sps_col_{canonical} uses index {spec.file_column} "
                    f"but {filepath} has only {n_cols} columns "
                    f"(valid indices 0..{n_cols - 1})."
                )
            raw[canonical] = np.asarray(data[:, spec.file_column], dtype=float)
        else:
            if not header_names:
                raise ValueError(
                    f"sps_col_{canonical} uses header name {spec.file_column!r} "
                    f"but no header row was detected in {filepath}.\n"
                    "Either add a header row at the top of the file, or "
                    "rewrite this line with a 0-based integer column "
                    f"index instead (valid range: 0..{n_cols - 1})."
                )
            if spec.file_column not in header_names:
                raise ValueError(
                    f"sps_col_{canonical} uses header name {spec.file_column!r} "
                    f"but {filepath}'s header has: {header_names}.\n"
                    "Check spelling, or use a 0-based integer column index."
                )
            idx = header_names.index(spec.file_column)
            raw[canonical] = np.asarray(data[:, idx], dtype=float)
    return raw
