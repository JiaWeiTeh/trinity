# -*- coding: utf-8 -*-
"""
Centralised force-colour palette for all ``_plots`` paper scripts.

Three ChromaPalette-based schemes are provided:

- ``"vibrance"`` — Vibrance + Elegant (original default)
- ``"eastern"``  — EasternHues
- ``"vintage"``  — VintageBlend

Switch globally by calling :func:`set_palette` **before** any plotting code
runs, or by setting the environment variable ``TRINITY_PALETTE``.

Usage
-----
>>> from src._plots.force_colors import C, FORCE_FIELDS_BASE, set_palette
>>> set_palette("eastern")      # switch palette at runtime
>>> print(C.GRAV)               # access a colour constant
"""

import os as _os

# ======================================================================
# Palette definitions  (ChromaPalette hex values)
# ======================================================================
# Each dict maps a *logical role* to a hex colour.
# Keys:  GRAV, DRIVE, RAD, PISM, WIND, SN, PHII,
#        THERMAL, GAS, ACC, NET,
#        DOM_GRAV, DOM_DRIVE, DOM_HII, DOM_WIND, DOM_SN, DOM_RAD, DOM_PISM

_PALETTES = {
    # ------------------------------------------------------------------
    # Vibrance + Elegant  (original scheme)
    # ------------------------------------------------------------------
    "vibrance": dict(
        GRAV     = "#1a1a1a",
        DRIVE    = "#508ab2",
        RAD      = "#a1d0c7",
        PISM     = "#FFFFFF",
        WIND     = "#8b6ca7",
        SN       = "#d5ba82",
        PHII     = "#e04050",
        # simplified 3-force plots
        THERMAL  = "blue",
        GAS      = "blue",
        ACC      = "orange",
        NET      = "gray",
        # dominant-feedback grid
        DOM_GRAV = "#1a1a1a",
        DOM_DRIVE= "#508ab2",
        DOM_HII  = "#b36a6f",
        DOM_WIND = "#8b6ca7",
        DOM_SN   = "#d5ba82",
        DOM_RAD  = "#a1d0c7",
        DOM_PISM = "#999999",
        # dominant-feedback grid (paper_dominantFeedback)
        GRID_GRAV    = "#2c3e50",
        GRID_WIND    = "#3498db",
        GRID_SN      = "#DAA520",
        GRID_ION_OUT = "#e74c3c",
        GRID_RAD     = "#9b59b6",
    ),

    # ------------------------------------------------------------------
    # EasternHues
    # ------------------------------------------------------------------
    "eastern": dict(
        GRAV     = "#563F2E",
        DRIVE    = "#2B5F75",
        RAD      = "#78C2C4",
        PISM     = "#FFFFFF",
        WIND     = "#0F4C3A",
        SN       = "#D9AB42",
        PHII     = "#C73E3A",
        THERMAL  = "#2B5F75",
        GAS      = "#2B5F75",
        ACC      = "#B47157",
        NET      = "#A35E47",
        DOM_GRAV = "#563F2E",
        DOM_DRIVE= "#2B5F75",
        DOM_HII  = "#C73E3A",
        DOM_WIND = "#0F4C3A",
        DOM_SN   = "#D9AB42",
        DOM_RAD  = "#78C2C4",
        DOM_PISM = "#A35E47",
        GRID_GRAV    = "#563F2E",
        GRID_WIND    = "#0F4C3A",
        GRID_SN      = "#D9AB42",
        GRID_ION_OUT = "#C73E3A",
        GRID_RAD     = "#78C2C4",
    ),

    # ------------------------------------------------------------------
    # VintageBlend
    # ------------------------------------------------------------------
    "vintage": dict(
        GRAV     = "#034960",
        DRIVE    = "#26889F",
        RAD      = "#AA8F42",
        PISM     = "#FFFFFF",
        WIND     = "#936569",
        SN       = "#AA8F42",
        PHII     = "#692E24",
        THERMAL  = "#26889F",
        GAS      = "#26889F",
        ACC      = "#936569",
        NET      = "#AA8F42",
        DOM_GRAV = "#034960",
        DOM_DRIVE= "#26889F",
        DOM_HII  = "#692E24",
        DOM_WIND = "#936569",
        DOM_SN   = "#AA8F42",
        DOM_RAD  = "#AA8F42",
        DOM_PISM = "#034960",
        GRID_GRAV    = "#034960",
        GRID_WIND    = "#936569",
        GRID_SN      = "#AA8F42",
        GRID_ION_OUT = "#692E24",
        GRID_RAD     = "#26889F",
    ),
}

DEFAULT_PALETTE = "vibrance"

# ======================================================================
# Runtime colour container
# ======================================================================

class _Colors:
    """Attribute-style access to the active colour mapping."""

    def __init__(self):
        self._map: dict = {}

    def _load(self, name: str):
        self._map = _PALETTES[name]

    def __getattr__(self, key: str) -> str:
        try:
            return self._map[key]
        except KeyError:
            raise AttributeError(
                f"No colour key {key!r} in the active palette"
            ) from None


C = _Colors()


# ======================================================================
# Derived structures (rebuilt on every palette switch)
# ======================================================================

# These module-level lists/dicts are *replaced in-place* by set_palette()
# so that any module that did ``from force_colors import FORCE_FIELDS_BASE``
# keeps a reference to the same mutable object.

FORCE_FIELDS_BASE: list = []       # paper_feedback
FORCE_FIELDS_MOMENTUM: list = []   # paper_momentum  (solid lines)
DASHED_FIELDS: list = []           # paper_momentum  (dashed lines)
DOMINANT_COLORS: dict = {}         # paper_momentum  (colour strip)
FORCE_FIELDS_GRID: list = []       # paper_dominantFeedback
FORCE_FIELDS_FRACTION: list = []   # paper_forceFraction
ACCEL_FIELDS: list = []            # paper_accelerationDecomposition


def _rebuild():
    """Rebuild all derived structures from the active palette."""

    # -- paper_feedback
    FORCE_FIELDS_BASE.clear()
    FORCE_FIELDS_BASE.extend([
        ("F_grav",   "Gravity",                 C.GRAV),
        ("F_drive",  r"$F_{\rm drive}$",        C.DRIVE),
        ("F_rad",    "Radiation (dir.+indir.)",  C.RAD),
        ("F_ion_in", "PISM (inner HII)",        C.PISM),
    ])

    # -- paper_momentum  (solid)
    FORCE_FIELDS_MOMENTUM.clear()
    FORCE_FIELDS_MOMENTUM.extend([
        ("F_grav",  "Gravity",                  C.GRAV),
        ("F_drive", r"$F_{\rm drive}$",         C.DRIVE),
        ("F_rad",   "Radiation (dir.+indir.)",  C.RAD),
        ("F_PISM",  "PISM (inner HII)",         C.DOM_PISM),
    ])

    # -- paper_momentum  (dashed)
    DASHED_FIELDS.clear()
    DASHED_FIELDS.extend([
        ("F_ion_out",  r"$P_{\rm HII}$", C.DOM_HII),
        ("F_ram_wind", "Wind",            C.DOM_WIND),
        ("F_ram_SN",   "Supernovae",      C.DOM_SN),
    ])

    # -- paper_momentum  dominant strip
    DOMINANT_COLORS.clear()
    DOMINANT_COLORS.update({
        "F_grav":     C.DOM_GRAV,
        "F_drive":    C.DOM_DRIVE,
        "F_ion_out":  C.DOM_HII,
        "F_ram_wind": C.DOM_WIND,
        "F_ram_SN":   C.DOM_SN,
        "F_rad":      C.DOM_RAD,
        "F_PISM":     C.DOM_PISM,
    })

    # -- paper_dominantFeedback grid
    FORCE_FIELDS_GRID.clear()
    FORCE_FIELDS_GRID.extend([
        ("F_grav",     "Gravity",          C.GRID_GRAV),
        ("F_ram_wind", "Winds",            C.GRID_WIND),
        ("F_ram_SN",   "Supernovae",       C.GRID_SN),
        ("F_ion_out",  "Photoionised gas", C.GRID_ION_OUT),
        ("F_rad",      "Radiation",        C.GRID_RAD),
    ])

    # -- paper_forceFraction
    FORCE_FIELDS_FRACTION.clear()
    FORCE_FIELDS_FRACTION.extend([
        ("F_grav",    r"$F_{\rm grav}$",    C.GRAV),
        ("F_thermal", r"$F_{\rm thermal}$", C.THERMAL),
        ("F_rad",     r"$F_{\rm rad}$",     C.RAD),
    ])

    # -- paper_accelerationDecomposition
    ACCEL_FIELDS.clear()
    ACCEL_FIELDS.extend([
        ("a_gas",  r"$a_{\rm gas}$",  C.GAS,  "-",  1.5),
        ("a_rad",  r"$a_{\rm rad}$",  C.RAD,  "-",  1.5),
        ("a_grav", r"$a_{\rm grav}$", C.GRAV, "-",  1.5),
        ("a_acc",  r"$a_{\rm acc}$",  C.ACC,  "-",  1.5),
        ("a_net",  r"$a_{\rm net}$",  C.NET,  "--", 2.5),
    ])


# ======================================================================
# Public API
# ======================================================================

def set_palette(name: str = DEFAULT_PALETTE):
    """
    Switch the active colour palette.

    Parameters
    ----------
    name : str
        One of ``"vibrance"``, ``"eastern"``, or ``"vintage"``.
    """
    name = name.lower()
    if name not in _PALETTES:
        raise ValueError(
            f"Unknown palette {name!r}. "
            f"Choose from: {', '.join(sorted(_PALETTES))}"
        )
    C._load(name)
    _rebuild()


def get_palette_names():
    """Return the list of available palette names."""
    return sorted(_PALETTES.keys())


# ======================================================================
# Initialise on import  (honours $TRINITY_PALETTE)
# ======================================================================
set_palette(_os.environ.get("TRINITY_PALETTE", DEFAULT_PALETTE))
