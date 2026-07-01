"""Load-time resolution of ``cooling_boost_kappa = auto``.

``auto`` resolves to the smallest Spitzer-conduction multiplier f_kappa that
made the ``cooling_balance`` energy->momentum trigger fire (theta =
Lloss/Lgain > 0.95, the Lancaster+2021 efficiently-cooled band), as MEASURED
on the 819-run (mCloud, sfe, nCore) sweep of 2026-06-29
(``docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv``, column
``f_kappa_fire_measured``; grid analysis in
``docs/dev/transition/pdv-trigger/data/make_fkappa_theta1_collapse.py``).

The sweep refuted a single-variable f_kappa(n_H) law (spread up to 32x across
mCloud/sfe at fixed density), so the lookup keeps all three axes: trilinear
interpolation in (log10 mCloud_input, log10 sfe, log10 nCore) of
log10 f_kappa_fire.  Notes:

* mCloud axis is the PRE-star-formation input mass (``mCloud_input``) -- the
  quantity the sweep folder names carried -- not the post-SFE ``mCloud``.
* Coordinates outside the calibrated hull are clamped to it, with a warning.
* Censored cells (the diffuse/high-SFE corner where nothing up to f_kappa=64
  fired) are filled with the sweep ceiling 64; a resolved value at that
  ceiling means the calibration could NOT demonstrate firing, and the
  resolver warns accordingly.
* Calibration was measured on flat power-law clouds (densPL, alpha=0),
  nISM=0.1, hybr solver.  Other profiles resolve on the same table with no
  measured guarantee (a warning is logged).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import trinity._functions.unit_conversions as cvt

logger = logging.getLogger(__name__)

# Sweep grid axes (log10). mCloud is the pre-SFE input mass.
_LOG_M = np.log10([1e5, 1e6, 1e7])
_LOG_SFE = np.log10([0.03, 0.1, 0.3])
_LOG_N = np.log10([1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5])

# Largest f_kappa the sweep tested; censored cells (never fired) carry it.
F_KAPPA_CEILING = 64.0
_C = F_KAPPA_CEILING

# f_kappa_fire_measured[mCloud, sfe, nCore] from fkappa_nH_sweep.csv.
_F_FIRE = np.array(
    [
        # mCloud = 1e5
        [
            [32.0, 16.0, 12.0, 4.0, 3.0, 1.0, 1.0],  # sfe = 0.03
            [64.0, 48.0, 32.0, 8.0, 6.0, 4.0, 1.5],  # sfe = 0.1
            [_C, _C, 48.0, 24.0, 16.0, 8.0, 4.0],
        ],  # sfe = 0.3  (2 censored)
        # mCloud = 1e6
        [
            [48.0, 16.0, 8.0, 4.0, 1.0, 1.0, 1.0],
            [64.0, 32.0, 24.0, 12.0, 4.0, 1.5, 1.0],
            [_C, 64.0, 48.0, 32.0, 12.0, 6.0, 1.0],
        ],  # 1 censored
        # mCloud = 1e7
        [
            [64.0, 32.0, 12.0, 1.0, 1.0, 1.0, 1.0],
            [_C, 48.0, 24.0, 8.0, 1.0, 1.0, 1.0],  # 1 censored
            [_C, _C, 48.0, 24.0, 6.0, 1.0, 1.0],
        ],  # 2 censored
    ]
)

_INTERP = RegularGridInterpolator((_LOG_M, _LOG_SFE, _LOG_N), np.log10(_F_FIRE), method="linear")


def fkappa_fire(mCloud_input: float, sfe: float, nCore: float) -> float:
    """Interpolated f_kappa needed for the cooling_balance trigger to fire.

    Pure lookup (no params dict): trilinear in log10 space, coordinates
    clamped to the calibrated hull. Returns a float >= 1.
    """
    lo = np.array([_LOG_M[0], _LOG_SFE[0], _LOG_N[0]])
    hi = np.array([_LOG_M[-1], _LOG_SFE[-1], _LOG_N[-1]])
    coords = np.log10([mCloud_input, sfe, nCore])
    clamped = np.clip(coords, lo, hi)
    if not np.array_equal(coords, clamped):
        logger.warning(
            "cooling_boost_kappa='auto': (mCloud_input=%.3g Msun, sfe=%.3g, "
            "nCore=%.3g cm-3) lies outside the calibrated grid "
            "(1e5-1e7, 0.03-0.3, 1e2-1e5); clamping to the hull.",
            mCloud_input,
            sfe,
            nCore,
        )
    return max(1.0, float(10.0 ** _INTERP(clamped)[0]))


def resolve_fkappa_auto(value, params):
    """Registry resolver for ``cooling_boost_kappa`` (read_param Step 7).

    Numeric values pass through UNTOUCHED (the default 1.0 path stays
    byte-identical).  The string 'auto' resolves via :func:`fkappa_fire`
    against mCloud_input / sfe / nCore.
    """
    if not (isinstance(value, str) and value.strip().lower() == "auto"):
        return value

    profile = params["dens_profile"].value
    alpha = params.get("densPL_alpha")
    if profile != "densPL" or (alpha is not None and alpha.value != 0.0):
        logger.warning(
            "cooling_boost_kappa='auto' was calibrated on flat power-law "
            "clouds (densPL, alpha=0); profile %r carries no measured "
            "guarantee that the trigger fires.",
            profile,
        )

    f_kappa = fkappa_fire(
        params["mCloud_input"].value,
        params["sfe"].value,
        # nCore is already in code units (pc^-3) by Step 7; the grid is cm^-3
        params["nCore"].value * cvt.ndens_au2cgs,
    )
    if f_kappa >= 0.999 * F_KAPPA_CEILING:
        logger.warning(
            "cooling_boost_kappa='auto' resolved to the calibration ceiling "
            "(%.3g): no tested f_kappa fired the cooling_balance trigger in "
            "this regime (diffuse/high-SFE corner); the run may stay "
            "energy-driven.",
            f_kappa,
        )
    else:
        logger.info("cooling_boost_kappa='auto' resolved to %.3g.", f_kappa)
    return f_kappa
