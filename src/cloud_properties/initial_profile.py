#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct the initial cloud profile ``(r_arr, n_arr, m_arr)`` from
the run-constant scalars stored in ``metadata.json``.

This is the inverse of ``src/phase0_init/get_InitCloudProp.py``: instead
of computing the arrays once at simulation start and persisting them
inline in every output file, we recreate them on demand from the run-
constant scalars (``mCloud``, ``nCore``, …) that are saved in
``metadata.json``.

Why
---
Storing the arrays inline costs ~71 KB per run snapshot and is
duplicated information — the arrays are a deterministic function of
~6 scalars.  This helper lets every consumer (plot scripts, cloudy
exporter) get the arrays back without paying the storage cost.

The implementation re-uses ``phase0_init._init_powerlaw_cloud`` /
``_init_bonnor_ebert_cloud`` via a minimal ``MockParam`` adapter so
the array-construction logic stays single-sourced inside the phase-0
initialiser.  Calling the constructor with post-correction scalars
is a no-op for the auto-correction branches (``nEdge < nISM`` etc.)
because those checks pass given the already-corrected inputs.

Public API
----------
``build_initial_cloud_profile(*, dens_profile, ...)`` → ``(r, n, m)``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


# Density profiles understood by the phase-0 initialiser.  Mirrors
# ``src/_input/read_param.py`` and ``src/_output/cloudy/run_loader.py``.
_VALID_DENS_PROFILES = ("densPL", "densBE")


class _MockItem:
    """Minimal DescribedItem stand-in: just ``.value`` read/write.

    The phase-0 constructors expect ``params[key].value`` access and
    occasionally write back via ``params[key].value = ...`` during the
    auto-correction branches.  This duplicates only the surface area
    they touch — no metadata, no JSON helpers.
    """
    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value


def build_initial_cloud_profile(
    *,
    dens_profile: str,
    mCloud: float,
    nCore: float,
    nISM: float,
    rCore: float,
    rCloud: float,
    densPL_alpha: float = 0.0,
    mu_convert: float,
    densBE_Omega: float | None = None,
    gamma_adia: float | None = None,
    nEdge: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct ``(r_arr, n_arr, m_arr)`` from run-constant scalars.

    Parameters
    ----------
    dens_profile
        Either ``"densPL"`` (power-law) or ``"densBE"`` (Bonnor-Ebert).
    mCloud, nCore, nISM, rCore, rCloud, mu_convert
        Scalars common to both profiles (internal units: Msun, pc, etc.).
    densPL_alpha
        Power-law slope; ignored for BE.  Default ``0.0`` (homogeneous).
    densBE_Omega, gamma_adia
        Required for ``dens_profile="densBE"``; ignored for PL.
    nEdge
        Optional pre-computed edge density (BE only — initialised by
        ``_init_bonnor_ebert_cloud`` from the Lane-Emden solution).

    Returns
    -------
    r_arr, n_arr, m_arr
        Each a 1-D ``np.ndarray`` of equal length.  Matches the layout
        produced by ``phase0_init.get_InitCloudProp``.

    Raises
    ------
    ValueError
        If ``dens_profile`` is not one of the supported values, or if a
        BE profile is requested without the BE-specific scalars.
    """
    if dens_profile not in _VALID_DENS_PROFILES:
        raise ValueError(
            f"unknown dens_profile {dens_profile!r}; "
            f"expected one of {_VALID_DENS_PROFILES}"
        )

    # The phase-0 constructors mutate a few keys (rCore, rCloud, nEdge)
    # via auto-correction.  Wrap every value in a _MockItem so the
    # mutations land on our transient dict, not on a real params
    # container.  Lazy import avoids a circular dependency
    # (cloud_properties → phase0_init → cloud_properties).
    from src.phase0_init.get_InitCloudProp import (
        _init_powerlaw_cloud,
        _init_bonnor_ebert_cloud,
    )

    params: dict[str, _MockItem] = {
        "dens_profile": _MockItem(dens_profile),
        "mCloud": _MockItem(mCloud),
        "nCore": _MockItem(nCore),
        "nISM": _MockItem(nISM),
        "rCore": _MockItem(rCore),
        "rCloud": _MockItem(rCloud),
        "mu_convert": _MockItem(mu_convert),
        "densPL_alpha": _MockItem(densPL_alpha),
        # nEdge placeholder — ``_init_powerlaw_cloud`` will populate it
        # from (nCore, rCloud, rCore, alpha).  Pre-seed for the rare
        # edge-correction path that may read it.
        "nEdge": _MockItem(nEdge if nEdge is not None else float("nan")),
    }

    if dens_profile == "densPL":
        props = _init_powerlaw_cloud(params)
    else:
        if densBE_Omega is None or gamma_adia is None:
            raise ValueError(
                "dens_profile='densBE' requires densBE_Omega and "
                "gamma_adia scalars (from metadata.json)."
            )
        params["densBE_Omega"] = _MockItem(densBE_Omega)
        params["gamma_adia"] = _MockItem(gamma_adia)
        # _init_bonnor_ebert_cloud writes these BE-specific keys back.
        # Pre-seed them so the .value = ... assignment lands cleanly.
        params["densBE_Teff"] = _MockItem(None)
        params["densBE_xi_out"] = _MockItem(None)
        params["densBE_f_rho_rhoc"] = _MockItem(None)
        params["densBE_f_m"] = _MockItem(None)
        props = _init_bonnor_ebert_cloud(params)

    return (
        np.asarray(props.r_arr, dtype=float),
        np.asarray(props.n_arr, dtype=float),
        np.asarray(props.M_arr, dtype=float),
    )
