# -*- coding: utf-8 -*-
"""
Cloud-scale physical quantities shared across analysis scripts.

Provides canonical implementations of cloud radius, surface density,
free-fall time, cumulative integration, and binding energy so that each
``src._calc`` script can import rather than duplicate these helpers.
"""

import numpy as np

from src._functions.unit_conversions import CGS, CONV, INV_CONV

# Mean molecular weight per H nucleus (molecular gas)
MU_MOL = 1.4

# Velocity conversion: internal (pc/Myr) -> km/s
V_AU2KMS = INV_CONV.v_au2kms  # ~0.978


def cloud_radius_pc(mCloud_Msun: float, nCore_cm3: float) -> float:
    """Cloud radius for a uniform sphere [pc]."""
    rho_cgs = MU_MOL * CGS.m_H * nCore_cm3
    M_g = mCloud_Msun / CONV.g2Msun
    R_cm = (3.0 * M_g / (4.0 * np.pi * rho_cgs)) ** (1.0 / 3.0)
    return R_cm * CONV.cm2pc


def cloud_radius_cgs(mCloud_Msun: float, nCore_cm3: float) -> float:
    """Cloud radius for a uniform sphere [cm]."""
    rho = MU_MOL * CGS.m_H * nCore_cm3
    M_g = mCloud_Msun / CONV.g2Msun
    return (3.0 * M_g / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


def surface_density(mCloud: float, rCloud: float) -> float:
    """Mean surface density Sigma = M / (pi R^2)  [Msun pc^-2]."""
    return mCloud / (np.pi * rCloud ** 2)


def freefall_time_Myr(nCore_cm3: float) -> float:
    """Free-fall time t_ff = sqrt(3 pi / (32 G rho))  [Myr]."""
    rho = MU_MOL * CGS.m_H * nCore_cm3
    t_ff_s = np.sqrt(3.0 * np.pi / (32.0 * CGS.G * rho))
    return t_ff_s * CONV.s2Myr


def binding_energy(mCloud_Msun: float, rCloud_pc: float) -> float:
    """Gravitational binding energy (3/5) G M^2 / R  [Msun pc^2 Myr^-2]."""
    G_au = CONV.G_cgs2au
    return 0.6 * G_au * mCloud_Msun ** 2 / rCloud_pc


def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with result[0] = 0."""
    dx = np.diff(x)
    incr = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(incr)
    return out
