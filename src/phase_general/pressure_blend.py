#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure blending utilities for TRINITY phases.

This module provides the convex blend weight calculation for
P_drive = (1-w)*P_b + w*P_IF, using an independent Strömgren-based
pressure to break the P_IF ∝ P_b degeneracy from shell structure.
"""

import numpy as np


def compute_blend_weight(
    Qi: float,
    caseB_alpha: float,
    R2: float,
    k_B: float,
    P_b: float,
    f_abs_ion: float,
    T_ion: float = 1e4,
) -> tuple:
    """
    Compute convex blend weight for P_drive = (1-w)*P_b + w*P_IF.

    Uses independent Strömgren pressure for the weight to break
    the P_IF ∝ P_b degeneracy from the shell structure boundary condition.

    The shell structure solver sets inner shell density via pressure equilibrium
    with P_b, so P_IF ∝ P_b. This makes the ratio P_IF/(P_IF + P_b) ≈ 0.5 always,
    regardless of physical regime. Using P_HII_Str (which depends on Qi and R2,
    NOT on P_b) allows the weight to genuinely track energy vs momentum regimes.

    Parameters
    ----------
    Qi : float
        Ionizing photon rate [internal units]
    caseB_alpha : float
        Case B recombination coefficient [internal units]
    R2 : float
        Shell radius [internal units, pc]
    k_B : float
        Boltzmann constant [internal units]
    P_b : float
        Bubble/ram pressure [internal units]
    f_abs_ion : float
        Fraction of ionizing photons absorbed by shell
    T_ion : float, optional
        Ionized gas temperature [K], default 1e4

    Returns
    -------
    w_blend : float
        Blend weight in [0, f_abs_ion]
    n_Str : float
        Strömgren density [internal units]
    P_HII_Str : float
        Strömgren pressure [internal units]

    Notes
    -----
    The Strömgren density is computed as:
        n_Str = sqrt(3 * Qi / (4 * pi * alpha_B * R2^3))

    This is the characteristic HII region density for ionization equilibrium
    at radius R2, independent of bubble pressure.

    All inputs are expected in TRINITY internal units (Msun, pc, Myr).
    No unit conversions are performed here.

    Regime behavior:
    - Energy-driven (P_b >> P_HII_Str): w → 0, P_drive → P_b
    - Transition (P_b ~ P_HII_Str): w ~ f_abs/2, blended
    - Momentum (P_b << P_HII_Str): w → f_abs_ion, P_drive → P_IF
    """
    # Strömgren density — independent of P_b, depends only on Q_i and R2
    # All quantities already in internal units (Msun, pc, Myr)
    n_Str = np.sqrt(3.0 * Qi / (4.0 * np.pi * caseB_alpha * R2**3))

    # Strömgren pressure (factor of 2 for electron + ion contributions)
    P_HII_Str = 2.0 * n_Str * k_B * T_ion

    # Blend weight: uses P_HII_Str to determine WHEN HII pressure matters
    denom = P_HII_Str + P_b
    if denom > 1e-30:
        w_blend = f_abs_ion * P_HII_Str / denom
    else:
        w_blend = 0.0

    return w_blend, n_Str, P_HII_Str
