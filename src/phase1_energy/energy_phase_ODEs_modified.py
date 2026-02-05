#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified energy phase ODEs with pure functions.

This module provides ODE functions that do NOT mutate the params dictionary.
This is essential for using adaptive ODE solvers like scipy.integrate.solve_ivp,
which take trial steps that can be rejected. If ODE functions mutate state during
rejected trial steps, the params dictionary becomes corrupted.

Key difference from energy_phase_ODEs.py:
- get_ODE_Edot_pure() returns only derivatives, never writes to params
- All parameters are read at the start and passed as a frozen snapshot
- Dictionary updates happen only after successful integration segments

@author: Jia Wei Teh
"""
import numpy as np
import scipy.optimize
import src.cloud_properties.mass_profile as mass_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.sb99.update_feedback import get_currentSB99feedback
from src.phase1_energy.energy_phase_ODEs import get_press_ion  # Use existing function
from src.phase_general.pressure_blend import compute_blend_weight
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _scalar(x):
    """Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x."""
    a = np.asarray(x)
    return a.item() if a.size == 1 else x


@dataclass
class ODESnapshot:
    """
    Frozen snapshot of parameters needed for ODE evaluation.

    This captures all values needed by the ODE function at the start of
    an integration segment, ensuring the ODE function never reads from
    or writes to the params dictionary during integration.
    """
    # Shell properties
    shell_fAbsorbedIon: float
    shell_F_rad: float
    rShell: float
    shell_mass: float
    isCollapse: bool
    n_IF: float  # Density at ionization front (for P_IF convex blend)

    # Cluster/bubble properties
    mCluster: float
    bubble_LTotal: float
    Qi: float
    Lmech_total: float
    v_mech_total: float

    # Physical constants
    G: float
    k_B: float
    gamma_adia: float
    caseB_alpha: float

    # ISM properties
    PISM: float
    nISM: float
    TShell_ion: float

    # Timing
    tSF: float

    # Phase info
    current_phase: str
    EarlyPhaseApproximation: bool

    # Cloud properties
    rCloud: float


def create_ODE_snapshot(params) -> ODESnapshot:
    """
    Create a frozen snapshot of all parameters needed for ODE evaluation.

    This should be called once at the start of each integration segment,
    not during ODE evaluation.
    """
    return ODESnapshot(
        shell_fAbsorbedIon=params['shell_fAbsorbedIon'].value,
        shell_F_rad=params['shell_F_rad'].value,
        rShell=params['rShell'].value,
        shell_mass=params['shell_mass'].value,
        isCollapse=params['isCollapse'].value,
        n_IF=params['n_IF'].value,
        mCluster=params['mCluster'].value,
        bubble_LTotal=params['bubble_LTotal'].value,
        Qi=params['Qi'].value,
        Lmech_total=params['Lmech_total'].value,
        v_mech_total=params['v_mech_total'].value,
        G=params['G'].value,
        k_B=params['k_B'].value,
        gamma_adia=params['gamma_adia'].value,
        caseB_alpha=params['caseB_alpha'].value,
        PISM=params['PISM'].value,
        nISM=params['nISM'].value,
        TShell_ion=params['TShell_ion'].value,
        tSF=params['tSF'].value,
        current_phase=params['current_phase'].value,
        EarlyPhaseApproximation=params['EarlyPhaseApproximation'].value,
        rCloud=params['rCloud'].value,
    )


def get_ODE_Edot_pure(t: float, y: list, snapshot: ODESnapshot, params_for_feedback):
    """
    Pure ODE function for bubble expansion.

    IMPORTANT: This function does NOT write to any dictionary.
    It only reads from the frozen snapshot and returns derivatives.

    Parameters
    ----------
    t : float
        Time in Myr
    y : list
        State vector [R2, v2, Eb] - radius, velocity, energy
    snapshot : ODESnapshot
        Frozen snapshot of parameters
    params_for_feedback : DescribedDict
        Original params dict, used ONLY for get_currentSB99feedback
        (which needs interpolation tables that are too large to copy)

    Returns
    -------
    list
        Derivatives [rd, vd, Ed]
    """
    R2, v2, Eb = y

    # Get feedback values (this reads from params but doesn't write)
    feedback = get_currentSB99feedback(t, params_for_feedback)
    Lmech_total = feedback.Lmech_total
    v_mech_total = feedback.v_mech_total

    # Calculate shell mass using existing mass_profile module
    # (only reads from params, safe for ODE evaluation)
    # Two conditions for freezing shell mass:
    # 1. During collapse (isCollapse=True): shell mass is frozen
    # 2. Shell mass can NEVER decrease - once mass is swept up, it stays in shell
    if snapshot.isCollapse:
        mShell = snapshot.shell_mass
        mShell_dot = 0.0
    else:
        mShell_new, mShell_dot = mass_profile.get_mass_profile(
            R2, params_for_feedback, return_mdot=True, rdot=v2
        )
        # Ensure shell mass never decreases
        prev_mShell = snapshot.shell_mass
        if prev_mShell > 0 and mShell_new < prev_mShell:
            mShell = prev_mShell
            mShell_dot = 0.0
        else:
            mShell = mShell_new

    # Gravity force (self + cluster)
    F_grav = snapshot.G * mShell / (R2**2) * (snapshot.mCluster + 0.5 * mShell)

    # Calculate R1 (inner bubble radius)
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * R2, R2,
        args=([Lmech_total, Eb, v_mech_total, R2])
    )

    # Bubble pressure calculation (uses shared helper for ODE/diagnostics consistency)
    press_bubble = get_bubbleParams.get_effective_bubble_pressure(
        current_phase=snapshot.current_phase,
        Eb=Eb, R2=R2, R1=R1, gamma=snapshot.gamma_adia,
        Lmech_total=Lmech_total, v_mech_total=v_mech_total,
        t=t, tSF=snapshot.tSF
    )

    # Inward pressure from photoionized gas outside shell
    # Uses existing get_press_ion() which only reads from params
    FABSi = snapshot.shell_fAbsorbedIon
    if FABSi < 1.0:
        press_HII_in = get_press_ion(snapshot.rShell, params_for_feedback)
    else:
        press_HII_in = 0.0

    # Add ISM pressure if shell beyond cloud
    if snapshot.rShell >= snapshot.rCloud:
        press_HII_in += snapshot.PISM * snapshot.k_B

    # ==========================================================================
    # WARM IONIZED GAS PRESSURE (P_IF) - CONVEX BLEND APPROACH
    # P_drive = (1-w)*P_b + w*P_IF
    # Weight uses INDEPENDENT Strömgren pressure to break P_IF ∝ P_b degeneracy:
    #   w = f_abs_ion * P_HII_Str / (P_HII_Str + P_b)
    # where P_HII_Str = 2 * n_Str * k_B * T_ion, n_Str = sqrt(3*Qi / (4*pi*alpha_B*R2^3))
    # P_IF from shell structure is used for the blend VALUE (physically correct IF pressure).
    # ==========================================================================
    T_ion = 1e4  # K — standard HII region temperature

    # Pressure at ionization front (from shell structure) - used for blend VALUE
    P_IF = 2.0 * snapshot.n_IF * snapshot.k_B * T_ion

    # Blending weight: uses independent Strömgren pressure (breaks P_IF ∝ P_b degeneracy)
    w_blend, n_Str, P_HII_Str = compute_blend_weight(
        Qi=snapshot.Qi,
        caseB_alpha=snapshot.caseB_alpha,
        R2=R2,
        k_B=snapshot.k_B,
        P_b=press_bubble,
        f_abs_ion=FABSi,
        T_ion=T_ion
    )

    # Driving pressure as convex blend: P_drive = (1-w)*P_b + w*P_IF
    P_drive = (1.0 - w_blend) * press_bubble + w_blend * P_IF

    # Force from HII pressure contribution (for diagnostics)
    F_HII = 4.0 * np.pi * R2**2 * (P_drive - press_bubble)

    # Radiation force
    F_rad = snapshot.shell_F_rad

    # Rename press_HII_in to press_ext for clarity (external/confining pressure)
    press_ext = press_HII_in

    # Time derivatives
    rd = v2
    vd = (4.0 * np.pi * R2**2 * (P_drive - press_ext)
          - mShell_dot * v2 - F_grav + F_rad) / mShell

    # Early phase approximation
    if snapshot.EarlyPhaseApproximation:
        vd = -1e8

    # Energy derivative
    L_bubble = snapshot.bubble_LTotal
    L_leak = 0  # Future: add cover fraction
    Ed = (Lmech_total - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak

    logger.debug(f'Pure ODE: t={t:.6f}, R2={R2:.6e}, v2={v2:.6e}, Eb={Eb:.6e}')
    logger.debug(f'  -> rd={rd:.6e}, vd={vd:.6e}, Ed={Ed:.6e}')

    return [rd, vd, Ed]


@dataclass
class ODEResult:
    """
    Result from ODE evaluation, containing values to update params with.

    This is returned after a successful integration segment and contains
    all the values that should be written to params.
    """
    # State variables
    R2: float
    v2: float
    Eb: float
    t_now: float

    # Derived quantities (optional, computed during last ODE evaluation)
    R1: Optional[float] = None
    Pb: Optional[float] = None
    shell_mass: Optional[float] = None
    shell_massDot: Optional[float] = None
    F_grav: Optional[float] = None
    F_ion_in: Optional[float] = None
    F_ion_out: Optional[float] = None
    F_ram: Optional[float] = None
    F_rad: Optional[float] = None

    # P_IF diagnostic quantities (ionization front pressure - convex blend)
    n_IF: Optional[float] = None
    R_IF: Optional[float] = None
    P_IF: Optional[float] = None
    w_blend: Optional[float] = None
    P_drive: Optional[float] = None
    F_HII: Optional[float] = None
    # Strömgren diagnostics for blend weight (independent of P_b)
    n_Str: Optional[float] = None
    P_HII_Str: Optional[float] = None


def compute_derived_quantities(t: float, y: list, snapshot: ODESnapshot, params_for_feedback) -> ODEResult:
    """
    Compute all derived quantities after a successful integration step.

    This is called ONCE after integration completes, not during ODE evaluation.
    Returns an ODEResult dataclass that can be used with updateDict.
    """
    R2, v2, Eb = y

    feedback = get_currentSB99feedback(t, params_for_feedback)
    Lmech_total = feedback.Lmech_total
    v_mech_total = feedback.v_mech_total

    # Shell mass using existing mass_profile module
    # Two conditions for freezing shell mass:
    # 1. During collapse (isCollapse=True): shell mass is frozen
    # 2. Shell mass can NEVER decrease - once mass is swept up, it stays in shell
    if snapshot.isCollapse:
        mShell = snapshot.shell_mass
        mShell_dot = 0.0
    else:
        mShell_new, mShell_dot = mass_profile.get_mass_profile(
            R2, params_for_feedback, return_mdot=True, rdot=v2
        )
        # Ensure shell mass never decreases
        prev_mShell = snapshot.shell_mass
        if prev_mShell > 0 and mShell_new < prev_mShell:
            mShell = prev_mShell
            mShell_dot = 0.0
        else:
            mShell = mShell_new

    # R1
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * R2, R2,
        args=([Lmech_total, Eb, v_mech_total, R2])
    )

    # Bubble pressure (uses shared helper for ODE/diagnostics consistency)
    # In momentum phase, this returns pRam; in energy phase, returns bubble_E2P
    Pb = get_bubbleParams.get_effective_bubble_pressure(
        current_phase=snapshot.current_phase,
        Eb=Eb, R2=R2, R1=R1, gamma=snapshot.gamma_adia,
        Lmech_total=Lmech_total, v_mech_total=v_mech_total,
        t=t, tSF=snapshot.tSF
    )

    # Forces
    F_grav = snapshot.G * mShell / (R2**2) * (snapshot.mCluster + 0.5 * mShell)

    FABSi = snapshot.shell_fAbsorbedIon
    if FABSi < 1.0:
        press_HII_in = get_press_ion(snapshot.rShell, params_for_feedback)
    else:
        press_HII_in = 0.0

    if snapshot.rShell >= snapshot.rCloud:
        press_HII_in += snapshot.PISM * snapshot.k_B

    # ==========================================================================
    # P_IF CALCULATION - CONVEX BLEND (same as in ODE function)
    # P_drive = (1-w)*P_b + w*P_IF
    # Weight uses INDEPENDENT Strömgren pressure to break P_IF ∝ P_b degeneracy:
    #   w = f_abs_ion * P_HII_Str / (P_HII_Str + P_b)
    # where P_HII_Str = 2 * n_Str * k_B * T_ion, n_Str = sqrt(3*Qi / (4*pi*alpha_B*R2^3))
    # P_IF from shell structure is used for the blend VALUE (physically correct IF pressure).
    # ==========================================================================
    T_ion = 1e4  # K — standard HII region temperature

    # Pressure at ionization front (from shell structure via snapshot) - used for blend VALUE
    n_IF = snapshot.n_IF
    P_IF = 2.0 * n_IF * snapshot.k_B * T_ion

    # Blending weight: uses independent Strömgren pressure (breaks P_IF ∝ P_b degeneracy)
    w_blend, n_Str, P_HII_Str = compute_blend_weight(
        Qi=snapshot.Qi,
        caseB_alpha=snapshot.caseB_alpha,
        R2=R2,
        k_B=snapshot.k_B,
        P_b=Pb,
        f_abs_ion=FABSi,
        T_ion=T_ion
    )

    # Driving pressure as convex blend: P_drive = (1-w)*P_b + w*P_IF
    P_drive = (1.0 - w_blend) * Pb + w_blend * P_IF

    # Forces
    F_HII = 4.0 * np.pi * R2**2 * (P_drive - Pb)
    # F_ion_out kept for backwards compatibility
    F_ion_out = F_HII

    return ODEResult(
        R2=R2,
        v2=v2,
        Eb=Eb,
        t_now=t,
        R1=R1,
        Pb=Pb,
        shell_mass=mShell,
        shell_massDot=mShell_dot,
        F_grav=F_grav,
        F_ion_in=press_HII_in * 4 * np.pi * R2**2,
        F_ion_out=F_ion_out,
        F_ram=Pb * 4 * np.pi * R2**2,
        F_rad=snapshot.shell_F_rad,
        # P_IF diagnostic quantities (convex blend)
        n_IF=n_IF,
        R_IF=snapshot.rShell,  # Use rShell as proxy for R_IF (updated by shell structure)
        P_IF=P_IF,
        w_blend=w_blend,
        P_drive=P_drive,
        F_HII=F_HII,
        # Strömgren diagnostics (independent of P_b)
        n_Str=n_Str,
        P_HII_Str=P_HII_Str,
    )
