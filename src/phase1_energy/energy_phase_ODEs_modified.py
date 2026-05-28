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
import src.cloud_properties.density_profile as density_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.sps.update_feedback import get_current_sps_feedback
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _scalar(x):
    """Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x."""
    a = np.asarray(x)
    return a.item() if a.size == 1 else x


def get_press_ion(r, params):
    """
    Pressure from photoionized part of cloud at radius r.

    Parameters
    ----------
    r : float
        Radius in pc.
    params : DescribedDict
        Parameter dictionary.

    Returns
    -------
    float
        Pressure of ionized gas in code units.
    """
    r = np.atleast_1d(r)
    n_r = density_profile.get_density_profile(r, params)
    P_ion = 2.0 * n_r * params['k_B'].value * params['TShell_ion'].value
    return _scalar(P_ion)


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
    F_rad: float
    rShell: float
    shell_mass: float
    isCollapse: bool
    n_IF: float  # Density at ionization front (from shell structure ODE)
    include_PHII: bool  # Gate all HII pressure
    R_IF: float  # Radius of ionization front (pc)

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

    # HII pressure from Strömgren ionization balance in shell (n_IF_Str)
    P_HII: float


def create_ODE_snapshot(params, shell_props) -> ODESnapshot:
    """
    Create a frozen snapshot of all parameters needed for ODE evaluation.

    This should be called once at the start of each integration segment,
    not during ODE evaluation.

    Parameters
    ----------
    params : DescribedDict
        Main parameter dictionary.
    shell_props : ShellProperties
        Shell properties from the current shell_structure_pure() call.
        Used to compute the radiation pressure force inline.
    """
    # Radiation pressure force (direct + IR-trapped)
    if shell_props.isDissolved:
        F_rad = 0.0
    else:
        F_rad = (shell_props.shell_fAbsorbedWeightedTotal
                 * params['Lbol'].value / params['c_light'].value
                 * (1.0 + shell_props.shell_tauKappaRatio * params['dust_KappaIR'].value))

    return ODESnapshot(
        shell_fAbsorbedIon=params['shell_fAbsorbedIon'].value,
        F_rad=F_rad,
        rShell=params['rShell'].value,
        shell_mass=params['shell_mass'].value,
        isCollapse=params['isCollapse'].value,
        n_IF=params['n_IF'].value,
        R_IF=params['R_IF'].value,
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
        include_PHII=params['include_PHII'].value,
        P_HII=params['P_HII'].value,
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
        Original params dict, used ONLY for get_current_sps_feedback
        (which needs interpolation tables that are too large to copy)

    Returns
    -------
    list
        Derivatives [rd, vd, Ed]
    """
    R2, v2, Eb = y

    # Get feedback values (this reads from params but doesn't write)
    feedback = get_current_sps_feedback(t, params_for_feedback)
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
    rShell = snapshot.rShell
    if FABSi < 1.0:
        P_ext = get_press_ion(rShell, params_for_feedback)
    else:
        P_ext = 0.0

    # Add ISM pressure if shell beyond cloud
    if rShell >= snapshot.rCloud:
        P_ext += snapshot.PISM * snapshot.k_B

    # ==========================================================================
    # WARM IONIZED GAS PRESSURE
    # P_HII from Strömgren ionization balance in shell (n_IF_Str)
    # Pre-computed in phase runner and stored in snapshot.
    # ==========================================================================
    P_HII = snapshot.P_HII

    if snapshot.current_phase == 'transition':
        P_ram = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
        P_drive = max(press_bubble, P_HII + P_ram)
    else:
        # energy / implicit phases: max(Pb, P_HII)
        P_drive = max(press_bubble, P_HII)

    # Radiation force
    F_rad = snapshot.F_rad

    # Time derivatives
    rd = v2
    vd = (4.0 * np.pi * R2**2 * (P_drive - P_ext)
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
    F_HII: Optional[float] = None
    F_ram: Optional[float] = None
    F_rad: Optional[float] = None

    # Pressure quantities
    n_IF: Optional[float] = None
    R_IF: Optional[float] = None
    P_HII: Optional[float] = None
    P_drive: Optional[float] = None
    P_ram: Optional[float] = None
    press_HII_in: Optional[float] = None


def compute_derived_quantities(t: float, y: list, snapshot: ODESnapshot, params_for_feedback) -> ODEResult:
    """
    Compute all derived quantities after a successful integration step.

    This is called ONCE after integration completes, not during ODE evaluation.
    Returns an ODEResult dataclass that can be used with updateDict.
    """
    R2, v2, Eb = y

    feedback = get_current_sps_feedback(t, params_for_feedback)
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
    rShell = snapshot.rShell
    if FABSi < 1.0:
        P_ext = get_press_ion(rShell, params_for_feedback)
    else:
        P_ext = 0.0

    if rShell >= snapshot.rCloud:
        P_ext += snapshot.PISM * snapshot.k_B

    # ==========================================================================
    # WARM IONIZED GAS PRESSURE
    # P_HII from Strömgren ionization balance in shell (n_IF_Str)
    # Pre-computed in phase runner and stored in snapshot.
    # ==========================================================================
    P_HII = snapshot.P_HII

    if snapshot.current_phase == 'transition':
        P_ram = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
        P_drive = max(Pb, P_HII + P_ram)
    else:
        # energy / implicit phases: max(Pb, P_HII)
        P_drive = max(Pb, P_HII)

    F_HII = 4.0 * np.pi * R2**2 * P_HII

    # P_ram: only relevant in transition; 0 in energy/implicit
    if snapshot.current_phase == 'transition':
        P_ram_val = P_ram
    else:
        P_ram_val = 0.0

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
        F_ion_in=P_ext * 4 * np.pi * R2**2,
        F_HII=F_HII,
        F_ram=Pb * 4 * np.pi * R2**2,
        F_rad=snapshot.F_rad,
        # Pressure quantities
        n_IF=snapshot.n_IF,
        R_IF=snapshot.R_IF,
        P_HII=P_HII,
        P_drive=P_drive,
        P_ram=P_ram_val,
        press_HII_in=P_ext,
    )
