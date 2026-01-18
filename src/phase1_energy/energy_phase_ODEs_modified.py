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
from src.sb99.update_feedback import get_currentSB99feedback
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

    # Density profile parameters (for mass calculation)
    density_profile_params: dict


def create_ODE_snapshot(params) -> ODESnapshot:
    """
    Create a frozen snapshot of all parameters needed for ODE evaluation.

    This should be called once at the start of each integration segment,
    not during ODE evaluation.
    """
    # Collect density profile parameters needed by mass_profile
    density_params = {
        'nCore': params['nCore'].value,
        'rCore': params['rCore'].value,
        'nISM': params['nISM'].value,
        'rCloud': params['rCloud'].value,
        'k_B': params['k_B'].value,
        'TShell_neu': params['TShell_neu'].value,
        'TShell_ion': params['TShell_ion'].value,
        'mu_n': params['mu_n'].value,
        'mu_p': params['mu_p'].value,
    }

    return ODESnapshot(
        shell_fAbsorbedIon=params['shell_fAbsorbedIon'].value,
        shell_F_rad=params['shell_F_rad'].value,
        rShell=params['rShell'].value,
        shell_mass=params['shell_mass'].value,
        isCollapse=params['isCollapse'].value,
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
        density_profile_params=density_params,
    )


def get_press_ion_pure(r: float, snapshot: ODESnapshot) -> float:
    """
    Pressure from photoionized part of cloud at radius r.

    Pure version that uses snapshot instead of params dictionary.
    """
    r = np.atleast_1d(r)
    # Simplified density calculation using snapshot parameters
    dp = snapshot.density_profile_params

    # Replicate density_profile logic inline to avoid params dependency
    nCore = dp['nCore']
    rCore = dp['rCore']
    nISM = dp['nISM']
    rCloud = dp['rCloud']

    # Simplified Plummer-like profile
    n_r = nCore / (1 + (r / rCore)**2)**(3/2)
    # Ensure density doesn't drop below ISM
    n_r = np.maximum(n_r, nISM)

    P_ion = 2.0 * n_r * snapshot.k_B * snapshot.TShell_ion
    return _scalar(P_ion)


def get_shell_mass_pure(R2: float, v2: float, snapshot: ODESnapshot):
    """
    Calculate shell mass and its time derivative.

    Pure version that uses snapshot instead of params dictionary.
    Returns (mShell, mShell_dot).
    """
    if snapshot.isCollapse:
        return snapshot.shell_mass, 0.0

    dp = snapshot.density_profile_params
    nCore = dp['nCore']
    rCore = dp['rCore']
    nISM = dp['nISM']
    rCloud = dp['rCloud']
    mu_n = dp['mu_n']

    # Simplified mass calculation (Plummer sphere)
    # M(r) = (4/3) * pi * nCore * rCore^3 * [r/rCore / sqrt(1 + (r/rCore)^2)]
    x = R2 / rCore
    mass_factor = x / np.sqrt(1 + x**2)
    mShell = (4/3) * np.pi * nCore * mu_n * rCore**3 * mass_factor

    # Mass derivative: dM/dt = dM/dr * dr/dt
    # dM/dr = 4 * pi * r^2 * rho(r)
    n_at_R2 = nCore / (1 + x**2)**(3/2)
    n_at_R2 = max(n_at_R2, nISM)
    dmdr = 4 * np.pi * R2**2 * n_at_R2 * mu_n
    mShell_dot = dmdr * v2

    return _scalar(mShell), _scalar(mShell_dot)


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

    # Calculate shell mass
    mShell, mShell_dot = get_shell_mass_pure(R2, v2, snapshot)

    # Gravity force (self + cluster)
    F_grav = snapshot.G * mShell / (R2**2) * (snapshot.mCluster + 0.5 * mShell)

    # Calculate R1 (inner bubble radius)
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * R2, R2,
        args=([Lmech_total, Eb, v_mech_total, R2])
    )

    # Bubble pressure calculation
    dt_switchon = 1e-3
    tmin = dt_switchon

    if snapshot.current_phase == 'momentum':
        press_bubble = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
    else:
        if t > (tmin + snapshot.tSF):
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, snapshot.gamma_adia)
        else:
            R1_tmp = (t - snapshot.tSF) / tmin * R1
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, snapshot.gamma_adia)

    # Inward pressure from photoionized gas outside shell
    FABSi = snapshot.shell_fAbsorbedIon
    if FABSi < 1.0:
        press_HII_in = get_press_ion_pure(snapshot.rShell, snapshot)
    else:
        press_HII_in = 0.0

    # Add ISM pressure if shell beyond cloud
    if snapshot.rShell >= snapshot.rCloud:
        press_HII_in += snapshot.PISM * snapshot.k_B

    # Photoionized gas from HII region
    if FABSi < 1:
        nR2 = snapshot.nISM
    else:
        nR2 = np.sqrt(snapshot.Qi / snapshot.caseB_alpha / R2**3 * 3 / 4 / np.pi)

    press_HII_out = 2 * nR2 * snapshot.k_B * 3e4

    # Radiation force
    F_rad = snapshot.shell_F_rad

    # Time derivatives
    rd = v2
    vd = (4 * np.pi * R2**2 * (press_bubble - press_HII_in + press_HII_out)
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

    # Shell mass
    mShell, mShell_dot = get_shell_mass_pure(R2, v2, snapshot)

    # R1
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * R2, R2,
        args=([Lmech_total, Eb, v_mech_total, R2])
    )

    # Bubble pressure
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, snapshot.gamma_adia)

    # Forces
    F_grav = snapshot.G * mShell / (R2**2) * (snapshot.mCluster + 0.5 * mShell)

    FABSi = snapshot.shell_fAbsorbedIon
    if FABSi < 1.0:
        press_HII_in = get_press_ion_pure(snapshot.rShell, snapshot)
    else:
        press_HII_in = 0.0

    if snapshot.rShell >= snapshot.rCloud:
        press_HII_in += snapshot.PISM * snapshot.k_B

    if FABSi < 1:
        nR2 = snapshot.nISM
    else:
        nR2 = np.sqrt(snapshot.Qi / snapshot.caseB_alpha / R2**3 * 3 / 4 / np.pi)
    press_HII_out = 2 * nR2 * snapshot.k_B * 3e4

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
        F_ion_out=press_HII_out * 4 * np.pi * R2**2,
        F_ram=Pb * 4 * np.pi * R2**2,
        F_rad=snapshot.shell_F_rad,
    )
