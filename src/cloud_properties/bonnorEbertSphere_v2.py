#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bonnor-Ebert Sphere Implementation - CORRECT VERSION v2

This module creates Bonnor-Ebert (BE) spheres, which are isothermal, self-gravitating
gas spheres in hydrostatic equilibrium. These model molecular cloud cores on the
verge of gravitational collapse.

User Inputs:
============
1. Cloud mass (mCloud) [Msun]
2. Core density (nCore) [cm^-3]
3. Density contrast (Omega = rho_core / rho_surface)

Outputs:
========
1. Cloud radius (rCloud) [pc]
2. Edge density (nEdge) [cm^-3]
3. Effective temperature (T_eff) [K]
4. Density profile function f_rho_rhoc(xi)

Physics:
========
The isothermal Lane-Emden equation:
    d²u/dξ² + (2/ξ) du/dξ = exp(-u)

Where:
    ξ = dimensionless radius = r × √(4πGρc/cs²)
    u(ξ) = dimensionless potential
    ρ(ξ)/ρc = exp(-u) = density contrast

Critical values:
    ξ_crit ≈ 6.451 (critical radius)
    Ω_crit ≈ 14.04 (critical density contrast)
    m_crit ≈ 1.182 (critical dimensionless mass)

CORRECT mass formula: m(ξ) = ξ² du/dξ

References:
===========
- Bonnor (1956): MNRAS 116, 351
- Ebert (1955): Z. Astrophys. 37, 217
- Rahner et al. (2017): MNRAS 470, 4453

@author: TRINITY Team (integrated from analysis/bonnorEbert/)
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import logging
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

# Import physical constants and unit conversions from central module
from src._functions.unit_conversions import CGS, INV_CONV

# Configure logging
logger = logging.getLogger(__name__)

# Physical constants in CGS (from central module)
G_CGS = CGS.G               # [cm³ g⁻¹ s⁻²]
K_B_CGS = CGS.k_B           # [erg K⁻¹]
M_H_CGS = CGS.m_H           # [g] hydrogen mass

# Unit conversions (from central module)
MSUN_TO_G = INV_CONV.Msun2g  # [g/Msun]
PC_TO_CM = INV_CONV.pc2cm    # [cm/pc]
MYR_TO_S = INV_CONV.Myr2s    # [s/Myr]

# ============================================================================
# BONNOR-EBERT SPHERE CONSTANTS
# ============================================================================

# Critical Bonnor-Ebert sphere parameters (from Lane-Emden solution)
OMEGA_CRITICAL = 14.04      # Critical density contrast ρc/ρsurf
XI_CRITICAL = 6.451         # Critical dimensionless radius

# Note on dimensionless mass conventions:
# - Bonnor (1956) definition: m_B = (1/√4π) × ξ² × du/dξ × √f ≈ 1.182 at critical
#   Formula: M = m_B × c_s⁴ / (G^(3/2) × √P_ext)
# - Integration-based definition (used here): m = ξ² × du/dξ ≈ 15.70 at critical
#   Formula: M = 4π × m × ρc × a³ (directly matches ∫4πr²ρ dr)
# Both give the SAME physical mass M, just different m conventions.
M_DIM_CRITICAL = 15.70      # Critical dimensionless mass (m = ξ² du/dξ)
M_BONNOR_CRITICAL = 1.182   # Bonnor's dimensionless mass (for reference)

# Integration parameters
XI_MIN = 1e-7               # Start point (near zero, avoid singularity)
XI_MAX = 20.0               # Maximum ξ (well beyond critical)
N_POINTS = 5000             # Number of integration points


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LaneEmdenSolution:
    """
    Container for Lane-Emden equation solution.

    Attributes
    ----------
    xi : np.ndarray
        Dimensionless radius array
    u : np.ndarray
        Dimensionless potential u(ξ)
    dudxi : np.ndarray
        Derivative du/dξ
    rho_rhoc : np.ndarray
        Density contrast ρ(ξ)/ρc = exp(-u)
    m : np.ndarray
        Dimensionless mass m(ξ) = ξ² du/dξ
    f_rho_rhoc : Callable
        Interpolation: ξ → ρ/ρc
    f_m : Callable
        Interpolation: ξ → m
    f_xi_from_rho : Callable
        Inverse interpolation: ρ/ρc → ξ
    """
    xi: np.ndarray
    u: np.ndarray
    dudxi: np.ndarray
    rho_rhoc: np.ndarray
    m: np.ndarray
    f_rho_rhoc: Callable
    f_m: Callable
    f_xi_from_rho: Callable


@dataclass
class BESphereResult:
    """
    Result of Bonnor-Ebert sphere creation.

    Attributes
    ----------
    xi_out : float
        Dimensionless outer radius
    r_out : float [pc]
        Physical outer radius
    n_out : float [cm⁻³]
        Surface number density
    T_eff : float [K]
        Effective isothermal temperature
    c_s : float [cm/s]
        Isothermal sound speed (CGS)
    m_dim : float
        Dimensionless mass
    M_cloud : float [Msun]
        Total cloud mass (input)
    n_core : float [cm⁻³]
        Core number density (input)
    Omega : float
        Density contrast (input)
    is_stable : bool
        Whether Omega < 14.04 (stable)
    """
    xi_out: float
    r_out: float
    n_out: float
    T_eff: float
    c_s: float
    m_dim: float
    M_cloud: float
    n_core: float
    Omega: float
    is_stable: bool


# ============================================================================
# LANE-EMDEN EQUATION SOLVER
# ============================================================================

def lane_emden_ode(y: np.ndarray, xi: float) -> np.ndarray:
    """
    Isothermal Lane-Emden equation as first-order ODE system.

    The equation: d²u/dξ² + (2/ξ) du/dξ = exp(-u)

    Rewritten as system:
        du/dξ = ω
        dω/dξ = exp(-u) - 2ω/ξ

    Parameters
    ----------
    y : array [u, ω]
        Current state
    xi : float
        Dimensionless radius

    Returns
    -------
    dydt : array [du/dξ, dω/dξ]
    """
    u, omega = y
    dudt = omega
    domegadt = np.exp(-u) - 2.0 * omega / xi
    return np.array([dudt, domegadt])


def get_initial_conditions(xi0: float = XI_MIN) -> Tuple[float, float]:
    """
    Get accurate initial conditions using series expansion.

    Near ξ = 0, the Lane-Emden equation has series solution:
        u(ξ) = ξ²/6 - ξ⁴/120 + ξ⁶/1890 + O(ξ⁸)
        du/dξ = ξ/3 - ξ³/30 + ξ⁵/315 + O(ξ⁷)

    This is much more accurate than arbitrary small values.
    """
    u0 = xi0**2 / 6.0 - xi0**4 / 120.0 + xi0**6 / 1890.0
    omega0 = xi0 / 3.0 - xi0**3 / 30.0 + xi0**5 / 315.0
    return u0, omega0


def solve_lane_emden(
    xi_max: float = XI_MAX,
    n_points: int = N_POINTS,
    xi_min: float = XI_MIN
) -> LaneEmdenSolution:
    """
    Solve the isothermal Lane-Emden equation.

    Solves: d²u/dξ² + (2/ξ) du/dξ = exp(-u)

    Parameters
    ----------
    xi_max : float, optional
        Maximum ξ (default: 20.0)
    n_points : int, optional
        Number of grid points (default: 5000)
    xi_min : float, optional
        Starting ξ (default: 1e-7)

    Returns
    -------
    solution : LaneEmdenSolution
        Complete solution with interpolation functions
    """
    logger.debug(f"Solving Lane-Emden: ξ ∈ [{xi_min}, {xi_max}], N={n_points}")

    # Get accurate initial conditions from series expansion
    u0, omega0 = get_initial_conditions(xi_min)

    # Integration grid (logarithmic spacing)
    xi = np.logspace(np.log10(xi_min), np.log10(xi_max), n_points)

    # Solve ODE
    solution = scipy.integrate.odeint(
        lane_emden_ode, [u0, omega0], xi, tfirst=False
    )

    u = solution[:, 0]
    dudxi = solution[:, 1]

    # Derived quantities
    rho_rhoc = np.exp(-u)

    # Dimensionless mass: m = ξ² × du/dξ
    m = xi**2 * dudxi

    logger.debug(f"Lane-Emden solved: u_max={u.max():.3f}, m_max={m.max():.3f}")

    # Create interpolation functions
    f_rho_rhoc = scipy.interpolate.interp1d(
        xi, rho_rhoc, kind='cubic',
        bounds_error=False, fill_value=(1.0, rho_rhoc[-1])
    )

    f_m = scipy.interpolate.interp1d(
        xi, m, kind='cubic',
        bounds_error=False, fill_value=(0.0, m[-1])
    )

    # Inverse interpolation: ρ/ρc → ξ (ρ/ρc decreases monotonically)
    rho_unique, idx_unique = np.unique(rho_rhoc[::-1], return_index=True)
    xi_unique = xi[::-1][idx_unique]
    f_xi_from_rho = scipy.interpolate.interp1d(
        rho_unique, xi_unique, kind='cubic',
        bounds_error=False, fill_value=(xi[-1], xi[0])
    )

    return LaneEmdenSolution(
        xi=xi, u=u, dudxi=dudxi, rho_rhoc=rho_rhoc, m=m,
        f_rho_rhoc=f_rho_rhoc, f_m=f_m, f_xi_from_rho=f_xi_from_rho
    )


# ============================================================================
# BONNOR-EBERT SPHERE CREATION
# ============================================================================

def create_BE_sphere(
    M_cloud: float,
    n_core: float,
    Omega: float,
    mu: float = 1.4,
    gamma: float = 5.0/3.0,
    validate: bool = True,
    lane_emden_solution: Optional[LaneEmdenSolution] = None
) -> BESphereResult:
    """
    Create Bonnor-Ebert sphere from user inputs.

    This uses the DIRECT analytical method - no nested optimization!

    User Inputs:
    -----------
    M_cloud : float [Msun]
        Total cloud mass
    n_core : float [cm⁻³]
        Core number density
    Omega : float
        Density contrast ρ_core/ρ_surface (must be < 14.04 for stability)

    Optional:
    ---------
    mu : float
        Mean molecular weight for mass conversion [m_H units] (default: 1.4 = mu_convert)
    gamma : float
        Adiabatic index (default: 5/3)
    validate : bool
        Perform input validation (default: True)
    lane_emden_solution : LaneEmdenSolution, optional
        Pre-computed solution for efficiency

    Returns
    -------
    result : BESphereResult
        Cloud radius (r_out), edge density (n_out), temperature (T_eff), etc.

    Algorithm:
    ----------
    1. Solve Lane-Emden equation (or use cached solution)
    2. Find ξ_out where ρ/ρc = 1/Omega (direct lookup)
    3. Get m(ξ_out) from Lane-Emden solution (direct lookup)
    4. Calculate c_s from: M = m × ρc × a³ where a = c_s/√(4πGρc)
    5. Convert to physical units

    No nested optimization needed!
    """
    # ========================================================================
    # VALIDATION
    # ========================================================================
    if validate:
        if not np.isfinite(M_cloud) or M_cloud <= 0:
            raise ValueError(f"M_cloud must be positive finite, got {M_cloud}")
        if not np.isfinite(n_core) or n_core <= 0:
            raise ValueError(f"n_core must be positive finite, got {n_core}")
        if not np.isfinite(Omega) or Omega <= 1.0:
            raise ValueError(f"Omega must be > 1, got {Omega}")
        if Omega > OMEGA_CRITICAL:
            logger.warning(
                f"Omega={Omega:.2f} > {OMEGA_CRITICAL:.2f} (critical). "
                f"Sphere will be gravitationally UNSTABLE!"
            )

    is_stable = Omega < OMEGA_CRITICAL

    logger.debug(f"Creating BE sphere: M={M_cloud:.3f} Msun, n_core={n_core:.2e} cm⁻³, Ω={Omega:.2f}")

    # ========================================================================
    # STEP 1: Solve Lane-Emden (or use cached)
    # ========================================================================
    if lane_emden_solution is None:
        solution = solve_lane_emden()
    else:
        solution = lane_emden_solution

    # ========================================================================
    # STEP 2: Find ξ_out where ρ/ρc = 1/Omega (DIRECT LOOKUP)
    # ========================================================================
    target_rho_rhoc = 1.0 / Omega

    # Check bounds
    if target_rho_rhoc < solution.rho_rhoc[-1]:
        raise ValueError(
            f"Omega={Omega:.2f} too high. Max supported: {1/solution.rho_rhoc[-1]:.2f}"
        )

    xi_out = float(solution.f_xi_from_rho(target_rho_rhoc))
    logger.debug(f"ξ_out = {xi_out:.4f} for Omega={Omega:.2f}")

    # ========================================================================
    # STEP 3: Get dimensionless mass m(ξ_out) (DIRECT LOOKUP)
    # ========================================================================
    m_dim = float(solution.f_m(xi_out))
    logger.debug(f"Dimensionless mass m = {m_dim:.4f}")

    # ========================================================================
    # STEP 4: Calculate sound speed (DIRECT - NO OPTIMIZATION!)
    # ========================================================================
    # Convert inputs to CGS
    M_cgs = M_cloud * MSUN_TO_G  # [g]
    rho_core_cgs = n_core * mu * M_H_CGS  # [g/cm³]

    # From M = 4π × m × ρc × a³ where a = c_s/√(4πGρc) and m = ξ² du/dξ
    # Solve for c_s:
    factor = G_CGS ** 1.5 * np.sqrt(4.0 * np.pi * rho_core_cgs)
    c_s_cubed = M_cgs / m_dim * factor
    c_s = c_s_cubed ** (1.0 / 3.0)  # [cm/s]

    logger.debug(f"Sound speed c_s = {c_s:.4e} cm/s")

    # ========================================================================
    # STEP 5: Convert to physical units
    # ========================================================================

    # Length scale a = c_s / √(4πGρc)
    a = c_s / np.sqrt(4.0 * np.pi * G_CGS * rho_core_cgs)  # [cm]

    # Physical outer radius
    r_out_cm = xi_out * a  # [cm]
    r_out = r_out_cm / PC_TO_CM  # [pc]

    # Surface number density
    n_out = n_core / Omega  # [cm⁻³]

    # Effective temperature: T = μ m_H c_s² / (γ k_B)
    T_eff = mu * M_H_CGS * c_s**2 / (gamma * K_B_CGS)  # [K]

    logger.debug(f"Result: r_out={r_out:.4f} pc, n_out={n_out:.2e} cm⁻³, T_eff={T_eff:.1f} K")

    return BESphereResult(
        xi_out=xi_out,
        r_out=r_out,
        n_out=n_out,
        T_eff=T_eff,
        c_s=c_s,
        m_dim=m_dim,
        M_cloud=M_cloud,
        n_core=n_core,
        Omega=Omega,
        is_stable=is_stable
    )


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def r_to_xi(r: float, c_s: float, rho_core: float) -> float:
    """
    Convert physical radius to dimensionless radius.

    Parameters
    ----------
    r : float [cm]
        Physical radius
    c_s : float [cm/s]
        Sound speed
    rho_core : float [g/cm³]
        Core mass density

    Returns
    -------
    xi : float
        Dimensionless radius
    """
    a = c_s / np.sqrt(4.0 * np.pi * G_CGS * rho_core)
    return r / a


def xi_to_r(xi: float, c_s: float, rho_core: float) -> float:
    """
    Convert dimensionless radius to physical radius.

    Parameters
    ----------
    xi : float
        Dimensionless radius
    c_s : float [cm/s]
        Sound speed
    rho_core : float [g/cm³]
        Core mass density

    Returns
    -------
    r : float [cm]
        Physical radius
    """
    a = c_s / np.sqrt(4.0 * np.pi * G_CGS * rho_core)
    return xi * a


# ============================================================================
# TRINITY INTEGRATION
# ============================================================================

def create_BE_sphere_from_params(params) -> BESphereResult:
    """
    Create BE sphere from TRINITY params dictionary.

    This is the main interface function for TRINITY.

    Required params:
    ---------------
    - 'mCloud' : Total cloud mass [Msun]
    - 'nCore' : Core number density [cm⁻³]
    - 'densBE_Omega' : Density contrast
    - 'mu_convert' : Mean molecular weight for mass conversion (=1.4)
    - 'gamma_adia' : Adiabatic index

    Updates params with:
    -------------------
    - 'densBE_Teff' : Effective temperature
    - 'densBE_xi_arr' : ξ array
    - 'densBE_u_arr' : u array
    - 'densBE_dudxi_arr' : du/dξ array
    - 'densBE_rho_rhoc_arr' : ρ/ρc array
    - 'densBE_f_rho_rhoc' : Interpolation function
    - 'densBE_f_m' : Mass interpolation function
    - 'densBE_xi_out' : Outer dimensionless radius
    - 'rCloud' : Cloud radius [pc]
    - 'nEdge' : Edge number density [cm⁻³]

    Returns
    -------
    result : BESphereResult
    """
    # Extract parameters
    M_cloud = params['mCloud'].value
    n_core = params['nCore'].value
    Omega = params['densBE_Omega'].value
    mu = params['mu_convert'].value  # Use mu_convert = 1.4 for mass conversion
    gamma = params['gamma_adia'].value

    # Solve Lane-Emden (cache for efficiency)
    solution = solve_lane_emden()

    # Create BE sphere
    result = create_BE_sphere(
        M_cloud=M_cloud,
        n_core=n_core,
        Omega=Omega,
        mu=mu,
        gamma=gamma,
        lane_emden_solution=solution
    )

    # Update params dictionary
    params['densBE_Teff'].value = result.T_eff
    params['densBE_xi_arr'].value = solution.xi
    params['densBE_u_arr'].value = solution.u
    params['densBE_dudxi_arr'].value = solution.dudxi
    params['densBE_rho_rhoc_arr'].value = solution.rho_rhoc
    params['densBE_f_rho_rhoc'].value = solution.f_rho_rhoc
    params['densBE_f_m'].value = solution.f_m
    params['densBE_xi_out'].value = result.xi_out

    # Also update derived cloud properties
    params['rCloud'].value = result.r_out
    params['nEdge'].value = result.n_out

    logger.info(f"BE sphere created: rCloud={result.r_out:.4f} pc, T_eff={result.T_eff:.1f} K")

    return result


def r2xi(r, params):
    """
    Convert physical radius to dimensionless radius (TRINITY interface).

    Parameters
    ----------
    r : float or array [pc]
        Physical radius
    params : dict
        TRINITY parameters (needs densBE_Teff, nCore, mu_convert, gamma_adia)

    Returns
    -------
    xi : float or array
        Dimensionless radius
    """
    # Get parameters
    T_eff = params['densBE_Teff'].value
    n_core = params['nCore'].value
    mu = params['mu_convert'].value  # Use mu_convert = 1.4 for mass density
    gamma = params['gamma_adia'].value

    # Calculate sound speed [cm/s]
    c_s = np.sqrt(gamma * K_B_CGS * T_eff / (mu * M_H_CGS))

    # Core mass density [g/cm³]
    rho_core = n_core * mu * M_H_CGS

    # Convert r from pc to cm
    r_cm = np.asarray(r) * PC_TO_CM

    # Calculate xi
    xi = r_to_xi(r_cm, c_s, rho_core)

    return xi


def xi2r(xi, params):
    """
    Convert dimensionless radius to physical radius (TRINITY interface).

    Parameters
    ----------
    xi : float or array
        Dimensionless radius
    params : dict
        TRINITY parameters (needs densBE_Teff, nCore, mu_convert, gamma_adia)

    Returns
    -------
    r : float or array [pc]
        Physical radius
    """
    # Get parameters
    T_eff = params['densBE_Teff'].value
    n_core = params['nCore'].value
    mu = params['mu_convert'].value  # Use mu_convert = 1.4 for mass density
    gamma = params['gamma_adia'].value

    # Calculate sound speed [cm/s]
    c_s = np.sqrt(gamma * K_B_CGS * T_eff / (mu * M_H_CGS))

    # Core mass density [g/cm³]
    rho_core = n_core * mu * M_H_CGS

    # Calculate r [cm]
    r_cm = xi_to_r(np.asarray(xi), c_s, rho_core)

    # Convert to pc
    r_pc = r_cm / PC_TO_CM

    return r_pc
