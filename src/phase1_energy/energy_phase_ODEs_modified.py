#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure ODE functions for TRINITY energy phase.

This module provides pure (side-effect-free) ODE functions for bubble expansion
that are compatible with scipy.integrate.solve_ivp.

Key features:
- No dictionary mutations (params dict is read-only)
- StaticODEParams dataclass for immutable parameter passing
- Gradual activation of mShell_dot term to fix v²/r singularity at small R2
- R1Cache for efficient caching of expensive brentq calculations

State vector: y = [R2, v2, Eb] (3 variables)
Note: T0 is NOT integrated - it's calculated externally via bubble_luminosity.

@author: TRINITY Team (refactored for pure functions)
"""

import numpy as np
import scipy.optimize
import scipy.interpolate
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# Import for R1 calculation
import src.bubble_structure.get_bubbleParams as get_bubbleParams
# Import existing mass profile function (side-effect-free)
from src.cloud_properties.mass_profile import get_mass_profile

logger = logging.getLogger(__name__)

# Constants
FOUR_PI = 4.0 * np.pi


# =============================================================================
# Static Parameters Container
# =============================================================================

@dataclass(frozen=True)
class StaticODEParams:
    """
    Immutable container for ODE parameters.

    These are computed OUTSIDE the ODE function and passed in as read-only.
    Using frozen=True ensures no accidental modifications.

    Attributes
    ----------
    # Physical constants
    gamma_adia : float
        Adiabatic index (typically 5/3)
    G : float
        Gravitational constant [pc³/Msun/Myr²]
    k_B : float
        Boltzmann constant [internal units]

    # Cloud properties (static during integration segment)
    rCloud : float
        Cloud outer radius [pc]
    rCore : float
        Core radius [pc]
    mCloud : float
        Cloud mass [Msun]
    mCluster : float
        Cluster mass [Msun]

    # Density profile parameters
    nCore : float
        Core number density [1/pc³] (internal units)
    nISM : float
        ISM number density [1/pc³] (internal units)
    mu_convert : float
        Mean molecular weight [Msun] (internal units)
    dens_profile : str
        Profile type ('densPL' or 'densBE')
    densPL_alpha : float
        Power-law exponent (for densPL)

    # Feedback (can be interpolators or constants)
    LWind : float
        Wind luminosity [internal units]
    vWind : float
        Wind velocity [internal units]

    # From previous bubble_luminosity calculation
    L_bubble : float
        Bubble cooling luminosity [internal units]
    F_rad : float
        Radiation force [internal units]
    FABSi : float
        Fraction of ionizing radiation absorbed

    # Pressures (from shell structure)
    press_HII_in : float
        Inward HII pressure [internal units]
    press_HII_out : float
        Outward HII pressure [internal units]

    # Cached R1 value (computed at segment start)
    R1_cached : float
        Inner bubble radius [pc]

    # Timing
    tSF : float
        Star formation time [Myr]

    # Phase info
    current_phase : str
        Current simulation phase
    is_collapse : bool
        Whether shell is collapsing
    shell_mass_frozen : float
        Frozen shell mass during collapse [Msun]
    """
    # Physical constants
    gamma_adia: float
    G: float
    k_B: float

    # Cloud properties
    rCloud: float
    rCore: float
    mCloud: float
    mCluster: float

    # Density profile
    nCore: float
    nISM: float
    mu_convert: float
    dens_profile: str
    densPL_alpha: float

    # Feedback
    LWind: float
    vWind: float

    # Bubble/shell properties
    L_bubble: float
    F_rad: float
    FABSi: float
    press_HII_in: float
    press_HII_out: float

    # Cached values
    R1_cached: float

    # Timing and phase
    tSF: float
    current_phase: str
    is_collapse: bool
    shell_mass_frozen: float


# =============================================================================
# R1 Cache for efficient brentq caching
# =============================================================================

class R1Cache:
    """
    Cache for inner radius R1 to avoid expensive brentq inside ODE.

    R1 changes slowly, so we:
    1. Compute R1 at segment start
    2. Use cached value during segment
    3. Optionally build interpolator for smoother variation
    """

    def __init__(self):
        self.t_values = []
        self.R1_values = []
        self._interp = None

    def update(self, t: float, R2: float, Eb: float, LWind: float, vWind: float) -> float:
        """
        Compute and cache R1 at given time.

        Parameters
        ----------
        t : float
            Current time [Myr]
        R2 : float
            Outer bubble radius [pc]
        Eb : float
            Bubble energy [internal units]
        LWind : float
            Wind luminosity [internal units]
        vWind : float
            Wind velocity [internal units]

        Returns
        -------
        R1 : float
            Inner bubble radius [pc]
        """
        try:
            R1 = scipy.optimize.brentq(
                get_bubbleParams.get_r1,
                1e-6 * R2, R2 * 0.999,
                args=([LWind, Eb, vWind, R2])
            )
        except ValueError:
            # Fallback if brentq fails (e.g., no root in interval)
            R1 = 0.01 * R2
            logger.warning(f"R1 brentq failed at t={t:.3e}, using R1=0.01*R2")

        self.t_values.append(t)
        self.R1_values.append(R1)

        # Build interpolator if we have enough points
        if len(self.t_values) >= 2:
            self._interp = scipy.interpolate.interp1d(
                self.t_values, self.R1_values,
                kind='linear', fill_value='extrapolate'
            )

        return R1

    def get(self, t: float) -> float:
        """
        Get R1 at time t via interpolation (or return last cached value).

        Parameters
        ----------
        t : float
            Time [Myr]

        Returns
        -------
        R1 : float
            Inner bubble radius [pc]
        """
        if self._interp is None:
            return self.R1_values[-1] if self.R1_values else 0.0
        return float(self._interp(t))

    def clear(self):
        """Clear the cache."""
        self.t_values = []
        self.R1_values = []
        self._interp = None


# =============================================================================
# ParamsWrapper for using existing mass_profile functions
# =============================================================================

class ParamsWrapper:
    """
    Minimal dict-like wrapper for using mass_profile.get_mass_profile().

    This allows the pure ODE function to use the existing side-effect-free
    mass profile functions without needing the full params dict.
    """

    def __init__(self, static: StaticODEParams):
        """
        Create wrapper from StaticODEParams.

        Parameters
        ----------
        static : StaticODEParams
            Immutable parameters container
        """
        self._static = static
        # Map params keys to StaticODEParams attributes
        self._key_map = {
            'rCloud': 'rCloud',
            'rCore': 'rCore',
            'mCloud': 'mCloud',
            'nCore': 'nCore',
            'nISM': 'nISM',
            'mu_convert': 'mu_convert',
            'dens_profile': 'dens_profile',
            'densPL_alpha': 'densPL_alpha',
            'isCollapse': 'is_collapse',
            'shell_mass': 'shell_mass_frozen',
            # Additional keys that mass_profile might need
            'initial_cloud_r_arr': None,
            'initial_cloud_m_arr': None,
        }

    def __getitem__(self, key):
        """Get parameter value wrapped in object with .value attribute."""
        if key in self._key_map:
            attr_name = self._key_map[key]
            if attr_name is None:
                # Return None for optional arrays not in static params
                return type('Param', (), {'value': None})()
            value = getattr(self._static, attr_name)
            return type('Param', (), {'value': value})()
        raise KeyError(f"Key '{key}' not found in ParamsWrapper")

    def __contains__(self, key):
        """Check if key exists."""
        return key in self._key_map

    def get(self, key, default=None):
        """Get with default value."""
        try:
            return self[key]
        except KeyError:
            return type('Param', (), {'value': default})()


def _get_mass_from_profile(R2: float, v2: float, static: StaticODEParams) -> Tuple[float, float]:
    """
    Get shell mass and mass accretion rate using existing mass_profile module.

    This wraps the existing get_mass_profile() function which is already
    side-effect-free.

    Parameters
    ----------
    R2 : float
        Shell radius [pc]
    v2 : float
        Shell velocity [pc/Myr]
    static : StaticODEParams
        Immutable parameters

    Returns
    -------
    mShell : float
        Enclosed mass at R2 [Msun]
    mShell_dot : float
        Mass accretion rate at R2 [Msun/Myr]
    """
    # Handle collapse mode
    if static.is_collapse:
        return static.shell_mass_frozen, 0.0

    # Create wrapper for mass_profile function
    params_wrapper = ParamsWrapper(static)

    # Call existing mass profile function
    result = get_mass_profile(R2, params_wrapper, return_mdot=True, rdot=v2)

    # Handle both tuple and single return
    if isinstance(result, tuple):
        mShell, mShell_dot = result
    else:
        mShell = result
        mShell_dot = 0.0

    # Ensure scalar outputs
    if hasattr(mShell, '__len__') and len(mShell) == 1:
        mShell = float(mShell[0])
    if hasattr(mShell_dot, '__len__') and len(mShell_dot) == 1:
        mShell_dot = float(mShell_dot[0])

    return float(mShell), float(mShell_dot)


def _get_mShell_dot_with_activation(mShell_dot_raw: float, R2: float,
                                     static: StaticODEParams) -> float:
    """
    Gradually activate mass accretion drag term as shell forms.

    This fixes the v²/r singularity at small R2 where the thin-shell
    approximation breaks down.

    Parameters
    ----------
    mShell_dot_raw : float
        Raw mass accretion rate [Msun/Myr]
    R2 : float
        Shell radius [pc]
    static : StaticODEParams
        Immutable parameters

    Returns
    -------
    mShell_dot : float
        Activated mass accretion rate [Msun/Myr]
    """
    # Activation radius: shell is well-formed above this
    R2_activate = static.rCore * 0.1  # 10% of core radius

    if R2 < R2_activate:
        # Linear ramp from 0 to 1 as R2 grows from 0 to R2_activate
        activation = R2 / R2_activate
        return mShell_dot_raw * activation
    else:
        return mShell_dot_raw


# =============================================================================
# Pure ODE Function
# =============================================================================

def get_ODE_Edot_pure(t: float, y: np.ndarray, static: StaticODEParams) -> np.ndarray:
    """
    Pure ODE function for bubble expansion - NO SIDE EFFECTS.

    This function is compatible with scipy.integrate.solve_ivp.
    It reads from the immutable StaticODEParams and returns derivatives.

    State vector: y = [R2, v2, Eb]
    Note: T0 is NOT in state - it's updated between integration segments.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb]
    static : StaticODEParams
        Immutable parameters container

    Returns
    -------
    dydt : ndarray
        Derivatives [dR2/dt, dv2/dt, dEb/dt]
    """
    R2, v2, Eb = y
    R2 = float(R2)
    v2 = float(v2)
    Eb = float(Eb)

    # Ensure positive values
    R2 = max(R2, 1e-10)
    Eb = max(Eb, 1e-10)

    # Calculate shell mass and mass accretion rate using existing mass_profile module
    mShell, mShell_dot_raw = _get_mass_from_profile(R2, v2, static)

    # Apply gradual activation to fix v²/r singularity
    mShell_dot = _get_mShell_dot_with_activation(mShell_dot_raw, R2, static)

    # Ensure positive shell mass
    mShell = max(mShell, 1e-10)

    # Gravity force (self + cluster)
    F_grav = static.G * mShell / (R2**2) * (static.mCluster + 0.5 * mShell)

    # Bubble pressure
    if static.current_phase == 'momentum':
        press_bubble = get_bubbleParams.pRam(R2, static.LWind, static.vWind)
    else:
        # Use cached R1
        R1 = static.R1_cached

        # Pressure switch-on at early times
        dt_switchon = 1e-3  # Myr
        t_elapsed = t - static.tSF

        if t_elapsed <= dt_switchon:
            # Gradually switch on: interpolate R1 from 0 to actual value
            R1_tmp = (t_elapsed / dt_switchon) * R1
            R1_tmp = max(R1_tmp, 1e-10)
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, static.gamma_adia)
        else:
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, static.gamma_adia)

    # Net pressure force
    press_HII_in = static.press_HII_in
    press_HII_out = static.press_HII_out
    F_pressure = FOUR_PI * R2**2 * (press_bubble - press_HII_in + press_HII_out)

    # Time derivatives
    rd = v2  # dR/dt = velocity

    # dv/dt = (F_pressure - momentum_drag - F_grav + F_rad) / mShell
    vd = (F_pressure - mShell_dot * v2 - F_grav + static.F_rad) / mShell

    # dE/dt = LWind - L_bubble - P*dV/dt (energy balance)
    # Note: L_leak = 0 for now (no fragmentation)
    L_leak = 0.0
    Ed = (static.LWind - static.L_bubble) - (FOUR_PI * R2**2 * press_bubble) * v2 - L_leak

    return np.array([rd, vd, Ed])


# =============================================================================
# Event Functions for solve_ivp
# =============================================================================

def velocity_floor_event(t: float, y: np.ndarray, static: StaticODEParams) -> float:
    """
    Event function: triggers when velocity drops below minimum.

    This is a physically motivated safeguard - if the shell stalls,
    we should halt rather than continue with unphysical collapse.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb]
    static : StaticODEParams
        Immutable parameters

    Returns
    -------
    value : float
        Positive when v2 > v_min, negative when v2 < v_min
    """
    R2, v2, Eb = y
    v_min = 0.01  # pc/Myr minimum velocity
    return v2 - v_min

# Set event attributes
velocity_floor_event.terminal = True
velocity_floor_event.direction = -1  # Trigger when v2 decreasing through v_min


def radius_exceeds_cloud_event(t: float, y: np.ndarray, static: StaticODEParams) -> float:
    """
    Event function: triggers when shell reaches cloud boundary.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb]
    static : StaticODEParams
        Immutable parameters

    Returns
    -------
    value : float
        Positive when R2 < rCloud, negative when R2 > rCloud
    """
    R2, v2, Eb = y
    return static.rCloud - R2

radius_exceeds_cloud_event.terminal = False  # Don't stop, just record
radius_exceeds_cloud_event.direction = -1


# =============================================================================
# Helper to extract static params from params dict
# =============================================================================

def extract_static_params(params, R1_cached: float = 0.01) -> StaticODEParams:
    """
    Extract StaticODEParams from params dictionary.

    This creates an immutable snapshot of the current parameter values
    for use during an integration segment.

    Parameters
    ----------
    params : dict
        Parameter dictionary with .value attributes
    R1_cached : float, optional
        Pre-computed R1 value (from R1Cache)

    Returns
    -------
    static : StaticODEParams
        Immutable parameters container
    """
    # Handle optional parameters with defaults
    def get_value(key, default=0.0):
        if key in params and hasattr(params[key], 'value'):
            return params[key].value
        elif key in params:
            return params[key]
        return default

    return StaticODEParams(
        # Physical constants
        gamma_adia=get_value('gamma_adia', 5.0/3.0),
        G=get_value('G', 4.49e-15),
        k_B=get_value('k_B', 6.94e-60),

        # Cloud properties
        rCloud=get_value('rCloud', 10.0),
        rCore=get_value('rCore', 1.0),
        mCloud=get_value('mCloud', 1e5),
        mCluster=get_value('mCluster', 1e4),

        # Density profile
        nCore=get_value('nCore'),
        nISM=get_value('nISM'),
        mu_convert=get_value('mu_convert'),
        dens_profile=get_value('dens_profile', 'densPL'),
        densPL_alpha=get_value('densPL_alpha', 0.0),

        # Feedback
        LWind=get_value('LWind'),
        vWind=get_value('vWind'),

        # Bubble/shell properties
        L_bubble=get_value('bubble_LTotal', 0.0),
        F_rad=get_value('shell_F_rad', 0.0),
        FABSi=get_value('shell_fAbsorbedIon', 1.0),
        press_HII_in=get_value('press_HII_in', 0.0),
        press_HII_out=get_value('press_HII_out', 0.0),

        # Cached R1
        R1_cached=R1_cached,

        # Timing and phase
        tSF=get_value('tSF', 0.0),
        current_phase=get_value('current_phase', 'energy'),
        is_collapse=get_value('isCollapse', False),
        shell_mass_frozen=get_value('shell_mass', 0.0),
    )


# =============================================================================
# Wrapper for backward compatibility (if needed)
# =============================================================================

def get_ODE_Edot_wrapper(y, t, params):
    """
    Wrapper that provides interface compatible with old code.

    DEPRECATED: Use get_ODE_Edot_pure with solve_ivp instead.

    This wrapper:
    1. Extracts static params from params dict
    2. Calls the pure function
    3. Updates params dict with computed values (for compatibility)

    Parameters
    ----------
    y : list
        State vector [R2, v2, Eb, T0]
    t : float
        Time [Myr]
    params : dict
        Parameter dictionary

    Returns
    -------
    derivs : list
        Derivatives [rd, vd, Ed, Td]
    """
    import warnings
    warnings.warn(
        "get_ODE_Edot_wrapper is deprecated. Use get_ODE_Edot_pure with solve_ivp.",
        DeprecationWarning,
        stacklevel=2
    )

    R2, v2, Eb, T0 = y

    # Get R1 for this state
    LWind = params['LWind'].value
    vWind = params['vWind'].value

    try:
        R1 = scipy.optimize.brentq(
            get_bubbleParams.get_r1,
            1e-6 * R2, R2 * 0.999,
            args=([LWind, Eb, vWind, R2])
        )
    except:
        R1 = 0.01 * R2

    # Extract static params
    static = extract_static_params(params, R1_cached=R1)

    # Call pure function
    y_arr = np.array([R2, v2, Eb])
    dydt = get_ODE_Edot_pure(t, y_arr, static)

    # Update params dict (for compatibility with old code)
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = Eb

    # Return with T0 derivative = 0 (T0 is external)
    return [dydt[0], dydt[1], dydt[2], 0]
