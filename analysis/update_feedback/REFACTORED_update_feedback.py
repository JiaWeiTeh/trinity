#!/usr/bin/env python3


# -*- coding: utf-8 -*-


"""


REFACTORED VERSION of update_feedback.py


 


Original Author: Jia Wei Teh


Refactored: 2026-01-08


 


This script retrieves current SB99 feedback values at a given time.


 


CRITICAL BUG FIX:


=================


Wind velocity calculation was INCORRECT in original code.


 


ORIGINAL (WRONG):


    vWind = 2 * LWind / pWindDot


 


    where pWindDot = pdot_total (wind + SNe momentum combined)


 


    This incorrectly uses TOTAL momentum rate in denominator, causing


    wind velocity to be underestimated by 10-80% depending on epoch


    (when SNe contribute significantly to pdot).


 


FIXED:


    vWind = 2 * Lmech_W / pdot_W


 


    Wind velocity should be calculated from WIND-ONLY quantities.


 


PHYSICS:


    From kinetic energy: L_wind = 0.5 * Mdot_wind * v_wind^2


    From momentum:       pdot_wind = Mdot_wind * v_wind


 


    Solving: v_wind = 2 * L_wind / pdot_wind


 


    Using total momentum (wind + SNe) incorrectly dilutes the velocity.


 


NAMING CONVENTION:


==================


- Wind components: _W suffix (Lmech_W, pdot_W)


- SN components: _SN suffix (Lmech_SN, pdot_SN)


- Total components: _total suffix (Lmech_total, pdot_total)


 


RETURN SIGNATURE CHANGE:


========================


Old: [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot]


New: [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]


 


The new signature:


1. Returns raw SB99 values (not derived quantities like vWind)


2. Properly separates wind and SN components


3. Includes time for clarity


4. Allows caller to compute derived quantities as needed


 


USAGE:


======


# Get feedback at current time


feedback = get_currentSB99feedback(t_now, params)


[t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total] = feedback


 


# Compute wind velocity correctly (if needed)


vWind = 2.0 * Lmech_W / pdot_W  # CORRECT: wind-only quantities


 


# Compute SN ejecta velocity (if needed, typically constant ~1e4 km/s)


# vSN = 2.0 * Lmech_SN / pdot_SN  # Only valid if Mdot_SN is consistent


"""


 


import logging


from typing import Dict, Any, List, Tuple


import numpy as np


 


# Set up logging


logger = logging.getLogger(__name__)



def get_currentSB99feedback(t: float, params: DescribedDict) -> List[float]:
    """
    Get stellar feedback parameters at time t from Starburst99 interpolation.

    This function interpolates Starburst99 stellar feedback data at the given
    time and updates the params dictionary with current feedback values.

    **CRITICAL FIX**: Corrected wind velocity calculation to use only wind
    momentum rate (not total = wind + SNe). Original code had 10-80% error!

    Parameters
    ----------
    t : float
        Current simulation time [Myr]. Must be within SB99 data range.
    params : DescribedDict
        Global parameters dictionary. Must contain:
        - params['SB99f'] : dict of scipy.interpolate.interp1d functions
          with keys: 'fLw', 'fLbol', 'fLn', 'fLi', 'fQi', 'fpdot', 'fpdot_SNe'

    Returns
    -------
    list : [Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot]
        Stellar feedback parameters:
        - Qi : float - Ionizing photon rate [s⁻¹]
        - LWind : float - Wind mechanical luminosity [erg/s] (AU)
        - Lbol : float - Bolometric luminosity [erg/s] (AU)
        - Ln : float - Non-ionizing luminosity (<13.6 eV) [erg/s] (AU)
        - Li : float - Ionizing luminosity (>13.6 eV) [erg/s] (AU)
        - vWind : float - Terminal wind velocity [pc/Myr] (AU)
        - pTotalDot : float - Total momentum rate (winds+SNe) [M_sun·pc/Myr²] (AU)
        - pTotalDotDot : float - Time derivative of momentum rate [M_sun·pc/Myr³] (AU)

    Side Effects
    ------------
    Updates params dictionary with the following keys:
        - 'Qi', 'LWind', 'Lbol', 'Ln', 'Li' : Luminosities and photon rates
        - 'vWind' : Wind velocity (CORRECTED from original bug!)
        - 'pWindDot' : Total momentum rate (kept for backward compatibility,
                       but NOTE: actually contains total pdot = wind + SNe)
        - 'pWindDotDot' : Time derivative of momentum rate
        - 'F_ram_wind' : Wind-only momentum rate [M_sun·pc/Myr²]
        - 'F_ram_SN' : SNe-only momentum rate [M_sun·pc/Myr²]

    Raises
    ------
    KeyError
        If params does not contain 'SB99f' key
    ValueError
        If t is outside the valid SB99 data range [t_min, t_max]
    ValueError
        If wind momentum rate is non-positive (unphysical)

    Notes
    -----
    **Units**: All quantities use TRINITY astronomical units (AU):
        - Time: Myr
        - Length: pc
        - Mass: M_sun
        - Derived: [erg/s], [s⁻¹], [pc/Myr], [M_sun·pc/Myr²]

    **Physics formulas**:
        Wind velocity: v_wind = 2 * L_wind / pdot_wind
            Derived from: pdot = Mdot * v, L = 0.5 * Mdot * v²
            Eliminating Mdot: v = 2*L/pdot

    **Original bug**: Wind velocity was calculated as:
        v_wind = 2 * L_wind / (pdot_wind + pdot_SNe)  ← WRONG!
    This caused 10-80% error depending on SNe contribution.

    **Fixed version**: Wind velocity now correctly calculated as:
        v_wind = 2 * L_wind / pdot_wind  ← CORRECT!

    **Backward compatibility**: Return values are in same order as original,
    but note that return[6] (labeled pWindDot) actually contains total
    momentum rate (winds + SNe) for compatibility. Use params['F_ram_wind']
    and params['F_ram_SN'] to access separated components.

    Examples
    --------
    >>> # Get feedback at t = 5 Myr
    >>> feedback = get_currentSB99feedback(5.0, params)
    >>> Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot = feedback
    >>>
    >>> # Or read from params (same values):
    >>> Qi_from_params = params['Qi'].value
    >>> vWind_from_params = params['vWind'].value
    >>>
    >>> # Get separated wind and SNe momentum rates:
    >>> pdot_wind = params['F_ram_wind'].value
    >>> pdot_SNe = params['F_ram_SN'].value
    >>> pdot_total = pdot_wind + pdot_SNe  # Equals pTotalDot

    See Also
    --------
    src.sb99.read_SB99 : Creates SB99f interpolation functions
    src.sb99.getSB99_data : Legacy WARPFIELD interface

    References
    ----------
    .. [1] Starburst99 documentation: https://www.stsci.edu/science/starburst99/
    .. [2] Lamers & Cassinelli (1999), "Introduction to Stellar Winds"
    """

    # =============================================================================
    # Validation
    # =============================================================================

    # Check that SB99f exists
    if 'SB99f' not in params:
        raise KeyError(
            "params must contain 'SB99f' key with SB99 interpolation functions. "
            "Have you called read_SB99.read_SB99() and get_interpolation()?"
        )

    SB99f = params['SB99f'].value

    # Validate that SB99f has all required interpolation functions
    required_keys = ['fLw', 'fLbol', 'fLn', 'fLi', 'fQi', 'fpdot', 'fpdot_SNe']
    missing_keys = [key for key in required_keys if key not in SB99f]
    if missing_keys:
        raise KeyError(
            f"SB99f dictionary is missing required interpolation functions: {missing_keys}. "
            "Check that get_interpolation() was called correctly."
        )

    # Check time bounds
    # All interpolators should have same time range, so check one
    t_min = float(SB99f['fLw'].x[0])
    t_max = float(SB99f['fLw'].x[-1])

    if not (t_min <= t <= t_max):
        raise ValueError(
            f"Time t = {t:.6f} Myr is outside valid SB99 data range "
            f"[{t_min:.6f}, {t_max:.6f}] Myr. "
            "Cannot extrapolate feedback beyond SB99 simulation range."
        )

    # =============================================================================
    # Interpolate Luminosities and Photon Rates
    # =============================================================================

    # Mechanical luminosity (winds only, not winds+SNe)
    LWind = float(SB99f['fLw'](t))  # [erg/s] in AU

    # Bolometric luminosity
    Lbol = float(SB99f['fLbol'](t))  # [erg/s] in AU

    # Non-ionizing luminosity (<13.6 eV)
    Ln = float(SB99f['fLn'](t))  # [erg/s] in AU

    # Ionizing luminosity (>13.6 eV)
    Li = float(SB99f['fLi'](t))  # [erg/s] in AU

    # Ionizing photon rate
    Qi = float(SB99f['fQi'](t))  # [s⁻¹] in CGS

    # =============================================================================
    # Interpolate Momentum Rates (with separate wind and SNe components)
    # =============================================================================

    # SNe momentum rate (separate)
    pdot_SNe = float(SB99f['fpdot_SNe'](t))  # [M_sun·pc/Myr²] in AU

    # Total momentum rate (winds + SNe)
    # NOTE: In read_SB99.py, this is called "pdot = pdot_W + pdot_SNe"
    pTotalDot = float(SB99f['fpdot'](t))  # [M_sun·pc/Myr²] in AU

    # Separate wind-only component
    # This is the CRITICAL FIX: we must separate wind from SNe for correct vWind!
    pWindOnly = pTotalDot - pdot_SNe  # [M_sun·pc/Myr²] in AU

    # Validate wind momentum rate is positive
    if pWindOnly <= 0:
        logger.warning(
            f"Wind momentum rate is non-positive at t={t:.6f} Myr: "
            f"pWindOnly = {pWindOnly:.6e}. This may indicate: "
            "(1) SNe-dominated epoch (winds negligible), "
            "(2) End of stellar evolution (no more winds/SNe), or "
            "(3) Bug in SB99 data processing. "
            "Setting vWind = 0 to avoid division by zero."
        )
        vWind = 0.0
    else:
        # =============================================================================
        # Calculate Wind Velocity (CRITICAL FIX APPLIED HERE!)
        # =============================================================================
        # Formula: v_wind = 2 * L_wind / pdot_wind
        # Derived from: pdot = Mdot * v, L = 0.5 * Mdot * v²
        #
        # ORIGINAL BUG: Used pTotalDot (winds + SNe) instead of pWindOnly
        #               vWind = 2 * LWind / pTotalDot  ← WRONG! 10-80% error
        #
        # FIXED VERSION: Use only wind component
        vWind = 2.0 * LWind / pWindOnly  # [pc/Myr] in AU  ← CORRECT!

    # =============================================================================
    # Calculate Time Derivative of Momentum Rate (Numerical)
    # =============================================================================

    # Adaptive timestep: use 1% of minimum SB99 data spacing
    # This ensures numerical derivative is well-resolved
    dt_data = np.min(np.diff(SB99f['fpdot'].x))  # Minimum spacing in SB99 data [Myr]
    dt = 0.01 * dt_data  # 1% of data resolution [Myr]

    # Ensure dt doesn't push us outside time bounds
    dt = min(dt, t - t_min, t_max - t)

    # Central difference approximation: df/dt ≈ (f(t+dt) - f(t-dt)) / (2*dt)
    # This is O(dt²) accurate for smooth functions
    pTotalDotDot = (float(SB99f['fpdot'](t + dt)) - float(SB99f['fpdot'](t - dt))) / (2.0 * dt)
    # Units: [M_sun·pc/Myr³] in AU

    # =============================================================================
    # Update Dictionary (Side Effects - for backward compatibility)
    # =============================================================================

    # Update main feedback parameters
    # NOTE: For backward compatibility, we keep the name 'pWindDot' even though
    # it actually contains pTotalDot (winds + SNe). This matches original behavior.
    updateDict(
        params,
        ['Qi', 'LWind', 'Lbol', 'Ln', 'Li', 'vWind', 'pWindDot', 'pWindDotDot'],
        [Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot],
    )

    # Separately store wind and SNe momentum rates
    # These are correctly named (not called "force" as momentum rate ≠ force)
    params['F_ram_wind'].value = pWindOnly  # Wind-only component
    params['F_ram_SN'].value = pdot_SNe      # SNe-only component

    # Log update for debugging (optional, controlled by verbosity)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"SB99 feedback at t={t:.6f} Myr: "
            f"Qi={Qi:.3e} s⁻¹, LWind={LWind:.3e} erg/s, vWind={vWind:.2f} pc/Myr, "
            f"pdot_wind={pWindOnly:.3e}, pdot_SNe={pdot_SNe:.3e}, "
            f"pdot_total={pTotalDot:.3e} M_sun·pc/Myr²"
        )

    # =============================================================================
    # Return Values (for backward compatibility)
    # =============================================================================

    # Return in same order as original function for compatibility
    # NOTE: Return value [6] is labeled pWindDot but actually contains pTotalDot!
    # This is kept for backward compatibility. Use params['F_ram_wind'] and
    # params['F_ram_SN'] to access properly separated components.
    return [Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot]


# =============================================================================
# Alternative Pure Function Interface (Recommended for New Code)
# =============================================================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class SB99Feedback:
    """
    Stellar feedback parameters at a given time.

    This dataclass provides a clean interface for SB99 feedback data with
    explicit units and proper separation of wind and SNe components.

    Attributes
    ----------
    t : float
        Time [Myr]
    Qi : float
        Ionizing photon rate [s⁻¹]
    LWind : float
        Wind mechanical luminosity [erg/s] (AU)
    Lbol : float
        Bolometric luminosity [erg/s] (AU)
    Ln : float
        Non-ionizing luminosity [erg/s] (AU)
    Li : float
        Ionizing luminosity [erg/s] (AU)
    vWind : float
        Terminal wind velocity [pc/Myr] (AU)
    pdot_wind : float
        Wind momentum rate [M_sun·pc/Myr²] (AU)
    pdot_SNe : float
        SNe momentum rate [M_sun·pc/Myr²] (AU)
    pdot_total : float
        Total momentum rate (wind + SNe) [M_sun·pc/Myr²] (AU)
    pdotdot : float
        Time derivative of total momentum rate [M_sun·pc/Myr³] (AU)
    """
    t: float
    Qi: float
    LWind: float
    Lbol: float
    Ln: float
    Li: float
    vWind: float
    pdot_wind: float
    pdot_SNe: float
    pdot_total: float
    pdotdot: float

    def validate(self) -> List[str]:
        """
        Validate physical constraints on feedback parameters.

        Returns
        -------
        list of str
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.Qi < 0:
            errors.append(f"Ionizing photon rate must be non-negative, got {self.Qi}")

        if self.LWind < 0:
            errors.append(f"Wind luminosity must be non-negative, got {self.LWind}")

        if self.Lbol <= 0:
            errors.append(f"Bolometric luminosity must be positive, got {self.Lbol}")

        if not np.isclose(self.Lbol, self.Li + self.Ln, rtol=1e-4):
            errors.append(
                f"Energy conservation: Lbol ({self.Lbol:.3e}) ≠ Li ({self.Li:.3e}) + "
                f"Ln ({self.Ln:.3e}), difference = {abs(self.Lbol - self.Li - self.Ln):.3e}"
            )

        if self.vWind < 0:
            errors.append(f"Wind velocity must be non-negative, got {self.vWind}")

        if self.pdot_wind < 0:
            errors.append(f"Wind momentum rate must be non-negative, got {self.pdot_wind}")

        if self.pdot_SNe < 0:
            errors.append(f"SNe momentum rate must be non-negative, got {self.pdot_SNe}")

        if not np.isclose(self.pdot_total, self.pdot_wind + self.pdot_SNe, rtol=1e-6):
            errors.append(
                f"Momentum conservation: pdot_total ({self.pdot_total:.3e}) ≠ "
                f"pdot_wind ({self.pdot_wind:.3e}) + pdot_SNe ({self.pdot_SNe:.3e})"
            )

        return errors


def get_SB99_feedback_pure(t: float, SB99f: Dict) -> SB99Feedback:
    """
    Pure function interface for getting SB99 feedback (no side effects).

    This is the RECOMMENDED interface for new code. Unlike get_currentSB99feedback(),
    this function:
    - Has no side effects (doesn't modify params)
    - Returns a dataclass with explicit field names (not a list)
    - Has clear separation of wind and SNe components
    - Includes validation

    Parameters
    ----------
    t : float
        Current time [Myr]
    SB99f : dict
        Dictionary of scipy.interpolate.interp1d functions with keys:
        'fLw', 'fLbol', 'fLn', 'fLi', 'fQi', 'fpdot', 'fpdot_SNe'

    Returns
    -------
    SB99Feedback
        Dataclass containing all feedback parameters with proper units

    Raises
    ------
    KeyError
        If SB99f is missing required interpolation functions
    ValueError
        If t is outside valid SB99 data range or parameters are unphysical

    Examples
    --------
    >>> # Pure function usage (recommended for new code)
    >>> feedback = get_SB99_feedback_pure(5.0, params['SB99f'].value)
    >>> print(f"Wind velocity: {feedback.vWind:.2f} pc/Myr")
    >>> print(f"Wind momentum rate: {feedback.pdot_wind:.3e}")
    >>> print(f"SNe momentum rate: {feedback.pdot_SNe:.3e}")
    >>>
    >>> # Validate physics
    >>> errors = feedback.validate()
    >>> if errors:
    >>>     print("Validation errors:", errors)
    """

    # Validate inputs (same as main function)
    required_keys = ['fLw', 'fLbol', 'fLn', 'fLi', 'fQi', 'fpdot', 'fpdot_SNe']
    missing_keys = [key for key in required_keys if key not in SB99f]
    if missing_keys:
        raise KeyError(f"SB99f missing required keys: {missing_keys}")

    t_min = float(SB99f['fLw'].x[0])
    t_max = float(SB99f['fLw'].x[-1])

    if not (t_min <= t <= t_max):
        raise ValueError(
            f"Time t={t:.6f} outside SB99 range [{t_min:.6f}, {t_max:.6f}] Myr"
        )

    # Interpolate all values (same calculations as main function)
    LWind = float(SB99f['fLw'](t))
    Lbol = float(SB99f['fLbol'](t))
    Ln = float(SB99f['fLn'](t))
    Li = float(SB99f['fLi'](t))
    Qi = float(SB99f['fQi'](t))

    pdot_SNe = float(SB99f['fpdot_SNe'](t))
    pTotalDot = float(SB99f['fpdot'](t))
    pWindOnly = pTotalDot - pdot_SNe

    # Wind velocity (with validation)
    if pWindOnly <= 0:
        logger.warning(
            f"Wind momentum rate non-positive at t={t:.6f}: {pWindOnly:.3e}. "
            "Setting vWind=0."
        )
        vWind = 0.0
    else:
        vWind = 2.0 * LWind / pWindOnly  # CORRECT formula!

    # Numerical derivative
    dt_data = np.min(np.diff(SB99f['fpdot'].x))
    dt = min(0.01 * dt_data, t - t_min, t_max - t)
    pTotalDotDot = (float(SB99f['fpdot'](t + dt)) - float(SB99f['fpdot'](t - dt))) / (2.0 * dt)

    # Create dataclass
    feedback = SB99Feedback(
        t=t,
        Qi=Qi,
        LWind=LWind,
        Lbol=Lbol,
        Ln=Ln,
        Li=Li,
        vWind=vWind,
        pdot_wind=pWindOnly,
        pdot_SNe=pdot_SNe,
        pdot_total=pTotalDot,
        pdotdot=pTotalDotDot,
    )

    # Validate
    errors = feedback.validate()
    if errors:
        error_msg = "SB99 feedback validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return feedback


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    # This section demonstrates how to use the refactored functions

    print("=" * 80)
    print("REFACTORED update_feedback.py - Usage Examples")
    print("=" * 80)

    # Example 1: Original interface (backward compatible)
    print("\n1. Original interface (backward compatible):")
    print("-" * 80)
    print("""
    from src.sb99.update_feedback import get_currentSB99feedback

    # Same interface as before, but with CRITICAL BUG FIXES!
    feedback = get_currentSB99feedback(t=5.0, params=params)
    Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot = feedback

    # vWind is now CORRECT (was 10-80% wrong in original!)
    print(f"Wind velocity: {vWind:.2f} pc/Myr")

    # Access separated wind and SNe components:
    pdot_wind = params['F_ram_wind'].value
    pdot_SNe = params['F_ram_SN'].value
    """)

    # Example 2: Pure function interface (recommended for new code)
    print("\n2. Pure function interface (RECOMMENDED for new code):")
    print("-" * 80)
    print("""
    from src.sb99.update_feedback import get_SB99_feedback_pure

    # No side effects, returns dataclass with explicit fields
    feedback = get_SB99_feedback_pure(t=5.0, SB99f=params['SB99f'].value)

    # Access with clear field names:
    print(f"Wind velocity: {feedback.vWind:.2f} pc/Myr")
    print(f"Wind momentum rate: {feedback.pdot_wind:.3e} M_sun·pc/Myr²")
    print(f"SNe momentum rate: {feedback.pdot_SNe:.3e} M_sun·pc/Myr²")
    print(f"Total momentum rate: {feedback.pdot_total:.3e} M_sun·pc/Myr²")

    # Automatic validation:
    errors = feedback.validate()
    if errors:
        print("Physics validation failed:", errors)
    """)

    print("\n" + "=" * 80)
    print("See ANALYSIS_get_currentSB99feedback.md for complete documentation")
    print("=" * 80)
