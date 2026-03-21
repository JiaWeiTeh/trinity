#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified shell structure with dataclass returns.

This module provides shell structure calculations that return a dataclass
instead of mutating the params dictionary. This is essential for use with
adaptive ODE solvers.

Key difference from shell_structure.py:
- shell_structure_pure() returns a ShellProperties dataclass
- No dictionary mutations during calculation
- Use updateDict(params, shell_data) after call returns

@author: Jia Wei Teh
"""

import numpy as np
import scipy.integrate
import scipy.optimize
from dataclasses import dataclass
from typing import Optional, Union
import logging

from src.shell_structure import get_shellODE

logger = logging.getLogger(__name__)


@dataclass
class ShellProperties:
    """
    Dataclass containing all shell structure properties.

    This can be used with updateDict(params, shell_properties) to
    update the params dictionary after shell calculation completes.
    """
    # Shell density
    shell_n0: float  # Density at inner edge of shell

    # Shell geometry
    rShell: float  # Outer radius of shell
    shell_thickness: float  # Thickness of shell

    # Absorption fractions
    shell_fAbsorbedIon: float  # Fraction of ionizing radiation absorbed
    shell_fAbsorbedNeu: float  # Fraction of non-ionizing radiation absorbed
    shell_fAbsorbedWeightedTotal: float  # Luminosity-weighted total absorption
    shell_fIonisedDust: float  # Fraction of ionizing radiation absorbed by dust

    # Shell properties
    shell_nMax: float  # Maximum density in shell
    shell_tauKappaRatio: float  # tau_IR / kappa_IR = integral(rho dr)
    shell_F_rad: float  # Radiation force with IR trapping

    # Gravitational properties
    shell_grav_r: Union[np.ndarray, float]  # Radius array for gravity
    shell_grav_phi: float  # Gravitational potential
    shell_grav_force_m: Union[np.ndarray, float]  # Gravitational force per unit mass

    # State flags
    isDissolved: bool  # Is the shell dissolved?
    is_fullyIonised: bool  # Is the shell fully ionized?
    diss_condition_met: bool  # Is shell_nMax < nISM this timestep?

    # Ionization front properties (for P_HII convex blend)
    n_IF: float  # Density at ionization front (code units)
    n_IF_ODE: float  # Raw ODE-derived n_IF before max() selection (code units)
    R_IF: float  # Radius of ionization front (pc)
    n_IF_Str: float  # Strömgren-based n_IF diagnostic (Lancaster+2025)

    # Independent ionization-equilibrium pressure (root-find diagnostic)
    P_HII_free: float      # Independent P_HII from root-find (code units)
    n0_HII_free: float     # Root-find inner-edge density (code units)

    # Shell density profile arrays (ionized + neutral)
    shell_r_arr: Union[np.ndarray, float]  # Radial grid through shell [pc]
    shell_n_arr: Union[np.ndarray, float]  # Number density through shell [1/pc^3]
    shell_ion_idx: int  # Last index of ionized region in shell_r/n_arr (-1 if empty)


def _integrate_ionized_to_mass_limit(n0, R2, M_shell_target, params):
    """
    Integrate the ionized shell ODE from density n0 at radius R2
    until phi drops to zero.  Return mass-based residual for root-find.

    Parameters
    ----------
    n0 : float
        Trial inner-edge density (code units, 1/pc³).
    R2 : float
        Inner shell radius (pc).
    M_shell_target : float
        Total shell mass to integrate through (code units).
    params : DescribedDict
        For physical constants only (alpha_B, sigma_d, Qi, Li, Ln, mu_ion,
        mu_atom, k_B, TShell_ion, c_light). NOT used for Pb.

    Returns
    -------
    residual : float
        (M_at_phi_zero - M_shell_target) / M_shell_target.
        Positive means n0 is too low  (photons ionize past M_shell).
        Negative means n0 is too high (photons absorbed before M_shell).
        Zero means n0 is the correct root.
    """
    Qi = params['Qi'].value
    alpha_B = params['caseB_alpha'].value

    # Maximum shell radius (Strömgren radius from n0)
    max_shellRadius = (3 * Qi / (4 * np.pi * alpha_B * n0**2))**(1/3) + R2

    # Integration parameters (same as shell_structure_pure)
    nsteps = 1e3
    sliceSize = np.min([1, (max_shellRadius - R2) / 10])
    rShell_step = sliceSize / nsteps

    # Guard against degenerate step sizes
    if rShell_step <= 0 or not np.isfinite(rShell_step):
        return 1.0  # n0 too low — photons would leak through infinite volume

    f_cover = 1
    rShell_start = R2
    nShell0 = n0
    phi0 = 1.0
    tau0_ion = 0.0
    mShell0 = 0.0

    # Integration limit: stop if mass exceeds 10x target (n0 clearly too low)
    max_mass = 10.0 * M_shell_target
    max_iter = 200  # prevent runaway loops

    for _ in range(max_iter):
        rShell_stop = rShell_start + sliceSize
        rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)
        if len(rShell_arr) < 2:
            break

        is_ionised = True
        y0 = [nShell0, phi0, tau0_ion]
        sol_ODE = scipy.integrate.odeint(
            get_shellODE.get_shellODE, y0, rShell_arr,
            args=(f_cover, is_ionised, params)
        )
        nShell_arr = sol_ODE[:, 0]
        phiShell_arr = sol_ODE[:, 1]
        tauShell_arr = sol_ODE[:, 2]

        # Guard against ODE divergence (NaN/Inf)
        if not np.all(np.isfinite(phiShell_arr)):
            # ODE diverged — density too extreme
            return -1.0

        # Mass of each thin spherical shell
        mShell_arr = np.empty_like(rShell_arr)
        mShell_arr[0] = mShell0
        mShell_arr[1:] = (nShell_arr[1:] * params['mu_ion'].value *
                         4 * np.pi * rShell_arr[1:]**2 * rShell_step)
        mShell_arr_cum = np.cumsum(mShell_arr)

        # Has phi dropped below threshold?
        phiCondition = phiShell_arr <= 1e-9
        if np.any(phiCondition):
            idx_phi = np.nonzero(phiCondition)[0][0]
            M_at_phi0 = mShell_arr_cum[idx_phi]
            return (M_at_phi0 - M_shell_target) / M_shell_target

        # Has mass exceeded our integration limit? (n0 clearly too low)
        if mShell_arr_cum[-1] >= max_mass:
            return (max_mass - M_shell_target) / M_shell_target  # large positive

        # Reinitialize for next slice
        idx = len(rShell_arr) - 1
        nShell0 = nShell_arr[idx]
        phi0 = max(0.0, phiShell_arr[idx])
        tau0_ion = tauShell_arr[idx]
        mShell0 = mShell_arr_cum[idx]
        rShell_start = rShell_arr[idx]

    # Fell through without phi→0: n0 is too low, photons leak
    return 1.0


def compute_P_HII_free(Qi, R2, M_shell, params):
    """
    Compute independent ionization-equilibrium pressure via root-find.

    Finds n0 such that the ionized shell ODE, starting from n(R2) = n0,
    absorbs all Qi photons exactly when cumulative mass = M_shell.

    This function NEVER reads params['Pb']. It depends only on Qi,
    M_shell, and the shell ODE physics constants.

    Parameters
    ----------
    Qi : float
        Ionizing photon rate (code units, 1/Myr).
    R2 : float
        Inner shell radius (pc).
    M_shell : float
        Total shell mass (code units).
    params : DescribedDict
        For physical constants only.

    Returns
    -------
    P_HII_free : float
        Independent ionization-equilibrium pressure (code units).
    n0_star : float
        Root-find inner-edge density (code units).
    """
    # Guard against degenerate inputs
    if Qi <= 0 or M_shell <= 0 or R2 <= 0:
        return (0.0, 0.0)

    alpha_B = params['caseB_alpha'].value
    mu_ion = params['mu_ion'].value
    k_B = params['k_B'].value
    TShell_ion = params['TShell_ion'].value

    # Analytic density scale: uniform-shell Strömgren equilibrium
    n_eq = Qi * mu_ion / (alpha_B * M_shell)

    # Geometry correction: for large R2, the thin-shell equilibrium density
    # is n ~ n_eq * 3 / (4π R2²), which can be orders of magnitude lower.
    geo_factor = 3.0 / (4.0 * np.pi * max(R2, 1e-6)**2)
    n_eq_geo = n_eq * min(geo_factor, 1.0)  # never increase above n_eq

    # Use the geometric estimate as bracket center
    n_center = n_eq_geo

    def residual(n0):
        return _integrate_ionized_to_mass_limit(n0, R2, M_shell, params)

    try:
        # Initial bracket: wide range around geometry-aware estimate
        n_low = 0.01 * n_center
        n_high = 100.0 * n_center
        r_low = residual(n_low)
        r_high = residual(n_high)

        if r_low * r_high >= 0:
            # Expand bracket much wider
            n_low = 1e-4 * n_center
            n_high = 1e4 * n_center
            r_low = residual(n_low)
            r_high = residual(n_high)

        if r_low * r_high >= 0:
            # Logarithmic scan across many orders of magnitude
            # centered on n_eq_geo, scanning 12 decades
            n_scan_lo, n_scan_hi = n_low, n_high
            r_scan_lo, r_scan_hi = r_low, r_high
            for exp in np.linspace(-8, 8, 33):
                n_test = n_center * 10.0**exp
                r_test = residual(n_test)
                if r_scan_lo * r_test < 0:
                    n_scan_hi = n_test
                    r_scan_hi = r_test
                    break
                if r_scan_hi * r_test < 0:
                    n_scan_lo = n_test
                    r_scan_lo = r_test
                    break
                # Update the bounds that are closest to a sign change
                if r_test > 0:
                    n_scan_lo = n_test
                    r_scan_lo = r_test
                elif r_test < 0:
                    n_scan_hi = n_test
                    r_scan_hi = r_test
            n_low, n_high = n_scan_lo, n_scan_hi
            r_low, r_high = r_scan_lo, r_scan_hi

        if r_low * r_high < 0:
            n0_star = scipy.optimize.brentq(residual, n_low, n_high, rtol=1e-3)
        else:
            # Fallback: use geometry-aware estimate (best guess)
            logger.warning(
                f'P_HII_free root-find bracket failed '
                f'(r_low={r_low:.3e}, r_high={r_high:.3e}); '
                f'falling back to n_eq_geo={n_eq_geo:.3e}'
            )
            n0_star = n_eq_geo
    except Exception as e:
        logger.warning(f'P_HII_free root-find failed: {e}; falling back to n_eq_geo={n_eq_geo:.3e}')
        n0_star = n_eq_geo

    P_HII_free = 2.0 * n0_star * k_B * TShell_ion
    return (P_HII_free, n0_star)


def shell_structure_pure(params) -> ShellProperties:
    """
    Evaluate shell structure and return properties as a dataclass.

    This is the pure version that does NOT mutate params.
    All calculated values are returned in a ShellProperties dataclass.

    Parameters
    ----------
    params : DescribedDict
        Parameter dictionary (read-only access)

    Returns
    -------
    ShellProperties
        Dataclass containing all shell properties
    """
    # Read input parameters (no mutations)
    mBubble = params['bubble_mass'].value
    pBubble = params['Pb'].value
    rShell0 = params['R2'].value
    mShell_end = params['shell_mass'].value
    Qi = params['Qi'].value
    Li = params['Li'].value
    Ln = params['Ln'].value

    # Capture previous rShell for dissolved case (original doesn't update rShell when dissolved)
    rShell_previous = params['rShell'].value

    # TODO: Add f_cover from fragmentation mechanics
    f_cover = 1

    # Initialize values at r = rShell0 = inner edge of shell
    rShell_start = rShell0
    phi0 = 1  # Attenuation function for ionizing flux (unitless)
    tau0_ion = 0  # tau(r) at ionized region
    mShell0 = 0

    # Density at inner edge of shell
    nShell0 = (params['mu_ion'].value / params['mu_atom'].value /
               (params['k_B'].value * params['TShell_ion'].value) * params['Pb'].value)
    shell_n0 = nShell0  # Store for output

    # Initialize logic gates
    is_allMassSwept = False
    is_shellDissolved = params.get('isDissolved', False)
    if hasattr(is_shellDissolved, 'value'):
        is_shellDissolved = is_shellDissolved.value
    is_fullyIonised = False

    # Arrays for ionized region
    mShell_arr_ion = np.array([])
    mShell_arr_cum_ion = np.array([])
    phiShell_arr_ion = np.array([])
    tauShell_arr_ion = np.array([])
    nShell_arr_ion = np.array([])
    rShell_arr_ion = np.array([])

    # Maximum shell radius (for integration bounds)
    max_shellRadius = (3 * Qi / (4 * np.pi * params['caseB_alpha'].value * nShell0**2))**(1/3) + rShell_start

    # Integration parameters
    nsteps = 1e3
    sliceSize = np.min([1, (max_shellRadius - rShell_start) / 10])
    rShell_step = sliceSize / nsteps

    logger.debug(f'sliceSize={sliceSize}, max_shellRadius={max_shellRadius}, '
                 f'rShell_start={rShell_start}, rShell_step={rShell_step}')

    # =============================================================================
    # Ionized region integration
    # =============================================================================
    while not is_allMassSwept and not is_fullyIonised:
        logger.debug('Ionised shell loop: not is_allMassSwept and not is_fullyIonised')

        rShell_stop = rShell_start + sliceSize
        rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)

        is_ionised = True
        y0 = [nShell0, phi0, tau0_ion]
        sol_ODE = scipy.integrate.odeint(
            get_shellODE.get_shellODE, y0, rShell_arr,
            args=(f_cover, is_ionised, params)
        )
        nShell_arr = sol_ODE[:, 0]
        phiShell_arr = sol_ODE[:, 1]
        tauShell_arr = sol_ODE[:, 2]

        # Mass of spherical shell
        mShell_arr = np.empty_like(rShell_arr)
        mShell_arr[0] = mShell0
        mShell_arr[1:] = (nShell_arr[1:] * params['mu_ion'].value *
                         4 * np.pi * rShell_arr[1:]**2 * rShell_step)
        mShell_arr_cum = np.cumsum(mShell_arr)

        # Find termination index
        massCondition = mShell_arr_cum >= mShell_end
        phiCondition = phiShell_arr <= 1e-9 #small positive threshold
        idx_array = np.nonzero((massCondition | phiCondition))[0]

        if len(idx_array) == 0:
            idx = len(rShell_arr) - 1
        else:
            idx = idx_array[0]

        mShell_arr_cum[idx + 1:] = 0.0
        is_allMassSwept = any(massCondition)
        is_fullyIonised = any(phiCondition)

        # Store values
        mShell_arr_ion = np.concatenate((mShell_arr_ion, mShell_arr[:idx]))
        mShell_arr_cum_ion = np.concatenate((mShell_arr_cum_ion, mShell_arr_cum[:idx]))
        phiShell_arr_ion = np.concatenate((phiShell_arr_ion, phiShell_arr[:idx]))
        tauShell_arr_ion = np.concatenate((tauShell_arr_ion, tauShell_arr[:idx]))
        nShell_arr_ion = np.concatenate((nShell_arr_ion, nShell_arr[:idx]))
        rShell_arr_ion = np.concatenate((rShell_arr_ion, rShell_arr[:idx]))

        # Reinitialize for next iteration
        nShell0 = nShell_arr[idx]
        phi0 = max(0.0, phiShell_arr[idx])   # guard against sub-threshold negative phi
        tau0_ion = tauShell_arr[idx]
        mShell0 = mShell_arr_cum[idx]
        rShell_start = rShell_arr[idx]

        # Dissolution condition is now evaluated after shell structure is computed
        # (see diss_condition_met below); shell_structure_pure is stateless.

    # Append final values
    mShell_arr_ion = np.append(mShell_arr_ion, mShell_arr[idx])
    mShell_arr_cum_ion = np.append(mShell_arr_cum_ion, mShell_arr_cum[idx])
    phiShell_arr_ion = np.append(phiShell_arr_ion, phiShell_arr[idx])
    tauShell_arr_ion = np.append(tauShell_arr_ion, tauShell_arr[idx])
    nShell_arr_ion = np.append(nShell_arr_ion, nShell_arr[idx])
    rShell_arr_ion = np.append(rShell_arr_ion, rShell_arr[idx])

    # Extract ionization front properties (for P_HII convex blend)
    n_IF = nShell_arr_ion[-1]  # Density at ionization front
    n_IF_ODE = n_IF            # Preserve raw ODE value before max() selection
    R_IF = rShell_arr_ion[-1]  # Radius of ionization front

    # ------------------------------------------------------------------
    # Two-branch n_IF: Strömgren correction (Lancaster+2025)
    #
    # The ODE-derived n_IF is anchored to Pb at the inner boundary via
    # pressure equilibrium (Rahner+2017 Eq.14), with density rising
    # outward as radiation pressure sets up a thermal pressure gradient.
    # This captures the bubble-dominated (ζ > 1) regime correctly.
    #
    # When the bubble is sub-dominant (ζ < 1), the HII region is
    # independently pressurised via ionisation balance. The Strömgren
    # density n_IF_Str = sqrt(3Qi / (4π αB ΔV)) is independent of Pb.
    # Using max() is conservative: existing behaviour is unchanged
    # unless n_IF_Str exceeds the ODE value.
    #
    # Guards:
    #   (a) is_fullyIonised=True  → photons escape; R_IF is the shell
    #       outer edge, not a physical pressure surface. The Strömgren
    #       volume would absorb zero photons. Skip correction.
    #   (b) rShell0 >= rCloud     → shell has left the cloud; ambient
    #       density structure changes and the ionisation balance inside
    #       the shell is no longer well-defined against a GMC background.
    #       Skip correction; external ISM pressure is handled separately
    #       via press_HII_in in the ODE.
    # ------------------------------------------------------------------
    _rCloud = params['rCloud'].value
    _vol_ion = R_IF**3 - rShell0**3   # rShell0 == params['R2'].value
    if (not is_fullyIonised) and (rShell0 < _rCloud) and (_vol_ion > 0.0) and (Qi > 0.0):
        n_IF_Str = np.sqrt(
            3.0 * Qi /
            (4.0 * np.pi * params['caseB_alpha'].value * _vol_ion)
        )
        # Two-branch selection: physics lives here.
        # P_drive = max(Pb, P_HII) in the ODE remains a safety floor.
        n_IF = max(n_IF, n_IF_Str)
    else:
        # Fully ionised, beyond cloud, or degenerate geometry:
        # keep ODE-derived n_IF unchanged.
        n_IF_Str = n_IF

    # =============================================================================
    # Continue computation if shell hasn't dissolved
    # =============================================================================
    if not is_shellDissolved:
        logger.debug('Shell not dissolved, computing gravitational potential')

        # Gravitational potential for ionized part
        grav_ion_rho = nShell_arr_ion * params['mu_ion'].value
        grav_ion_r = rShell_arr_ion
        grav_ion_m = grav_ion_rho * 4 * np.pi * grav_ion_r**2 * rShell_step
        grav_ion_m_cum = np.cumsum(grav_ion_m) + mBubble
        grav_ion_phi = -4 * np.pi * params['G'].value * scipy.integrate.simpson(
            grav_ion_r * grav_ion_rho, x=grav_ion_r
        )
        grav_phi = grav_ion_phi
        grav_ion_force_m = params['G'].value * grav_ion_m_cum / grav_ion_r**2

        grav_force_m = grav_ion_force_m
        grav_r = grav_ion_r

        # Dust vs hydrogen absorption
        dr_ion_arr = rShell_arr_ion[1:] - rShell_arr_ion[:-1]
        phi_dust = np.sum(
            -nShell_arr_ion[:-1] * params['dust_sigma'].value * phiShell_arr_ion[:-1] * dr_ion_arr
        )
        phi_hydrogen = np.sum(
            -4 * np.pi * rShell_arr_ion[:-1]**2 / Qi *
            params['caseB_alpha'].value * nShell_arr_ion[:-1]**2 * dr_ion_arr
        )

        if (phi_dust + phi_hydrogen) == 0.0:
            f_ionised_dust = 0.0
        else:
            f_ionised_dust = phi_dust / (phi_dust + phi_hydrogen)

        # Arrays for neutral region
        mShell_arr_neu = np.array([])
        mShell_arr_cum_neu = np.array([])
        tauShell_arr_neu = np.array([])
        nShell_arr_neu = np.array([])
        rShell_arr_neu = np.array([])
        rShell_start = rShell_arr_ion[-1]

        logger.debug('Ready to evaluate neutral shell region')

        # =============================================================================
        # Neutral region integration (if not fully ionized)
        # =============================================================================
        if not is_fullyIonised:
            logger.debug('Shell not fully ionised, calculating neutral region')

            # Temperature/density discontinuity at boundary
            nShell0 = (nShell0 * params['mu_atom'].value / params['mu_ion'].value *
                      params['TShell_ion'].value / params['TShell_neu'].value)
            tau0_neu = tau0_ion

            tau_max = 100
            nsteps = 5e3
            sliceSize = np.min([1, (max_shellRadius - rShell_start) / 10])
            rShell_step = sliceSize / nsteps

            while not is_allMassSwept:
                logger.debug('Neutral shell loop: not is_allMassSwept')

                rShell_stop = rShell_start + sliceSize
                rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)
                is_ionised = False

                y0 = [nShell0, tau0_neu]
                sol_ODE = scipy.integrate.odeint(
                    get_shellODE.get_shellODE, y0, rShell_arr,
                    args=(f_cover, is_ionised, params)
                )
                nShell_arr = sol_ODE[:, 0]
                tauShell_arr = sol_ODE[:, 1]

                mShell_arr = np.empty_like(rShell_arr)
                mShell_arr[0] = mShell0
                mShell_arr[1:] = (nShell_arr[1:] * params['mu_atom'].value *
                                 4 * np.pi * rShell_arr[1:]**2 * rShell_step)
                mShell_arr_cum = np.cumsum(mShell_arr)

                massCondition = mShell_arr_cum >= mShell_end
                idx_array = np.nonzero(massCondition)[0]

                if len(idx_array) == 0:
                    idx = len(rShell_arr) - 1
                else:
                    idx = idx_array[0]

                is_allMassSwept = any(massCondition)

                mShell_arr_neu = np.concatenate((mShell_arr_neu, mShell_arr[:idx]))
                mShell_arr_cum_neu = np.concatenate((mShell_arr_cum_neu, mShell_arr_cum[:idx]))
                tauShell_arr_neu = np.concatenate((tauShell_arr_neu, tauShell_arr[:idx]))
                nShell_arr_neu = np.concatenate((nShell_arr_neu, nShell_arr[:idx]))
                rShell_arr_neu = np.concatenate((rShell_arr_neu, rShell_arr[:idx]))

                nShell0 = nShell_arr[idx]
                tau0_neu = tauShell_arr[idx]
                mShell0 = mShell_arr_cum[idx]
                rShell_start = rShell_arr[idx]

            # Append final neutral values
            mShell_arr_neu = np.append(mShell_arr_neu, mShell_arr[idx])
            mShell_arr_cum_neu = np.append(mShell_arr_cum_neu, mShell_arr_cum[idx])
            tauShell_arr_neu = np.append(tauShell_arr_neu, tauShell_arr[idx])
            nShell_arr_neu = np.append(nShell_arr_neu, nShell_arr[idx])
            rShell_arr_neu = np.append(rShell_arr_neu, rShell_arr[idx])

            # Gravitational potential for neutral part
            grav_neu_rho = nShell_arr_neu * params['mu_atom'].value
            grav_neu_r = rShell_arr_neu
            grav_neu_m = grav_neu_rho * 4 * np.pi * grav_neu_r**2 * rShell_step
            grav_neu_m_cum = np.cumsum(grav_neu_m) + grav_ion_m_cum[-1]
            grav_neu_phi = -4 * np.pi * params['G'].value * scipy.integrate.simpson(
                grav_neu_r * grav_neu_rho, x=grav_neu_r
            )
            grav_phi = grav_neu_phi + grav_ion_phi
            grav_neu_force_m = params['G'].value * grav_neu_m_cum / grav_neu_r**2

            grav_force_m = np.concatenate([grav_force_m, grav_neu_force_m])
            grav_r = np.concatenate([grav_r, grav_neu_r])

        logger.debug(f'Checking shell phiShell_arr_ion[:10]: {phiShell_arr_ion[:10]}')

        # =============================================================================
        # Compute final shell properties
        # =============================================================================
        if is_fullyIonised:
            shellThickness = rShell_arr_ion[-1] - rShell0
            tau_rEnd = tauShell_arr_ion[-1]
            phi_rEnd = phiShell_arr_ion[-1]
            if phi_rEnd < 0:
                phi_rEnd = 0
            nShell_max = np.max(nShell_arr_ion)
            tau_kappa_IR = params['mu_ion'].value * np.sum(nShell_arr_ion[:-1] * dr_ion_arr)
        else:
            shellThickness = rShell_arr_neu[-1] - rShell0
            tau_rEnd = tauShell_arr_neu[-1]
            phi_rEnd = 0
            nShell_max = max(np.max(nShell_arr_ion), np.max(nShell_arr_neu))
            dr_neu_arr = rShell_arr_neu[1:] - rShell_arr_neu[:-1]
            tau_kappa_IR = (params['mu_ion'].value * np.sum(nShell_arr_ion[:-1] * dr_ion_arr) +
                params['mu_atom'].value * np.sum(nShell_arr_neu[:-1] * dr_neu_arr))

        # Absorption fractions
        f_absorbed_ion = 1 - phi_rEnd
        f_absorbed_neu = 1 - np.exp(-tau_rEnd)
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln) / (Li + Ln)

        rShell = grav_r[-1]

        # Radiation force with IR trapping
        shell_F_rad = (f_absorbed * params['Lbol'].value / params['c_light'].value * (1 + tau_kappa_IR * params['dust_KappaIR'].value))

        # Combined shell density profile (ionized + neutral)
        # shell_ion_idx: last index belonging to the ionized region.
        # If shell_ion_idx == len(shell_r_arr)-1, the entire shell is ionized
        # (either is_fullyIonised, or all mass swept with photons leaking out).
        shell_ion_idx = len(rShell_arr_ion) - 1
        if is_fullyIonised or (is_allMassSwept and len(rShell_arr_neu) == 0):
            shell_r_arr = rShell_arr_ion
            shell_n_arr = nShell_arr_ion
        else:
            shell_r_arr = np.concatenate([rShell_arr_ion, rShell_arr_neu])
            shell_n_arr = np.concatenate([nShell_arr_ion, nShell_arr_neu])

        # === Independent ionization-equilibrium pressure (root-find) ===
        # Compute P_HII_free: the pressure the ionized layer would exert
        # if its density were set by ionization balance alone (no Pb dependence).
        # This is always computed as a diagnostic. When use_Pb_BC_for_PHii=False,
        # the momentum-phase runner uses this value upstream to override the
        # shell BC BEFORE calling shell_structure_pure.
        _Qi_for_rootfind = Qi
        _R2_for_rootfind = rShell0    # rShell0 == params['R2'].value == inner shell edge
        _Msh_for_rootfind = mShell_end
        if _Qi_for_rootfind > 0.0 and _Msh_for_rootfind > 0.0 and _R2_for_rootfind > 0.0:
            P_HII_free, n0_HII_free = compute_P_HII_free(
                _Qi_for_rootfind, _R2_for_rootfind, _Msh_for_rootfind, params
            )
        else:
            P_HII_free = 0.0
            n0_HII_free = 0.0

    elif is_shellDissolved:
        f_absorbed_ion = 0.0 # dissolved shell = no absorber; ionizing photons escape freely
        f_absorbed_neu = 0.0
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln) / (Li + Ln)
        f_ionised_dust = np.nan
        is_fullyIonised = True
        shellThickness = np.nan
        nShell_max = params['nISM'].value
        tau_kappa_IR = 0
        grav_r = np.nan
        grav_phi = np.nan
        grav_force_m = np.nan
        # Keep previous rShell value when dissolved (matches original behavior)
        rShell = rShell_previous
        shell_F_rad = 0.0
        # No ionization front when dissolved
        n_IF = 0.0
        n_IF_ODE = 0.0
        R_IF = 0.0
        n_IF_Str = 0.0
        P_HII_free = 0.0
        n0_HII_free = 0.0
        shell_r_arr = np.array([])
        shell_n_arr = np.array([])
        shell_ion_idx = -1

        logger.warning('Shell dissolved.')

    # Evaluate instantaneous dissolution condition: shell_nMax < nISM
    nISM = params['nISM'].value
    allow_dissolution = params.get('allowShellDissolution', True)
    if hasattr(allow_dissolution, 'value'):
        allow_dissolution = allow_dissolution.value
    diss_condition_met = bool(allow_dissolution and nShell_max < nISM)

    # Return dataclass with all properties
    return ShellProperties(
        shell_n0=shell_n0,
        rShell=rShell,
        shell_thickness=shellThickness,
        shell_fAbsorbedIon=f_absorbed_ion,
        shell_fAbsorbedNeu=f_absorbed_neu,
        shell_fAbsorbedWeightedTotal=f_absorbed,
        shell_fIonisedDust=f_ionised_dust,
        shell_nMax=nShell_max,
        shell_tauKappaRatio=tau_kappa_IR,
        shell_F_rad=shell_F_rad,
        shell_grav_r=grav_r,
        shell_grav_phi=grav_phi,
        shell_grav_force_m=grav_force_m,
        isDissolved=is_shellDissolved,
        is_fullyIonised=is_fullyIonised,
        diss_condition_met=diss_condition_met,
        n_IF=n_IF,
        n_IF_ODE=n_IF_ODE,
        R_IF=R_IF,
        n_IF_Str=n_IF_Str,
        P_HII_free=P_HII_free,
        n0_HII_free=n0_HII_free,
        shell_r_arr=shell_r_arr,
        shell_n_arr=shell_n_arr,
        shell_ion_idx=shell_ion_idx,
    )

