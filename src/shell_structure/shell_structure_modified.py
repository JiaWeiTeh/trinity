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

    # Ionization front properties (for P_HII convex blend)
    n_IF: float  # Density at ionization front (code units)
    R_IF: float  # Radius of ionization front (pc)


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
    is_shellDissolved = False
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
        phi0 = phiShell_arr[idx]
        tau0_ion = tauShell_arr[idx]
        mShell0 = mShell_arr_cum[idx]
        rShell_start = rShell_arr[idx]

        # Check for shell dissolution
        if nShell_arr[0] < params['stop_n_diss'].value:
            is_shellDissolved = True

    # Append final values
    mShell_arr_ion = np.append(mShell_arr_ion, mShell_arr[idx])
    mShell_arr_cum_ion = np.append(mShell_arr_cum_ion, mShell_arr_cum[idx])
    phiShell_arr_ion = np.append(phiShell_arr_ion, phiShell_arr[idx])
    tauShell_arr_ion = np.append(tauShell_arr_ion, tauShell_arr[idx])
    nShell_arr_ion = np.append(nShell_arr_ion, nShell_arr[idx])
    rShell_arr_ion = np.append(rShell_arr_ion, rShell_arr[idx])

    # Extract ionization front properties (for P_HII convex blend)
    n_IF = nShell_arr_ion[-1]  # Density at ionization front
    R_IF = rShell_arr_ion[-1]  # Radius of ionization front

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

    elif is_shellDissolved:
        f_absorbed_ion = 1.0
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
        R_IF = rShell_previous

        logger.info('Shell dissolved.')

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
        n_IF=n_IF,
        R_IF=R_IF,
    )

