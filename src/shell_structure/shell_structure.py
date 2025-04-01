#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 20:53:03 2022

@author: Jia Wei Teh

This script contains a function that evaluates the shell structure.
"""
# libraries
import numpy as np
import astropy.constants as c
import scipy.integrate 
import astropy.units as u
import os
import sys
from astropy.table import Table
import src._functions.unit_conversions as cvt

#--
from src.shell_structure import get_shellODE, get_shellParams

# get parameters
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()


def shell_structure(rShell0, 
                    pBubble,
                    mBubble, 
                    Ln, Li, Qi,
                    mShell_end,
                    f_cover,
                    params
                    ):
    
    
    """
    This function evaluates the shell structure. Includes the ability to 
    also treat the shell as composite region (i.e., ionised + neutral region).

    Assumes cgs! Will take in other units, but will change into cgs.

    Parameters
    ----------
    rShell0 : float
        Radius of inner shell.
    pBubble : float
        Bubble pressure.
    mBubble : float  (Minterior in old code)
        Bubble mass.
    Ln : float
        Non-ionising luminosity.
    Li : float
        Ioinising luminosity.
    Qi : float
        Ionising photon rate.
    mShell_end : float
        Maximum total shell mass. (Msh_fix_au)
    sigma_dust : float
        Dust cross section after scaling with metallicity.
    f_cover : float
        DESCRIPTION.
    warpfield_params : Object
        Object containing WARPFIELD parameters.

    Returns
    -------
    "f_absorbed_ion": float
        Fraction of absorbed ionising radiations.
    "f_absorbed_neu": float
        Fraction of absorbed non-ionising radiations.
    "f_absorbed": float
        Total absorption fraction, defined as luminosity weighted average of 
        f_absorbed_ion and f_absorbed_neu.
    "f_ionised_dust": float
        How much ionising radiation is absorbed by dust?
    "is_fullyIonised": boolean
        Is the shell fully ionised?
    "shellThickness": float
        The thickness of the shell.
    "nShell0": float
        The density of shell at inner edge/radius
    "nShell0_cloud": float
        The density of shell at inner edge/radius, but including B-field, as
        this will be passed to CLOUDY.
    "nShell_max": float
        The maximum density across the shell.
    "tau_kappa_IR": float
        The ratio between optical depth and dust opacity, tau_IR/kappa_IR  = \int rho dr
    "grav_r": array
        The array containing radius at which the gravitational potential is evaluated.
    "grav_phi": float
        Gravitational potential 
    "grav_force_m": array
        The array containing gravitational force per unit mass evaluated at grav_r.
    """
    # Notes: 
    # old code: shell_structure2()
    
    # TODO: Add also f_cover.
    # TODO: Check also neutral region.
    
    # initialise values at r = rShell0 = inner edge of shell
    rShell_start = rShell0.to(u.cm)
    # attenuation function for ionising flux. Unitless.
    phi0 = 1 
    # tau(r) at ionised region?
    tau0_ion = 0
    mShell0 = 0 * u.g
    mShell_end = mShell_end.to(u.g)
    
    # Obtain density at the inner edge of shell
    nShell0 = get_shellParams.get_nShell0(pBubble, warpfield_params.t_ion)
    # define for future use, as nShell0 constantly changes in the loop.
    nShellInner = nShell0
    
    
    # =============================================================================
    # Before beginning the integration, initialise some logic gates.
    # =============================================================================
    # 1. Have we accounted all the shell mass in our integration? 
    # I.e., is cumulative_mass(r = R) >= mShell?
    is_allMassSwept = False
    # 2. Are all ionising photons being used up? 
    # I.e., is phi(r = R) = 0?
    is_phiZero = False
    # 3. Has the shell dissolved?
    is_shellDissolved = False
    # 4. Is the shell fully ionised at r = R?
    is_fullyIonised = False
    # Create arrays to store values
    # shell mass at r (ionised region)
    mShell_arr_ion = np.array([])
    # cumulative shell mass at r (ionised region)
    mShell_arr_cum_ion = np.array([])
    # phi at r (ionised region)
    # Note here that phiShell_arr_ion is redundant; this is because
    # the phi parameter (where phi > 0 ) naturally denotes an ionised region.
    # However, we initialise them here... just cause.
    phiShell_arr_ion = np.array([])
    # tau at r (ionised region)
    tauShell_arr_ion = np.array([])
    # density at r (ionised region)
    nShell_arr_ion = np.array([])
    # r array of ionised region
    rShell_arr_ion = np.array([])
    
    # First, set the end of integration, rShell_stop:
    # The integration range should not be larger than 1pc. However,
    # if the shell is lesser than 1pc, we will obviously use that instead.
    # Assuming constant density, what is the maximum possible shell thickness, 
    # assuming that density does not change?
    # Old version:
    max_shellThickness = (3 * Qi / (4 * np.pi * warpfield_params.alpha_B * nShell0**2) + rShell_start**3)**(1/3)
    max_shellThickness = max_shellThickness.to(u.cm)
    # New version(?):
    # max_shellThickness = r_stromgren - rShell0
    # max_shellThickness = (3 * Qi / (4 * np.pi * alpha_B * nShell0**2))**(1/3) - rShell_start
   
    # First, set the end of integration, rShell_stop:
    # The integration range should not be larger than 1pc. However,
    # if the shell step is lesser than 1pc, we will obviously use that instead.   
    mydr = np.min([1.0 * u.pc.to(u.cm), np.abs(max_shellThickness - rShell_start).value]) * u.cm

    # Then, set the step size. This will just be a very small number
    # i.e., 5e-4pc, unless the thickness itself is not sufficient to support 
    # at least 2000 steps cross the shell
    
    # this is basically saying, rShell_step > mydr/1e6, but <  5e-4 or mydr/1e3
    # since r = range(start, start+mydr, step), this means len(r) < 1e4 and len(r) > 1e3. 
    rShell_step = np.max([
        np.min([ 5e-4 * u.pc.to(u.cm), mydr.value/1e3]),
        mydr.value/1e6
        ]) * u.cm
    
    
                    #     # restate just for clarity
                    # L_total, T_rgoal, L_bubble, L_conduction, L_intermediate, dMdt_factor_out, Tavg = output
                    
                    # print('\n\nFinish bubble\n\n')
                    # print('L_total', L_total.to(u.M_sun*u.pc**2/u.Myr**3))
                    # print('T_rgoal', T_rgoal)
                    # print('L_bubble', L_bubble.to(u.M_sun*u.pc**2/u.Myr**3))
                    # print('L_conduction', L_conduction.to(u.M_sun*u.pc**2/u.Myr**3))
                    # print('L_intermediate', L_intermediate.to(u.M_sun*u.pc**2/u.Myr**3))
                    # print('dMdt_factor_out', dMdt_factor_out)
                    # print('Tavg', Tavg)
        
    print('\n\nwe are now in shell_structure.\n\n')
    print(f'rShell0: {rShell0.to(u.cm)}')
    print(f'pBubble: {pBubble.to(u.M_sun/u.pc/u.Myr**2)}')
    print(f'mBubble: {mBubble}')
    print(f'Ln: {Ln.to(u.M_sun*u.pc**2/u.Myr**3)}')
    print(f'Li: {Li.to(u.M_sun*u.pc**2/u.Myr**3)}')
    print(f'Qi: {Qi.to(1/u.Myr)}')
    print(f'mShell_end: {mShell_end}')
    print(f'max_shellThickness: {max_shellThickness}')
    print(f'nShell0: {nShell0}')
    print(f'rShell_step: {rShell_step}')
    # sys.exit()
    
    
    
    while not is_allMassSwept and not is_phiZero:
        
        # =============================================================================
        # Define the range at which integration occurs.
        # This is necessary because, unfortunately, the ODE is very stiff and 
        # requires small steps of r. 
        # This loop, we deal with situations where not all masses are swept into 
        # the shell, and at this rStep range not all ionisation is being used up (phi !=0).
        # =============================================================================
        
        # Therefore the end of integration is just rThickness + rStart
        # if the allowed shell thickness is more than 1pc, take 1pc as each stepsize. The ODE is 
        # very stiff, and cannot deal with bigger step size. If the shell is smaller than 
        # 1pc, then it can do it in one go. rShell_stop < 1pc. 
        # rShell_stop = np.min([max_shellThickness.value, 1]) * u.pc + rShell_start
        rShell_stop = mydr + rShell_start
        # We now have the array at which we integrate
        rShell_arr = np.arange(rShell_start.value, rShell_stop.value, rShell_step.value) * u.cm
        
        # Get arguments and parameters for integration:
        # ionised region    
        is_ionised = True
        # initial values
        y0 = [nShell0.to(1/u.cm**3).value, phi0, tau0_ion]
        # constants
        cons = [Ln, Li, Qi]
        # Run integration
        # TODO: problem here!
        sol_ODE = scipy.integrate.odeint(get_shellODE.get_shellODE, y0, rShell_arr.value,
                              args=(cons, f_cover, is_ionised),
                              rtol=1e-3, hmin=1e-7)
        # solved for n(r), phi(r), and tau(r)
        nShell_arr = sol_ODE[:,0] / u.cm**3
        phiShell_arr = sol_ODE[:,1] 
        tauShell_arr = sol_ODE[:,2]

        # mass of spherical shell. Volume given by V = 4 pi r**2 * thickness
        mShell_arr = np.empty_like(rShell_arr.value) * u.g
        mShell_arr[0] = mShell0.to(u.g)
        mShell_arr[1:] = (nShell_arr[1:] * warpfield_params.mu_n * 4 * np.pi * rShell_arr[1:]**2 * rShell_step).to(u.g)
        mShell_arr_cum = np.cumsum(mShell_arr)
        
        # =============================================================================
        # Now, find the index at which M(r = R) = Mshell, or phi(r = R) = 0 
        # If exists, then terminate the loop. Otherwise, repeat the loop
        # with new (a continuation of) sets of steps and start/end values.
        # =============================================================================
        massCondition = mShell_arr_cum.value >= mShell_end.value
        phiCondition = phiShell_arr <= 0
        idx_array = np.nonzero(( massCondition | phiCondition ))[0]
        # If there is none, then take as last index
        if len(idx_array) == 0:
            idx = len(rShell_arr) - 1
        else:
            idx = idx_array[0]
        # if such idx exists, set anything after that to 0.
        mShell_arr_cum[idx+1:] = 0.0
        # Associated condition
        # True if any part of the array is true
        is_allMassSwept = any(massCondition) 
        is_fullyIonised = any(massCondition)
        is_phiZero = any(phiCondition)
        
        # Store values into arrays representing profile in the ionised region. 
        mShell_arr_ion = np.concatenate(( mShell_arr_ion, mShell_arr[:idx-1]))
        mShell_arr_cum_ion = np.concatenate(( mShell_arr_cum_ion, mShell_arr_cum[:idx-1]))
        phiShell_arr_ion = np.concatenate(( phiShell_arr_ion, phiShell_arr[:idx-1]))
        tauShell_arr_ion = np.concatenate(( tauShell_arr_ion, tauShell_arr[:idx-1]))
        nShell_arr_ion = np.concatenate(( nShell_arr_ion, nShell_arr[:idx-1]))
        rShell_arr_ion = np.concatenate(( rShell_arr_ion, rShell_arr[:idx-1]))

        # Reinitialise values for next integration
        nShell0 = nShell_arr[idx - 1]
        phi0 = phiShell_arr[idx - 1]
        tau0_ion = tauShell_arr[idx - 1]
        mShell0 = mShell_arr_cum[idx - 1]
        rShell_start = rShell_arr[idx - 1]
        
        # Consider the shell dissolved if the followings occur:
        # 1. The density of shell is far lower than the density of ISm.
        # 2. The shell has expanded too far.
        # TODO: output message to tertminal depending on verbosity
        if nShellInner < (0.001 * warpfield_params.nISM) or\
            rShell_stop == (1.2 * warpfield_params.stop_r * u.pc.to(u.cm)) or\
                (rShell_start - rShell_stop) > (10 * rShell_start):
                    is_shellDissolved = True
                    break
        
        # begin next iteration if shell is not all ionised and mshell is not all accounted for.
        # if either condition is not met, move on.
        
    # append the last few values that are otherwise missed in the while loop.
    mShell_arr_ion = np.append(mShell_arr_ion, mShell_arr[idx-1])
    mShell_arr_cum_ion = np.append(mShell_arr_cum_ion, mShell_arr_cum[idx-1])
    phiShell_arr_ion = np.append(phiShell_arr_ion, phiShell_arr[idx-1])
    tauShell_arr_ion = np.append(tauShell_arr_ion, tauShell_arr[idx-1])
    nShell_arr_ion = np.append(nShell_arr_ion, nShell_arr[idx-1])
    rShell_arr_ion = np.append(rShell_arr_ion, rShell_arr[idx-1])

    # print('rShell_arr_ion')
    # print(rShell_arr_ion[-1])
    # print('nShell_arr_ion')
    # print(nShell_arr_ion[-1])
    # print('tauShell_arr_ion')
    # print(tauShell_arr_ion[-1])
    # print('phiShell_arr_ion')
    # print(phiShell_arr_ion[-1])
    # print('mShell_arr_ion')
    # print(mShell_arr_ion[-1])
    # print('mShell_arr_cum_ion')
    # print(mShell_arr_cum_ion[-1])
    # sys.exit()
    
    # =============================================================================
    # If shell hasn't dissolved, continue some computation to prepare for 
    # further evaulation.
    # =============================================================================
    if not is_shellDissolved:
    
        # =============================================================================
        # First, compute the gravitational potential for the ionised part of shell
        # =============================================================================
        grav_ion_rho = (nShell_arr_ion * warpfield_params.mu_n).to(u.g/u.cm**3)
        grav_ion_r = rShell_arr_ion
        # mass of the thin spherical shell
        grav_ion_m = grav_ion_rho * 4 * np.pi * grav_ion_r**2 * rShell_step
        # cumulative mass
        grav_ion_m_cum = np.cumsum(grav_ion_m) + mBubble
        # gravitational potential
        grav_ion_phi = - 4 * np.pi * c.G.cgs * (scipy.integrate.simps(grav_ion_r * grav_ion_rho, x = grav_ion_r) * u.cm**2 * u.g/u.cm**3)
        # it is now in cgs
        grav_ion_phi = grav_ion_phi.decompose(bases = u.cgs.bases)
        # mark for future use
        grav_phi = grav_ion_phi
        # gravitational potential force per unit mass
        grav_ion_force_m = c.G.cgs * grav_ion_m_cum / grav_ion_r**2
        
        # Now, modify the array so that it matches the potential file.
        # I am not entirely sure what this section does, but it was in the
        # old code. 
        potentialFile_internalLength = 10000
        skip_idx = max(
            int(
                len(grav_ion_r)/potentialFile_internalLength
                ), 1
            )        
        # modify array
        grav_ion_r = grav_ion_r[:-1:skip_idx]
        grav_ion_force_m = grav_ion_force_m[:-1:skip_idx]
        
        # For the whole shell; but for now we have only calculated the ionised part.
        grav_force_m = grav_ion_force_m
        grav_r = grav_ion_r
        
        # How much ionising radiation is absorbed by dust and how much by hydrogen?
        dr_ion_arr = rShell_arr_ion[1:] - rShell_arr_ion[:-1]
        # We do so by integrating using left Riemann sums 
        # dust term in dphi/dr
        phi_dust = np.sum(
                        - nShell_arr_ion[:-1] * warpfield_params.sigma_d * phiShell_arr_ion[:-1] * dr_ion_arr
                        )
        # recombination term in dphi/dr
        phi_hydrogen = np.sum(
                        - 4 * np.pi * rShell_arr_ion[:-1]**2 / Qi * warpfield_params.alpha_B * nShell_arr_ion[:-1]**2 * dr_ion_arr
                        )
        # If there is no ionised shell (e.g., because the ionising radiation is too weak)
        if (phi_dust + phi_hydrogen).value == 0.0:
            f_ionised_dust = 0.0
            f_ionised_hydrogen = 0.0
        # If there is, compute the fraction.
        else:
            f_ionised_dust = phi_dust / (phi_dust + phi_hydrogen)
            f_ionised_hydrogen  = phi_hydrogen / (phi_dust + phi_hydrogen)
            
        # Create arrays to store values
        # shell mass at r (neutral region)
        mShell_arr_neu = np.array([])
        # cumulative shell mass at r (neutral region)
        mShell_arr_cum_neu = np.array([])
        # tau at r (neutral region)
        tauShell_arr_neu = np.array([])
        # density at r (neutral region)
        nShell_arr_neu = np.array([])
        # r array of neutral region
        rShell_arr_neu = np.array([])
        # reinitialise
        rShell_start = rShell_arr_ion[-1]
        
        # =============================================================================
        # If the shell is not fully ionised, calculate structure of 
        # non-ionized (neutral) part
        # =============================================================================
        if not is_fullyIonised:
            
            # Pressure equilibrium dictates that there will be a temperature and density
            # discontinuity at boundary between ionised and neutral region.
            nShell0 = nShell0 * warpfield_params.mu_n / warpfield_params.mu_p * warpfield_params.t_ion / warpfield_params.t_neu
            # tau(r) at neutral shell region
            tau0_neu = tau0_ion
            
            # =============================================================================
            # Evaluate the remaining neutral shell until all masses are accounted for.
            # =============================================================================
            # Entering this loop means not all mShell has been accounted for. Thus
            # is_phiZero can either be True or False here.
            while not is_allMassSwept:
    
                # if tau is already 100, there is no point in integrating more.
                tau_max = 100
                # the maximum width of the neutral shell, assuming constant density.
                max_shellThickness = np.abs((tau_max - tau0_ion)/(nShell0 * warpfield_params.sigma_d))
                # the end range of integration 
                rShell_stop = np.min([ 1 * u.pc.to(u.cm), max_shellThickness.to(u.cm).value ])*u.cm + rShell_start.to(u.cm)
                # Step size
                rShell_step = np.max([
                    np.min([ 5e-5 * u.pc.to(u.cm), max_shellThickness.to(u.cm).value/1e3]),
                    max_shellThickness.to(u.cm).value/1e6
                    ]) * u.cm
                # range of r values
                rShell_arr = np.arange(rShell_start.to(u.cm).value, rShell_stop.value, rShell_step.value) * u.cm
                # Get arguments and parameters for integration:
                # neutral region
                is_ionised = False
                # initial values
                y0 = [nShell0.to(1/u.cm**3).value, tau0_neu]
                # constants
                cons = [Ln, Qi]
                # Run integration
                sol_ODE = scipy.integrate.odeint(get_shellODE.get_shellODE, y0, rShell_arr.value,
                                      args=(cons, f_cover, is_ionised),
                                      rtol=1e-3, hmin=1e-7)
                # solved for n(r) and tau(r)
                nShell_arr = sol_ODE[:,0] / u.cm**3
                tauShell_arr = sol_ODE[:,1]
                        
                # mass of spherical shell. Volume given by V = 4 pi r**2 * thickness
                mShell_arr = np.empty_like(rShell_arr.value) * u.g
                mShell_arr[0] = mShell0.to(u.g)
                # FIXME: Shouldnt we use mu_p?
                mShell_arr[1:] = (nShell_arr[1:] * warpfield_params.mu_n * 4 * np.pi * rShell_arr[1:]**2 * rShell_step).to(u.g)
                mShell_arr_cum = np.cumsum(mShell_arr)
                
                # =============================================================================
                # Again, find the index at which M(r = R) = Mshell.
                # If exists, then terminate the loop. Otherwise, repeat the loop
                # with new (a continuation of) set of steps and start/end values.
                # =============================================================================                
                massCondition = mShell_arr_cum >= mShell_end
                idx_array = np.nonzero(massCondition)[0]
                # If there is none, then take as last index
                if len(idx_array) == 0:
                    idx = len(rShell_arr) - 1
                else:
                    idx = idx_array[0]
                # Associated condition
                # True if any part of the array is true
                is_allMassSwept = any(massCondition) 
                
                # Store values into arrays representing profile in the ionised region. 
                mShell_arr_neu = np.concatenate(( mShell_arr_neu, mShell_arr[:idx-1]))
                mShell_arr_cum_neu = np.concatenate(( mShell_arr_cum_neu, mShell_arr_cum[:idx-1]))
                tauShell_arr_neu = np.concatenate(( tauShell_arr_neu, tauShell_arr[:idx-1]))
                nShell_arr_neu = np.concatenate(( nShell_arr_neu, nShell_arr[:idx-1]))
                rShell_arr_neu = np.concatenate(( rShell_arr_neu, rShell_arr[:idx-1]))
                
                # Reinitialise values for next integration
                nShell0 = nShell_arr[idx - 1]
                tau0_neu = tauShell_arr[idx - 1]
                mShell0 = mShell_arr[idx - 1]
                rShell_start = rShell_arr[idx - 1]
                
            # append the last few values that are otherwise missed in the while loop.
            mShell_arr_neu = np.append(mShell_arr_neu, mShell_arr[idx-1])
            mShell_arr_cum_neu = np.append(mShell_arr_cum_neu, mShell_arr_cum[idx-1])
            tauShell_arr_neu = np.append(tauShell_arr_neu, tauShell_arr[idx-1])
            nShell_arr_neu = np.append(nShell_arr_neu, nShell_arr[idx-1])
            rShell_arr_neu = np.append(rShell_arr_neu, rShell_arr[idx-1])
            
            # =============================================================================
            # Now, compute the gravitational potential for the neutral part of shell
            # =============================================================================
            # FIXME: Shouldnt we use mu_p?
            grav_neu_rho = nShell_arr_neu * warpfield_params.mu_n
            grav_neu_r = rShell_arr_neu
            # mass of the thin spherical shell
            grav_neu_m = grav_neu_rho * 4 * np.pi * grav_neu_r**2 * rShell_step
            # cumulative mass
            grav_neu_m_cum = np.cumsum(grav_neu_m) + grav_ion_m_cum[-1]
            # gravitational potential
            grav_neu_phi = - 4 * np.pi * c.G.cgs * (scipy.integrate.simps(grav_neu_r * grav_neu_rho, x = grav_neu_r)* u.cm**2 * u.g/u.cm**3)
            grav_phi = grav_neu_phi + grav_ion_phi
            # gravitational potential force per unit mass
            grav_neu_force_m = c.G.cgs * grav_neu_m_cum / grav_neu_r**2
            
            # Now, modify the array so that it matches the potential file.
            # I am not entirely sure what this section does, but it was in the
            # old code. 
            potentialFile_internalLength = 10000
            skip_idx = max(
                int(
                    len(grav_neu_r)/potentialFile_internalLength
                    ), 1
                )        
            # modify array
            grav_neu_r = grav_neu_r[:-1:skip_idx]
            grav_neu_force_m = grav_neu_force_m[:-1:skip_idx]
            # concatenate to an array which represents the whole shell
            grav_force_m = np.concatenate([grav_force_m, grav_neu_force_m])
            grav_r = np.concatenate([grav_r, grav_neu_r])
            
        #thickness of shell
        try:
            dRs = rShell_arr_neu[-1]/u.pc.to(u.cm) - rShell0
        except:
            dRs = rShell_arr_neu/u.pc.to(u.cm) - rShell0
        os.environ["ShTh"] = str(dRs)
        
        # =============================================================================
        # Shell is fully evaluated. Compute shell properties now.
        # =============================================================================
        if is_fullyIonised:
            # What is the final thickness of the shell?
            shellThickness = rShell_arr_ion[-1] - rShell0
            # What is tau and phi at the outer edge of the shell?
            tau_rEnd = tauShell_arr_ion[-1]
            phi_rEnd = phiShell_arr_ion[-1]
            # What is the maximum shell density?
            nShell_max = np.max(nShell_arr_ion)
            # The ratio tau_IR/kappa_IR  = \int rho dr
            # Integrating using left Riemann sums.
            # See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf page 45 Eq 9
            tau_kappa_IR = warpfield_params.mu_n * np.sum(nShell_arr_ion[:-1] * dr_ion_arr) 
            
        else:
            shellThickness = rShell_arr_neu[-1] - rShell0
            tau_rEnd = tauShell_arr_neu[-1] 
            phi_rEnd = 0
            nShell_max = np.max(nShell_arr_ion)
            dr_neu_arr = rShell_arr_neu[1:] - rShell_arr_neu[:-1]
            # FIXME: Shouldnt we use mu_p?
            tau_kappa_IR = warpfield_params.mu_n * (np.sum(nShell_arr_neu[:-1] * dr_neu_arr) + np.sum(nShell_arr_ion[:-1] * dr_ion_arr))
            
        # fraction of absorbed ionizing and non-ionizing radiations:
        f_absorbed_ion = 1 - phi_rEnd
        f_absorbed_neu = 1 - np.exp( - tau_rEnd )
        # total absorption fraction, defined as luminosity weighted average of 
        # f_absorbed_ion and f_absorbed_neu.
        # See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf page 47 Eq 22, 23, 24.
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln)/(Li + Ln)
        
        if warpfield_params.write_shell:

            # save shell structure as .txt file (radius, density, temperature)
            # only save Ndat entries (equally spaced in index, skip others)
            Ndat = 500
            Nskip_ion = int(max(1, len(rShell_arr_ion) / Ndat))
            Nskip_noion = int(max(1, len(rShell_arr_ion) / Ndat))
            TShell_arr_ion = warpfield_params.t_ion * np.ones(len(rShell_arr_ion)) 
            TShell_arr_neu = warpfield_params.t_neu * np.ones(len(rShell_arr_neu))

            if is_fullyIonised:
                r_save = np.append(rShell_arr_ion[0:-1:Nskip_ion], rShell_arr_ion[-1])
                n_save = np.append(nShell_arr_ion[0:-1:Nskip_ion], nShell_arr_ion[-1])
                T_save = np.append(TShell_arr_ion[0:-1:Nskip_ion], TShell_arr_ion[-1])

            else:
                r_save = np.append(np.append(rShell_arr_ion[0:-1:Nskip_ion], rShell_arr_ion[-1]), np.append(rShell_arr_neu[0:-1:Nskip_noion], rShell_arr_neu[-1]))
                n_save = np.append(np.append(nShell_arr_ion[0:-1:Nskip_ion], nShell_arr_ion[-1]), np.append(nShell_arr_neu[0:-1:Nskip_noion], nShell_arr_neu[-1]))
                T_save = np.append(np.append(TShell_arr_ion[0:-1:Nskip_ion], TShell_arr_ion[-1]), np.append(TShell_arr_neu[0:-1:Nskip_noion], TShell_arr_neu[-1]))

            sh_savedata = {"r (cm)": r_save, "n (cm-3)": n_save,
                            "T (K)": T_save}
            name_list = ["r (cm)", "n (cm-3)", "T (K)"]
            tab = Table(sh_savedata, names=name_list)
            outname = warpfield_params.out_dir + "shell/shell_structure.txt"
            formats = {'r (cm)': '%1.6e', 'n (cm-3)': '%1.4e', 'T (K)': '%1.4e'}
            tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)
            
            
    elif is_shellDissolved:
        f_absorbed_ion = 1.0
        f_absorbed_neu = 0.0
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln)/(Li + Ln)
        f_ionised_dust = np.nan
        is_fullyIonised = True
        shellThickness = np.nan
        nShell_max = warpfield_params.nISM.value * cvt.ndens_cgs2au
        tau_kappa_IR = 0
        grav_r = np.nan
        grav_phi = np.nan
        grav_force_m = np.nan
        
        print('(non) shell attributes')
        return f_absorbed_ion, f_absorbed_neu, f_absorbed, f_ionised_dust, is_fullyIonised, shellThickness, nShellInner, nShell_max, tau_kappa_IR, grav_r, grav_phi, grav_force_m
            
    print(type(f_absorbed_ion))
    print(f_absorbed_ion, f_absorbed_neu, f_absorbed, f_ionised_dust, is_fullyIonised, shellThickness, nShellInner, nShell_max, tau_kappa_IR,)
    return f_absorbed_ion, f_absorbed_neu, f_absorbed.value, f_ionised_dust.value, \
                is_fullyIonised, shellThickness.value * cvt.cm2pc, \
                nShellInner.value * cvt.ndens_cgs2au, nShell_max.value * cvt.ndens_cgs2au, \
                tau_kappa_IR.value * cvt.tau_cgs2au, grav_r.value * cvt.cm2pc, \
                grav_phi.value * cvt.gravPhi_cgs2au, grav_force_m.value * cvt.grav_force_m_cgs2au
        
    
# add these values too:
    
    
    














