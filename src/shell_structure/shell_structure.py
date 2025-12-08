#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 20:53:03 2022

@author: Jia Wei Teh

This script contains a function that evaluates the shell structure.
"""
# libraries
import numpy as np
import scipy.integrate 
import os
import sys
from astropy.table import Table
import src._functions.unit_conversions as cvt

#--
from src.shell_structure import get_shellODE, get_shellParams


# SHELL THICKNESS IS NEGATIVE???

from numba import jit

# @jit
def shell_structure(params):
    
    
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
    
    # should be safe because its not being altered here
    mBubble = params['bubble_mass'].value
    pBubble = params['Pb'].value
    rShell0 = params['R2'].value
    mShell_end = params['shell_mass'].value
    Qi = params['Qi'].value
    Li = params['Li'].value
    Ln = params['Ln'].value
    
    # TODO: Add also f_cover. from fragmentation mechanics.
    f_cover = 1 
    
    # TODO: Check also neutral region.
    
    # initialise values at r = rShell0 = inner edge of shell
    rShell_start = rShell0
    # attenuation function for ionising flux. Unitless.
    phi0 = 1 
    # tau(r) at ionised region?
    tau0_ion = 0
    
    mShell0 = 0 
    mShell_end = mShell_end
    
    # Obtain density at the inner edge of shell
    nShell0 = get_shellParams.get_nShell0(params)
    # define for future use, as nShell0 constantly changes in the loop.
    # nShellInner = nShell0
    
    
    # =============================================================================
    # Before beginning the integration, initialise some logic gates.
    # =============================================================================
    # 1. Have we accounted all the shell mass in our integration? 
    # I.e., is cumulative_mass(r = R) >= mShell?
    is_allMassSwept = False
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







    # =============================================================================
    # name conventions:
        # rshell_stop = the end of local radius integration
        # maxshellRadius = the global end of shell radius (R2+stromgren)
        # slice = subsection of a shell
        # nsteps = steps in a slice when integrating
    # =============================================================================


    # Assuming constant density, what is the maximum possible shell thickness.
    
    # this is just to set a radiusmax for integration
    
    # max_shellRadius = r_stromgren + rShell0
    max_shellRadius = (3 * Qi / (4 * np.pi * params['caseB_alpha'].value * nShell0**2))**(1/3) + rShell_start
    
    
    # because of the nature of stiff ODE, we need to break down shell into 
    # slices, and for each slices we have 1e3 steps.
    # the slice should also have minimum of 1pc. 
    
    # number of steps in a slice
    nsteps = 1e3
    
    sliceSize = np.min([1, (max_shellRadius - rShell_start)/10])
    
    rShell_step = sliceSize/nsteps
    
    #--
    
    print(f'slizesize {sliceSize}')
    print(f'max_shellRadius {max_shellRadius}')
    print(f'rShell_start {rShell_start}')
    print(f'shellthickness {max_shellRadius - rShell_start}')
    print(f'rShell_step {rShell_step}')
    
    
    while not is_allMassSwept and not is_fullyIonised:
        
        print('1-- not is_allMassSwept and not is_fullyIonised')
        
        # =============================================================================
        # Define the range at which integration occurs.
        # This is necessary because, unfortunately, the ODE is very stiff and 
        # requires small steps of r. 
        # This loop, we deal with situations where not all masses are swept into 
        # the shell, and at this rStep range not all ionisation is being used up (phi !=0).
        # =============================================================================
        
        # stop value of slice        
        rShell_stop = rShell_start + sliceSize
        # define radius of slice
        rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step) 
        
        # Get arguments and parameters for integration:
        # ionised region    
        is_ionised = True
        
        #--- solved for n(r), phi(r), and tau(r)
        y0 = [nShell0, phi0, tau0_ion]
        sol_ODE = scipy.integrate.odeint(get_shellODE.get_shellODE, y0, rShell_arr,
                              args=(f_cover, is_ionised, params),
                              # rtol=1e-3, hmin=1e-7
                              )
        nShell_arr = sol_ODE[:,0] 
        phiShell_arr = sol_ODE[:,1] 
        tauShell_arr = sol_ODE[:,2]
                
        # mass of spherical shell. Volume given by V = 4 pi r**2 * thickness
        mShell_arr = np.empty_like(rShell_arr)
        mShell_arr[0] = mShell0
        mShell_arr[1:] = (nShell_arr[1:] * params['mu_ion'].value * 4 * np.pi * rShell_arr[1:]**2 * rShell_step)
        mShell_arr_cum = np.cumsum(mShell_arr)
        
        # =============================================================================
        # Now, find the index at which M(r = R) = mShell, or phi(r = R) = 0 
        # If exists, then terminate the loop. Otherwise, repeat the loop
        # with new (a continuation of) sets of steps and start/end values.
        # =============================================================================
        massCondition = mShell_arr_cum >= mShell_end
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
        # all mass if swept up if any part of the radius has mass = mass shell
        is_allMassSwept = any(massCondition) 
        # the shell is fully ionised if any part of the radius has fraction 0.
        is_fullyIonised = any(phiCondition)
        
        # Store values into arrays representing profile in the ionised region. 
        mShell_arr_ion = np.concatenate(( mShell_arr_ion, mShell_arr[:idx]))
        mShell_arr_cum_ion = np.concatenate(( mShell_arr_cum_ion, mShell_arr_cum[:idx]))
        phiShell_arr_ion = np.concatenate(( phiShell_arr_ion, phiShell_arr[:idx]))
        tauShell_arr_ion = np.concatenate(( tauShell_arr_ion, tauShell_arr[:idx]))
        nShell_arr_ion = np.concatenate(( nShell_arr_ion, nShell_arr[:idx]))
        rShell_arr_ion = np.concatenate(( rShell_arr_ion, rShell_arr[:idx]))

        # Reinitialise values for next integration
        nShell0 = nShell_arr[idx]
        phi0 = phiShell_arr[idx]
        tau0_ion = tauShell_arr[idx]
        mShell0 = mShell_arr_cum[idx]
        rShell_start = rShell_arr[idx]
        
        # IDEA: here we remove the condition and put it to energy events
        # TODO: make sure this works
        
        # ---------------------
        # # Consider the shell dissolved if the followings occur:
        # # 1. The density of shell is far lower than the density of ISm.
        # # 2. The shell has expanded too far.
        # # TODO: output message to tertminal depending on verbosity
    
        # if nShellInner < (0.001 * params['nISM_au'].value) or\
        #     rShell_stop == (1.2 * params['stop_r'].value) or\
        #         (rShell_start - rShell_stop) > (10 * rShell_start):
        #             is_shellDissolved = True
        #             break
        
        # begin next iteration if shell is not all ionised and mshell is not all accounted for.
        # if either condition is not met, move on.
        # ---------------------
        
        # right now the shell is considered dissolved.
        if nShell_arr[0] < params['stop_n_diss']:
            is_shellDissolved = True


    # append the last few values that are otherwise missed in the while loop.
    mShell_arr_ion = np.append(mShell_arr_ion, mShell_arr[idx])
    mShell_arr_cum_ion = np.append(mShell_arr_cum_ion, mShell_arr_cum[idx])
    phiShell_arr_ion = np.append(phiShell_arr_ion, phiShell_arr[idx])
    tauShell_arr_ion = np.append(tauShell_arr_ion, tauShell_arr[idx])
    nShell_arr_ion = np.append(nShell_arr_ion, nShell_arr[idx])
    rShell_arr_ion = np.append(rShell_arr_ion, rShell_arr[idx])
    
    
    # =============================================================================
    # If shell hasn't dissolved, continue some computation to prepare for 
    # further evaulation.
    # =============================================================================
    if not is_shellDissolved:
        
        print('2-- not is_shellDissolved')
    
        # =============================================================================
        # First, compute the gravitational potential for the ionised part of shell
        # =============================================================================
        
        grav_ion_rho = (nShell_arr_ion * params['mu_ion'].value)
        grav_ion_r = rShell_arr_ion
        # mass of the thin spherical shell
        grav_ion_m = grav_ion_rho * 4 * np.pi * grav_ion_r**2 * rShell_step
        # cumulative mass
        grav_ion_m_cum = np.cumsum(grav_ion_m) + mBubble
        # gravitational potential
        grav_ion_phi = - 4 * np.pi * params['G'].value * scipy.integrate.simps(grav_ion_r * grav_ion_rho, x = grav_ion_r)
        # mark for future use
        grav_phi = grav_ion_phi
        # gravitational potential force per unit mass
        grav_ion_force_m = params['G'].value * grav_ion_m_cum / grav_ion_r**2
        
        
        # For the whole shell; but for now we have only calculated the ionised part.
        grav_force_m = grav_ion_force_m
        grav_r = grav_ion_r
        
        # How much ionising radiation is absorbed by dust and how much by hydrogen?
        dr_ion_arr = rShell_arr_ion[1:] - rShell_arr_ion[:-1]
        # We do so by integrating using left Riemann sums 
        # dust term in dphi/dr
        phi_dust = np.sum(
                        - nShell_arr_ion[:-1] * params['dust_sigma'].value * phiShell_arr_ion[:-1] * dr_ion_arr
                        )
        # recombination term in dphi/dr
        phi_hydrogen = np.sum(
                        - 4 * np.pi * rShell_arr_ion[:-1]**2 / Qi * params['caseB_alpha'].value * nShell_arr_ion[:-1]**2 * dr_ion_arr
                        )
        
        # If there is no ionised shell (e.g., because the ionising radiation is too weak)
        if (phi_dust + phi_hydrogen) == 0.0:
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
        
        print('2-- ready to go into 3--')
        
        # =============================================================================
        # If the shell is not fully ionised, calculate structure of 
        # non-ionized (neutral) part
        # =============================================================================
        if not is_fullyIonised:
            
            print('3-- not is_fullyIonised')
            
            
            # Pressure equilibrium dictates that there will be a temperature and density
            # discontinuity at boundary between ionised and neutral region.
            nShell0 = nShell0 * params['mu_neu'].value / params['mu_ion'].value * params['TShell_ion'].value / params['TShell_neu'].value
            # tau(r) at neutral shell region
            tau0_neu = tau0_ion
            
            # =============================================================================
            # Evaluate the remaining neutral shell until all masses are accounted for.
            # =============================================================================
            # Entering this loop means not all mShell has been accounted for. Thus
            # is_phiZero can either be True or False here.
            
            # # if tau is already 100, there is no point in integrating more.
            tau_max = 100
            
            
            # # --- OLD
            # mydr = np.min([ 1, np.abs((tau_max - tau0_ion)/(nShell0 * params['sigma_d_au'].value))])
            

            # rShell_step = np.max([
            #         np.min([ 5e-5, mydr/1e3]),
            #         mydr/1e4
            #         ]) 
            # # ---
            
                
            # new step size
            # TODO: idea: do a mesh up of logspace and reverse logspace. This way
            # it contains high dense points for both ends
            nsteps = 5e3
            
            # # step size for the shell
            # added a factor of 10 here so that the loop does not exceed 10 times. 
            # speed up for unnecessary calculations
            
            # maybe instead of a tau condition, we use instead a mass condtion
            # option 1---
            # derived from dtau/dr = n*sigma where r = 0 , tau = 0
            # sliceSize = np.min([ 1, np.abs((tau_max - tau0_ion)/(nShell0 * params['sigma_d_au'].value))/10])
            # option 2---
            sliceSize = np.min([1, (max_shellRadius - rShell_start)/10])            
            
            rShell_step = sliceSize/nsteps
            
            
            while not is_allMassSwept:
                
                print('4-- not is_allMassSwept')
                
                # there seem to be an update issue
    
                # ---
                # # if tau is already 100, there is no point in integrating more.
                # tau_max = 100
                # # the maximum width of the neutral shell, assuming constant density.
                # max_shellRadius = np.abs((tau_max - tau0_ion)/(nShell0 * params['sigma_d_au'].value))
                # # the end range of integration 
                # rShell_stop = np.min([ 1 * u.pc.to(u.cm), max_shellRadius.to(u.cm).value ])*u.cm + rShell_start.to(u.cm)
                # # Step size
                # rShell_step = np.max([
                #     np.min([ 5e-5 * u.pc.to(u.cm), max_shellRadius.to(u.cm).value/1e3]),
                #     max_shellRadius.to(u.cm).value/1e6
                #     ]) * u.cm
                # # range of r values
                # rShell_arr = np.arange(rShell_start.to(u.cm).value, rShell_stop.value, rShell_step.value) * u.cm
                # # Get arguments and parameters for integration:
                # # neutral region
                # is_ionised = False
                
                # ---
                # # if tau is already 100, there is no point in integrating more.
                # tau_max = 100
                # # the maximum width of the neutral shell, assuming constant density.
                # mydr = np.abs((tau_max - tau0_ion)/(nShell0 * params['sigma_d_au'].value))
                # the end range of integration 
                rShell_stop = rShell_start + sliceSize
                
                # # Step size
                # rShell_step = np.max([
                #     np.min([ 5e-5, max_shellRadius/1e3]),
                #     max_shellRadius/1e4
                #     ]) 
                # range of r values
                rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)
                # Get arguments and parameters for integration:
                # neutral region
                is_ionised = False
                
                print('this is how long the shell array is in the second loop: ', len(rShell_arr), ', and here is the stepsize', rShell_step,\
                      'and the shell thickness', max_shellRadius - rShell_start, 'and the slicesize', sliceSize)
                
                    
                y0 = [nShell0, tau0_neu]
                sol_ODE = scipy.integrate.odeint(get_shellODE.get_shellODE, y0, rShell_arr,
                                      args=(f_cover, is_ionised, params),
                                      # rtol=1e-3, hmin=1e-7
                                      )
                # solved for n(r), phi(r), and tau(r)
                nShell_arr = sol_ODE[:,0]
                tauShell_arr = sol_ODE[:,1]
                
                        
                # mass of spherical shell. Volume given by V = 4 pi r**2 * thickness
                mShell_arr = np.empty_like(rShell_arr)
                mShell_arr[0] = mShell0
                # FIXME: Shouldnt we use mu_p?
                mShell_arr[1:] = (nShell_arr[1:] * params['mu_neu'].value * 4 * np.pi * rShell_arr[1:]**2 * rShell_step)
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
                    
                    
                print('mass condition:', idx, mShell_arr_cum[idx], mShell_end)
                
                        
                # ----NEW----
                # Associated condition
                # True if any part of the array is true
                is_allMassSwept = any(massCondition) 
                
                # Store values into arrays representing profile in the ionised region. 
                mShell_arr_neu = np.concatenate(( mShell_arr_neu, mShell_arr[:idx]))
                mShell_arr_cum_neu = np.concatenate(( mShell_arr_cum_neu, mShell_arr_cum[:idx]))
                tauShell_arr_neu = np.concatenate(( tauShell_arr_neu, tauShell_arr[:idx]))
                nShell_arr_neu = np.concatenate(( nShell_arr_neu, nShell_arr[:idx]))
                rShell_arr_neu = np.concatenate(( rShell_arr_neu, rShell_arr[:idx]))
                
                # Reinitialise values for next integration
                nShell0 = nShell_arr[idx]
                tau0_neu = tauShell_arr[idx]
                mShell0 = mShell_arr[idx]
                rShell_start = rShell_arr[idx]
                
            # append the last few values that are otherwise missed in the while loop.
            mShell_arr_neu = np.append(mShell_arr_neu, mShell_arr[idx])
            mShell_arr_cum_neu = np.append(mShell_arr_cum_neu, mShell_arr_cum[idx])
            tauShell_arr_neu = np.append(tauShell_arr_neu, tauShell_arr[idx])
            nShell_arr_neu = np.append(nShell_arr_neu, nShell_arr[idx])
            rShell_arr_neu = np.append(rShell_arr_neu, rShell_arr[idx])
            
            # =============================================================================
            # Now, compute the gravitational potential for the neutral part of shell
            # =============================================================================
            # FIXME: Shouldnt we use mu_p?
            grav_neu_rho = nShell_arr_neu * params['mu_neu'].value
            grav_neu_r = rShell_arr_neu
            # mass of the thin spherical shell
            grav_neu_m = grav_neu_rho * 4 * np.pi * grav_neu_r**2 * rShell_step
            # cumulative mass
            grav_neu_m_cum = np.cumsum(grav_neu_m) + grav_ion_m_cum[-1]
            # gravitational potential
            grav_neu_phi = - 4 * np.pi * params['G'].value * (scipy.integrate.simps(grav_neu_r * grav_neu_rho, x = grav_neu_r))
            grav_phi = grav_neu_phi + grav_ion_phi
            # gravitational potential force per unit mass
            grav_neu_force_m = params['G'].value * grav_neu_m_cum / grav_neu_r**2
            
            
            # concatenate to an array which represents the whole shell
            grav_force_m = np.concatenate([grav_force_m, grav_neu_force_m])
            grav_r = np.concatenate([grav_r, grav_neu_r])
            
        
        # =============================================================================
        # Shell is fully evaluated. Compute shell properties now.
        # =============================================================================
        if is_fullyIonised:
            # What is the actual thickness of the shell?
            shellThickness = rShell_arr_ion[-1] - rShell0
            # What is tau and phi at the outer edge of the shell?
            tau_rEnd = tauShell_arr_ion[-1]
            phi_rEnd = phiShell_arr_ion[-1]
            # What is the maximum shell density?
            nShell_max = np.max(nShell_arr_ion)
            # The ratio tau_IR/kappa_IR  = \int rho dr
            # Integrating using left Riemann sums.
            # See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf page 45 Eq 9
            tau_kappa_IR = params['mu_ion'].value * np.sum(nShell_arr_ion[:-1] * dr_ion_arr) 
            
        else:
            shellThickness = rShell_arr_neu[-1] - rShell0
            tau_rEnd = tauShell_arr_neu[-1] 
            phi_rEnd = 0
            nShell_max = np.max(nShell_arr_ion)
            dr_neu_arr = rShell_arr_neu[1:] - rShell_arr_neu[:-1]
            # FIXME: Shouldnt we use mu_p?
            tau_kappa_IR = params['mu_neu'].value * (np.sum(nShell_arr_neu[:-1] * dr_neu_arr) + np.sum(nShell_arr_ion[:-1] * dr_ion_arr))
            
        # fraction of absorbed ionizing and non-ionizing radiations:
        f_absorbed_ion = 1 - phi_rEnd
        f_absorbed_neu = 1 - np.exp( - tau_rEnd )
        # total absorption fraction, defined as luminosity weighted average of 
        # f_absorbed_ion and f_absorbed_neu.
        # See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf page 47 Eq 22, 23, 24.
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln)/(Li + Ln)
        
        
            
    elif is_shellDissolved:
        params['isDissolved'].value = True
        # f_absorbed_ion = 1.0
        # f_absorbed_neu = 0.0
        # f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln)/(Li + Ln)
        # f_ionised_dust = np.nan
        # is_fullyIonised = True
        # shellThickness = np.nan
        # nShell_max = params['nISM'].value
        # tau_kappa_IR = 0
        # grav_r = np.nan
        # grav_phi = np.nan
        # grav_force_m = np.nan
        
        print('Shell dissolved.')
        
        
    # finally, record
    params['shell_fAbsorbedIon'].value = f_absorbed_ion
    params['shell_fAbsorbedNeu'].value = f_absorbed_neu
    params['shell_fAbsorbedWeightedTotal'].value = f_absorbed
    params['shell_fIonisedDust'].value = f_ionised_dust
    # old: seems to not include indirect radiation
    # params['shell_fRad'].value = f_absorbed_ion * params['Lbol'].value / params['c_light'].value 
    # new: has radiation
    params['shell_fRad'].value = f_absorbed_ion * params['Lbol'].value / params['c_light'].value * (1 + params['shell_tauKappaRatio'] * params['dust_KappaIR'])

    params['shell_thickness'].value = shellThickness 
    # params['shell_nShellInner'].value = nShellInner
    params['shell_nMax'].value = nShell_max
    params['shell_tauKappaRatio'].value = tau_kappa_IR
    
    params['shell_grav_r'].value = grav_r 
    params['shell_grav_phi'].value = grav_phi
    params['shell_grav_force_m'].value = grav_force_m
    
    params['rShell'].value = grav_r[-1]
        
    return
        
    













