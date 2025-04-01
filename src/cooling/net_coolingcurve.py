#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:08:23 2023

@author: Jia Wei Teh

This is a master script which craetes a NET cooling rate (dudt) curve containing both CIE and non-CIE
conditions.

old code: coolnoeq.cool_interp_master()
"""
import scipy.interpolate
import numpy as np
import astropy.units as u
import sys

import src.cooling.CIE.read_coolingcurve as CIE
# get_Lambda
import src._functions.unit_conversions as cvt


def get_dudt(age, ndens, T, phi, params_dict):
    """
    Calculates dudt in cgs, but input and ouput in au. 

    Parameters (input)
    ----------
    age [Myr]: TYPE
    ndens [1/pc3]: TYPE
    T [K]: TYPE
    phi [1/pc2/Myr]: TYPE

    Returns
    -------
    dudt is [M_sun/pc/yr3] (erg/cm3/s), because cooling is in units of (erg cm3/s) [M_sun*pc5/s3] 
    TYPE
        DESCRIPTION.

    """
    
    # These value should not be logged!
    # double checking units
    # ndens = ndens * (1/u.cm**3)
    # phi = phi * (1/u.cm**2/u.s)
    ndens /= cvt.ndens_cgs2au #(pc-3 to cm-3)
    phi /= cvt.phi_cgs2au #(1/pc2/yr to 1/cm2/s)
    
    # print('ndens', ndens)
    # print('phi', phi)
    
    # New idea, since non-CIE curve is only up to 10^5.5K, which is exactly
    # what our threshold is, we create an if/else function that returns 
    # Lambda(T)_CIE if above, and Lambda(n,T,phi) if below. 
    # If between max of nonCIE and min of CIE, take interpolation between two Lambda values.
    
    
    # In order to improve speed, here we use dictionary. This means that the age will not be 
    # as accurate, since the cooling structure only updates every once in a while or so.
    # E.g., lets say 5e4 years according to run_implicit_energy.py.
    
    
    cooling_nonCIE = params_dict['cStruc_cooling_nonCIE'].value
    # heating_nonCIE = params_dict['cStruc_heating_nonCIE'].value
    netcool_interp = params_dict['cStruc_net_nonCIE_interpolation'].value
    
    CIE_interp = params_dict['cStruc_cooling_CIE_interpolation'].value
    logT_CIE = params_dict['cStruc_cooling_CIE_logT'].value
    
    # import values from two cooling curves
    # _timer.begin()
    # depreciated
    # cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(age)
    # print(cooling_nonCIE)
    # _timer.end()
    Lambda_CIE = CIE.get_Lambda(T, CIE_interp) # Lambda is returned in units of erg/s * cm3
    
    # we take the cutoff at 10e5.5 K. 
    # These are all in log-space. 
    # cutoff at which temperature above switches to CIE file:
    nonCIE_Tcutoff = max(cooling_nonCIE.temp[cooling_nonCIE.temp <= 5.5])
    # cutoff at which temperature below switches to non-CIE file:
    CIE_Tcutoff = min(logT_CIE[logT_CIE > 5.5])
    # output
    # print(f'{cpr.WARN}Taking net-cooling curve from non-CIE condition at T <= {nonCIE_Tcutoff}K and CIE condition at T >= {CIE_Tcutoff}K.{cpr.END}')
    # if nonCIE_Tcutoff != CIE_Tcutoff:
        # print(f'{cpr.WARN}Net cooling for temperature values in-between will be interpolated{cpr.END}.')

    # if temperature is lower than the non-CIE temperature, use non-CIE
    if np.log10(T) <= nonCIE_Tcutoff and np.log10(T) >= min(cooling_nonCIE.temp):
        # print(f'{cpr.WARN}Entering non-CIE regime...{cpr.END}')
        # All this does here is to interpolate for values of Lambda based on
        # T, dens and phi.
        
        # netcooling grid (depreciated)
        # netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube
        # create interpolation function (depreciated)
        # f_dudt = scipy.interpolate.RegularGridInterpolator((cooling_nonCIE.ndens, cooling_nonCIE.temp, cooling_nonCIE.phi), netcooling)
        # get net cooling rate
        # remember that these have to be logged!
        # print(cooling_nonCIE.ndens)
        # print(cooling_nonCIE.temp)
        # print(cooling_nonCIE.phi)
        # print(netcooling)
        # print(ndens, T, phi)
        dudt = netcool_interp([np.log10(ndens), np.log10(T), np.log10(phi)])[0] #* u.erg / u.cm**3 / u.s
        # return in negative sign for convension (since the rate of change is negative due to net cooling)
        return -1 * dudt * cvt.dudt_cgs2au
        
    # if temperature is higher than the CIE curve, use CIE.
    elif np.log10(T) >= CIE_Tcutoff:
        # print(f'{cpr.WARN}Entering CIE regime...{cpr.END}')
        # get CIE cooling rate
        dudt = ndens**2 * Lambda_CIE
        return -1 * dudt * cvt.dudt_cgs2au
        
    # if temperature is between, do interpolation
    elif (np.log10(T) > nonCIE_Tcutoff) and (np.log10(T) < CIE_Tcutoff):
        # print(f'{cpr.WARN}Entering interpolation regime...{cpr.END}')
        # =============================================================================
        # This part is just for non-CIE, and slight-modification from above
        # Get the maximum point of non-CIE. 
        # =============================================================================
        # netcooling grid (depreciated)
        # netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube
        # create interpolation function (depreciated)
        # f_dudt = scipy.interpolate.RegularGridInterpolator((cooling_nonCIE.ndens, cooling_nonCIE.temp, cooling_nonCIE.phi), netcooling)
        # get net cooling rate
        dudt_nonCIE = netcool_interp([np.log10(ndens), nonCIE_Tcutoff, np.log10(phi)])[0] #* u.erg / u.cm**3 / u.s
        
        # =============================================================================
        # This part is just for CIE
        # =============================================================================
    
        # # get CIE cooling rate
        Lambda = CIE.get_Lambda(10**CIE_Tcutoff, CIE_interp)
        dudt_CIE = (ndens**2 * Lambda)#.to(u.erg / u.cm**3 / u.s)
        
        # =============================================================================
        # Do interpolation now
        # =============================================================================
        
        # print(np.log10(T), [nonCIE_Tcutoff, CIE_Tcutoff],[dudt_nonCIE, dudt_CIE])
        dudt = np.interp(np.log10(T), [nonCIE_Tcutoff, CIE_Tcutoff],[dudt_nonCIE, dudt_CIE])
    
        return -1 * dudt * cvt.dudt_cgs2au
    
    
    # if temperature is lower than the available non-CIE curve, error (or better, provide some interpolation in the future?)
    else:
        raise Exception(f'Temperature T = {T} not understood. Cooling curve and dudt cannot be computed.')
        
    
    
