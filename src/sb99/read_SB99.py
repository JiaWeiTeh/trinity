#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:06:39 2023

@author: Jia Wei Teh

This script contains functions that will help reading in Starburst99 data.

"""

import numpy as np
import scipy
import sys

import src._functions.unit_conversions as cvt

# TODO: Implement interpolation function for in-between metallicities/cluster 
    # : Add fmet, where metallicity scaling due to non-existent SB99 file

def read_SB99(f_mass, params):
    """
    This function retrieves data from the Starburst99 files. 
    
    Returns
    ----------
    Here are the parameters directly from Starburst99 runs:
    
    t: time [yr]; however, saved as [Myr] in the output of this function.
    
    Qi: emission rate of ionizing photons log[1/s] (au)
    
    fi: fraction of ionising radiation
    
    Lbol: bolometric luminosity [erg/s] (au)
    
    Lmech: mechanical luminosity (Winds + SNe) [erg/s] (au)
    
    pdot_W: momtntum rate (Winds) [g/cm/s2] (au)
    
    Lmech_W: mechanical luminosity (Winds) [erg/s] (au)

    Returns
    -------
    In addition, calculate and include parameters that will be useful for the run:
        
    Li: luminosity in the ionizing part of the spectrum (>13.6 eV)
    
    Ln: luminosity in the non-ionizing part of the spectrum (<13.6 eV)
    
    """
    
    # =============================================================================
    # Step1: find and read the SB99 file. 
    # =============================================================================
    # grab the file name based on simulation input
    filename = get_filename(params)
    path2sps = params['path_sps'].value
    # read file
    # SB99_file = np.loadtxt(warpfield_params.path_sps + filename)
    SB99_file = np.loadtxt(path2sps + filename)
    # read columns
    # change to in Myr instead
    # u.Myr
    t = SB99_file[:,0] /1e6 
    # the rest, translate to linear, then scale with actual cluster mass
    # / u.s
    Qi = 10**SB99_file[:,1] * f_mass / cvt.s2Myr
    fi = 10**SB99_file[:,2]
    # u.erg/u.s
    Lbol = 10**SB99_file[:,3] * f_mass * cvt.L_cgs2au
    # u.erg/u.s
    Lmech = 10**SB99_file[:,4] * f_mass * cvt.L_cgs2au
    # u.g * u.cm/(u.s**2)
    pdot_W = 10**SB99_file[:,5] * f_mass * cvt.pdot_cgs2au
    # u.erg/u.s
    Lmech_W = 10**SB99_file[:,6] * f_mass * cvt.L_cgs2au

    # =============================================================================
    # Step2: calculate other derived values
    # =============================================================================
    # Ionising and non-ionising luminosity (13.5 eV)
    Li = Lbol * fi
    Ln = Lbol * (1-fi)
    # mechanical luminosity (SNe)
    Lmech_SN = Lmech - Lmech_W
    
    # =============================================================================
    # Scale values for WIND mass loss rate and terminal velocity (g/s, m/s)
    # thus consequently the mechanical luminosity and momentum injection rate.
    # =============================================================================
    # first break down into mass loss and velocity
    Mdot_W = pdot_W ** 2 / (2 * Lmech_W)
    velocity_W = 2 * Lmech_W / pdot_W
    # Add fraction of mass injected into the cloud due to sweeping of cold material
    # from protostars and disks inside star clusters?
    Mdot_W *= (1 + params['f_Mcold_wind'].value)
    # Modifiy terminal velocity according to 
    # 1) thermal efficiency and 2) cold mass content in cluster?
    velocity_W *= np.sqrt(params['thermcoeff_wind'].value / (1. + params['f_Mcold_wind'].value)) 
    # convert back
    pdot_W = Mdot_W * velocity_W
    Lmech_W = 0.5 * Mdot_W * velocity_W**2
    
    # =============================================================================
    # Scale values for SN mass loss rate and terminal velocity (g/s, m/s)
    # thus consequently the mechanical luminosity and momentum injection rate.
    # =============================================================================
    # first break down into mass loss and velocity
    # TODO: get time-dependent velocity, e.g. when mass of ejecta are known
    # convert to cgs
    velocity_SN = params['v_SN'].value
    Mdot_SN = 2 * Lmech_SN / velocity_SN**2
    # Add fraction of mass injected into the cloud due to sweeping of cold material
    # from protostars and disks inside star clusters?
    Mdot_SN *= (1 + params['f_Mcold_SN'].value)
    # Modifiy terminal velocity according to 
    # 1) thermal efficiency and 2) cold mass content in cluster?
    velocity_SN *= np.sqrt(params['thermcoeff_SN'].value / (1. + params['f_Mcold_SN'].value)) 
    # convert back
    pdot_SN = Mdot_SN * velocity_SN
    Lmech_SN = 0.5 * Mdot_SN * velocity_SN**2
    # =============================================================================
    # Final touchups
    # =============================================================================
    # total energy and momentum injection rate
    Lmech = Lmech_SN + Lmech_W
    pdot = pdot_SN + pdot_W
    
    # insert 1 element at t=0 for interpolation purposes
    t = np.insert(t, 0, 0.0)
    Qi = np.insert(Qi, 0, Qi[0])
    Li = np.insert(Li, 0, Li[0])
    Ln = np.insert(Ln, 0, Ln[0])
    Lbol = np.insert(Lbol, 0, Lbol[0])
    Lmech= np.insert(Lmech, 0, Lmech[0])
    pdot = np.insert(pdot, 0, pdot[0])
    pdot_SN = np.insert(pdot_SN, 0, pdot_SN[0])
    
    return [t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN]
    

def get_filename(params):
    """
    Creates filename (str) based on simulation parameters
    """
    
    # All filenames have convention of [mass]cluster_[rotation]_[metallicity]_[blackholeCutoffMass].txt
    # Right now, only solar metallicity, 1e6, BH120, and rotation is considered. 
    try:
        # cluster mass in SB99 run?
        # turn float into simple string. e.g., 1000000 -> 1e6
        def format_e(n):
            a = '%E' % n
            return a.split('E')[0].rstrip('0').rstrip('.') + 'e' + a.split('E')[1].strip('+').strip('0')
        SBmass_str = format_e(params['SB99_mass'].value)
        # with rotation?
        if params['SB99_rotation'].value == True:
            rot_str = 'rot'
        else:
            rot_str = 'norot'
        # what metallicity?
        if float(params['metallicity'].value) == 1.0:
            # solar
            z_str = 'Z0014'
        elif float(params['metallicity'].value) == 0.15:
            # 0.15 solar
            z_str = 'Z0002'
        # what blackhole cutoff mass?
        if int(params['SB99_BHCUT'].value) == 120:
            # solar
            BH_str = 'BH120'
        elif int(params['SB99_BHCUT'].value) == 40:
            # 0.15 solar
            BH_str = 'BH40'            
            
        filename = SBmass_str + 'cluster_' + rot_str + '_' + z_str + '_' + BH_str + '.txt'
        return filename
    except:
        raise Exception(f"Starburst99 file {filename} not found. Make sure to double check parameters in the 'parameters for Starburst99 operations' section.")



def get_interpolation(SB99, ftype = 'cubic'):
    """
    This function creates interpolation function for further use. 

    Parameters
    ----------
    SB99 : array
        Data array of SB99.
    ftype : str, optional
        The default is 'cubic'. Fed into scipy.interpolate.interp1d. Accept also 
        values like 'linear'. Cubic is important for small-value interpolations.

    Returns
    -------
    SB99f : dict
        A dictionary of interpolation functions for SB99 data.

    """
    # Old code: make_interpfunc()
    
    # obtain all SB99 values
    [t_Myr, Qi, Li, Ln, Lbol, Lw, pdot, pdot_SNe] = SB99
    # get interpolation functions
    fQi = scipy.interpolate.interp1d(t_Myr, Qi, kind = ftype) 
    fLi = scipy.interpolate.interp1d(t_Myr, Li, kind = ftype)
    fLn = scipy.interpolate.interp1d(t_Myr, Ln, kind = ftype)
    fLbol = scipy.interpolate.interp1d(t_Myr, Lbol, kind = ftype)
    fLw = scipy.interpolate.interp1d(t_Myr, Lw, kind = ftype)
    fpdot = scipy.interpolate.interp1d(t_Myr, pdot, kind = ftype)
    fpdot_SNe = scipy.interpolate.interp1d(t_Myr, pdot_SNe, kind = ftype)

    SB99f = {'fQi': fQi, 'fLi': fLi, 'fLn': fLn, 'fLbol': fLbol, 'fLw': fLw,
              'fpdot': fpdot, 'fpdot_SNe': fpdot_SNe}
    
    return SB99f









