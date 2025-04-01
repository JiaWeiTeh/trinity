#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:49:43 2023

@author: Jia Wei Teh

This script adds astropy units into the dictionary. 
"""

import astropy.units as u
import astropy.constants as c
# Well.. this is actually the modules name
# import fuckit

# decorator to avoid lengthy try blocks
# @fuckit









def append_units(path2file):
    
    with open(path2file, 'w') as f:

        lines = f.readlines()
        
        for l in lines:
            
            if l == 'log_mCloud =':
                l.append()
    
    
    return



def old_append_units(wp):
    """
        Takes in warpfield_params from get_param.py, add astropy units. 
    """
    
    wp.log_mCloud *= u.M_sun
    
    wp.nCore /= u.cm**3
    
    wp.rCore *= u.pc

    wp.rand_log_mCloud *= u.M_sun
    
    wp.rand_n_cloud /= u.cm**3

    wp.r_coll *= u.pc
    
    wp.SB99_mass *= u.M_sun  
    
    wp.SB99_BHCUT *= u.M_sun  
    
    wp.SB99_age_min *=  u.yr  
    
    wp.v_SN *= u.km/u.s
    
    wp.dens_navg_pL /= u.cm**3
      
    wp.stop_n_diss /= u.cm**3  
    
    wp.stop_t_diss *= u.Myr  
    
    wp.stop_r *= u.pc  
    
    wp.stop_v *= u.km/u.s
    
    if wp.stop_t_unit == 'Myr':
        wp.stop_t *= u.Myr
    
    wp.phase_Emin *= u.erg
    
    wp.sigma0 *= u.cm**2
    
    wp.mu_n *= u.g
    
    wp.mu_p *= u.g
    
    wp.t_ion *= u.K
    
    wp.t_neu *= u.K

    wp.nISM /= u.cm**3
    
    wp.kappa_IR *= u.cm**2 / u.g
    
    wp.alpha_B *= u.cm**3 / u.s
    
    wp.c_therm *= u.erg / u.cm / u.s * u.K**(-7/2)
    
    wp.sigma_d *= u.cm**2
    
    wp.tff *= u.Myr
    
    wp.mCloud *= u.M_sun

    wp.mCluster *= u.M_sun
    
    wp.T_r2Prime *= u.K
    
    # TODO
    # Finally, aadd a method to double check all parameters have astrophy attribute
    # also compare to length in .yaml to make sure all is done. 
    
    return wp

