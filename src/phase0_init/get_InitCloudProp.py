# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Sun Jul 24 23:42:14 2022

@author: Jia Wei Teh

This script contains a function that returns initial radius and edge density of the cloud.
"""
import numpy as np
import sys
import src.cloud_properties.bonnorEbertSphere as bonnorEbertSphere
import src.cloud_properties.powerLawSphere as powerLawSphere
import src.cloud_properties.density_profile as density_profile
import src.cloud_properties.mass_profile as mass_profile
import src._functions.unit_conversions as cvt
#--



def get_InitCloudProp(params):
    
    # get cloud radius and number density at cloud radius
    # get initial density profile
    
    
    if params['dens_profile'].value == 'densBE':
        _, rCloud, nEdge, _ = bonnorEbertSphere.create_BESphere(params)
        
    
    elif params['dens_profile'].value == 'densPL':
        rCloud, nEdge = powerLawSphere.create_PLSphere(params)
        
    # logspace radius array
    r_arr_inCloud = np.logspace(-3, np.log10(rCloud), 1000)
    r_arr_beyondCloud = np.logspace(np.log10(rCloud), np.log10(rCloud*1.5), 100)
    # start with zero
    r_arr = np.concatenate(([1e-10], r_arr_inCloud, r_arr_beyondCloud))
    
    r_arr = np.unique(r_arr)
    
    # initial cloud values
    params['rCloud'].value = rCloud
    params['nEdge'].value = nEdge
    print(f"Cloud radius is {np.round(rCloud, 3)}pc.")
    print(f"Cloud edge density is {np.round(nEdge * cvt.ndens_au2cgs, 3)} cm-3.")
    # radius array
    params['initial_cloud_r_arr'].value = r_arr
    # density array
    params['initial_cloud_n_arr'].value = density_profile.get_density_profile(r_arr, params)
    print(params['initial_cloud_n_arr'].value)
    # mass array
    params['initial_cloud_m_arr'].value = mass_profile.get_mass_profile(r_arr, params, return_mdot = False)
    
    return 
    

