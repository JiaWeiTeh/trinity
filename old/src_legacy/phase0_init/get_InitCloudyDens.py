#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:48:08 2023

@author: Jia Wei Teh

This script contains functions that create density law for cloudy, and 
write initial density profile to a txt. file.
"""

import numpy as np
import astropy.constants as c
import astropy.units as u
import os
import sys
import src.cloud_properties.density_profile as density_profile
from src._output.terminal_prints import cprint as cpr

def create_InitCloudyDens(path2output,
                       rCloud, mCloud,
                       alpha, 
                        nAvg,
                        rCore,
                        nISM,
                        nCore,
                        cloudy_dict,
                       coll_counter=0):

    # Note:
        # old code: __cloudy__.create_dlaw()
    
    dx_small = 1e-4 * u.pc
    rLast = (rCloud - dx_small).to(u.pc)
    # last radius just outside cloud
    # TODO: shouldn't this be +dx_small then?
    r = np.logspace(-1.,np.log10(rLast.value), endpoint=True, num=200) * u.pc
    # join with different endpoints
    r = np.concatenate([[dx_small],r,[r[-1]+0.001 * u.pc], [r[-1]+5.0* u.pc], [r[-1]+500.0* u.pc]])
    # get density profile
    n = density_profile.get_density_profile(r.value, cloudy_dict
                                            ) / u.pc**3
    
    # print('Checking initial density')
    # print(n)
    # sys.exit()
    logn = np.log10((n/u.cm**-3)) * u.cm**-3
    logr = np.log10(r/u.pc) * u.pc
    
    # old: dlaw.txt
    # save into csv
    full_path = os.path.join(path2output, 'init_density_profile_col'+ str(coll_counter) + '.csv')
    rel_path = os.path.relpath(full_path, os.getcwd())
    np.savetxt(full_path,
               np.transpose([logr, logn]), fmt='%.6e',
               delimiter = ',',
               header="Log10 radius [pc],Log10 density [1/cm3]", comments='')
    
    # TODO: make this sound better, and also check if the logr is in correct unit.
    return print(f'{cpr.FILE}Density for CLOUDY: {rel_path}{cpr.END}')
