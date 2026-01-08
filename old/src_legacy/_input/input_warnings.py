#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:56:22 2022

@author: Jia Wei Teh

This script contains a function which checks if input the parameters make sense.
"""


import sys

def input_warnings(params_dict):
    """
    A function that checks if input parameters make sense.

    Parameters
    ----------
    params_dict : dict
        Dictionary of input parameters.

    Returns
    -------
    None.

    """
    
    # TODO. E.g., if metalicity is >0, dens profile str is correct, etc. 
    # E.g., -2<nalpha<0
    # rCore < rCloud
    # only pL and bE allowed as str
    # nedge < nintercl
    
    # =============================================================================
    # Parameters with only True or False
    # =============================================================================
    # A list for { True | False} parameters
    trueFalseValues = [0 , 1]
    trueFalseParams = ['is_mCloud_beforeSF', 
                       'rand_input',
                       'stochastic_sampling',
                       'mult_exp',
                       'frag_enabled',
                       'frag_grav',
                       'frag_RTinstab', 
                       'frag_densInhom',
                       'frag_enable_timescale',
                       'inc_grav',
                       ]
    # check
    for pars in trueFalseParams:
        if params_dict[pars] not in trueFalseValues:
            sys.exit('The parameter \'%s\' accepts only 0 or 1 as input'%pars)
            
    # =============================================================================
    # Parameters that accepts only one or two input
    # =============================================================================
    
    
    
    # =============================================================================
    # Parameters with specific range
    # =============================================================================
    
    
    # =============================================================================
    # Parameters with specific keyword    
    # =============================================================================
          
    
    
    
    return
        

#%%


    
# params_dict = {'model_name': 'example', 
#                 'out_dir': 'def_dir', 
#                 'verbose': 1.0, 
#                 'output_format': 'ASCII', 
#                 'rand_input': 0.0, 
#                 'log_mCloud': 6.0, 
#                 'is_mCloud_beforeSF': 1.0, 
#                 'sfe': 0.01, 
#                 'nCore': 1000.0, 
#                 'rCore': 0.099, 
#                 'metallicity': 1.0, 
#                 'stochastic_sampling': 0.0, 
#                 'n_trials': 1.0, 
#                 'rand_log_mCloud': ['5', ' 7.47'], 
#                 'rand_sfe': ['0.01', ' 0.10'], 
#                 'rand_n_cloud': ['100.', ' 1000.'], 
#                 'rand_metallicity': ['0.15', ' 1'], 
#                 'mult_exp': 0.0, 
#                 'r_coll': 1.0, 
#                 'mult_SF': 1.0, 
#                 'sfe_tff': 0.01, 
#                 'imf': 'kroupa.imf', 
#                 'stellar_tracks': 'geneva', 
#                 'dens_profile': 'bE_prof', 
#                 'dens_g_bE': 14.1, 
#                 'dens_a_pL': -2.0, 
#                 'dens_navg_pL': 170.0, 
#                 'frag_enabled': 0.0, 
#                 'frag_r_min': 0.1, 
#                 'frag_grav': 0.0, 
#                 'frag_grav_coeff': 0.67, 
#                 'frag_RTinstab': 0.0, 
#                 'frag_densInhom': 0.0, 
#                 'frag_cf': 1.0, 
#                 'frag_enable_timescale': 1.0, 
#                 'stop_n_diss': 1.0, 
#                 'stop_t_diss': 1.0, 
#                 'stop_r': 1000.0, 
#                 'stop_t': 15.05, 
#                 'stop_t_unit': 'Myr', 
#                 'write_main': 1.0, 
#                 'write_stellar_prop': 0.0, 
#                 'write_bubble': 0.0, 
#                 'inc_grav': 1.0, 
#                 'f_Mcold_W': 0.0, 
#                 'f_Mcold_SN': 0.0, 
#                 'v_SN': 1000000000.0, 
#                 'sigma0': 1.5e-21, 
#                 'z_nodust': 0.05, 
#                 'mu_n': 2.1287915392418182e-24, 
#                 'mu_p': 1.0181176926808696e-24, 
#                 't_ion': 10000.0, 
#                 't_neu': 100.0, 
#                 'n_ISM': 0.1, 
#                 'kappa_IR': 4.0, 
#                 'gamma': 1.6666666666666667, 
#                 'thermcoeff_wind': 1.0, 
#                 'thermcoeff_SN': 1.0,
#                 }

# input_warnings(params_dict)