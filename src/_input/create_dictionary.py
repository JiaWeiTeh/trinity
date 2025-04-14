#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:04:16 2025

@author: Jia Wei Teh

This script handles the creation of dictionary
"""

from src._input.dictionary import DescribedItem, DescribedDict, updateDict
import os
import numpy as np
import importlib
import src._functions.unit_conversions as cvt
import astropy.constants as c
import astropy.units as u


trinity_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])



# Maybe move this to read_params so that we dont have to call trinity_params and can completely
# get rid of the dependency

def create():
    
    
    

    
# TODO: in future for multiple collapses, here we can add an if/else where counter 0 = reset, counter > 0 = retain most values.
    
    # --- initialise and prepare dictionary ---
    
    main_dict = DescribedDict()
    # how often do we save into json file?
    # 2 = 10(2-1) = 10 saves per snapshot.
    main_dict.snapshot_interval = 2 
    
    
    # main
    main_dict['current_phase'] = DescribedItem('1a', 'Which phase is the simulation in? 1a: energy, 1b: implicit energy, 2: momentum')
    main_dict['isCollapse'] = DescribedItem(False, 'is the bubble currently collapsing?')
    main_dict['isDissolution'] = DescribedItem(False, 'is the bubble currently dissolving')
    main_dict['collapse_counter'] = DescribedItem(0, 'How many times recollapse has happened.')
    main_dict['metallicity'] = DescribedItem(trinity_params.metallicity, 'Cloud metallicity of the simulation')
    main_dict['model_name'] = DescribedItem(trinity_params.model_name, 'Model name')
    main_dict['dens_profile'] = DescribedItem(trinity_params.dens_profile, 'density profile')
    main_dict['sfe'] = DescribedItem(trinity_params.sfe, 'sfe')
    
    # constants
    main_dict['mu_n_au'] = DescribedItem(trinity_params.mu_n.value * cvt.g2Msun, 'Msun.')
    main_dict['mu_p_au'] = DescribedItem(trinity_params.mu_p.value * cvt.g2Msun, 'Msun.')
    main_dict['k_B_au'] = DescribedItem(c.k_B.cgs.value * cvt.k_B_cgs2au, 'Boltzmann constant. Msun*pc2/Myr2/K.')
    main_dict['c_therm_au'] = DescribedItem(trinity_params.c_therm.value * cvt.c_therm_cgs2au, 'Msun*pc/Myr3/K(7/2).')
    main_dict['c_au'] = DescribedItem(c.c.cgs.value * cvt.v_cms2au , 'speed of light.')
    main_dict['G_au'] = DescribedItem(c.G.cgs.value * cvt.G_cgs2au , 'pc3/Msun/Myr2.')
    main_dict['alpha_pL'] = DescribedItem(trinity_params.dens_a_pL, 'Coefficient of density power-law.')
    main_dict['rCore_au'] = DescribedItem(trinity_params.rCore.value, 'pc. Core radius.')
    main_dict['nCore_au'] = DescribedItem(trinity_params.nCore.value * cvt.ndens_cgs2au, '1/pc3. Core density.')
    main_dict['nAvg_au'] = DescribedItem(trinity_params.dens_navg_pL.value * cvt.ndens_cgs2au, '1/pc3.')
    main_dict['nISM_au'] = DescribedItem(trinity_params.nISM.value * cvt.ndens_cgs2au, '1/pc3.')
    main_dict['t_ion'] = DescribedItem(trinity_params.t_ion.value, 'K')
    main_dict['t_neu'] = DescribedItem(trinity_params.t_neu.value, 'K')
    main_dict['alpha_B_au'] = DescribedItem(trinity_params.alpha_B.value * cvt.cm2pc**3 / cvt.s2Myr, 'au (cm3/s)')
    main_dict['sigma_d_au'] = DescribedItem(trinity_params.sigma_d.value * cvt.cm2pc**2, 'pc^2, dust cross section at solar metallicity')
    
    # cloud and cluster properties
    main_dict['mCluster_au'] = DescribedItem(trinity_params.mCluster.value, 'M_sun')
    main_dict['mCloud_au'] = DescribedItem(trinity_params.mCloud.value, 'M_sun. Cloud mass now.')
    main_dict['rCloud_au'] = DescribedItem(np.nan, 'pc. Cloud radius.')
    main_dict['nEdge_au'] = DescribedItem(np.nan, '1/pc3. Desity at the end of cloud.')
    
    # sequential star formation recordings
    main_dict['mCluster_au_list'] = DescribedItem(np.array([trinity_params.mCluster.value]), 'Records cluster mass')
    main_dict['tSF'] = DescribedItem(0, 'Records current time of star formation')
    main_dict['tSF_list'] = DescribedItem(np.array([0]), 'Records time of star formation')
    
    # Starburst99 outputs
    main_dict['SB99_mass'] = DescribedItem(trinity_params.SB99_mass.value, 'SB99 data mass')
    main_dict['SB99_data'] = DescribedItem(np.nan, 'SB99 outputs')
    main_dict['SB99f'] = DescribedItem(np.nan, 'SB99 Interpolation function.')
    
    main_dict['L_wind'] = DescribedItem(np.nan, 'Msun*pc2/Myr3)')
    main_dict['Qi'] = DescribedItem(np.nan, '/Myr')
    main_dict['v_wind'] = DescribedItem(np.nan, 'pc/Myr')
    main_dict['pwdot'] = DescribedItem(np.nan, 'Msun*pc/Myr2')
    main_dict['pwdot_dot'] = DescribedItem(np.nan, 'Msun*pc/Myr3')
    main_dict['Lbol'] = DescribedItem(np.nan,  'Msun*pc2/Myr3. Bolometric luminosity.') #not added
    main_dict['Ln'] = DescribedItem(np.nan,  'Msun*pc2/Myr3. Non-ionizing luminosity Lbol*(1-fi).') #not added
    main_dict['Li'] = DescribedItem(np.nan,  'Msun*pc2/Myr3. Ionizing luminosity Lbol*fi.') #not added
    
    # most important parameter for ODEs
    main_dict['t_now'] = DescribedItem(np.nan, 'Myr. Current time.')
    main_dict['v2'] = DescribedItem(np.nan, 'pc/Myr. Velocity at outer bubble/inner shell radius.')
    main_dict['R2'] = DescribedItem(np.nan, 'pc. Outer bubble/inner shell radius.')
    main_dict['T0'] = DescribedItem(np.nan, 'K. This is current T_rgoal in bubble_luminosity.')
    main_dict['Eb'] = DescribedItem(np.nan,  'Msun*pc2/Myr2. Bubble energy.')

    # cooling parameters
    main_dict['alpha'] = DescribedItem(trinity_params.cooling_alpha, 'Unitless. value being: v2 * t_now / R2')
    main_dict['beta'] = DescribedItem(trinity_params.cooling_beta, 'cooling related.')
    main_dict['delta'] = DescribedItem(trinity_params.cooling_delta, 'cooling related.')
        
    # tracking solving cooling parameters
    main_dict['beta_Edot_residual'] = DescribedItem(0, 'tbd')
    main_dict['delta_T_residual'] = DescribedItem(0, 'tbd')
    
    main_dict['Edot1_guess'] = DescribedItem(np.nan, 'Value of Edot obtained via beta.')
    main_dict['Edot2_guess'] = DescribedItem(np.nan, 'Value of Edot obtained via energy balance equation.')
    main_dict['T1_guess'] = DescribedItem(np.nan, 'Value of T obtained via bubble_Trgoal.')
    main_dict['T2_guess'] = DescribedItem(np.nan, 'Value of T obtained via T0.')
    
    main_dict['transformation_beta_cent_a'] = DescribedItem([], 'Transformation parameters for beta. [previous_beta (center), sigma (symmetric about center)]')
    main_dict['transformation_delta_cent_a'] = DescribedItem([], 'Transformation parameters for delta. [previous_delta (center), sigma (symmetric about center)]')
    
    main_dict['cStruc_cooling_CIE_interpolation'] = DescribedItem(np.nan, 'CIE structure interpolations')
    main_dict['cStruc_cooling_CIE_logT'] = DescribedItem(np.nan, 'CIE structure log temperature')
    main_dict['cStruc_cooling_CIE_logLambda'] = DescribedItem(np.nan, 'CIE structure log lambda values')
    main_dict['cStruc_cooling_nonCIE'] = DescribedItem(np.nan, 'non-CIE, cooling cube')
    main_dict['cStruc_heating_nonCIE'] = DescribedItem(np.nan, 'non-CIE, heating cube')
    main_dict['cStruc_net_nonCIE_interpolation'] = DescribedItem(np.nan, 'non-CIE, netcooling cube interpolations')

    main_dict['time_last_cooling_update'] = DescribedItem(-1e30, 'arbitrary large negative number as initiation.')

    # paths
    main_dict['path_cooling_CIE'] = DescribedItem(trinity_params.path_cooling_CIE, 'path to CIE data')
    main_dict['path2output'] = DescribedItem(trinity_params.out_dir, 'Output path')

    # mass loss calculations in bubble
    main_dict['dMdt'] = DescribedItem(np.nan, 'Msun/Myr. mass loss from region c (shell) into region b (shocked winds) due to thermal conduction.')
    main_dict['dMdt_factor'] = DescribedItem(1.646, 'Tbd.')
    main_dict['v0'] = DescribedItem(0, 'pc/Myr. velocity at r1')
    main_dict['v0_residual'] = DescribedItem(0, 'pc/Myr. residual for v0 - 0/v0')
    
    main_dict['bubble_v_arr'] = DescribedItem(np.array([]), 'pc/Myr. velocity structure. Decreasing, since radius array is decreasing.')
    main_dict['bubble_T_arr'] = DescribedItem(np.array([]), 'K. velocity structure. Paired with decreasing radius array.')
    main_dict['bubble_dTdr_arr'] = DescribedItem(np.array([]), 'K/pc. T gradient structure. Paired with decreasing radius array.')
    main_dict['bubble_r_arr'] = DescribedItem(np.array([]), 'pc. radius structure. High to low. Paired with decreasing radius array.')
    main_dict['bubble_n_arr'] = DescribedItem(np.array([]), '1/pc3. density structure. Paired with decreasing radius array.')
    main_dict['bubble_dMdt'] = DescribedItem(np.nan, 'dMdt scipy guesses. Not the final guess.')
  
    # bubble calculations
    main_dict['R1'] = DescribedItem(np.nan, 'pc. Inner radius of bubble')
    main_dict['Pb'] = DescribedItem(np.nan, 'Msun/Myr2/pc). Bubble pressure')
    main_dict['L_leak'] = DescribedItem(0, 'Leaking luminosity.')
    
    main_dict['r_goal'] = DescribedItem(np.nan, 'pc. Where to calculate T. This is R2prime.')
    main_dict['T_goal'] = DescribedItem(np.nan, 'K. Initial guess of T at R2prime. Usually 3e4K.')
    main_dict['r_inner'] = DescribedItem(np.nan, 'pc. Inner radius of where to evaluate dMdt. Usually is just R1.')
  
    main_dict['cs_avg'] = DescribedItem(np.nan, 'pc/Myr. Sound speed.')
    
  
    # bubble output values
    main_dict['bubble_L_total'] = DescribedItem(0, 'Total luminosity lost to cooling.')
    main_dict['bubble_T_rgoal'] = DescribedItem(np.nan, 'Current guess temperature at R2prime. This becomes T0.')
    main_dict['bubble_L_bubble'] = DescribedItem(np.nan, 'Total luminosity lost to cooling (bubble zone)')
    main_dict['bubble_L_conduction'] = DescribedItem(np.nan, 'Total luminosity lost to cooling (conduction zone)')
    main_dict['bubble_L_intermediate'] = DescribedItem(np.nan, 'Total luminosity lost to cooling (intermediate zone)')
    main_dict['bubble_Tavg'] = DescribedItem(np.nan, 'Average temperature across bubble.')
    main_dict['bubble_mBubble'] = DescribedItem(np.nan, 'Bubble mass')
    
    # simulation constraints
    
    main_dict['Lgain'] = DescribedItem(np.nan, 'au, luminosity gain in get_betadelta')
    main_dict['Lloss'] = DescribedItem(np.nan, 'au, luminosity lost (due to cooling/leaking) in get_betadelta')
    
    
    # shell calculations
    # set dissolution time to arbitrary high number (i.e. the shell has not yet dissolved)
    main_dict['Rsh_max'] = DescribedItem(np.nan, 'maximum shell radius at any given time.')
    main_dict['inc_grav'] = DescribedItem(trinity_params.inc_grav, 'maximum shell radius at any given time.')
    
    # shell output values
    main_dict['shell_f_absorbed_ion'] = DescribedItem(np.nan, 'unitless')
    main_dict['shell_f_absorbed_neu'] = DescribedItem(np.nan, 'unitless')
    main_dict['shell_f_absorbed'] = DescribedItem(np.nan, 'unitless')
    main_dict['shell_f_ionised_dust'] = DescribedItem(np.nan, 'unitless')
    main_dict['shell_thickness'] = DescribedItem(np.nan, 'pc')
    main_dict['shell_nShellInner'] = DescribedItem(np.nan, '/pc3')
    main_dict['shell_nShell_max'] = DescribedItem(np.nan, '/pc3')
    main_dict['shell_tau_kappa_IR'] = DescribedItem(np.nan, 'Msun / pc2')
    main_dict['shell_grav_r'] = DescribedItem(np.nan, 'pc')
    main_dict['shell_grav_phi'] = DescribedItem(np.nan, 'pc2 / Myr2')
    main_dict['shell_grav_force_m'] = DescribedItem(np.nan, 'pc / Myr2')
    main_dict['shell_f_rad'] = DescribedItem(np.nan, 'Radiation pressure coupled to the shell. f_abs * Lbol / c')
    
    
    main_dict['mShell'] = DescribedItem(np.nan, 'Msol. Shell mass')
    main_dict['mShell_dot'] = DescribedItem(np.nan, 'Msol/Myr. Rate of change of shell mass')
    main_dict['isLowdense'] = DescribedItem(False, 'is the shell currently in low density?')
    main_dict['t_Lowdense'] = DescribedItem(1e30, 'Myr, time of most recent isLowdense==True')
    
    
    main_dict['stop_n_diss'] = DescribedItem(trinity_params.stop_n_diss.value * cvt.ndens_cgs2au, 'au, density such that after stop_t_diss the shell is considered dissolved')
    main_dict['stop_t_diss'] = DescribedItem(trinity_params.stop_t_diss.value, 'Myr, time sustained below stop_n_diss after which shell is considered dissolved')
    
    
    # simulation
    main_dict['tStop'] = DescribedItem(trinity_params.stop_t.value, 'Myr. Maximum time.') 
    # TODO:
    # i thinjk this is not used? also there seem to be shell termination within shell_strcuture.
    main_dict['t_dissolve'] = DescribedItem(1e30, 'Time after which consider simulation as dissolved.') 
    main_dict['dEdt'] = DescribedItem(np.nan, 'Constant energy gradient over time; used for phase 1c and beyond.')

    main_dict['r_coll'] = DescribedItem(trinity_params.r_coll.value, 'Radius below which cloud is considered completely collapse during collapse event')
    main_dict['stop_r'] = DescribedItem(trinity_params.stop_r.value, 'Radius above which cloud is considered too large')
    # TODO:
    # i think this wasnt reached because in shell it has its own termination
    main_dict['completed_reason'] = DescribedItem('', 'What caused simulation to complete')
    
    # Force calculations
    main_dict['F_grav'] = DescribedItem(0, '')
    main_dict['F_SN'] = DescribedItem(0, '')
    main_dict['F_wind'] = DescribedItem(0, '')
    main_dict['F_ram'] = DescribedItem(0, 'F_wind + F_SN')
    main_dict['F_rad'] = DescribedItem(0, 'Radiation pressure = direct + indirect ~ f_abs * Lbol/c * (1 + tau_IR)')
    
    
    
    
    
    return main_dict
    
