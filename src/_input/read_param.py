#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:33:31 2022

@author: Jia Wei Teh

This script contains a function that reads in parameter file and passes it to TRINITY. 
The function will also create a summary.txt file in the output directory.

"""

import sys
from datetime import datetime
from pathlib import Path
from fractions import Fraction
import os
import src._functions.unit_conversions as cvt
from src._input.dictionary import DescribedItem, DescribedDict

def read_param(path2file, write_summary = True):    
    """
    This function takes in the path to .param file, and returns an object containing parameters.
    Additionally, this function filters out non-useful parameters, then writes
    useful parameters into a .txt summary file in the output directory.

    Parameters
    ----------
    path2file : str
        Path to the .param file.
    write_summary: boolean
        Whether or not to write a summary .txt file.

    Returns
    -------
    params : Object
        An object describing WARPFIELD parameters.
        Example: To extract value for `sfe`, simply invoke params.sfe

    """
    
    
    # categorise 'True' to True, '123' to 123, and 'abc' to 'abc'.
    def parse_value(val):
        val = val.strip()
        # Check for boolean
        if val.lower() == 'true':
            return True
        elif val.lower() == 'false':
            return False
        # Check for float (or int)
        try:
            num = float(val)
            return num
        except ValueError:
            try: 
                # It could be a fraction (i.e., 3/5)
                return float(Fraction(val))
            except ValueError:
                # Fallback: treat as string
                return val
    
    
    # =============================================================================
    # Create list from input
    # =============================================================================
    
    input_dict = {}
    
    with open(path2file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines and comments
            parts = line.split(None)  # split only on the first whitespace
            if len(parts) == 2:
                key, val = parts
                parsed_val = parse_value(val)
                input_dict[key] = parsed_val
            else:
                raise ParameterFileError(f'Input parameter file formatting error: -> {line}')

    path2default = r'/Users/jwt/unsync/Code/Trinity/param/default.param'

    default_dict = {}
    with open(path2default, "r") as f:
        lines = f.readlines()
        comment = None
        unit = None
        for line in lines:
            line = line.strip()
            if line.startswith("# INFO: "):
                comment = line[len("# INFO: "):]
            elif line.startswith("# UNIT:"):
                unit = line[len("# UNIT:"):].strip().replace('[', '').replace(']', '')
            elif line and not line.startswith("#") and comment:
                parts = line.strip().split(None)
                if len(parts) == 2:
                    key, val = parts
                    parsed_val = parse_value(val)
                    default_dict[key] = (comment, unit, parsed_val)
                    comment = None  # Reset for next entry
                    unit = None  # Reset for next entry
                else:
                    raise ParameterFileError(f'Default parameter file formatting error: -> {line}')
    
    
    # Step1: check keys exist. 
    # Step2: merge two libraries
    for key, new_val in input_dict.items():
        if key not in default_dict.keys():
            raise ParameterFileError(f"Parameter '{key}' does not exist.")
        else:
            comment, unit, ori_val = default_dict[key]
            default_dict[key] = (comment, unit, new_val)
            
    
    param = DescribedDict()
    
    # Step3: change default parameter units into astronomy units (_au; e.g., s = Myr, cm = pc, g = Msun)
    for key, (comment, unit, val) in default_dict.items():
        
        factor = cvt.convert2au(unit)
        param[key] = DescribedItem(val * factor, comment, unit)
      
    # print(param)
    # import sys
    # sys.exit()

    # =============================================================================
    # Check with default parameters
    # =============================================================================
    
    # TODO: remove _summary. Provide only the yaml file, and then 
    # make a python file that turns it into something that is human-readable!
    # Then, print in terminal that there is two files, one yaml, one for human.
    
    
    # =============================================================================
    # Check if parameters given in .param file makes sense
    # =============================================================================
    # First, for parameters specified in .param file, update dictionary and use the
    # specified values instead of the default.
    
    # TODO:
        # What do if randomised? Should show both randomised range, and randomised result. 
    
    # TODO
    # give warning if parameter does not make sense
    # input_warnings.input_warnings(params_dict)
    
    # warnings
    if param['ZCloud'] != 1:
        raise ParameterFileError(f"metallicity of {param['ZCloud'].value} is not implemented.")
    
    # =============================================================================
    # Here we deal with additional parameters that will be recorded in summary.txt.
    # For those that are not recorded, scroll down to the final section of this 
    # script.
    # =============================================================================
    # We have assumed the dust cross section scales linearly with metallicity. However,
    # below a certain metallicity, there is no dust
    if param['ZCloud'] >= param['dust_noZ']:
        param['dust_sigma'].value = param['dust_sigma'] * param['ZCloud']
    else:
        param['dust_sigma'].value = 0
        
    # print(param['model_name'])
    # print(param['dust_sigma'])
    
    # param['test'] = 1

    # =============================================================================
    # Store only useful parameters into the summary.txt file
    # =============================================================================
    # First, grab directories
    # 1. Output directory:
    if param['path2output'] == 'def_dir':
        # If user did not specify, the directory will be set as ./outputs/ 
        # check if directory exists; if not, create one.
        # TODO: Add smart system that adds 1, 2, 3 if repeated default to avoid overwrite.
        path2output = os.path.join(os.getcwd(), 'outputs/'+param['model_name']+'/')
        Path(path2output).mkdir(parents=True, exist_ok = True)
        param['path2output'].value = path2output
    else:
        # if instead given a path, then use that instead
        path2output = str(param['out_dir'])
        Path(path2output).mkdir(parents=True, exist_ok = True)
        param['path2output'].value = path2output
    
    # TODO: put into environment(?). This is the only one that matters.
    
    
    # 2. Cooling table directory - nonCIE:
    if param['path_cooling_nonCIE'] == 'def_dir':
        # If user did not specify, the directory will be set as ./lib/cooling/opiate/
        path2cooling_nonCIE = os.path.join(os.getcwd(), 'lib/cooling/opiate/')
        param['path_cooling_nonCIE'].value = path2cooling_nonCIE
    else:
        # if instead given a path, then use that instead
        path2cooling_nonCIE = str(param['path_cooling_nonCIE'])
        Path(path2cooling_nonCIE).mkdir(parents=True, exist_ok = True)
        param['path_cooling_nonCIE'].value = path2cooling_nonCIE
        
    # 3. Cooling table directory - CIE:
    if param['ZCloud'] == 1:
        # option 1, 2, 3 are for solar metallicity
        if param['path_cooling_CIE'] == 1:
            # If user did not specify, the directory will be set as ./lib/cooling/CIE/
            param['path_cooling_CIE'].value = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_1_Cloudy.dat')
        elif param['path_cooling_CIE'] == 2:
            # If user did not specify, the directory will be set as ./lib/cooling/CIE/
            param['path_cooling_CIE'].value = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_2_Cloudy_grains.dat')
        elif param['path_cooling_CIE'] == 3:
            # If user did not specify, the directory will be set as ./lib/cooling/CIE/
            param['path_cooling_CIE'].value = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_3_Gnat-Ferland2012.dat')
    elif param['ZCloud'] == 0.15:
        param['path_cooling_CIE'].value = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_4_Sutherland-Dopita1993.dat')
    else:
        # if instead given a path, then use that instead
        path2cooling_CIE = str(param['path_cooling_CIE'])
        Path(path2cooling_CIE).mkdir(parents=True, exist_ok = True)
        param['path2cooling_CIE'].value = path2cooling_CIE
        
    # 4. Starburst99 (sps) table directory:
    if param['path_sps'] == 'def_dir':
        # If user did not specify, the directory will be set as ./lib/sps/starburst99/
        path2sps = os.path.join(os.getcwd(), 'lib/sps/starburst99/')
        param['path_sps'].value = path2sps
    else:
        # if instead given a path, then use that instead
        path2sps = str(param['path_sps'])
        Path(path2sps).mkdir(parents=True, exist_ok = True)
        param['path_sps'].value = path2sps        
        
    # ----
    
    # Then, organise dictionary so that it does not include useless info
    
    # Remove unrelated parameters depending on selected density profile
    
    if param['dens_profile'].value not in ['densBE', 'densPL']:
        raise ParameterFileError(f"{param['dens_profile']} not in \'dens_profile\'")
    
    if param['dens_profile'] == 'densBE':
        param.pop('densPL_alpha')
        param['densBE_Omega'].exclude_from_snapshot = True
    elif param['dens_profile'] == 'densPL':
        param.pop('densBE_Omega')
        param['densPL_alpha'].exclude_from_snapshot = True
        
    # ----
    
    # ----
        
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # Store into summary file as output.
    if write_summary:
        with open(path2output+param['model_name']+'_summary.txt', 'w') as f:
            # header
            f.writelines('\
# =============================================================================\n\
# Summary of parameters in the \'%s\' run. Values are in units of [Myr, Msun, pc].\n\
# Created at %s.\n\
# =============================================================================\n\n\
'%(param['model_name'].value, dt_string))
            # body
            for key, val in param.items():
                f.writelines(key+'    '+"".join(str(val.value))+'\n')
            # close 
            f.close()
   
    # =============================================================================
    # This section contains information which you do not want to appear in the 
    # final summary.txt file, but want it included. Mostly mini-conversions.
    # =============================================================================
            
    # TODO:
    # Here we deal with tStop based on units. If the unit is Myr, then
    # tStop is tStop. If it is in free-frall time, then we need to change
    # to a proper unit of time. Add a parameter 'tff' that calculates the 
    # free-fall time. This may be used even if stop_t_unit is not in tff time, 
    # as it may be called if mult_SF = 2, where the second startburst is
    # characterised by the free-fall time.
    # tff = np.sqrt(3. * np.pi / (32. * c.G.cgs.value * params_dict['dens_navg_pL'] * params_dict['mu_n'])) * u.s.to(u.Myr)
    # params_dict['tff'] = float(tff)
    # if params_dict['stop_t_unit'] == 'tff':
        # params_dict['stop_t'] = params_dict['stop_t'] * params_dict['tff']
    # elif params_dict['stop_t_unit'] == 'Myr':
        # params_dict['stop_t'] = params_dict['stop_t'] 
    # if params_dict['stop_t_unit'] == 'Myr', it is fine; Myr is also the 
    # unit in all other calculations.
    
    # ----
    
    # Here we calculate mCloud 
    # cluster mass
    mCluster = param['mCloud'] * param['sfe']
    # cloud mass after SF
    mCloud = param['mCloud'] - mCluster
    # save
    param['mCloud'].value = mCloud
    param['mCluster'] = DescribedItem(mCluster, 'Cluster mass')
    
    # exclude snapshots?
    for key, val in param.items():
        # only keep track of these. The rest in initial file are just constants that 
        # will not change, and makes no sense to keep track and waste memory.
        if key not in ['model_name', 'mCloud', 'cool_alpha', 'cool_beta', 'cool_delta']:
            val.exclude_from_snapshot = True
            

    param['SimulationEndReason'] = DescribedItem('', 'What caused simulation to complete')

    
    import numpy as np
    # add tSF to track for multi expansion
    param['tSF'] = DescribedItem(0, 'Time of star formation.')
    # track most important bubble parameters:
    param['t_now'] = DescribedItem(0, 'Myr. Current time.')
    param['v2'] = DescribedItem(0, 'pc/Myr. Velocity at outer bubble/inner shell radius.')
    param['R2'] = DescribedItem(0, 'pc. Outer bubble/inner shell radius.')
    param['T0'] = DescribedItem(0, 'K. This is current T_rgoal in bubble_luminosity.')
    param['Eb'] = DescribedItem(0,  'Msun*pc2/Myr2. Bubble energy.')
    param['R1'] = DescribedItem(0, 'pc. Inner radius of bubble')
    param['Pb'] = DescribedItem(0, 'Msun/Myr2/pc. Bubble pressure')    
    param['c_sound'] = DescribedItem(0, 'Sound speed.')
    # --
    param['rCloud'] = DescribedItem(0, 'Cloud radius')
    param['nEdge'] = DescribedItem(0, 'Number density at cloud radius')
    
    # feedback values from SB99
    param['SB99_data'] = DescribedItem(0, 'SB99 datacube from read_SB99.py', exclude_from_snapshot = True)
    param['SB99f'] = DescribedItem(0, 'SB99 interpolation function from read_SB99.py', exclude_from_snapshot = True)
    # --
    param['LWind'] = DescribedItem(0, 'Msun*pc2/Myr3. wind mechanical luminosity')
    param['Qi'] = DescribedItem(0, '/Myr. Ionising photon rate.')
    param['vWind'] = DescribedItem(0, 'pc/Myr. Terminal wind velocity')
    param['pWindDot'] = DescribedItem(0, 'Msun*pc/Myr2. Wind momentum rate')
    param['pWindDotDot'] = DescribedItem(0, 'Msun*pc/Myr3. Rate of wind momentum rate.')
    param['Lbol'] = DescribedItem(0,  'Msun*pc2/Myr3. Bolometric luminosity.') 
    param['Ln'] = DescribedItem(0,  'Msun*pc2/Myr3. Non-ionizing luminosity Lbol*(1-fi).') 
    param['Li'] = DescribedItem(0,  'Msun*pc2/Myr3. Ionizing luminosity Lbol*fi.') 
    # cooling
    param['t_previousCoolingUpdate'] = DescribedItem(1e30, 'Myr. At what time is the previous cooling update?')
    param['cStruc_cooling_nonCIE'] = DescribedItem(0, 'non-CIE, cooling cube', exclude_from_snapshot = True)
    param['cStruc_heating_nonCIE'] = DescribedItem(0, 'non-CIE, heating cube', exclude_from_snapshot = True)
    param['cStruc_net_nonCIE_interpolation'] = DescribedItem(0, 'non-CIE, netcooling cube interpolations', exclude_from_snapshot = True)
    # --
    param['cStruc_cooling_CIE_logT'] = DescribedItem(0, 'CIE structure log temperature', exclude_from_snapshot = True)
    param['cStruc_cooling_CIE_logLambda'] = DescribedItem(0, 'CIE structure log lambda values', exclude_from_snapshot = True)
    param['cStruc_cooling_CIE_interpolation'] = DescribedItem(0, 'CIE structure interpolations', exclude_from_snapshot = True)
 
    # shell
    param['shell_fAbsorbedIon'] = DescribedItem(1, 'Fraction of absorbed ionizing radiations')
    param['shell_fAbsorbedNeu'] = DescribedItem(0, 'Fraction of absorbed non-ionizing radiations')
    param['shell_fAbsorbedWeightedTotal'] = DescribedItem(0, 'Total absorption fraction, defined as luminosity weighted average of shell_fAbsorbedIon and shell_fAbsorbedNeu')
    param['shell_fIonisedDust'] = DescribedItem(0, 'unitless')
    param['shell_fRad'] = DescribedItem(0, 'Radiation pressure coupled to the shell. f_abs * Lbol / c')
    param['shell_thickness'] = DescribedItem(0, 'pc. Thickness of shell.')
    # main_dict['shell_nInner'] = DescribedItem(0, '/pc3')
    param['shell_nMax'] = DescribedItem(0, '/pc3. Maximum density of shell currently.')
    param['shell_tauKappaRatio'] = DescribedItem(0, 'Msun / pc2. The ratio tau_IR/kappa_IR  = \int rho dr')
    param['shell_grav_r'] = DescribedItem(np.array([]), 'pc. Radius array of gravitational potential calculations')
    param['shell_grav_phi'] = DescribedItem(np.array([]), 'pc2 / Myr2. Gravitational potential')
    param['shell_grav_force_m'] = DescribedItem(np.array([]), 'pc / Myr2. Gravitational potential force per unit mass')
    param['shell_mass'] = DescribedItem(0, 'Msol. Shell mass')
    param['shell_massDot'] = DescribedItem(0, 'Msol/Myr. Rate of change of shell mass')

    # Force calculation in shell dynamics
    
    # Force calculations
    param['F_grav'] = DescribedItem(0, 'Force on shell due to gravity')
    param['F_SN'] = DescribedItem(0, 'Force on shell due to SN')
    param['F_ram'] = DescribedItem(0, 'Force on shell due to ram pressure (wind+SN?)')
    param['F_wind'] = DescribedItem(0, 'Force on shell due to winds')
    param['F_ion'] = DescribedItem(0, 'Force on shell due to photoionisation pressure')
    param['F_rad'] = DescribedItem(0, 'Radiation pressure = direct + indirect ~ f_abs * Lbol/c * (1 + tau_IR)')
    
    # bubble parameters 

    param['bubble_LTotal'] = DescribedItem(0, 'Total luminosity lost to cooling.')
    # main_dict['bubble_T_rgoal'] = DescribedItem(0, 'Current guess temperature at R2prime. This becomes T0.')
    param['bubble_L1Bubble'] = DescribedItem(0, 'Total luminosity lost to cooling (bubble zone)')
    param['bubble_L2Conduction'] = DescribedItem(0, 'Total luminosity lost to cooling (conduction zone)')
    param['bubble_L3Intermediate'] = DescribedItem(0, 'Total luminosity lost to cooling (intermediate zone)')
    param['bubble_Tavg'] = DescribedItem(0, 'Average temperature across bubble.')
    param['bubble_mass'] = DescribedItem(0, 'Bubble mass')
    param['bubble_r_Tb'] = DescribedItem(0, 'True radius obtained from bubble_xi_Tb * R2.')
    param['bubble_T_r_Tb'] = DescribedItem(0, 'Temperature at r_Tb')


    param['bubble_r_arr'] = DescribedItem(np.array([]), 'pc. radius structure. High to low. Paired with decreasing radius array.')
    param['bubble_v_arr'] = DescribedItem(np.array([]), 'pc/Myr. velocity structure. Decreasing, since radius array is decreasing.')
    param['bubble_T_arr'] = DescribedItem(np.array([]), 'K. velocity structure. Paired with decreasing radius array.')
    param['bubble_dTdr_arr'] = DescribedItem(np.array([]), 'K/pc. T gradient structure. Paired with decreasing radius array.')
    param['bubble_n_arr'] = DescribedItem(np.array([]), '1/pc3. density structure. Paired with decreasing radius array.')
    param['bubble_dMdtGuess'] = DescribedItem(0, 'dMdt scipy guesses. Not the final guess.')
    param['bubble_dMdt'] = DescribedItem(np.nan, 'Current dMdt. Mass loss from region c (shell) into region b (shocked winds) due to thermal conduction.')
  
    param['bubble_Lgain'] = DescribedItem(np.nan, 'au, luminosity gain in cooling phase due to winds')
    param['bubble_Lloss'] = DescribedItem(np.nan, 'au, luminosity lost in cooling phase due to cooling (and possibly leaking)')
    param['bubble_Leak'] = DescribedItem(0, 'au, leaking luminosity')
 
    # state
    param['isCollapse'] = DescribedItem(False, 'Check if the cloud is collapsing')
   
    # initial(?)
    param['initial_cloudDens'] = DescribedItem([], 'Initial cloud density profile')
    param['initial_cloudMass'] = DescribedItem([], 'Initial cloud density profile')
    
   
    # To be removed
    param['residual_deltaT'] = DescribedItem(0, '(T1-T2)/T2')
    param['residual_betaEdot'] = DescribedItem(0, '(Edot - Edot2)/Edot')
    param['residual_Edot1_guess'] = DescribedItem(np.nan, 'Value of Edot obtained via beta.')
    param['residual_Edot2_guess'] = DescribedItem(np.nan, 'Value of Edot obtained via energy balance equation.')
    param['residual_T1_guess'] = DescribedItem(np.nan, 'Value of T obtained via bubble_Trgoal.')
    param['residual_T2_guess'] = DescribedItem(np.nan, 'Value of T obtained via T0.')
   
    
    

    return param


class ParameterFileError(Exception):
    """Raised when a parameter file entry is invalid."""






