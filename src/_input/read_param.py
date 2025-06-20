#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:33:31 2022

@author: Jia Wei Teh

This script contains a function that reads in parameter file and passes it to TRINITY. 
The function will also create a summary.txt file in the output directory.

"""


from datetime import datetime
from pathlib import Path
import random # for random numbers
import sys
import numpy as np
import os
import yaml
import astropy.units as u
import astropy.constants as c 
import src._input.unify_units as unify_units
import src._input.default_values as default_values
import src._functions.unit_conversions as cvt

from src._input.dictionary import DescribedItem, DescribedDict, updateDict

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
            pass
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
        param[key] = DescribedItem(val * factor, comment)

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
    if param['ZCloud'].value != 2:
        raise ParameterFileError(f"metallicity of {param['ZCloud'].value} is not implemented.")
    
    
    # =============================================================================
    # Here we deal with additional parameters that will be recorded in summary.txt.
    # For those that are not recorded, scroll down to the final section of this 
    # script.
    # =============================================================================
    # We have assumed the dust cross section scales linearly with metallicity. However,
    # below a certain metallicity, there is no dust
    if param['ZCloud'].value >= param['dust_noZ'].value:
        param['dust_sigma'].value = param['dust_sigma'].value * param['ZCloud'].value
    else:
        param['dust_sigma'].value = 0

    # =============================================================================
    # Store only useful parameters into the summary.txt file
    # =============================================================================
    # First, grab directories
    # 1. Output directory:
    if param['out_dir'].value == 'def_dir':
        # If user did not specify, the directory will be set as ./outputs/ 
        # check if directory exists; if not, create one.
        # TODO: Add smart system that adds 1, 2, 3 if repeated default to avoid overwrite.
        path2output = os.path.join(os.getcwd(), 'outputs/'+params_dict['model_name']+'/')
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output
    else:
        # if instead given a path, then use that instead
        path2output = str(params_dict['out_dir'])
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output
    
    # put into environment. This is the only one that matters.
    os.environ['path2trinity'] = path2output
 
    # 2. Cooling table directory - nonCIE:
    if params_dict['path_cooling_nonCIE'] == 'def_dir':
        # If user did not specify, the directory will be set as ./lib/cooling/opiate/
        path2cooling_nonCIE = os.path.join(os.getcwd(), 'lib/cooling/opiate/')
        params_dict['path_cooling_nonCIE'] = path2cooling_nonCIE
    else:
        # if instead given a path, then use that instead
        path2cooling_nonCIE = str(params_dict['path_cooling_nonCIE'])
        Path(path2cooling_nonCIE).mkdir(parents=True, exist_ok = True)
        params_dict['path_cooling_nonCIE'] = path2cooling_nonCIE
        
    # 3. Cooling table directory - CIE:
    if params_dict['metallicity'] == 1:
        # option 1, 2, 3 are for solar metallicity
        if params_dict['path_cooling_CIE'] == 1:
            # If user did not specify, the directory will be set as ./lib/cooling/CIE/
            params_dict['path_cooling_CIE'] = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_1_Cloudy.dat')
        elif params_dict['path_cooling_CIE'] == 2:
            # If user did not specify, the directory will be set as ./lib/cooling/CIE/
            params_dict['path_cooling_CIE'] = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_2_Cloudy_grains.dat')
        elif params_dict['path_cooling_CIE'] == 3:
            # If user did not specify, the directory will be set as ./lib/cooling/CIE/
            params_dict['path_cooling_CIE'] = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_3_Gnat-Ferland2012.dat')
    elif params_dict['metallicity'] == 0.15:
        params_dict['path_cooling_CIE'] = os.path.join(os.getcwd(), 'lib/cooling/CIE/coolingCIE_4_Sutherland-Dopita1993.dat')
    else:
        # if instead given a path, then use that instead
        path2cooling_CIE = str(params_dict['path_cooling_CIE'])
        Path(path2cooling_CIE).mkdir(parents=True, exist_ok = True)
        params_dict['path2cooling_CIE'] = path2cooling_CIE
        
    # 4. Starburst99 (sps) table directory:
    if params_dict['path_sps'] == 'def_dir':
        # If user did not specify, the directory will be set as ./lib/sps/starburst99/
        path2sps = os.path.join(os.getcwd(), 'lib/sps/starburst99/')
        params_dict['path_sps'] = path2sps
    else:
        # if instead given a path, then use that instead
        path2sps = str(params_dict['path_sps'])
        Path(path2sps).mkdir(parents=True, exist_ok = True)
        params_dict['path_sps'] = path2sps        
        
    # ----
    
    # Then, organise dictionary so that it does not include useless info
    
    # Remove unrelated parameters depending on selected density profile
    if params_dict['dens_profile'] == 'bE_prof':
        params_dict.pop('dens_a_pL')
        params_dict.pop('dens_navg_pL')
    elif params_dict['dens_profile'] == 'pL_prof':
        params_dict.pop('dens_g_bE')
        
    # ----
    
    # ----
        
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # Store into summary file as output.
    if write_summary:
        with open(path2output+params_dict['model_name']+'_summary.txt', 'w') as f:
            # header
            f.writelines('\
# =============================================================================\n\
# Summary of parameters in the \'%s\' run. Values are in units of [Myr, Msun, pc].\n\
# Created at %s.\n\
# =============================================================================\n\n\
'%(params_dict['model_name'], dt_string))
            # body
            for key, val in params_dict.items():
                f.writelines(key+'    '+"".join(str(val))+'\n')
            # close
            f.close()
   
    # =============================================================================
    # This section contains information which you do not want to appear in the 
    # final summary.txt file, but want it included. Mostly mini-conversions.
    # =============================================================================
            
    # Here we deal with tStop based on units. If the unit is Myr, then
    # tStop is tStop. If it is in free-frall time, then we need to change
    # to a proper unit of time. Add a parameter 'tff' that calculates the 
    # free-fall time. This may be used even if stop_t_unit is not in tff time, 
    # as it may be called if mult_SF = 2, where the second startburst is
    # characterised by the free-fall time.
    tff = np.sqrt(3. * np.pi / (32. * c.G.cgs.value * params_dict['dens_navg_pL'] * params_dict['mu_n'])) * u.s.to(u.Myr)
    params_dict['tff'] = float(tff)
    if params_dict['stop_t_unit'] == 'tff':
        params_dict['stop_t'] = params_dict['stop_t'] * params_dict['tff']
    elif params_dict['stop_t_unit'] == 'Myr':
        params_dict['stop_t'] = params_dict['stop_t'] 
    # if params_dict['stop_t_unit'] == 'Myr', it is fine; Myr is also the 
    # unit in all other calculations.
    
    # ----
    
    # Here we include calculations for mCloud, for future ease.
    params_dict['mCloud'] = 10**params_dict['log_mCloud']
    
    # Here we calculate mCloud, depending on whether the user specifies of this
    # is given as before or after the star formation event.
    if params_dict['is_mCloud_beforeSF'] == True:
        mCloud = params_dict['mCloud']
    else:
        mCloud = params_dict['mCloud'] / (1 - params_dict['sfe'])
    # cloud mass before SF. Saved as another parameter.
    params_dict['mCloud_beforeSF'] = mCloud
    # cluster mass
    mCluster = mCloud * params_dict['sfe']
    # cloud mass after SF
    mCloud = mCloud - mCluster
    # save
    params_dict['mCloud'] = mCloud
    params_dict['mCluster'] = mCluster

    # ----

    # Here for the magnatic field related constants
    params_dict['BMW'] = 10**params_dict['log_BMW']
    params_dict['nMW'] = 10**params_dict['log_nMW']
    
    # TODO: this seems unnecessary
    # Is there a density gradient?
    params_dict['density_gradient'] = float((params_dict['dens_profile'] == 'pL_prof') and (params_dict['dens_a_pL'] != 0))
    
    # =============================================================================
    # Save output to yaml. This contains parameters in which you do not wish
    # user to see in the output summary.txt.
    # =============================================================================
    # relative path to yaml
    path2yaml = r'./param/'
    # Write this into a file
    filename =  path2output + params_dict['model_name'] + '_config.yaml'
    with open(filename, 'w',) as file :
        # header
        file.writelines('\
# =============================================================================\n\
# Summary of parameters in the \'%s\' run.\n\
# Created at %s.\n\
# =============================================================================\n\n\
'%(params_dict['model_name'], dt_string))
        yaml.dump(params_dict, file, sort_keys=False) 
    
    # save path to object
    # TODO: delete this after warpfield is finished.
    # save file
    os.environ['PATH_TO_CONFIG'] = filename
    
    
    # =============================================================================
    # Try writting a settings.py file
    # It is also here that we append units
    # =============================================================================
        
    settings_name = path2output+params_dict['model_name']+'_settings.py'
    # TODO: remove thsi. the unit is mkaing everything much complicated especially
    # when new parameters are added. 
    with open(settings_name, 'w') as f:
            # header
            f.writelines('\
# =============================================================================\n\
# Summary of parameters in the \'%s\' run.\n\
# Created at %s.\n\
# =============================================================================\n\n\
#-- import library for units\n\
import astropy.units as u\n\n\n\
'%(params_dict['model_name'], dt_string))
            # body
            for key, val in params_dict.items():
                try:
                    if key == 'model_name':
                        # make sure it is string
                        f.writelines(f'{key} = "{val}"\n')
                    else:
                        float(val)
                        #check if value can be changed into float (therefore could require units)
                        # a series of tests
                        if key == 'log_mCloud':
                            f.writelines(f'{key} = {float(val)} * u.M_sun\n')
                            
                        elif key == 'nCore':
                            f.writelines(f'{key} = {float(val)} / u.cm**3\n')
                            
                        elif key == 'rCore':
                            f.writelines(f'{key} = {float(val)} * u.pc\n')
                            
                        elif key == 'rand_log_mCloud':
                            f.writelines(f'{key} = {float(val)} * u.M_sun\n')
                            
                        elif key == 'rand_n_cloud':
                            f.writelines(f'{key} = {float(val)} / u.cm**3\n')
                            
                        elif key == 'r_coll':
                            f.writelines(f'{key} = {float(val)} * u.pc\n')
                            
                        elif key == 'SB99_mass':
                            f.writelines(f'{key} = {float(val)} * u.M_sun\n')
                            
                        elif key == 'SB99_BHCUT':
                            f.writelines(f'{key} = {float(val)} * u.M_sun\n')
                            
                        elif key == 'SB99_age_min':
                            f.writelines(f'{key} = {float(val)} * u.yr\n')
                            
                        elif key == 'v_SN':
                            f.writelines(f'{key} = {float(val)} * u.km/u.s\n')
                            
                        elif key == 'dens_navg_pL':
                            f.writelines(f'{key} = {float(val)} / u.cm**3\n')
                            
                        elif key == 'stop_n_diss':
                            f.writelines(f'{key} = {float(val)} / u.cm**3  \n')
                            
                        elif key == 'stop_t_diss':
                                f.writelines(f'{key} = {float(val)} * u.Myr\n')
                            
                        elif key == 'stop_r':
                            f.writelines(f'{key} = {float(val)} * u.pc\n')
                            
                        elif key == 'stop_v':
                            f.writelines(f'{key} = {float(val)} * u.km/u.s\n')
                            
                        elif key == 'stop_t':
                            if params_dict['stop_t_unit'] == 'Myr':
                                f.writelines(f'{key} = {float(val)} * u.Myr\n')
                            else:
                                f.writelines(f'{key} = {float(val)}\n')
                                
                        elif key == 'phase_Emin':
                            f.writelines(f'{key} = {float(val)} * u.erg\n')
                            
                        elif key == 'sigma0':
                            f.writelines(f'{key} = {float(val)} * u.cm**2\n')
                            
                        elif key == 'mu_n':
                            f.writelines(f'{key} = {float(val)} * u.g\n')
                            
                        elif key == 'mu_p':
                            f.writelines(f'{key} = {float(val)} * u.g\n')
                            
                        elif key == 'TShell_ion':
                            f.writelines(f'{key} = {float(val)} * u.K\n')
    
                        elif key == 'TShell_neu':
                            f.writelines(f'{key} = {float(val)} * u.K\n')
    
                        elif key == 'nISM':
                            f.writelines(f'{key} = {float(val)} / u.cm**3\n')
    
                        elif key == 'kappa_IR':
                            f.writelines(f'{key} = {float(val)} * u.cm**2 / u.g\n')
    
                        elif key == 'alpha_B':
                            f.writelines(f'{key} = {float(val)} * u.cm**3 / u.s\n')
    
                        elif key == 'c_therm':
                            f.writelines(f'{key} = {float(val)} * u.erg / u.cm / u.s * u.K**(-7/2)\n')
    
                        elif key == 'sigma_d':
                            f.writelines(f'{key} = {float(val)} * u.cm**2\n')
    
                        elif key == 'tff':
                            f.writelines(f'{key} = {float(val)} * u.Myr\n')
    
                        elif key == 'mCloud':
                            f.writelines(f'{key} = {float(val)} * u.M_sun\n')
    
                        elif key == 'mCluster':
                            f.writelines(f'{key} = {float(val)} * u.M_sun\n')
    
                        elif key == 'T_r2Prime':
                            f.writelines(f'{key} = {float(val)} * u.K\n')
                        
                        # simply write without units
                        else:
                            f.writelines(f'{key} = {float(val)}\n')

                except ValueError:
                    # strings dont need units
                    f.writelines(key+'='+'\"'+"".join(str(val))+'\"'+'\n')
            # close
            f.close()
    
    # TODO: perhaps check if user input would cause crash, e.g., when not absolute path is given.
    # convert to relative path
    # changes output/params/settings.py into output.params.settings for future module import.
    settings_module = os.path.relpath(settings_name, os.getcwd()).replace('/', '.')
    
    os.environ['TRINITY_SETTING_MODULE'] = settings_module[:-3]
    
    return params_dict



class ParameterFileError(Exception):
    """Raised when a parameter file entry is invalid."""






