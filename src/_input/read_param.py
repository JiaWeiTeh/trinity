#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:33:31 2022
Rewritten: January 2026 - Complete restructure with robust parsing and unit handling

@author: Jia Wei Teh

This script reads parameter files and creates a DescribedDict for TRINITY simulations.

Key features:
- Reads default.param for all parameter definitions with INFO/UNIT metadata
- User .param file overrides defaults (missing parameters use defaults)
- Automatic unit conversion to [Msun, pc, Myr] via unit_conversions.py
- Inline comment support (e.g., "mCloud 1e6 # cloud mass")
- Robust error handling with line numbers and helpful messages
- Each parameter stored as DescribedItem(value, info, ori_units)
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from fractions import Fraction
import numpy as np
import src._functions.unit_conversions as cvt
from src._input.dictionary import DescribedItem, DescribedDict

# Initialize logger for this module
logger = logging.getLogger(__name__)


def read_param(path2file, write_summary=True):
    """
    Read parameter file and return DescribedDict with all TRINITY parameters.
    
    Parameters
    ----------
    path2file : str or Path
        Path to the user .param file.
    write_summary : bool, optional
        Whether to write a summary .txt file in the output directory.
    
    Returns
    -------
    params : DescribedDict
        Dictionary of all parameters as DescribedItem objects.
        Access values via: params['mCloud'].value
        Access info via: params['mCloud'].info
        Access units via: params['mCloud'].ori_units
    
    Raises
    ------
    ParameterFileError
        If parameter file has formatting errors or invalid parameters.
    FileNotFoundError
        If default.param cannot be found.
    """
    
    # =============================================================================
    # Helper function: parse value from string
    # =============================================================================
    
    def parse_value(val_str):
        """
        Parse a string value into appropriate Python type.
        
        Precedence: boolean → number → fraction → string
        """
        val_str = val_str.strip()
        
        # Boolean
        if val_str.lower() == 'true':
            return True
        elif val_str.lower() == 'false':
            return False
        
        # Number (float or int)
        try:
            return float(val_str)
        except ValueError:
            pass
        
        # Fraction (e.g., 5/3)
        try:
            return float(Fraction(val_str))
        except (ValueError, ZeroDivisionError):
            pass
        
        # String (fallback)
        return val_str
    
    # =============================================================================
    # Step 1: Read default.param with INFO and UNIT metadata
    # =============================================================================
    
    # Get path to default.param relative to this script
    script_dir = Path(__file__).parent.resolve()
    path2default = script_dir.parent.parent / 'param' / 'default.param'
    
    if not path2default.exists():
        raise FileNotFoundError(
            f"Default parameter file not found at: {path2default}\n"
            f"Expected: <trinity_root>/param/default.param"
        )
    
    # Storage: key -> (info, unit, default_value)
    default_dict = {}
    
    with open(path2default, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        current_info = None
        current_unit = None
        
        for line_num, line in enumerate(lines, start=1):
            # Remove inline comments
            if '#' in line:
                comment_pos = line.find('#')
                before_comment = line[:comment_pos].strip()
                full_line = line.strip()
                
                # Check if this is an INFO or UNIT line
                if full_line.startswith('# INFO:'):
                    current_info = full_line[len('# INFO:'):].strip()
                    continue
                elif full_line.startswith('# UNIT:'):
                    current_unit = full_line[len('# UNIT:'):].strip()
                    # Remove surrounding brackets if present
                    current_unit = current_unit.strip('[]').strip()
                    continue
                else:
                    # Regular line with inline comment
                    line = before_comment
            else:
                line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse parameter line (format: key value)
            parts = line.split(None, 1)  # Split on first whitespace only
            
            if len(parts) != 2:
                continue  # Skip malformed lines in default.param
            
            key, val_str = parts
            value = parse_value(val_str)
            
            # Store with metadata
            info = current_info if current_info else "INFO not specified"
            unit = current_unit if current_unit else None
            default_dict[key] = (info, unit, value)
            
            # Reset metadata for next parameter
            current_info = None
            current_unit = None
    
    logger.debug(f"Loaded {len(default_dict)} parameters from default.param")
    
    # =============================================================================
    # Step 2: Read user parameter file
    # =============================================================================
    
    user_dict = {}
    
    with open(path2file, 'r', encoding='utf-8') as f:
        filename = Path(f.name).stem
        
        for line_num, line in enumerate(f, start=1):
            # Remove inline comments
            if '#' in line:
                line = line[:line.find('#')]
            
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse parameter line
            parts = line.split(None, 1)
            
            if len(parts) != 2:
                raise ParameterFileError(
                    f"{Path(path2file).name}, line {line_num}: "
                    f"Expected format 'key value', got: '{line}'"
                )
            
            key, val_str = parts
            value = parse_value(val_str)
            user_dict[key] = value
    
    logger.debug(f"Loaded {len(user_dict)} parameters from {Path(path2file).name}")
    
    # =============================================================================
    # Step 3: Validate user parameters and merge with defaults
    # =============================================================================
    
    # Check that all user-specified keys exist in default.param
    invalid_keys = []
    for key in user_dict.keys():
        if key not in default_dict:
            invalid_keys.append(key)
    
    if invalid_keys:
        available = ', '.join(sorted(default_dict.keys())[:10])
        raise ParameterFileError(
            f"Invalid parameter(s) in {Path(path2file).name}: {', '.join(invalid_keys)}\n"
            f"Available parameters include: {available}..."
        )
    
    # Merge: user values override defaults
    merged_dict = {}
    for key, (info, unit, default_val) in default_dict.items():
        if key in user_dict:
            # User specified this parameter
            value = user_dict[key]
            merged_dict[key] = (info, unit, value)
        else:
            # Use default
            merged_dict[key] = (info, unit, default_val)
    
    # Report which parameters were overridden
    overridden = [k for k in user_dict.keys()]
    if overridden:
        logger.debug(f"Overridden {len(overridden)} parameters from user file")
    
    # =============================================================================
    # Step 4: Create DescribedDict with unit conversions
    # =============================================================================
    
    params = DescribedDict()
    
    for key, (info, unit, value) in merged_dict.items():
        # Convert units to astronomy units [Msun, pc, Myr]
        # Only convert numeric values; strings remain unchanged
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            conversion_factor = cvt.convert2au(unit)
            converted_value = value * conversion_factor
        else:
            # Strings, booleans, etc. don't get unit conversion
            converted_value = value
        
        # Create DescribedItem
        unit_str = unit if unit else "UNIT not specified"
        params[key] = DescribedItem(
            value=converted_value,
            info=info,
            ori_units=unit_str
        )

    
    logger.debug(f"Created DescribedDict with {len(params)} parameters")
    
    # =============================================================================
    # Step 5: Validate critical parameters
    # =============================================================================
    
    # Check metallicity
    if params['ZCloud'].value != 1:
        raise ParameterFileError(
            f"Metallicity Z={params['ZCloud'].value} not implemented. "
            f"Currently only Z=1 (solar) is supported."
        )
    
    # Validate density profile
    if params['dens_profile'].value not in ['densBE', 'densPL']:
        raise ParameterFileError(
            f"Invalid dens_profile '{params['dens_profile'].value}'. "
            f"Must be 'densBE' or 'densPL'."
        )
    
    # =============================================================================
    # Step 6: Compute derived parameters
    # =============================================================================
    
    # Dust cross-section scaling with metallicity
    if params['ZCloud'].value >= params['dust_noZ'].value:
        params['dust_sigma'].value = params['dust_sigma'].value * params['ZCloud'].value
    else:
        params['dust_sigma'].value = 0
    
    # Use filename as model name if not specified
    if params['model_name'].value == "default":
        params['model_name'].value = filename
    
    # Cluster and cloud masses after star formation
    mCluster = params['mCloud'].value * params['sfe'].value
    mCloud_after_SF = params['mCloud'].value - mCluster
    params['mCloud'].value = mCloud_after_SF
    params['mCluster'] = DescribedItem(
        value=mCluster,
        info="Cluster mass (mCloud * sfe)",
        ori_units="Msun"
    )
    
    # =============================================================================
    # Step 7: Set up directory paths
    # =============================================================================
    
    # Output directory
    if params['path2output'].value == 'def_dir':
        path2output = os.path.join(os.getcwd(), 'outputs', params['model_name'].value)
        Path(path2output).mkdir(parents=True, exist_ok=True)
        params['path2output'].value = path2output
    else:
        path2output = str(params['path2output'].value)
        Path(path2output).mkdir(parents=True, exist_ok=True)
        params['path2output'].value = path2output
    
    # Cooling directory - non-CIE
    if params['path_cooling_nonCIE'].value == 'def_dir':
        params['path_cooling_nonCIE'].value = os.path.join(os.getcwd(), 'lib/cooling/opiate/')
    else:
        path_cooling = str(params['path_cooling_nonCIE'].value)
        Path(path_cooling).mkdir(parents=True, exist_ok=True)
        params['path_cooling_nonCIE'].value = path_cooling
    
    # Cooling directory - CIE
    if params['ZCloud'].value == 1:
        cie_files = {
            1: 'lib/cooling/CIE/coolingCIE_1_Cloudy.dat',
            2: 'lib/cooling/CIE/coolingCIE_2_Cloudy_grains.dat',
            3: 'lib/cooling/CIE/coolingCIE_3_Gnat-Ferland2012.dat'
        }
        cie_choice = int(params['path_cooling_CIE'].value)
        if cie_choice in cie_files:
            params['path_cooling_CIE'].value = os.path.join(os.getcwd(), cie_files[cie_choice])
    elif params['ZCloud'].value == 0.15:
        params['path_cooling_CIE'].value = os.path.join(
            os.getcwd(), 'lib/cooling/CIE/coolingCIE_4_Sutherland-Dopita1993.dat'
        )
    
    # Starburst99 directory
    if params['path_sps'].value == 'def_dir':
        params['path_sps'].value = os.path.join(os.getcwd(), 'lib/sps/starburst99/')
    else:
        path_sps = str(params['path_sps'].value)
        Path(path_sps).mkdir(parents=True, exist_ok=True)
        params['path_sps'].value = path_sps
    
    # =============================================================================
    # Step 8: Handle density profile-specific parameters
    # =============================================================================
    
    if params['dens_profile'].value == 'densBE':
        # Bonnor-Ebert sphere
        params.pop('densPL_alpha')
        params['densBE_Omega'].exclude_from_snapshot = True
        
        # Add BE-specific runtime parameters
        params['densBE_Teff'] = DescribedItem(0, info="Effective temperature of BE sphere", ori_units="K")
        params['densBE_xi_arr'] = DescribedItem([], info="Lane-Emden xi array", ori_units="dimensionless")
        params['densBE_u_arr'] = DescribedItem([], info="Lane-Emden u array", ori_units="dimensionless")
        params['densBE_dudxi_arr'] = DescribedItem([], info="Lane-Emden du/dxi array", ori_units="dimensionless")
        params['densBE_rho_rhoc_arr'] = DescribedItem([], info="Density contrast array", ori_units="dimensionless")
        params['densBE_f_rho_rhoc'] = DescribedItem(0, info="Interpolation function for density contrast", ori_units="dimensionless")
    
    elif params['dens_profile'].value == 'densPL':
        # Power-law
        params.pop('densBE_Omega')
        params['densPL_alpha'].exclude_from_snapshot = True
    
    # =============================================================================
    # Step 9: Set snapshot exclusions for constants
    # =============================================================================
    
    # Only track time-varying quantities in snapshots
    # Exclude initial conditions and constants to save memory
    time_varying_keys = ['model_name', 'mCloud', 'cool_alpha', 'cool_beta', 'cool_delta']
    
    for key, val in params.items():
        if key not in time_varying_keys:
            val.exclude_from_snapshot = True
    
    # =============================================================================
    # Step 10: Initialize runtime parameters (not from .param files)
    # =============================================================================
    
    # Simulation state
    params['current_phase'] = DescribedItem('', info="Current simulation phase: energy/implicit/transition/momentum", ori_units="N/A")
    params['EndSimulationDirectly'] = DescribedItem(False, info="Flag to immediately end simulation", ori_units="N/A")
    params['SimulationEndReason'] = DescribedItem('', info="Reason for simulation completion", ori_units="N/A")
    params['EarlyPhaseApproximation'] = DescribedItem(True, info="Using approximations for early phase?", ori_units="N/A")
    
    # Time tracking
    params['tSF'] = DescribedItem(0, info="Time of star formation", ori_units="Myr")
    params['t_now'] = DescribedItem(0, info="Current simulation time", ori_units="Myr")
    
    # Main bubble parameters
    params['v2'] = DescribedItem(0, info="Velocity at outer bubble/inner shell radius", ori_units="pc/Myr")
    params['R2'] = DescribedItem(0, info="Outer bubble/inner shell radius", ori_units="pc")
    params['T0'] = DescribedItem(0, info="Bubble temperature at R2", ori_units="K")
    params['Eb'] = DescribedItem(0, info="Bubble energy", ori_units="Msun*pc**2/Myr**2")
    params['R1'] = DescribedItem(0, info="Inner bubble radius", ori_units="pc")
    params['Pb'] = DescribedItem(0, info="Bubble pressure", ori_units="Msun/Myr**2/pc")
    params['c_sound'] = DescribedItem(0, info="Sound speed", ori_units="pc/Myr")
    
    # Arrays for shell mass interpolation
    params['t_next'] = DescribedItem(0, info="Next time for mShell interpolation", ori_units="Myr", exclude_from_snapshot=False)
    
    # Cloud and shell geometry
    params['rCloud'] = DescribedItem(0, info="Cloud radius", ori_units="pc")
    params['rShell'] = DescribedItem(0, info="Shell outer radius", ori_units="pc")
    params['nEdge'] = DescribedItem(0, info="Number density at cloud edge", ori_units="1/pc**3")
    
    # Feedback from Starburst99
    params['SB99_data'] = DescribedItem(0, info="SB99 datacube", ori_units="N/A", exclude_from_snapshot=True)
    params['SB99f'] = DescribedItem(0, info="SB99 interpolation function", ori_units="N/A", exclude_from_snapshot=True)
    params['Lmech_W'] = DescribedItem(0, info="Wind mechanical luminosity", ori_units="Msun*pc**2/Myr**3")
    params['Lmech_SN'] = DescribedItem(0, info="SN mechanical luminosity", ori_units="Msun*pc**2/Myr**3")
    params['Lmech_total'] = DescribedItem(0, info="Total mechanical luminosity", ori_units="Msun*pc**2/Myr**3")
    params['v_mech_total'] = DescribedItem(0, info="mechanical velocity (winds+SN)", ori_units="pc/Myr")
    params['pdot_W'] = DescribedItem(0, info="Wind momentum rate", ori_units="Msun*pc/Myr**2")
    params['pdot_SN'] = DescribedItem(0, info="Supernova momentum rate", ori_units="Msun*pc/Myr**2")
    params['pdot_total'] = DescribedItem(0, info="Total momentum rate", ori_units="Msun*pc/Myr**2")
    params['pdotdot_total'] = DescribedItem(0, info="Rate of wind momentum rate", ori_units="Msun*pc/Myr**3")
    params['Qi'] = DescribedItem(0, info="Ionizing photon rate", ori_units="1/Myr")
    params['Lbol'] = DescribedItem(0, info="Bolometric luminosity", ori_units="Msun*pc**2/Myr**3")
    params['Ln'] = DescribedItem(0, info="Non-ionizing luminosity", ori_units="Msun*pc**2/Myr**3")
    params['Li'] = DescribedItem(0, info="Ionizing luminosity", ori_units="Msun*pc**2/Myr**3")
    
    # Cooling
    params['t_previousCoolingUpdate'] = DescribedItem(1e30, info="Time of previous cooling update", ori_units="Myr")
    params['cStruc_cooling_nonCIE'] = DescribedItem(0, info="Non-CIE cooling cube", ori_units="N/A", exclude_from_snapshot=True)
    params['cStruc_heating_nonCIE'] = DescribedItem(0, info="Non-CIE heating cube", ori_units="N/A", exclude_from_snapshot=True)
    params['cStruc_net_nonCIE_interpolation'] = DescribedItem(0, info="Non-CIE net cooling interpolation", ori_units="N/A", exclude_from_snapshot=True)
    params['cStruc_cooling_CIE_logT'] = DescribedItem(0, info="CIE log temperature array", ori_units="N/A", exclude_from_snapshot=True)
    params['cStruc_cooling_CIE_logLambda'] = DescribedItem(0, info="CIE log lambda array", ori_units="N/A", exclude_from_snapshot=True)
    params['cStruc_cooling_CIE_interpolation'] = DescribedItem(0, info="CIE cooling interpolation", ori_units="N/A", exclude_from_snapshot=True)
    
    # Shell properties
    params['shell_fAbsorbedIon'] = DescribedItem(1, info="Fraction of absorbed ionizing radiation", ori_units="dimensionless")
    params['shell_fAbsorbedNeu'] = DescribedItem(0, info="Fraction of absorbed non-ionizing radiation", ori_units="dimensionless")
    params['shell_fAbsorbedWeightedTotal'] = DescribedItem(0, info="Total absorption fraction (luminosity weighted)", ori_units="dimensionless")
    params['shell_fIonisedDust'] = DescribedItem(0, info="Ionized dust fraction", ori_units="dimensionless")
    params['shell_F_rad'] = DescribedItem(0, info="Radiation pressure on shell", ori_units="Msun*pc/Myr**2")
    params['shell_thickness'] = DescribedItem(0, info="Shell thickness", ori_units="pc")
    params['shell_nMax'] = DescribedItem(0, info="Maximum shell density", ori_units="1/pc**3")
    params['shell_tauKappaRatio'] = DescribedItem(0, info="tau_IR / kappa_IR ratio", ori_units="Msun/pc**2")
    params['shell_grav_r'] = DescribedItem(np.array([]), info="Radius array for gravitational calculations", ori_units="pc")
    params['shell_grav_phi'] = DescribedItem(np.array([]), info="Gravitational potential", ori_units="pc**2/Myr**2")
    params['shell_grav_force_m'] = DescribedItem(np.array([]), info="Gravitational force per unit mass", ori_units="pc/Myr**2")
    params['shell_mass'] = DescribedItem(0, info="Shell mass", ori_units="Msun")
    params['shell_massDot'] = DescribedItem(0, info="Shell mass accretion rate", ori_units="Msun/Myr")
    params['shell_interpolate_massDot'] = DescribedItem(False, info="Use shell mass interpolation?", ori_units="N/A")
    params['shell_n0'] = DescribedItem(0, info="Shell inner density (pressure balance)", ori_units="1/pc**3")
    
    # Forces on shell
    params['F_grav'] = DescribedItem(0, info="Gravitational force", ori_units="Msun*pc/Myr**2")
    params['F_SN'] = DescribedItem(0, info="Supernova force", ori_units="Msun*pc/Myr**2")
    params['F_ram'] = DescribedItem(0, info="Ram pressure force (from Pb-Eb relation)", ori_units="Msun*pc/Myr**2")
    params['F_ram_wind'] = DescribedItem(0, info="Wind ram pressure force (from SB99)", ori_units="Msun*pc/Myr**2")
    params['F_ram_SN'] = DescribedItem(0, info="SN ram pressure force (from SB99)", ori_units="Msun*pc/Myr**2")
    params['F_wind'] = DescribedItem(0, info="Wind force", ori_units="Msun*pc/Myr**2")
    params['F_ion_in'] = DescribedItem(0, info="Inward photoionization pressure", ori_units="Msun*pc/Myr**2")
    params['F_ion_out'] = DescribedItem(0, info="Outward photoionization pressure", ori_units="Msun*pc/Myr**2")
    params['F_rad'] = DescribedItem(0, info="Radiation pressure", ori_units="Msun*pc/Myr**2")
    params['F_ISM'] = DescribedItem(0, info="ISM pressure", ori_units="Msun*pc/Myr**2")
    
    # Bubble structure
    params['bubble_LTotal'] = DescribedItem(0, info="Total luminosity lost to cooling", ori_units="Msun*pc**2/Myr**3")
    params['bubble_L1Bubble'] = DescribedItem(0, info="Cooling in bubble zone", ori_units="Msun*pc**2/Myr**3")
    params['bubble_L2Conduction'] = DescribedItem(0, info="Cooling in conduction zone", ori_units="Msun*pc**2/Myr**3")
    params['bubble_L3Intermediate'] = DescribedItem(0, info="Cooling in intermediate zone", ori_units="Msun*pc**2/Myr**3")
    params['bubble_Tavg'] = DescribedItem(0, info="Average bubble temperature", ori_units="K")
    params['bubble_mass'] = DescribedItem(0, info="Bubble mass", ori_units="Msun")
    params['bubble_r_Tb'] = DescribedItem(0, info="Radius at bubble_xi_Tb * R2", ori_units="pc")
    params['bubble_T_r_Tb'] = DescribedItem(0, info="Temperature at r_Tb", ori_units="K")
    
    # Bubble structure arrays
    params['bubble_r_arr'] = DescribedItem(np.array([]), info="Bubble radius structure", ori_units="pc")
    params['bubble_v_arr'] = DescribedItem(np.array([]), info="Bubble velocity structure", ori_units="pc/Myr")
    params['bubble_T_arr'] = DescribedItem(np.array([]), info="Bubble temperature structure", ori_units="K")
    params['bubble_dTdr_arr'] = DescribedItem(np.array([]), info="Bubble temperature gradient", ori_units="K/pc")
    params['bubble_n_arr'] = DescribedItem(np.array([]), info="Bubble density structure", ori_units="1/pc**3")
    params['bubble_dMdtGuess'] = DescribedItem(0, info="Bubble dM/dt guess", ori_units="Msun/Myr")
    params['bubble_dMdt'] = DescribedItem(np.nan, info="Bubble mass loss rate (thermal conduction)", ori_units="Msun/Myr")
    params['bubble_Lgain'] = DescribedItem(np.nan, info="Luminosity gain from winds", ori_units="Msun*pc**2/Myr**3")
    params['bubble_Lloss'] = DescribedItem(np.nan, info="Luminosity loss from cooling/leaking", ori_units="Msun*pc**2/Myr**3")
    params['bubble_Leak'] = DescribedItem(0, info="Leaking luminosity", ori_units="Msun*pc**2/Myr**3")
    
    # State flags
    params['isCollapse'] = DescribedItem(False, info="Is cloud collapsing?", ori_units="N/A")
    params['isDissolved'] = DescribedItem(False, info="Has shell dissolved?", ori_units="N/A")
    params['is_fullyIonised'] = DescribedItem(False, info="Is shell fully ionised?", ori_units="N/A")
    
    # Diagnostic residuals
    params['residual_deltaT'] = DescribedItem(0, info="Temperature residual (T1-T2)/T2", ori_units="dimensionless")
    params['residual_betaEdot'] = DescribedItem(0, info="Energy rate residual", ori_units="dimensionless")
    params['residual_Edot1_guess'] = DescribedItem(np.nan, info="Edot from beta", ori_units="Msun*pc**2/Myr**3")
    params['residual_Edot2_guess'] = DescribedItem(np.nan, info="Edot from energy balance", ori_units="Msun*pc**2/Myr**3")
    params['residual_T1_guess'] = DescribedItem(np.nan, info="T from bubble_Trgoal", ori_units="K")
    params['residual_T2_guess'] = DescribedItem(np.nan, info="T from T0", ori_units="K")
    
    # =============================================================================
    # Step 11: Write summary file
    # =============================================================================
    
    if write_summary:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        summary_path = os.path.join(path2output, f"{params['model_name'].value}_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# {'=' * 77}\n")
            f.write(f"# Summary of parameters for run: '{params['model_name'].value}'\n")
            f.write(f"# Units: [Msun, pc, Myr] unless otherwise specified\n")
            f.write(f"# Created: {dt_string}\n")
            f.write(f"# {'=' * 77}\n\n")
            
            for key, item in params.items():
                f.write(f"{key:<30}  {item.value}\n")
        
        logger.info(f"Summary written to: {summary_path}")
    
    return params


class ParameterFileError(Exception):
    """Raised when parameter file has formatting or validation errors."""
    pass


# =============================================================================
# Quick test (commented out)
# =============================================================================
if __name__ == "__main__":
    # Configure logging for standalone test
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    params = read_param('param/Orion_M43_EON.param')
    logger.info(f"mCloud = {params['mCloud'].value} {params['mCloud'].ori_units}")
    logger.info(f"  Info: {params['mCloud'].info}")
