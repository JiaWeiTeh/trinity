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
from pathlib import Path
from fractions import Fraction
import numpy as np
import src._functions.unit_conversions as cvt
from src._input.dictionary import DescribedItem, DescribedDict
from src._input.errors import ParameterFileError
from src._input.registry import validate_all
import src.sps.sps_columns as sps_columns

# Anchor bundled-asset lookups to the repo root, not the CWD: users may launch
# run.py from anywhere, and the `lib/default/...` defaults must still resolve.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Initialize logger for this module
logger = logging.getLogger(__name__)


def read_param(path2file):
    """
    Read parameter file and return DescribedDict with all TRINITY parameters.

    Parameters
    ----------
    path2file : str or Path
        Path to the user .param file.

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

        Precedence: None → boolean → number → fraction → string
        """
        val_str = val_str.strip()

        # None
        if val_str.lower() == 'none':
            return None

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
    
    # Get path to default.param (the schema + defaults file lives next to this
    # script in src/_input/, not in the user-facing param/ directory).
    script_dir = Path(__file__).parent.resolve()
    path2default = script_dir / 'default.param'

    if not path2default.exists():
        raise FileNotFoundError(
            f"Default parameter file not found at: {path2default}\n"
            f"Expected: <trinity_root>/src/_input/default.param"
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
        # Only convert numeric values; strings, booleans, and None remain unchanged
        if value is None:
            # None values (e.g., for disabled termination conditions) pass through
            converted_value = None
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
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

    # Snapshot: record the DescribedItem instances that originated from
    # default.param.  Later steps (6/8/10) add runtime-only parameters and
    # must NOT silently replace any of these — if they do, the user's value
    # from their .param file is lost.  The post-Step-10 guard below compares
    # object identity to catch that drift loudly.  Value mutations (e.g.
    # params['mCloud'].value = ...) don't replace the DescribedItem and
    # are legitimate.
    _default_items_before = {k: params[k] for k in default_dict if k in params}

    # =============================================================================
    # Step 5: Validate critical parameters (driven by the registry)
    # =============================================================================
    # Each spec carrying a ``validator`` callable is invoked here.  Validators
    # may raise ``ParameterFileError`` on bad input and/or normalize the value
    # in place (e.g. coerce whole-number floats to int).  See the validator
    # definitions in ``src/_input/registry.py``.
    validate_all(params)


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
    
    # Cluster and cloud masses after star formation.
    #
    # NOTE: params['mCloud'] is rebound here.  Upstream of this block —
    # in the .param file and the folder name — mCloud is the pre-SFE
    # input GMC mass.  Downstream — throughout the simulation, in
    # metadata.json, and in every rehydrated snapshot — it is the
    # post-SFE residual cloud mass.  The pre-SFE input is preserved as
    # mCloud_input and the star-formed portion as mCluster; invariant:
    # mCloud_input == mCloud + mCluster.  Downstream analysis that
    # wants the input value should read mCloud_input, not back out
    # mCloud / (1 - sfe).
    mCloud_input_value = params['mCloud'].value
    mCluster = mCloud_input_value * params['sfe'].value
    mCloud_after_SF = mCloud_input_value - mCluster
    params['mCloud'].value = mCloud_after_SF
    params['mCloud_input'] = DescribedItem(
        value=mCloud_input_value,
        info=("Pre-SFE input cloud mass (= mCloud + mCluster). "
              "Matches the .param file and the sweep folder-name tag."),
        ori_units="Msun"
    )
    params['mCluster'] = DescribedItem(
        value=mCluster,
        info="Cluster mass (mCloud_input * sfe)",
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
    
    # Cooling directory - non-CIE.
    # Default sentinel 'def_dir' resolves to the shipped OPIATE cube folder
    # under lib/default/opiate/.
    if params['path_cooling_nonCIE'].value == 'def_dir':
        params['path_cooling_nonCIE'].value = str(_REPO_ROOT / 'lib' / 'default' / 'opiate') + os.sep
    else:
        path_cooling = str(params['path_cooling_nonCIE'].value)
        Path(path_cooling).mkdir(parents=True, exist_ok=True)
        params['path_cooling_nonCIE'].value = path_cooling

    # Cooling directory - CIE.
    # Integer-index preset {1, 2, 3} (under ZCloud == 1) selects between the
    # bundled CIE tables; ZCloud == 0.15 auto-pins to the Sutherland-Dopita
    # file. All resolved paths live under lib/default/CIE/.
    if params['ZCloud'].value == 1:
        cie_files = {
            1: 'lib/default/CIE/coolingCIE_1_Cloudy.dat',
            2: 'lib/default/CIE/coolingCIE_2_Cloudy_grains.dat',
            3: 'lib/default/CIE/coolingCIE_3_Gnat-Ferland2012.dat'
        }
        cie_choice = int(params['path_cooling_CIE'].value)
        if cie_choice in cie_files:
            params['path_cooling_CIE'].value = str(_REPO_ROOT / cie_files[cie_choice])
    elif params['ZCloud'].value == 0.15:
        params['path_cooling_CIE'].value = str(
            _REPO_ROOT / 'lib' / 'default' / 'CIE' / 'coolingCIE_4_Sutherland-Dopita1993.dat'
        )
    
    # sps_path: full path to the SPS data file. Sentinel 'def_path' resolves
    # to the bundled lib/default/sps/starburst99/1e6cluster_default.csv —
    # an SB99 grid at rotation=1, ZCloud=1, mass=1e6 Msun in CSV form with
    # the canonical 7-column SB99 layout (DEFAULT_SPS_COLUMN_MAP). The
    # default rejects combinations the bundled cooling tables can't fulfill;
    # users who need a different metallicity or rotation must set sps_path
    # explicitly. See analysis/sb99-refactor-audit.md §9.
    sps_path_is_default = params['sps_path'].value == 'def_path'
    if sps_path_is_default:
        if params['ZCloud'].value != 1.0:
            raise ValueError(
                f"ZCloud={params['ZCloud'].value} is not supported with the "
                "default SPS fallback (only ZCloud=1.0 is bundled). Set "
                "sps_path explicitly to use a non-solar metallicity SPS file."
            )
        if not params['SB99_rotation'].value:
            raise ValueError(
                "SB99_rotation=0 is not supported with the default SPS "
                "fallback (only rot cooling tables are bundled). Set "
                "sps_path explicitly and supply matching cooling tables "
                "for the norot case."
            )
        params['sps_path'].value = str(
            _REPO_ROOT / 'lib' / 'default' / 'sps' / 'starburst99' / '1e6cluster_default.csv'
        )
        column_map = sps_columns.DEFAULT_SPS_COLUMN_MAP
        logger.info(
            f"sps_path unset → using default SPS file: {params['sps_path'].value}"
        )
    else:
        params['sps_path'].value = str(params['sps_path'].value)
        try:
            column_map = sps_columns.build_user_column_map(params)
            sps_columns.validate_user_column_map(
                column_map, params['sps_path'].value
            )
        except ValueError as err:
            logger.error(f"SPS column map error:\n{err}")
            raise
        logger.info(f"Using user-defined sps_path = {params['sps_path'].value}")

    # sps_refmass: reference cluster mass used by f_mass = mCluster / sps_refmass.
    # Default 'def_value' is only meaningful for the bundled SPS file (its
    # reference mass is 1e6 Msun). When the user supplies sps_path, the
    # bundled 1e6 is almost certainly wrong for their file — silent fallback
    # would produce silently-wrong f_mass scaling — so require an explicit
    # value in that case.
    if params['sps_refmass'].value == 'def_value':
        if sps_path_is_default:
            params['sps_refmass'].value = 1e6
        else:
            raise ParameterFileError(
                f"sps_refmass is required when sps_path is user-set "
                f"(got sps_path={params['sps_path'].value!r}). The "
                f"default sps_refmass=1e6 only matches the bundled SPS "
                f"file; supplying your own sps_path means you must "
                f"declare the reference cluster mass it was normalized to."
            )

    params['sps_column_map'] = DescribedItem(
        column_map,
        info="SPS column mapping (canonical -> ColumnSpec)",
        ori_units="N/A",
        exclude_from_snapshot=True,
    )

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
        params['densBE_f_m'] = DescribedItem(None, info="Lane-Emden mass interpolation function", ori_units="N/A", exclude_from_snapshot=True)
        params['densBE_xi_out'] = DescribedItem(0, info="Dimensionless outer radius at cloud edge", ori_units="dimensionless")
    
    elif params['dens_profile'].value == 'densPL':
        # Power-law
        params.pop('densBE_Omega')
    
    # =============================================================================
    # Step 9: Set snapshot exclusions for constants
    # =============================================================================
    
    # Only track time-varying quantities in snapshots
    # Exclude initial conditions and constants to save memory
    time_varying_keys = [
        'model_name', 'mCloud', 'cool_alpha', 'cool_beta', 'cool_delta',
        # Cloud profile constants — needed for radial profile reconstruction
        'nCore', 'nISM', 'rCore', 'dens_profile', 'densPL_alpha',
    ]
    
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
    # Numeric exit code paired with SimulationEndReason. Set at the same site
    # that decides to end the run; consumed by write_simulation_end. None until
    # set; treated as UNKNOWN if the run finishes without it being assigned.
    params['SimulationEndCode'] = DescribedItem(None, info="Exit code (SimulationEndCode enum) for simulation completion", ori_units="N/A")
    params['EarlyPhaseApproximation'] = DescribedItem(True, info="Using approximations for early phase?", ori_units="N/A")

    # Counter for stop_at_rCloud_nSnap. Incremented inside the segment loops of
    # phases 1b/1c/2 each time a snapshot is saved with R2 > rCloud. Reset to 0
    # at run start; not part of any saved snapshot.
    rcloud_counter = DescribedItem(0, info="Snapshots saved with R2 > rCloud (used by stop_at_rCloud_nSnap)", ori_units="N/A")
    rcloud_counter.exclude_from_snapshot = True
    params['_snapshots_after_rCloud'] = rcloud_counter

    # Time tracking
    params['tSF'] = DescribedItem(0, info="Time of star formation", ori_units="Myr")
    params['t_now'] = DescribedItem(0, info="Current simulation time", ori_units="Myr")
    
    # Main bubble parameters
    params['v2'] = DescribedItem(0, info="Velocity at R2 (outer bubble radius = inner shell edge)", ori_units="pc/Myr")
    params['R2'] = DescribedItem(0, info="Outer bubble radius (= inner shell edge)", ori_units="pc")
    params['T0'] = DescribedItem(0, info="Characteristic bubble temperature (at xi_Tb fraction of bubble thickness)", ori_units="K")
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

    # Initial cloud arrays (set once in phase0, constant thereafter)
    params['initial_cloud_r_arr'] = DescribedItem(np.array([]), info="Initial cloud radius array", ori_units="pc")
    params['initial_cloud_n_arr'] = DescribedItem(np.array([]), info="Initial cloud density array", ori_units="1/cm**3")
    params['initial_cloud_m_arr'] = DescribedItem(np.array([]), info="Initial cloud enclosed mass array", ori_units="Msun")
    
    # Feedback from SPS (Starburst99 by default; arbitrary via sps_path).
    params['sps_data'] = DescribedItem(0, info="SPS raw 11-array datacube", ori_units="N/A", exclude_from_snapshot=True)
    params['sps_f'] = DescribedItem(0, info="SPS interpolators (dict of scipy interp1d)", ori_units="N/A", exclude_from_snapshot=True)
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
    params['shell_thickness'] = DescribedItem(0, info="Shell thickness", ori_units="pc")
    params['shell_nMax'] = DescribedItem(0, info="Maximum shell density", ori_units="1/pc**3")
    params['shell_tauKappaRatio'] = DescribedItem(0, info="tau_IR / kappa_IR ratio", ori_units="Msun/pc**2")
    params['shell_grav_r'] = DescribedItem(np.array([]), info="Radius array for gravitational calculations", ori_units="pc")
    params['shell_grav_phi'] = DescribedItem(np.array([]), info="Gravitational potential", ori_units="pc**2/Myr**2")
    params['shell_grav_force_m'] = DescribedItem(np.array([]), info="Gravitational force per unit mass", ori_units="pc/Myr**2")
    params['shell_r_arr'] = DescribedItem(np.array([]), info="Radial grid through ionized+neutral shell", ori_units="pc")
    params['shell_n_arr'] = DescribedItem(np.array([]), info="Number density through ionized+neutral shell", ori_units="1/pc**3")
    params['shell_ion_idx'] = DescribedItem(-1, info="Last index of ionized region in shell_r/n_arr (-1 if empty)", ori_units="N/A")
    params['shell_mass'] = DescribedItem(0, info="Shell mass", ori_units="Msun")
    params['shell_massDot'] = DescribedItem(0, info="Shell mass accretion rate", ori_units="Msun/Myr")
    params['shell_interpolate_massDot'] = DescribedItem(False, info="Use shell mass interpolation?", ori_units="N/A")
    params['shell_n0'] = DescribedItem(0, info="Shell inner density (pressure balance)", ori_units="1/pc**3")
    
    # Forces on shell
    params['F_grav'] = DescribedItem(0, info="Gravitational force", ori_units="Msun*pc/Myr**2")
    params['F_ram'] = DescribedItem(0, info="Ram pressure force (from Pb-Eb relation)", ori_units="Msun*pc/Myr**2")
    params['F_ram_wind'] = DescribedItem(0, info="Wind ram pressure force (from SPS interpolators)", ori_units="Msun*pc/Myr**2")
    params['F_ram_SN'] = DescribedItem(0, info="SN ram pressure force (from SPS interpolators)", ori_units="Msun*pc/Myr**2")
    params['F_ion_in'] = DescribedItem(0, info="Inward photoionization pressure", ori_units="Msun*pc/Myr**2")
    params['F_HII'] = DescribedItem(0, info="Outward HII pressure force (= P_HII * 4piR2^2)", ori_units="Msun*pc/Myr**2")
    params['F_rad'] = DescribedItem(0, info="Radiation pressure", ori_units="Msun*pc/Myr**2")
    params['F_ISM'] = DescribedItem(0, info="ISM pressure force (placeholder, never computed — always 0)", ori_units="Msun*pc/Myr**2")

    # HII region / ionization front diagnostic parameters
    params['n_IF'] = DescribedItem(0.0, info="Density at ionization front from shell ODE", ori_units="1/pc**3")
    params['n_IF_ODE'] = DescribedItem(0.0, info="Raw ODE-derived n_IF (same as n_IF, kept for diagnostics)", ori_units="1/pc**3")
    params['R_IF'] = DescribedItem(0.0, info="Radius of ionization front", ori_units="pc")
    params['n_IF_Str'] = DescribedItem(0.0, info="Stroemgren ionization balance density (Lancaster+2025), sole source of P_HII", ori_units="1/pc**3")
    params['P_HII'] = DescribedItem(0.0, info="HII pressure from Stroemgren ionization balance in shell (n_IF_Str)", ori_units="Msun/Myr**2/pc")
    params['P_drive'] = DescribedItem(0.0, info="Total driving pressure", ori_units="Msun/Myr**2/pc")
    params['P_ram'] = DescribedItem(0.0, info="Ram pressure from freely-streaming wind", ori_units="Msun/Myr**2/pc")
    params['press_HII_in'] = DescribedItem(0.0, info="Inward HII pressure at shell (confining)", ori_units="Msun/Myr**2/pc")
    
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
    params['is_phiDepleted'] = DescribedItem(False, info="Are ionising photons exhausted inside shell (phi->0)?", ori_units="N/A")
    
    # Diagnostic residuals
    params['residual_deltaT'] = DescribedItem(0, info="Temperature residual (T1-T2)/T2", ori_units="dimensionless")
    params['residual_betaEdot'] = DescribedItem(0, info="Energy rate residual", ori_units="dimensionless")
    params['residual_Edot1_guess'] = DescribedItem(np.nan, info="Edot from beta", ori_units="Msun*pc**2/Myr**3")
    params['residual_Edot2_guess'] = DescribedItem(np.nan, info="Edot from energy balance", ori_units="Msun*pc**2/Myr**3")
    params['residual_T1_guess'] = DescribedItem(np.nan, info="T from bubble_Trgoal", ori_units="K")
    params['residual_T2_guess'] = DescribedItem(np.nan, info="T from T0", ori_units="K")

    # =============================================================================
    # Guard: runtime init must not silently overwrite default.param keys
    # =============================================================================
    # A key from default.param that has been replaced (not just mutated) with
    # a fresh DescribedItem has lost the user's value — the most recent offender
    # was `include_PHII`, which meant every run integrated with include_PHII=True
    # regardless of what the .param file said. Fail loudly so this never ships
    # silently again.
    _stomped = [
        k for k, v_before in _default_items_before.items()
        if k in params and params[k] is not v_before
    ]
    if _stomped:
        raise RuntimeError(
            f"read_param: runtime init silently overwrote user-facing "
            f"default.param key(s): {sorted(_stomped)}. User parameters must "
            f"flow through the default.param merge (Step 4); remove the "
            f"conflicting assignment(s) from Step 6/8/10."
        )

    # Phase 5 drop: the legacy ``<model_name>_summary.txt`` is no longer
    # written here.  All run-constants land in ``metadata.json`` (written
    # by ``DescribedDict.flush()``); ``python -m src._output.show_run``
    # formats them for human consumption.

    return params


# ParameterFileError now lives in src/_input/errors.py — imported at the
# top of this module and re-exported here for any external caller using
# ``from src._input.read_param import ParameterFileError``.
__all__ = ["read_param", "ParameterFileError"]


# =============================================================================
# Quick test (commented out)
# =============================================================================
if __name__ == "__main__":
    # Configure logging for standalone test
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    params = read_param('param/simple_cluster.param')
    logger.info(f"mCloud = {params['mCloud'].value} {params['mCloud'].ori_units}")
    logger.info(f"  Info: {params['mCloud'].info}")
