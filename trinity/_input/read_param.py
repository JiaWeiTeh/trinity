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
from pathlib import Path
from fractions import Fraction
import trinity._functions.unit_conversions as cvt
from trinity._input.dictionary import DescribedItem, DescribedDict
from trinity._input.errors import ParameterFileError
from trinity._input.registry import (
    apply_active_when,
    materialize_runtime,
    resolve_all,
    validate_all,
    validate_companions,
)

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
    # script in trinity/_input/, not in the user-facing param/ directory).
    script_dir = Path(__file__).parent.resolve()
    path2default = script_dir / 'default.param'

    if not path2default.exists():
        raise FileNotFoundError(
            f"Default parameter file not found at: {path2default}\n"
            f"Expected: <trinity_root>/trinity/_input/default.param"
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

    # Enforce trigger/companion bundles (e.g. dens_profile=densPL must
    # be accompanied by an explicit densPL_alpha).  Runs against the
    # raw user dict so it fires only when the user actually typed the
    # trigger, not when it inherited the default.
    validate_companions(user_dict)
    
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
    # definitions in ``trinity/_input/registry.py``.
    validate_all(params)


    # =============================================================================
    # Step 6: Compute derived parameters
    # =============================================================================

    # ---- Composition: derive mu_* and chi_e from x_He, Z_He ----
    # x_He (n_He/n_H) and Z_He (helium ionisation state) are the single source
    # of truth for the gas composition. Exact-rational (Fraction) arithmetic
    # keeps the mu_* values byte-identical to the historical 14/11, 14/23,
    # 14/6, 1.4 encodings when x_He=0.1, Z_He=2 (verified). The 'm_H' unit
    # factor matches what Step 4 applied to the former numeric defaults.
    _xHe = Fraction(params['x_He'].value).limit_denominator(10**6)   # 0.1 -> 1/10
    _ZHe = Fraction(params['Z_He'].value).limit_denominator(10**6)   # 2.0 -> 2
    _muH    = 1 + 4 * _xHe                       # mass per H nucleus [m_H]
    _mu_n   = _muH / (1 + _xHe)                  # neutral mean mass/particle
    _mu_p   = _muH / (2 + _xHe * (1 + _ZHe))     # ionised mean mass/particle
    _mu_mol = _muH / (Fraction(1, 2) + _xHe)     # molecular mean mass/particle
    _chi_e  = 1 + _ZHe * _xHe                    # electrons per H nucleus, n_e/n_H
    _mH_au  = cvt.convert2au('m_H')              # m_H in Msun
    params['mu_convert'].value = float(_muH)    * _mH_au
    params['mu_atom'].value    = float(_mu_n)   * _mH_au
    params['mu_ion'].value     = float(_mu_p)   * _mH_au
    params['mu_mol'].value     = float(_mu_mol) * _mH_au
    params['chi_e'] = DescribedItem(
        value=float(_chi_e),
        info=('Electron-per-hydrogen-nucleus factor n_e/n_H = 1 + Z_He*x_He '
              'for the HOT bubble (doubly-ionised He). Derived at load from '
              'x_He, Z_He. Multiplies n_H^2 in the bubble CIE cooling.'),
        ori_units='dimensionless',
    )

    # Shell / HII region (~1e4 K) is singly ionised (Z_He_shell), unlike the
    # hot doubly-ionised bubble: derive its ionised mu and electron factor.
    _ZHe_sh = Fraction(params['Z_He_shell'].value).limit_denominator(10**6)  # 1.0 -> 1
    _mu_p_sh = _muH / (2 + _xHe * (1 + _ZHe_sh))    # singly-ionised mean mass/particle
    _chi_e_sh = 1 + _ZHe_sh * _xHe                  # electrons per H nucleus (singly)
    params['mu_ion_shell'] = DescribedItem(
        value=float(_mu_p_sh) * _mH_au,
        info=('Ionised mean mass per particle for the ~1e4 K shell / HII region '
              '(singly-ionised He): mu_H/(2 + x_He*(1+Z_He_shell)). Derived at '
              'load from x_He, Z_He_shell. Used for the HII gas pressure and the '
              'shell structure (the hot bubble uses mu_ion instead).'),
        ori_units='m_H',
    )
    params['chi_e_shell'] = DescribedItem(
        value=float(_chi_e_sh),
        info=('Electron-per-hydrogen-nucleus factor n_e/n_H = 1 + Z_He_shell*x_He '
              'for the ~1e4 K shell / HII region (singly-ionised He). Derived at '
              'load from x_He, Z_He_shell. Multiplies n_H^2 in shell recombination '
              'and the Stroemgren balance.'),
        ori_units='dimensionless',
    )

    # caseB_alpha (the case-B recombination coefficient, default 2.59e-13 cm^3/s)
    # is fixed at its ~1e4 K value and is NOT recomputed from TShell_ion. Since
    # alpha_B(T) ~ T^-0.7, moving the ionised-shell temperature far from ~1e4 K
    # leaves the Stroemgren balance (n_IF_Str) and P_HII/F_HII internally
    # inconsistent unless caseB_alpha is adjusted to match. Warn once at load.
    _T_shell_ion = params['TShell_ion'].value
    if not (8000.0 <= _T_shell_ion <= 1.1e4):
        logger.warning(
            f"TShell_ion = {_T_shell_ion:.4g} K is outside the ~8000-11000 K range "
            f"that the default caseB_alpha (case-B recombination coefficient) assumes. "
            f"alpha_B is temperature-dependent (~T^-0.7) and is NOT auto-adjusted, so "
            f"the Stroemgren n_IF_Str and P_HII/F_HII may be internally inconsistent. "
            f"Set caseB_alpha to the case-B value at your TShell_ion."
        )

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
    # Step 7: Resolve sentinel ('def_*') defaults
    # =============================================================================
    # Path + SPS-bundle sentinels resolve via their registry resolvers
    # (path2output, path_cooling_nonCIE, sps_path). sps_path's resolver
    # owns the coupled bundle — sps_refmass and the sps_col_* family
    # (consumed_by='sps_path') — and injects params['sps_column_map'].
    # Must run after Step 6 (model_name resolved; path2output depends on it).
    resolve_all(params)

    # Cooling directory - CIE (NOT a def_* sentinel: an integer-index
    # preset keyed on ZCloud, so it stays inline rather than in a resolver).
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

    # =============================================================================
    # Step 8: Apply active_when (conditional schema)
    # =============================================================================
    # Pops keys whose active_when predicate is False (e.g. densPL_alpha on
    # a densBE run, densBE_Omega on a densPL run) and adds keys whose
    # predicate is True but which aren't yet present (e.g. the 9
    # densBE_* runtime params on a densBE run). All metadata (info, unit,
    # exclude_from_snapshot) comes from the spec; mutable defaults are
    # deep-copied so runs don't share state. Step 9 below sweeps
    # exclude_from_snapshot for the final non-time-varying set.
    apply_active_when(params)
    
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
    # Step 10: Materialize runtime / derived-init parameters
    # =============================================================================
    # For every spec not yet in params (i.e. not from default.param,
    # resolve_all, or apply_active_when) and not gated by active_when
    # (Phase 8) or consumed_by (Phase 7's sps_path bundle), create a
    # fresh DescribedItem from spec metadata.  Default 106 adds today:
    # 9 with exclude_from_snapshot=True (cooling cubes, sps_data/sps_f,
    # rcloud counter) and 97 with False (time-varying simulation state
    # -- bubble_*, shell_*, forces, residuals -- that snapshots stream).
    # Must run after Step 9; new items' exclude_from_snapshot comes
    # straight from the spec and bypasses Step 9's sweep, matching the
    # pre-Phase-9 behavior where Step 10 also constructed post-sweep.
    materialize_runtime(params)

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
    # by ``DescribedDict.flush()``); ``python -m trinity._output.show_run``
    # formats them for human consumption.

    return params


# ParameterFileError now lives in trinity/_input/errors.py — imported at the
# top of this module and re-exported here for any external caller using
# ``from trinity._input.read_param import ParameterFileError``.
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
