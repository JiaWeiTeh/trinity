#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:02:19 2023

@author: Jia Wei Teh


Note:
    Old code: warp_writedata.py and warp_nameparser.py
    
    
This script contains function which creates directory for outputs, and function
which create .fits file for outputs.
    
"""

import os
from astropy.io import fits
# get parameter
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

def init_dir():
    """
    This function handles the creation of output files and directories.
    
    Creates
    -------
    warpfield_dir : str
        The path to output. This is the main directory where everything
        from a run is being dump into. 
    cloudy_dir : str
        The path to what will be input into cloudy (and output).
    output_filename : str
        The path to (and including) the output filename.

    In addition, this function also creates directory to /bubbles, /potential, /figures, 
    and /stellar_prop if desired.
    
    - /stellar_prop: contains stellar (e.g., starburst99) feedback parameters as a 
    function of time as were used by WARPFIELD.
    - /potential: stores file with gravitational potential.

    """
    
    # Old code: getmake_dir(), savedir(), create_inputfile(), check_outdir(), get_cloudypath(), get_fname()

    # =============================================================================
    # Create paths to output
    # =============================================================================

    # Old code: get_mypath(), modelprop_to_string() removed. 
    # mypath = warpfield_dir
    warpfield_dir = warpfield_params.out_dir
    # create folder where project data is stored if it does not exist
    # Old code: check_outdir()
    if not os.path.isdir(warpfield_dir):
        os.makedirs(warpfield_dir)

    #---------------------------------

    if warpfield_params.write_potential == True:
        potential_dir = os.path.join(warpfield_dir, "potential")
        os.makedirs(potential_dir)

    if warpfield_params.write_bubble == True:
        bubble_dir = os.path.join(warpfield_dir, "bubble")
        os.makedirs(bubble_dir)

    if warpfield_params.write_figures == True:
        figures_dir = os.path.join(warpfield_dir, "figures")
        os.makedirs(figures_dir)
  
    # This was write_SB99, but changed because SB99 will not be the only option. We
    # might have SLUG, CIGALE etc in the future.
    if warpfield_params.write_stellar_prop == True:
        figures_dir = os.path.join(warpfield_dir, "stellar_prop")
        os.makedirs(figures_dir)
        
    # Old: input is saved in input.dat, and output is saved in evo.dat.
    output_summary_filename = os.path.join(warpfield_dir, warpfield_params.model_name+'_summary.txt')
    
    return output_summary_filename


def write_evolution(data):
    
    # writes evolution. This is previously evo.dat. But now we try to save in fits format. 
    
    
    #  function inspired by SLUG2 (M. Krumholz)
    
    
    # convert data to FITS columns
    cols = []
    # 'format' keyword, description, and 8-bit bytes.
    # L                        logical (Boolean)               1
    # K                        64-bit integer                  8
    # A                        character                       1
    # D                        double precision float (64-bit) 8

    # evolution
    # Time
    cols.append(fits.Column(name="t", format='1D', 
                            unit='Myr', array = data['t']))
    # Shell_radius 
    cols.append(fits.Column(name="r", format='1D', 
                            unit='pc', array = data['r']))
    # Shell_velocity
    cols.append(fits.Column(name="v", format='1D', 
                            unit='km/s', array = data['v']))
    # Bubble energy
    cols.append(fits.Column(name="Eb", format='1D', 
                            unit='erg', array = data['E']))
    # Shell_mass
    cols.append(fits.Column(name="M", format='1D', 
                            unit='log Msun', array = data['logMshell']))
    # Total cooling luminosity
    cols.append(fits.Column(name="LumC_t", format='1D', 
                            unit='erg/s', array = data['Lcool']))    
    # Bubble cooling luminosity - inner bubble
    cols.append(fits.Column(name="LumC_b", format='1D', 
                            unit='erg/s', array = data['Lbb']))    
    # Bubble cooling luminosity - conduction zone
    cols.append(fits.Column(name="LumC_c", format='1D', 
                            unit='erg/s', array = data['Lbcz']))   
    # Bubble cooling luminosity - intermediate zone
    cols.append(fits.Column(name="LumC_i", format='1D', 
                            unit='erg/s', array = data['Lb3']))
    # Bubble temperature
    cols.append(fits.Column(name="T", format='1D', 
                            unit='K', array = data['Tb'])) 
    # dMdt factor
    # fabs, fabs_n, fabs_i
    # alpha, beta, delta
    
    fitscols = fits.ColDefs(cols)
    
    # Create the binary table HDU
    tbhdu = fits.BinTableHDU.from_columns(fitscols)

    # Create dummy primary HDU
    prihdu = fits.PrimaryHDU()

    # Create HDU list and write to file
    hdulist = fits.HDUList([prihdu, tbhdu])
    hdulist.writeto(warpfield_params.out_dir + '/' + warpfield_params.model_name+'_evolution.fits',
                    overwrite=True)
            
    return 





