#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:09:46 2022

@author: Jia Wei Teh

This script contains functions which compute the cooling function Lambda, given T.

old code: cool.py
"""
# libraries
import numpy as np
import sys
import scipy
import astropy.units as u
#--
# get parameter
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

# This is the simple case when CIE is achieved, so Lambda depends only on T. 
# TODO: add for non-solar metallicity 


# TODO: add file saving for quicker computation time.

def get_Lambda(T, cooling_CIE_interpolation):
    """
    This function calculates Lambda assuming CIE conditions.

    Parameters
    ----------
    T : float/array
        Temperature.

    Available libraries (specified in .param file) include:
        1: CLOUDY cooling curve for HII region, solar metallicity.
        2: CLOUDY cooling curve for HII region, solar metallicity. 
            Includes the evaporative (sublimation) cooling of icy interstellar 
            grains (occurs e.g., when heated by cosmic-ray particle)
        
        3. Gnat and Ferland 2012 (slightly interpolated for values)
        4. Sutherland and Dopita 1993, for [Fe/H] = -1].
    
    These files are by default stored in path/to/warpfield/lib/cooling_tables/CIE/current/.

    Returns
    -------
    Lambda [erg/s * cm3]: float.
        Cooling.
    These values are from the file directly:
        logT: temperature (log).
        logLambda: Lambda-values (log).
    cooling_CIE_interpolation [log(K)]: 
        Interpolation function that takes temperature.

    """
    
    # Might be a problem here because this does not support extrapolation. If
    # this happens, implement a function that does that.
    
    if warpfield_params.metallicity != 1:
        sys.exit('Need to implement non-solar metallicity.')
    # # get path to library
    # # See example_pl.param for more information.
    # path2cooling = warpfield_params.path_cooling_CIE
    # # unpack from file
    # logT, logLambda = np.loadtxt(path2cooling, unpack = True)
    # # create interpolation
    # cooling_CIE_interpolation = scipy.interpolate.interp1d(logT, logLambda, kind = 'linear')
    # change temperature to log for interpolation
    T = np.log10(T)
    # find lambda
    Lambda = 10**(cooling_CIE_interpolation(T))

    return Lambda

