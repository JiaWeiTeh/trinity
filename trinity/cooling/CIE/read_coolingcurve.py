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

# This is the simple case when CIE is achieved, so Lambda depends only on T. 
# TODO: add for non-solar metallicity 


# TODO: add file saving for quicker computation time.

def get_Lambda(T, cooling_CIE_interpolation, metallicity):
    """
    This function calculates Lambda assuming CIE conditions.

    Parameters
    ----------
    T : float/array
        Temperature.

    Available libraries (set via `path_cooling_CIE` in .param) include:
        1: CLOUDY cooling curve for HII region, solar metallicity.
        2: CLOUDY cooling curve for HII region, solar metallicity.
            Includes the evaporative (sublimation) cooling of icy interstellar
            grains (occurs e.g., when heated by cosmic-ray particle).
        3: Gnat and Ferland 2012 (slightly interpolated for values).
        4: Sutherland and Dopita 1993, for [Fe/H] = -1. Auto-pinned when
            ZCloud == 0.15 regardless of `path_cooling_CIE`.

    These files are bundled under lib/default/CIE/.

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

    # change temperature to log for interpolation
    T = np.log10(T)
    # find lambda
    Lambda = 10**(cooling_CIE_interpolation(T))

    return Lambda

