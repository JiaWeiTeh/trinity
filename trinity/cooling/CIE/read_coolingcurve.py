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
from bisect import bisect_left as _bisect_left
import sys
import scipy
import astropy.units as u
#--

# This is the simple case when CIE is achieved, so Lambda depends only on T. 
# TODO: add for non-solar metallicity 


# TODO: add file saving for quicker computation time.

class _ScalarLinear:
    """interp1d(kind='linear', bounds_error=True) at one scalar point."""
    __slots__ = ('x', 'y', 'n')

    def __init__(self, f):
        self.x = [float(v) for v in f.x]
        self.y = np.asarray(f.y, dtype=float).reshape(-1)
        self.n = len(self.x)

    def __call__(self, xn):
        x = self.x
        if xn < x[0]:
            raise ValueError(f"A value ({xn}) in x_new is below the interpolation range's "
                             f"minimum value ({x[0]}).")
        if xn > x[-1]:
            raise ValueError(f"A value ({xn}) in x_new is above the interpolation range's "
                             f"maximum value ({x[-1]}).")
        k = _bisect_left(x, xn)                 # np.searchsorted(side='left')
        k = 1 if k < 1 else (self.n - 1 if k > self.n - 1 else k)
        lo, hi = k - 1, k
        slope = (self.y[hi] - self.y[lo]) / (x[hi] - x[lo])
        # Return a 0-d ARRAY exactly as interp1d does: the caller's 10**(...) then runs
        # numpy's ufunc power loop. 10**np.float64 would take numpy's scalar-math path
        # (libm pow), and the two differ in the last bit on x86 -- not bit-identical.
        return np.array(slope * (xn - x[lo]) + self.y[lo])


def _scalar_evaluator(interp, cls):
    fast = getattr(interp, '_scalar_eval', None)
    if fast is None:
        fast = cls(interp)
        interp._scalar_eval = fast
    return fast


def get_Lambda(T, cooling_CIE_interpolation, metallicity):
    """
    This function calculates Lambda assuming CIE conditions.

    Parameters
    ----------
    T : float/array
        Temperature.
    cooling_CIE_interpolation : callable
        Interpolation function (log K -> log Lambda).
    metallicity : float
        Cloud metallicity (selects/validates against the CIE library).

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
        Cooling rate (per the CIE curve at temperature T).

    """
    
    # Might be a problem here because this does not support extrapolation. If
    # this happens, implement a function that does that.

    # change temperature to log for interpolation
    T = np.log10(T)
    # find lambda. A scalar T (the bubble-ODE RHS, once per evaluation) takes the
    # op-for-op scalar evaluator cached on the interpolator (_ScalarLinear above,
    # HOTPATH W1); arrays keep the vectorised interp1d call.
    if np.ndim(T) == 0:
        Lambda = 10**(_scalar_evaluator(cooling_CIE_interpolation, _ScalarLinear)(T))
    else:
        Lambda = 10**(cooling_CIE_interpolation(T))

    return Lambda

