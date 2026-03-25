#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:36:36 2023

@author: Jia Wei Teh

This script contains useful functions that help compute stuffs
"""

from functools import reduce
from typing import Tuple, Union, Sequence

import numpy as np
import src._functions.unit_conversions as cvt

def _simplify(
    x_arr: Union[np.ndarray, Sequence[float]],
    y_arr: Union[np.ndarray, Sequence[float]],
    nmin: int = 100,
    grad_inc: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic downsampling of a curve y(x) to approximately nmin points,
    preserving important features of the curve shape.

    The algorithm keeps:
    - both endpoints
    - points where the gradient changes sharply (large fractional second derivative)
    - local extrema (sign changes in the first derivative)
    - evenly spaced samples in cumulative-y-distance space

    Parameters
    ----------
    x_arr : array-like
        Independent variable (e.g. position, time).
    y_arr : array-like
        Dependent variable (e.g. temperature, density).
    nmin : int, optional
        Target minimum number of output samples. Default is 100.
    grad_inc : float, optional
        Fractional gradient-change threshold. Points where the gradient
        changes by more than this fraction are kept. Default is 1.0 (100%).

    Returns
    -------
    x_out, y_out : np.ndarray
        Downsampled x and y arrays.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)

    if x.size == 0 or y.size == 0:
        return x, y
    if x.size != y.size:
        raise ValueError(
            f"_simplify(): x and y must have the same length. "
            f"Got {x.size} and {y.size}"
        )
    if nmin >= x.size:
        return x, y
    nmin = max(int(nmin), 100)

    # --- 1) Gradient-based feature detection ---
    grad = np.gradient(y)
    eps = 1e-30
    # Safe denominator to avoid division by zero in flat regions
    denom = np.where(
        np.abs(grad[:-1]) < eps, eps, grad[:-1]
    )
    pct = np.diff(grad) / denom
    # Points with large fractional gradient change
    important_grad = np.where(np.abs(pct) > grad_inc)[0]
    # Points where the derivative changes sign (local extrema)
    important_sign = np.where(np.diff(np.sign(grad)) != 0)[0]

    # --- 2) Cumulative-distance sampling in y ---
    yrng = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(yrng) or yrng == 0:
        # Flat curve: fall back to uniform spacing
        idx = np.unique(np.linspace(0, x.size - 1, nmin).astype(int))
        return x[idx], y[idx]

    maxdist = yrng / nmin
    y_cum = np.cumsum(np.abs(np.diff(y)))
    bins = (y_cum / maxdist).astype(int)
    idx_dist = np.where(bins[:-1] != bins[1:])[0]

    # --- 3) Merge all candidate indices + endpoints ---
    merged = reduce(
        np.union1d,
        [
            np.array([0], dtype=int),
            important_grad.astype(int),
            important_sign.astype(int),
            idx_dist.astype(int),
            np.array([x.size - 1], dtype=int),
        ],
    )
    merged = np.unique(np.clip(merged, 0, x.size - 1))
    return x[merged], y[merged]


def find_nearest(array, value):
    """
    finds index idx in array for which array[idx] is closest to value
    """
    # make sure that we deal with an numpy array
    array = np.array(array)
    # index
    idx = (np.abs(array-value)).argmin()
    # return
    return idx

def find_nearest_lower(array, value):
    """
    This fucntion finds idx in array for which array[idx] satisfies:
        1) smaller or equal to value; and
        2) closest to value.
    Elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonic 
    # debug
    if any(array < 0):
        print(array)
        
    if not monotonic(array):
        print(f"array has to be monotonic! Instead got {array}.")
        # np.save(warpfield_params.out_dir + 'T_array_monotonic_check.npy', array)
        raise MonotonicError()
    
    # is it increasing?
    mon_incr = kindof_increasing(array)
    
    # get index
    idx = find_nearest(array, value)
    #---
    #---
    if array[idx] - value > 0: # then this element is the closest, but it is larger than value
        if mon_incr: 
            idx += -1 # take the element before, it will be smaller than value (if array is monotonically increasing)
        else: 
            idx += 1
    # Notes: boundary conditions, just in case. Although when these happen, it means that
    # the returned idx is actually higher than the value instead of the desired 
    # lower. Not quite sure what to do with that for now, but this part of 
    # the code shouldnt need to run anyway.
    if idx >= len(array): 
        idx = len(array) - 1
    if idx < 0: 
        idx = 0
    # return
    return idx

#  kind of, because includes equal values like [1,2,3,3,4]
def kindof_increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def kindof_decreasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return kindof_increasing(L) or kindof_decreasing(L)


def find_nearest_higher(array, value):
    """
    This fucntion finds idx in array for which array[idx] satisfies:
        1) higher or equal to value; and
        2) closest to value.
    Elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonic 
    # debug
    if any(array < 0):
        print(array)
        
    if not monotonic(array):
        print(f"array has to be monotonic! Instead got {array}.")
        # np.save(warpfield_params.out_dir + 'T_array_monotonic_check.npy', array)
        raise MonotonicError()

    # is it increasing?
    mon_incr = kindof_increasing(array)
    
    
    # get index
    idx = find_nearest(array, value)
    #---
    #---
    if array[idx] - value < 0: # then this element is the closest, but it is larger than value
        if mon_incr: 
            idx += 1 # take the element before, it will be smaller than value (if array is monotonically increasing)
        else: 
            idx += -1
    # Notes: boundary conditions, just in case. Although when these happen, it means that
    # the returned idx is actually higher than the value instead of the desired 
    # lower. Not quite sure what to do with that for now, but this part of 
    # the code shouldnt need to run anyway.
    if idx >= len(array): 
        idx = len(array) - 1
    if idx < 0: 
        idx = 0
    # return
    return idx

class MonotonicError(Exception):
    pass

def get_soundspeed(T, params):
    """
    This function computes the isothermal soundspeed, c_s, given temperature
    T and mean molecular weight mu.

    Parameters
    ----------
    T : float (Units: K)
        Temperature of the gas.

    Returns
    -------
    The isothermal soundspeed c_s (Units: Myr/pc)

    """    
    if T > 1e4:
        mu = params['mu_ion'] * cvt.Msun2g
    else:
        mu = params['mu_atom'] * cvt.Msun2g
    
    return  np.sqrt(params['gamma_adia'] * (params['k_B'] * cvt.k_B_au2cgs) * T / mu) * cvt.v_cms2au

