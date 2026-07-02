#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:36:36 2023

@author: Jia Wei Teh

This script contains useful functions that help compute stuffs
"""

import logging

import numpy as np
import trinity._functions.unit_conversions as cvt

logger = logging.getLogger(__name__)


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
    This function finds idx in array for which array[idx] satisfies:
        1) smaller or equal to value; and
        2) closest to value.
    Elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonic. DEBUG log, not print: the raise is caught
    # by the beta-delta trial wrapper (get_betadelta) as a penalised, retried
    # trial, so this firing is benign per-trial noise that must not spam stdout.
    if not monotonic(array):
        logger.debug(f"array has to be monotonic! Instead got {array}.")
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


# --- Tolerant monotonicity for find_nearest_higher --------------------------
# RETAINED FALLBACK: the bubble-luminosity solver is moving to a solve_ivp
# event-based regime split that does not call find_nearest_higher, so this
# guard may become unused by production. It is kept deliberately (like
# find_nearest_lower above) as a robustness fallback for any grid-based code
# path; do not remove it as "dead code".
#
# A backward bubble-temperature integration can leave tiny, provably-numerical
# non-monotonicities in T_array: a sub-percent dip at the T_init=3e4 outer edge
# (startup transient) or an isolated single-point spike from LSODA dense-output
# interpolation. These must not crash find_nearest_higher's directional search,
# but a *sustained* interior inversion (a possible real feature, or a dead
# integrator's zero/non-finite tail) must still be rejected. We tolerate only
# non-monotonicity that is both shallow (relative drawdown <= MONOTONIC_RTOL)
# and localized (confined to the leading BOUNDARY_FRAC of the array, or an
# isolated run <= MAX_SPIKE_LEN long).
MONOTONIC_RTOL = 1e-2     # max relative drawdown treated as numerical noise
BOUNDARY_FRAC = 0.01      # leading fraction treated as a startup transient
MAX_SPIKE_LEN = 2         # longest wrong-direction run treated as an isolated spike


def _is_monotonic_or_tolerable(L, rtol=MONOTONIC_RTOL,
                               boundary_frac=BOUNDARY_FRAC,
                               max_spike_len=MAX_SPIKE_LEN):
    """
    True if L is monotonic, or non-monotonic only as numerical noise: an
    isolated single-point spike (any depth -- a single point cannot be a
    physical inversion) or a shallow, localized multi-point wiggle. False for a
    non-finite profile and for deep or sustained interior non-monotonicity, so
    the caller still raises MonotonicError on genuinely bad profiles.
    """
    L = np.asarray(L, dtype=float)
    if not np.all(np.isfinite(L)):
        return False
    n = L.size
    if n < 2 or monotonic(L):
        return True
    increasing = L[-1] >= L[0]
    # signed step in the intended direction; wrong-direction steps are < 0
    step = np.diff(L) if increasing else -np.diff(L)
    wrong = step < 0
    if not wrong.any():
        return True
    boundary_cut = max(1, int(np.ceil(boundary_frac * n)))
    i = 0
    while i < wrong.size:
        if not wrong[i]:
            i += 1
            continue
        start = i
        while i < wrong.size and wrong[i]:
            i += 1
        end = i  # wrong-direction run covers steps [start, end); values L[start..end]
        if end - start == 1:
            # isolated single point: a numerical glitch, never a physical
            # inversion -> tolerate regardless of depth
            continue
        # relative depth of the dip/spike across this run
        drop = abs(L[start] - L[end])
        if drop / max(abs(L[start]), 1e-300) > rtol:
            return False
        within_boundary = end <= boundary_cut
        isolated = (end - start) <= max_spike_len
        if not (within_boundary or isolated):
            return False
    return True


def find_nearest_higher(array, value):
    """
    This function finds idx in array for which array[idx] satisfies:
        1) higher or equal to value; and
        2) closest to value.
    Elements in array should be monotonically increasing or decreasing. A
    shallow, localized numerical non-monotonicity (e.g. a sub-percent
    single-point spike, or a startup dip in the leading fraction) is tolerated;
    a deep or sustained-interior inversion still raises MonotonicError.
    """
    # check whether array is monotonic. DEBUG log, not print (see find_nearest_lower).
    if not _is_monotonic_or_tolerable(array):
        logger.debug(f"array has to be monotonic! Instead got {array}.")
        raise MonotonicError()

    # is it increasing? (use endpoints: robust to a tolerated local spike that
    # would otherwise make the all-pairs kindof_increasing() return False)
    mon_incr = array[-1] >= array[0]


    # get index
    idx = find_nearest(array, value)
    #---
    #---
    if array[idx] - value < 0: # then this element is the closest, but it is smaller than value
        if mon_incr:
            idx += 1 # take the next element, it will be larger than value (if array is monotonically increasing)
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
    Compute the adiabatic soundspeed
        c_s = sqrt(gamma_adia * k_B * T / mu),
    using the mean mass per particle mu = mu_ion (T > 1e4 K, ionised) or
    mu_atom (T <= 1e4 K, neutral) -- NOT mu_convert (mass per H nucleus).

    Parameters
    ----------
    T : float (Units: K)
        Temperature of the gas.

    Returns
    -------
    The adiabatic soundspeed c_s (Units: pc/Myr)

    """
    if T > 1e4:
        mu = params['mu_ion'] * cvt.Msun2g
    else:
        mu = params['mu_atom'] * cvt.Msun2g

    return  np.sqrt(params['gamma_adia'] * (params['k_B'] * cvt.k_B_au2cgs) * T / mu) * cvt.v_cms2au
