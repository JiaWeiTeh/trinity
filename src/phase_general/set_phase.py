#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:14:46 2023

@author: Jia Wei Teh

This python script contains function which will set the condition
of current phase. The output will then be read by other functions
to tell which phase TRINITY is currently in.
"""

# libraries
import numpy as np
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])


# =============================================================================
# Here we store fates that determines whether to 
# continue simulation or to stop entirely
# =============================================================================
coll = -1
cont = 0
stop = 1

# =============================================================================
# Here we associate phases with values.
# =============================================================================

# negative velocity
collapse = -1.0 
# enegry-driven winds
weaver = 1.0 
# gradient in energy driven phase
Egrad = 1.1 
# momentum-driven expansion into constant density (cloud core)
core = 2.0 
# momentum-driven expansion down a density gradient (power law)
gradient = 2.1 
# momentum-driven expansion into constant density (ambient medium)
ambient = 3.0 
# shell has dissolved
dissolve = 0.0 


# not sure if this is being used now

# def set_phase(r,rcore,rcloud,t, v, dens_grad = True):
#     #set the current phase
#     if type(r) is not np.ndarray:
#         r = np.array([r])
#     if type(t) is not np.ndarray:
#         t = np.array([t])
#     if type(v) is not np.ndarray:
#         v = np.array([v])

#     # set as nan values
#     phase = np.nan*r

#     ph_collapse = (v < 0.0)
#     #ph_weaver = (t<tcool and ((dens_grad and r < rcore) | (not dens_grad and r < rcloud)))
#     ph_amb = (r >= rcloud) & (~ph_collapse)
#     ph_grad = dens_grad & (r < rcloud) & (r > rcore) & (~ph_collapse)
#     ph_core = ~(ph_collapse | ph_amb | ph_grad)

#     phase[ph_collapse] = collapse #collapse
#     #phase[ph_weaver] = 1.0
#     phase[ph_amb] = ambient
#     phase[ph_grad] = gradient
#     phase[ph_core] = core

#     # dissolved phase: 0.0
#     return phase



# (aux.check_continue(t[-1], r[-1], v[-1], i.tStop) == ph.coll):

def check_simulation_status(t, r, v, tstop):
    """
    This function checks if WARPFIELD should stop, continue, or recollapse-reform cluster.

    Parameters
    ----------
    t : float
        Current time.
    r : float
        current shell radius.
    v : float
        current velocity.
    tStop : one of the following:
        -1: collapse (more more stars!)
        0: continue normally (probably in a new phase)
        1: stop simulation (end time reached, max radius reached, or shell dissolved)

    Returns
    -------
    Status of the simulaion. Available values:
        (Collapse) coll = -1
        (continue) cont = 0
        (Stop sim) stop = 1
    """
    
    # Note:
        # old code: aux.check_continue()

    eps = 1e-2

    if (v < 0. and r < warpfield_params.r_coll.value + eps and t < tstop - eps):
        # collapse!
        if warpfield_params.mult_exp:
            return coll
        else:  # if simulation should be stopped after re-collapse, return stop command
            return stop
    elif (r >  warpfield_params.stop_r.value - eps or t > tstop - eps): 
        # TODO MISSING: shell has dissolved
        # stop simulation (very large shell radius or end time reached or shell dissolved)
        return stop
    else:
        # continue simulation in new phase
        return cont
    
    
    
    
    
    
    

