#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:14:13 2023

@author: Jia Wei Teh
"""

import os
import numpy as np

from src._output.terminal_prints import cprint as cpr

# get parameter

import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()


def get_InitBubStruc():
    """
    This function initialises environmental variables to help calculate/track
    bubble structures.

    Parameters
    ----------

    Returns
    -------

    """
    # Notes
    # old code: optimalbstrux in aux_func()
    
    #-----------
    # Here, we initialise a file to track bubble properties
    # The ratio between R1(inner bubble) and R2 (outer bubble/inner shell)
    R1_R2 = np.array([0])
    # The ratio between R2prime (radius slightly smaller than R2, at which T(R2prime) = TR2_prime. See bubble_luminosity.py)
    R2p_R2 = np.array([0])
    # TODO: in the future, add more properties to track!
    
    # old: "/BubDetails/Bstrux.txt"
    # save into csv
    # TODO:
    # question: why not adding [coll_counter] for subsequent sf? perhaps its dealt with later?
    full_path = os.path.join(warpfield_params.out_dir, 'bubble_structure' + '.csv')
    rel_path = os.path.relpath(full_path, os.getcwd())
    np.savetxt(full_path,
               np.c_[R1_R2, R2p_R2],
               delimiter = '\t',
               header='R1/R2 (inner/outer bubble)'+'\t'+'R2prime/R2', comments='')
    print(f'{cpr.FILE}Bubble structure tracking (radius): {rel_path}{cpr.END}')
    #-----------
    
    # Here, initialise a file to track beta and delta especially for phase 1b onwards
    full_path = os.path.join(warpfield_params.out_dir, 'phase1b_details' + '.csv')
    rel_path = os.path.relpath(full_path, os.getcwd())
    np.savetxt(full_path,
               # i add two here just to dodge the error from reading np.loadtxt. This is brute forcing.
               # in future: record every beta delta from phase1a all the way to avoid havinf this linehere.
               np.c_[np.array([0, 0]), np.array([0.830, 0.830]),  np.array([-0.1785, -0.1785])],
               delimiter = '\t',
               header='time (yr)'+'\t'+'beta'+'\t'+'delta')
    
    # initialise some environment variables. The majority of these are to reduce
    # computation time, so that the solvers do not have to restart from scratch but
    # with an updated guess from previous solves. (e.g., beta, delta, dMdt.)
    
    # path. This is removed because it's redundant
    # os.environ["Bstrpath"] = warpfield_params.out_dir
    # dMdt. This is calculated in bubble luminosity: the rate of material that is being evaporated from shell to shocked stellar-wind bubble.
    os.environ["DMDT"] = 'None'
    # count
    os.environ["COUNT"] = str(0)
    # Lcool/gain
    os.environ["Lcool_event"] = str(0)
    os.environ["Lgain_event"] = str(0)
    # If coverfraction
    os.environ["Coverfrac?"] = str(0)
    # keeping track if beta, delta calculations have already been done (e.g., see get_betadelta.py)
    os.environ["beta_delta_result"] = 'None'
    # ??
    os.environ["BD_res_count"] = str(0)
    # ??
    os.environ["Mcl_aux"] = str(warpfield_params.mCloud)
    os.environ["SF_aux"]= str(warpfield_params.sfe)
    # ??
    dic_res={'Lb': 0, 'Trgoal': 0, 'dMdt_factor': 0, 'Tavg': 0, 'beta': 0, 'delta': 0, 'residual': 0}
    os.environ["BD_res"]=str(dic_res)
    
    # return
    return 0
