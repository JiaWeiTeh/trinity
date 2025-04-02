#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:04:16 2023

@author: Jia Wei Teh

This script finds the real value of beta-delta.
"""

# libraries
import numpy as np
import copy
import scipy.interpolate
import sys
import scipy.optimize
#--
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.bubble_structure.bubble_luminosity as bubble_luminosity
import src._functions.operations as operations
from src._input.dictionary import updateDict

#--
from src._functions.clock import _timer


def get_beta_delta_wrapper(beta_guess, delta_guess, params):
    """
    # old code: rootfinder_bd_wrap()
    
    This wrapper handles get_beta_delta(), which deals with 
    getting better estimates of delta, beta and other bubble properties. 
    
    Parameters
    ----------
    General: beta = - dPb/dt; delta = dT/dt at xi. See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf,
    pg92 (A4-5). This is used to resolve the velocity and temperature structure,
    v' (dvdr) and T'' (dTdrr). 
    
    """
    
    # this one is without solver
    beta_delta_outputs_main, final_params = get_betadelta(beta_guess, delta_guess, params)
    
    return beta_delta_outputs_main, final_params


def get_betadelta(beta_guess, delta_guess, params):
    
    
    beta_max = 10 # very large number. Maybe beta doesnt have a maximum! It used to be 1.
    beta_min = 0
    delta_max = 0
    delta_min = -1
    
    
    def generate_combinations(beta, delta):
        
        beta_range, delta_range = [], []
        
        # 0.05 seems to be too big a leap.
        epsilon = 0.02
        
        beta_range_min = max(beta_min, beta - epsilon)
        beta_range_max = min(beta_max, beta + epsilon)
        
        delta_range_min = max(delta_min, delta - epsilon)
        delta_range_max = min(delta_max, delta + epsilon)

        # create range
        beta_range = np.linspace(beta_range_min, beta_range_max, 5)
        delta_range = np.linspace(delta_range_min, delta_range_max, 5)
        
        # create all possible tuple combinations of elements from two arrays
        beta_grid, delta_grid = np.meshgrid(beta_range, delta_range, indexing='ij')  
        
        # Flatten and pair
        return np.column_stack([beta_grid.ravel(), delta_grid.ravel()])  
    
    
    # how is the current residual doing?
    
    test_params = copy.deepcopy(params)
    
    residual = get_residual([beta_guess, delta_guess], test_params)
    
    residual_sq = np.sum(np.square(residual))
    
    # only calculate beta/delta again if the residual is larger than 1e-4
    
    if residual_sq < 1e-4:
        
        for key in params.keys():
            # dont have to care about past snapshots
            if key.startswith('_sS'):
                continue
            updateDict(params, [key], [test_params[key].value])
   
        return [beta_guess, delta_guess], params
        
    else:
    
        bd_pairings = generate_combinations(beta_guess, delta_guess)
        dictionary_residual_pair = {}
        for bd_pair in bd_pairings:
            
            test_params = copy.deepcopy(params)
            
            try:
                residual = get_residual(bd_pair, test_params)
            except operations.MonotonicError as e:
                print(e)
                residual = (100,100)
            except Exception as e:
                print(e)
                sys.exit()
            
            # print('residual', residual)
            
            residual_sq = np.sum(np.square(residual))

            dictionary_residual_pair[residual_sq] = test_params
            
        # check residuals
        # for str_residual, test_dictionary in dictionary_residual_pair.items():
            
        sorted_keys = sorted(dictionary_residual_pair)
            
        # for key in sorted_keys:
        #     print('These are the residuals and beta-delta pairs')
        #     print('residual', key, 'beta', dictionary_residual_pair[key]['beta'].value, 'delta', dictionary_residual_pair[key]['delta'].value)
        
        smallest_residual = sorted_keys[0]
        
        for key in params.keys():
            # dont have to care about past snapshots
            if key.startswith('_sS'):
                continue
            updateDict(params, [key], [dictionary_residual_pair[smallest_residual][key].value])
    
        # print('we chose these beta delta values', params['beta'].value, params['delta'].value)
        # print('chosen:', params)
        
        beta, delta = params['beta'].value, params['delta'].value

        return [beta, delta], params


def get_residual(beta_delta_guess, params):
    
    beta_guess, delta_guess = beta_delta_guess
    
    _timer.begin('begin finding beta delta')
        
    # update
    params['beta'].value = beta_guess
    params['delta'].value = delta_guess
    
    # =============================================================================
    # Main equation to check current bubble structure
    # =============================================================================
    
    # copy_params = params
    
    params = bubble_luminosity.get_bubbleproperties(params)
    
    # =============================================================================
    # Part 1: calculate residual of Edot for beta
    # =============================================================================
    # Recalculate R1 based on boundary conditions, to get pressure, to get beta.
    # TODO: can't we just skip this since we have dictionary from previous calculation?

    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                           1e-3 * params['R2'].value, params['R2'].value, 
                           args=([params['L_wind'].value, 
                                  params['Eb'].value, 
                                  params['v_wind'].value,
                                  params['R2'].value,
                                  ]))
    
    params['R1'].value = R1

    # The bubble Pbure [cgs - g/cm/s2, or dyn/cm2]
    Pb = get_bubbleParams.bubble_E2P(params['Eb'].value,
                                    params['R2'].value, 
                                    params['R1'].value)
        
    params['Pb'].value = Pb

    # Now, compare Edot obtained from beta and Edot obtained from balance of energy
        #-- method 1 of calculating Edot, from beta
    Edot = get_bubbleParams.beta2Edot(params)
    
        #-- method 2 of calculating Edot, directly from equation
    L_gain = params['L_wind'].value
    L_loss = params['bubble_L_total'].value + params['L_leak'].value
    
    # these should be R2, v2 and press_bubble
    # gain - loss + work done
    Edot2 = L_gain - L_loss - 4 * np.pi * params['R2'].value**2 * params['v2'].value * Pb
    # residual
    Edot_residual = (Edot - Edot2)/Edot

    # =============================================================================
    # Part 2: calculate residual of T for delta
    # =============================================================================
    
    T_residual = (params['bubble_T_rgoal'].value - params['T0'].value)/params['T0'].value    

    _timer.end()
    
    # record runs
    # params['beta'].value = b_params['beta'].value
    # params['delta'].value = b_params['delta'].value
    params['delta_T_residual'].value = T_residual
    params['beta_Edot_residual'].value = Edot_residual
    params['Edot1_guess'].value = Edot
    params['Edot2_guess'].value = Edot2
    params['T1_guess'].value = params['bubble_T_rgoal'].value
    params['T2_guess'].value = params['T0'].value
    params['Lloss'].value = L_loss
    params['Lgain'].value = L_gain
    
    # or we could do this: first we make a deepcopy of the dictionary. 
    # then, we run it. if the residual is small, then we ruturn that,
    # oherwise, we continue the loop. No recording is needed to avoid
    # any duplication or any wrong information being brought into 
    # the next loop. 
    
    # wait, what if there is just no small value? what if the 
    # solver just ends because it cannot find a solution?
    
    # if Edot_residual < 0.05 and T_residual < 0.05:
    
    return Edot_residual, T_residual
    


