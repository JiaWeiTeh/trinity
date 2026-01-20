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
import logging
#--
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.bubble_structure.bubble_luminosity as bubble_luminosity
import src._functions.operations as operations
from src._input.dictionary import updateDict

#--
from src._functions.clock import _timer

logger = logging.getLogger(__name__)


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
    
     
    beta_max = 1 # very large number. Maybe beta doesnt have a maximum! It used to be 1.
    beta_min = 0
    delta_max = 0
    delta_min = -1 # same with this
    
    
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
            updateDict(params, [key], [test_params[key].value])

        logger.info(
            f"Beta-delta result (input): β={beta_guess:.4f}, δ={delta_guess:.4f}, "
            f"residual={residual_sq:.2e}, converged=True"
        )

        return [beta_guess, delta_guess], params
        
    else:
    
        bd_pairings = generate_combinations(beta_guess, delta_guess)
        dictionary_residual_pair = {}
        for bd_pair in bd_pairings:
            
            test_params = copy.deepcopy(params)
            
            try:
                residual = get_residual(bd_pair, test_params)
            except operations.MonotonicError as e:
                logger.warning(f"MonotonicError at β={bd_pair[0]:.4f}, δ={bd_pair[1]:.4f}: {e}")
                residual = (100, 100)
            except Exception as e:
                logger.warning(f"Error at β={bd_pair[0]:.4f}, δ={bd_pair[1]:.4f}: {e}")
            
            # print('residual', residual)
            
            residual_sq = np.sum(np.square(residual))

            dictionary_residual_pair[residual_sq] = test_params
            
        # check residuals
        sorted_keys = sorted(dictionary_residual_pair)

        # Log all candidates at DEBUG level
        for res_key in sorted_keys:
            logger.debug(
                f"Grid candidate: β={dictionary_residual_pair[res_key]['cool_beta'].value:.4f}, "
                f"δ={dictionary_residual_pair[res_key]['cool_delta'].value:.4f}, "
                f"residual={res_key:.2e}"
            )

        smallest_residual = sorted_keys[0]

        for key in params.keys():
            updateDict(params, [key], [dictionary_residual_pair[smallest_residual][key].value])

        beta, delta = params['cool_beta'].value, params['cool_delta'].value
        converged = smallest_residual < 1e-4

        # Log final result at INFO level
        logger.info(
            f"Beta-delta result (grid): β={beta:.4f}, δ={delta:.4f}, "
            f"residual={smallest_residual:.2e}, converged={converged}"
        )

        return [beta, delta], params


def get_residual(beta_delta_guess, params):
    
    beta_guess, delta_guess = beta_delta_guess
    
    # _timer.begin('begin finding beta delta')
        
    # update
    params['cool_beta'].value = beta_guess
    params['cool_delta'].value = delta_guess
    
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
                           args=([params['Lmech_total'].value, 
                                  params['Eb'].value, 
                                  params['v_mech_total'].value,
                                  params['R2'].value,
                                  ]))
    
    params['R1'].value = R1

    # The bubble Pbure [cgs - g/cm/s2, or dyn/cm2]
    Pb = get_bubbleParams.bubble_E2P(params['Eb'].value,
                                    params['R2'].value, 
                                    params['R1'].value,
                                    params['gamma_adia'].value)
        
    params['Pb'].value = Pb

    # Now, compare Edot obtained from beta and Edot obtained from balance of energy
        #-- method 1 of calculating Edot, from beta
    Edot = get_bubbleParams.beta2Edot(params)
    
        #-- method 2 of calculating Edot, directly from equation
    L_gain = params['Lmech_total'].value
    L_loss = params['bubble_LTotal'].value + params['bubble_Leak'].value
    
    # these should be R2, v2 and press_bubble
    # gain - loss + work done
    Edot2 = L_gain - L_loss - 4 * np.pi * params['R2'].value**2 * params['v2'].value * Pb
    # residual
    Edot_residual = (Edot - Edot2)/Edot

    # =============================================================================
    # Part 2: calculate residual of T for delta
    # =============================================================================
    
    T_residual = (params['bubble_T_r_Tb'].value - params['T0'].value)/params['T0'].value    

    # _timer.end()
    
    # record runs
    # params['beta'].value = b_params['beta'].value
    # params['delta'].value = b_params['delta'].value
    
    
    params['residual_deltaT'].value = T_residual
    params['residual_betaEdot'].value = Edot_residual
    params['residual_Edot1_guess'].value = Edot
    params['residual_Edot2_guess'].value = Edot2
    params['residual_T1_guess'].value = params['bubble_T_r_Tb'].value
    params['residual_T2_guess'].value = params['T0'].value
    
    
    params['bubble_Lloss'].value = L_loss
    params['bubble_Lgain'].value = L_gain
    
    # print('here to check beta delta residual why np.nan')
    # print(params)
    # print('residuals', Edot_residual, T_residual)
    # or we could do this: first we make a deepcopy of the dictionary. 
    # then, we run it. if the residual is small, then we ruturn that,
    # oherwise, we continue the loop. No recording is needed to avoid
    # any duplication or any wrong information being brought into 
    # the next loop. 
    
    # wait, what if there is just no small value? what if the 
    # solver just ends because it cannot find a solution?
    
    # if Edot_residual < 0.05 and T_residual < 0.05:
    
    return Edot_residual, T_residual
    


