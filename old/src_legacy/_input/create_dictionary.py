#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:04:16 2025

@author: Jia Wei Teh

This script handles the creation of dictionary
"""

from src._input.dictionary import DescribedItem, DescribedDict
import numpy as np


# Maybe move this to read_params so that we dont have to call trinity_params and can completely
# get rid of the dependency

def create():
    
    
# TODO: in future for multiple collapses, here we can add an if/else where counter 0 = reset, counter > 0 = retain most values.
    
    # --- initialise and prepare dictionary ---
    
    main_dict = DescribedDict()
    # how often do we save into json file?
    main_dict.snapshot_interval = 100
    
    
    # main
    main_dict['current_phase'] = DescribedItem('1a', 'Which phase is the simulation in? 1a: energy, 1b: implicit energy, 2: transition, 3: momentum')
    main_dict['isDissolution'] = DescribedItem(False, 'is the bubble currently dissolving')
    main_dict['collapse_counter'] = DescribedItem(0, 'How many times recollapse has happened.')
    
    
    # sequential star formation recordings
    main_dict['tSF_list'] = DescribedItem(np.array([0]), 'Records time of star formation')
    

    # tracking solving cooling parameters
    

    # mass loss calculations in bubble
    main_dict['v0'] = DescribedItem(0, 'pc/Myr. velocity at r1')
    main_dict['v0_residual'] = DescribedItem(0, 'pc/Myr. residual for v0 - 0/v0')
    
  
    # bubble calculations


    # simulation constraints
    
    
    # shell calculations
    # set dissolution time to arbitrary high number (i.e. the shell has not yet dissolved)
    main_dict['Rsh_max'] = DescribedItem(np.nan, 'maximum shell radius at any given time.')
    
    # shell output values
    # TODO: make sure these area are properly initiated with physically motivated values.
    # e.g.,  fraction starts with 1 or 0. np.nan may cause problem in fugture logic gates

    main_dict['isLowdense'] = DescribedItem(False, 'is the shell currently in low density?')
    main_dict['t_Lowdense'] = DescribedItem(1e30, 'Myr, time of most recent isLowdense==True')
    
    
    # TODO:
    # i thinjk this is not used? also there seem to be shell termination within shell_strcuture.
    main_dict['t_dissolve'] = DescribedItem(1e30, 'Time after which consider simulation as dissolved.') 
    main_dict['dEdt'] = DescribedItem(np.nan, 'Constant energy gradient over time; used for phase 1c and beyond.')

    
    
    
    return main_dict
    
