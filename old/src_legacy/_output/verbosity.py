#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:25:49 2023

@author: Jia Wei Teh

This script contains functions that outputs parameter depending on verbosity.
"""

import numpy as np
import os
import sys
import astropy.units as u
import astropy.constants as c
import time

test_param = {}
# test_param['thisisaverylongstring'] = ''
test_param['model_name'] = 'test_model'
test_param['time'] = 0.1  * u.Myr
test_param['r'] = 1e-4  * u.pc
test_param['v'] = 3000  * u.km / u.s
test_param['E'] = 1e51  * u.erg
test_param['Lbol'] = 1e46  * u.erg/u.s

def print_parameter(params_dict):
    
    # First, check how many parameters
    chunky_str, dict_length = create_printing_block(params_dict)
    
    # print parameters
    print(chunky_str)
    # \x1b[1A: move cursor up one line
    # \x1b[2K: delete last line
    sys.stdout.write('\x1b[1A\x1b[2K'*dict_length)
    
    return


def create_printing_block(params_dict):
    
    # initialise the chonky block of string
    chunky_str = ""
    
    # the longest string for formatting.
    longest_key = len(max(params_dict, key = len))
    
    for ii, (key, val) in enumerate(params_dict.items()):
        
        # check if it has astropy units, but not dimensionless. 
        hasUnit = False
        
        if hasattr(val, '_unit'):
            if val.unit != u.dimensionless_unscaled:
                hasUnit = True
            
        # if has unit, make sure to format them as well
        if hasUnit:
            chunky_str += f'{key:{longest_key+1}}: {val.value:.6e} [{val.unit}]'
        else:
            try:
                chunky_str += f'{key:{longest_key+1}}: {val:.6e}'
            except:
                chunky_str += f'{key:{longest_key+1}}: {val}'
        # add newline if the key isn't the last.
        if ii != len(params_dict) - 1:
            chunky_str += '\n'
    
    # length of dictionary
    dict_length = ii + 1
    
    return chunky_str, dict_length


def test():
    global test_param
    
    for i in range(30):
        time.sleep(0.5)
        test_param['time'] += 0.01 * u.Myr
        test_param['r'] *= 1.21  
        test_param['v'] *= 0.99
        test_param['E'] *= 0.78
        test_param['Lbol'] *= 1.15
    
        print_parameter(test_param)
    
    return




