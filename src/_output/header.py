#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:37:22 2023

@author: Jia Wei Teh
"""
import os
import numpy as np
from src._output.terminal_prints import cprint as cpr

def display(params):
    
    # display logo for WARPFIELD
    show_logo()
    print(f'\t\t      --------------------------------------------------')
    print(f'\t\t      Welcome to'+' \033[32m'+link('https://github.com/JiaWeiTeh/trinity', 'TRINITY')+'\033[0m!\n')
    print(f'\t\t      Notes:')
    print(f'\t\t         - Documentation can be found \033[32m'+link('https://trinitysf.readthedocs.io/en/latest/index.html', 'here')+'\033[0m.')
    print(f'\t\t         - \033[1m\033[96mBold text{cpr.END} indicates that a file is saved.')
    print(f'\t\t         - {cpr.WARN}Warning message{cpr.END}. Code runs still.')
    print(f'\t\t         - {cpr.FAIL}Error encountered.{cpr.END} Code terminates.\n')
    print(f'\t\t      [Version 3.0] 2022. All rights reserved.')
    print(f'\t\t      --------------------------------------------------')
    # show initial parameters
    show_param(params)
    
    return


def show_logo():
    
    print(r"""
          ,          ______   ______     __     __   __     __     ______   __  __   
       \  :  /      /\__  _\ /\  == \   /\ \   /\ "-.\ \   /\ \   /\__  _\ /\ \_\ \  
    `. __/ \__ .'   \/_/\ \/ \ \  __<   \ \ \  \ \ \-.  \  \ \ \  \/_/\ \/ \ \____ \ 
    _ _\     /_ _      \ \_\  \ \_\ \_\  \ \_\  \ \_\\"\_\  \ \_\    \ \_\  \/\_____\
       /_   _\          \/_/   \/_/ /_/   \/_/   \/_/ \/_/   \/_/     \/_/   \/_____/
     .'  \ /  `.      
          '             © J.W. Teh, R.S. Klessen                            
        """)

    return 


def link(url, label = None):
    if label is None: 
        label = url
    parameters = ''
    # OSC 8 ; params ; URL ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, url, label)

def show_param(params):
    # print some useful information
    print(f"{cpr.BLINK}Loading parameters:{cpr.END}")
    print(f"\tmodel name: {params['model_name'].value}")
    print(f"\tlog_mCloud: {np.log10(params['mCloud']/(1-params['sfe']))} Msun")
    print(f"\tSFE: {params['sfe'].value}")
    print(f"\tmetallicity: {params['ZCloud'].value} Zsun")
    print(f"\tdensity profile: {params['dens_profile'].value}")
    # shorten
    relpath = os.path.relpath(params['path2output'].value, os.getcwd())
    print(f"{cpr.FILE}Summary: {relpath}/{params['model_name'].value}{'_summary.txt'}{cpr.END}")

    return





