#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:37:22 2023

@author: Jia Wei Teh
"""
import time
import os
import sys
from src._output.terminal_prints import cprint as cpr
# get parameter
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])
# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

def display():
    
    # display logo for WARPFIELD
    show_logo()
    print(f'\t\t      --------------------------------------------------')
    print(f'\t\t      Welcome to'+' \033[32m'+link('https://github.com/JiaWeiTeh/warpfield3', 'TRINITY')+'\033[0m!\n')
    print(f'\t\t      Notes:')
    print(f'\t\t         - Documentation can be found \033[32m'+link('https://warpfield3.readthedocs.io/en/latest/', 'here')+'\033[0m.')
    print(f'\t\t         - \033[1m\033[96mBold text{cpr.END} indicates that a file is saved,')
    print(f'\t\t           and shows where it is saved.')
    print(f'\t\t         - {cpr.WARN}Warning message{cpr.END}. Code runs still.')
    print(f'\t\t         - {cpr.FAIL}Error encountered.{cpr.END} Code terminates.\n')
    print(f'\t\t      [Version 3.0] 2022. All rights reserved.')
    print(f'\t\t      --------------------------------------------------')
    # show initial parameters
    show_param()
    
    return


def show_logo():
    
    print(r"""
          ,          ______   ______     __     __   __     __     ______   __  __   
       \  :  /      /\__  _\ /\  == \   /\ \   /\ "-.\ \   /\ \   /\__  _\ /\ \_\ \  
    `. __/ \__ .'   \/_/\ \/ \ \  __<   \ \ \  \ \ \-.  \  \ \ \  \/_/\ \/ \ \____ \ 
    _ _\     /_ _      \ \_\  \ \_\ \_\  \ \_\  \ \_\\"\_\  \ \_\    \ \_\  \/\_____\
       /_   _\          \/_/   \/_/ /_/   \/_/   \/_/ \/_/   \/_/     \/_/   \/_____/
     .'  \ /  `.      
          '             © J.W. Teh, R.S. Klessen, S.O. Glover, K. Kreckel.                             
        """)

    return 


def link(url, label = None):
    if label is None: 
        label = url
    parameters = ''
    # OSC 8 ; params ; URL ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, url, label)

def show_param():
    # print some useful information
    print(f"{cpr.BLINK}Loading parameters:{cpr.END}")
    print(f'\tmodel name: {warpfield_params.model_name}')
    print(f'\tlog_mCloud: {warpfield_params.log_mCloud}')
    print(f'\tSFE: {warpfield_params.sfe}')
    print(f'\tmetallicity: {warpfield_params.metallicity}')
    print(f'\tdensity profile: {warpfield_params.dens_profile}')
    # shorten
    relpath = os.path.relpath(warpfield_params.out_dir, os.getcwd())
    print(f"{cpr.FILE}Summary: {relpath}/{warpfield_params.model_name}{'_summary.txt'}{cpr.END}")
    filename =  relpath + '/' + str(warpfield_params.model_name)+ '_config.yaml'
    print(f'{cpr.FILE}Verbose yaml: {filename}{cpr.END}')

    return





