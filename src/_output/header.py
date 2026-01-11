#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:37:22 2023

@author: Jia Wei Teh

This module handles the display of the TRINITY header and initial parameters.
Uses direct print() for header display (shown before logging is configured).
"""
import os
import numpy as np
from src._output.terminal_prints import cprint as cpr
import src._functions.unit_conversions as cvt


def display():
    """
    Display the TRINITY welcome header and initial parameter summary.

    Note: This function uses print() instead of logging because it should
    display BEFORE logging is configured. This ensures the header appears
    first in the terminal output.

    Parameters
    ----------
    params : DescribedDict
        Dictionary of simulation parameters
    """
    # display logo for TRINITY
    show_logo()
    print('\t\t      --------------------------------------------------')
    print('\t\t      Welcome to' + ' \033[32m' + link('https://github.com/JiaWeiTeh/trinity', 'TRINITY') + '\033[0m!\n')
    print('\t\t      Notes:')
    print('\t\t         - Documentation can be found \033[32m' + link('https://trinitysf.readthedocs.io/en/latest/index.html', 'here') + '\033[0m.')
    print(f'\t\t         - \033[1m\033[96mBold text{cpr.END} indicates that a file is saved.')
    print(f'\t\t         - {cpr.WARN}Warning message{cpr.END}. Code runs still.')
    print(f'\t\t         - {cpr.FAIL}Error encountered.{cpr.END} Code terminates.\n')
    print('\t\t      [Version 3.0] 2022. All rights reserved.')
    print('\t\t      --------------------------------------------------')

    return


def show_logo():
    """Display the TRINITY ASCII art logo."""
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


def link(url, label=None):
    """
    Create a clickable hyperlink for terminal output.

    Parameters
    ----------
    url : str
        The URL to link to
    label : str, optional
        Display text for the link. Defaults to the URL.

    Returns
    -------
    str
        OSC 8 escape sequence for clickable link
    """
    if label is None:
        label = url
    parameters = ''
    # OSC 8 ; params ; URL ST <name> OSC 8 ;; ST
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, url, label)


def show_param(params):
    """
    Display initial parameter summary.

    Parameters
    ----------
    params : DescribedDict
        Dictionary of simulation parameters
    """
    print(f"{cpr.BLINK}Loading parameters:{cpr.END}")
    print(f"\tmodel name: {params['model_name'].value}")
    print(f"\tlog_mCloud: {np.log10(params['mCloud']/(1-params['sfe']))} Msun")
    print(f"\tSFE: {params['sfe'].value}")
    print(f"\tmetallicity: {params['ZCloud'].value} Zsun")
    print(f"\tlog10 Core density: {np.round(np.log10(params['nCore'].value*cvt.ndens_au2cgs), 1)} cm-3")
    print(f"\tdensity profile: {params['dens_profile'].value}")
    # shorten
    relpath = os.path.relpath(params['path2output'].value, os.getcwd())
    print(f"{cpr.FILE}Summary: {relpath}/{params['model_name'].value}{'_summary.txt'}{cpr.END}")

    return





