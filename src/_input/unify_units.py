#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:49:43 2023

@author: Jia Wei Teh

This script adds astropy units into the dictionary. 
"""

import src._functions.unit_conversions as cvt



import re

def parse_and_map_units(unit_string):
    # Step 1: Clean and split units safely (excluding '**')
    # whitespace
    unit_string = re.sub(r'\s+', '', unit_string)
    units = [uu.strip() for uu in re.split(r'(?<!\*)\*(?!\*)', unit_string)]

    # Step 2: Mapping for replacements
    unit_map = {
        'g': str(cvt.g2Msun),
        's': str(cvt.s2Myr),
        'cm': str(cvt.cm2pc),
        'km': str(cvt.cm2pc*1e-5),
        'K': str(1), # *1 because there is no additional factor (no unit change required)
        'Zsun': str(1),
        'pc': str(1),
        'Myr': str(1),
        'erg': str(cvt.E_cgs2au),
    }

    # Step 3: Replace base units while preserving exponents
    mapped_units = []
    for uu in units:
        match = re.match(r'^([a-zA-Z]+)(\*\*.*)?$', uu)
        if match:
            base, exp = match.groups()
            mapped_base = unit_map.get(base, base)
            new_unit = mapped_base + (exp if exp else '')
            mapped_units.append(new_unit)
        else:
            mapped_units.append(uu)  # fallback

    # a function that only allows string including 0-9, ., +-/*, e, and ().
    # this is a small check for safer eval().
    def is_valid_expression(s):
        pattern = r'^[0-9e+\-*/().]*$'
        return bool(re.fullmatch(pattern, s))

    # evaluate
    factor = 1
    for element in mapped_units:
        if is_valid_expression(element):
            factor *= eval(element)
        else:
            raise Exception(f'Unit contains odd expression: \'{element}\'. If the expression is right, include it in unit_map.')

    return factor











