#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:27:38 2022

@author: Jia Wei Teh

This script contains the main file to run WARPFIELD.

In the main directory, type (as an example):
   python3 ./run.py param/example.param
"""

import os
import logging
import argparse
import yaml
from src._input import read_param


# =============================================================================
# Read in parameter files
# =============================================================================
# parser
parser = argparse.ArgumentParser()
# Add option to read in file
parser.add_argument('path2param')
# grab argument
args = parser.parse_args()

from src._output import header
header.display()

# Get class and write summary file
# Note: read_param uses logging which is not yet configured,
# so messages are deferred until after header display
params = read_param.read_param(args.path2param, write_summary=True)

header.show_param(params)



from src import main
import src._input.create_dictionary as create_dictionary

# main_dict = create_dictionary.create()


# =============================================================================
# Configure logging AFTER header display
# =============================================================================
# This ensures header appears first, then logging messages follow
from src._functions.logging_setup import setup_logging

logger = setup_logging(
    log_level='INFO',
    console_output=True,
    file_output=True,
    log_file_path=params['path2output'].value,
    log_file_name='trinity.log',
    use_colors=True,
)
logger.info(f"Parameter file loaded: {args.path2param}")


main.start_expansion(params)

