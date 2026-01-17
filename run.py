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
# Configure EARLY logging so read_param messages are captured
# =============================================================================
# This is a minimal setup - will be reconfigured after params are loaded
logging.basicConfig(
    level=logging.DEBUG,  # Start with DEBUG to capture everything
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
)
early_logger = logging.getLogger(__name__)
early_logger.debug("Early logging configured (pre-params)")


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
# Note: read_param logging is now captured with early config above
params = read_param.read_param(args.path2param, write_summary=True)

header.show_param(params)



from src import main
import src._input.create_dictionary as create_dictionary

# main_dict = create_dictionary.create()


# =============================================================================
# Reconfigure logging with params settings
# =============================================================================
# Now reconfigure with user's preferred log level from params
from src._functions.logging_setup import setup_logging

# Get log_level from params (default to INFO if not set)
log_level = 'INFO'
if 'log_level' in params:
    log_level = params['log_level'].value if hasattr(params['log_level'], 'value') else params['log_level']

logger = setup_logging(
    log_level=log_level,
    console_output=True,
    file_output=True,
    log_file_path=params['path2output'].value,
    log_file_name='trinity.log',
    use_colors=True,
)
logger.info(f"Parameter file loaded: {args.path2param}")
logger.info(f"Log level set to: {log_level}")


main.start_expansion(params)

