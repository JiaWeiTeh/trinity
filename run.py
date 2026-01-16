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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Parameter file loaded: {args.path2param}")

# Add file logging to output directory
out_dir = params['path2output'].value
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(out_dir, 'trinity.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {os.path.join(out_dir, 'trinity.log')}")


main.start_expansion(params)

