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

# Get class and write summary file
# Note: read_param uses logging which is not yet configured,
# so messages are deferred until after header display
params = read_param.read_param(args.path2param, write_summary=True)


from src import main
import src._input.create_dictionary as create_dictionary

# main_dict = create_dictionary.create()

# =============================================================================
# Display header FIRST (before logging is configured)
# =============================================================================
from src._output import header
header.display(params)

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


# # test if dictionary is working
# for i in range(2):
#     print(i)
#     params['verbose'].value += 1
#     params.save_snapshot()
# params.flush()
# for i in range(2):
#     print(i)
#     params['verbose'].value += 1
#     params.save_snapshot()
# params.flush()
    

# import sys
# sys.exit()

# try:
#     main.start_expansion(params)
# except Exception as e:
#     print(f'Simulation exited. Error:\n{e}')
#     params.flush()
#     pass


# test

main.start_expansion(params)

# try:
#     params.flush()
# except:
#     pass 


# try:
#     main.start_expansion(main_dict)
#     main_dict.flush()
#     print("Done!")
# except:
#     pass


# Done!

# TODO: add a way to plot from the dictionary. include maybe a json dictionary file pathname in the dictionary
# import src._plots.plot_examples as plot_examples
# plot_examples.plot()



# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

# import src.output_tools.write_outputs as write_outputs
# write_outputs.init_dir()