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
params = read_param.read_param(args.path2param, write_summary = True)

from src._output import header
header.display()



from src import main
import src._input.create_dictionary as create_dictionary

main_dict = create_dictionary.create()

main.start_expansion(main_dict)


try:
    main_dict.flush()
except:
    pass

# try:
# except Exception as e:
#     print(f'Simulation exited. Error:\n{e}')
#     main_dict.flush()
#     pass
# # test

# Done!
print("Done!")

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

# import src.output_tools.write_outputs as write_outputs
# write_outputs.init_dir()