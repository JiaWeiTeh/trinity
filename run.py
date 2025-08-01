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



from src import main
import src._input.create_dictionary as create_dictionary

# main_dict = create_dictionary.create()

from src._output import header
header.display(params)


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

try:
    main.start_expansion(params)
except Exception as e:
    print(f'Simulation exited. Error:\n{e}')
    params.flush()
    pass


# test

# main.start_expansion(params)

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