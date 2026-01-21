#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:49:38 2025

@author: Jia Wei Teh

Paper figure: ISM pressure (PISM) comparison.

USAGE:
    Configure the output base directory before running, or set environment variable:
    - TRINITY_OUTPUT_DIR: Base directory containing simulation outputs

    Expected directory structure (searches for .jsonl first, then .json):
        {TRINITY_OUTPUT_DIR}/1e5_sfe030_n1e4_PISM0/dictionary.jsonl (or .json)
        {TRINITY_OUTPUT_DIR}/1e5_sfe030_n1e4_PISM1e4/dictionary.jsonl (or .json)
        {TRINITY_OUTPUT_DIR}/1e5_sfe030_n1e4_PISM1e5/dictionary.jsonl (or .json)
        {TRINITY_OUTPUT_DIR}/1e5_sfe030_n1e4_PISM1e6/dictionary.jsonl (or .json)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import src._functions.unit_conversions as cvt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_path

# Output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA PATH - Configure this before running
# =============================================================================
# Base directory containing simulation outputs
OUTPUT_DIR = os.environ.get('TRINITY_OUTPUT_DIR', 'outputs')

plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))



#------

fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 150,)


alist = [0.3, 0.4, 0.6, 0.8, 1]

pressurelist = ['0', '10^4', '10^5', '10^6']

for ii, pressure in enumerate(['0', '1e4', '1e5', '1e6']):
    # Path to data file (prioritizes .jsonl over .json)
    base_path = f'{OUTPUT_DIR}/1e5_sfe030_n1e4_PISM{pressure}/dictionary'
    path2data = find_data_path(base_path)

    # Load using TrinityOutput reader
    output = load_output(path2data)

    rlist = list(output.get('R2'))
    tlist = list(output.get('t_now'))
    snaps = output  # Keep reference for rCloud access

    label = r"$P_{\mathrm{ISM}}/k_B = " + pressurelist[ii] + "\ \mathrm{K}\ \mathrm{cm}^{-3}$"

    plt.plot(tlist, rlist, c = 'k', alpha = alist[ii], label = label)


# Use rCloud from the last loaded run
rCloud = snaps[0].get('rCloud', 10.0)  # default if not found

plt.axhline(rCloud, linestyle = '--', c = 'b')

print('cloud radius is', rCloud)

plt.axhspan(0, rCloud, color = 'lightblue', alpha = 0.8)
plt.axhspan(rCloud, 1e3, color = 'lightblue', alpha = 0.3)
# plt.text(8, 10, 'ISM')

plt.yscale('log')
plt.xlabel('$t$ [Myr]')
plt.ylabel('$R_2$ [pc]')
plt.legend()
plt.xlim(min(tlist), 10)
plt.ylim(1e-1, 5e2)

plt.savefig(FIG_DIR / 'PISM.pdf', bbox_inches='tight')
print(f"Saved: {FIG_DIR / 'PISM.pdf'}")
plt.show()











