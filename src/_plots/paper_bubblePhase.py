#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:52:11 2025

@author: Jia Wei Teh

Paper figure: Bubble phase visualization.

USAGE:
    Configure the data paths below before running, or set environment variables:
    - TRINITY_DATA_PATH1: Path to first simulation output (.json or .jsonl)
    - TRINITY_DATA_PATH2: Path to second simulation output (.json or .jsonl)
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


# =============================================================================
# DATA PATHS - Configure these before running
# =============================================================================
# Option 1: Set paths directly (without extension - will search for .jsonl first, then .json)
# base_path1 = '/path/to/your/outputs/simulation1/dictionary'
# base_path2 = '/path/to/your/outputs/simulation2/dictionary'

# Option 2: Use environment variables (recommended for portability)
base_path1 = os.environ.get('TRINITY_DATA_PATH1', 'outputs/simulation1/dictionary')
base_path2 = os.environ.get('TRINITY_DATA_PATH2', 'outputs/simulation2/dictionary')

# Find data files (prioritizes .jsonl over .json)
path2data1 = find_data_path(base_path1)
path2data2 = find_data_path(base_path2)



    
#--------------

import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))




# find index at which values change.

def change_points(arr):
    arr = np.asarray(arr)
    return np.where(arr[1:] != arr[:-1])[0] + 1




#--

fig, axs = plt.subplots(2, 1, figsize = (5,5), dpi = 200)
    
    
for pp, path2data in enumerate([path2data1, path2data2]):

    # Load using TrinityOutput reader
    output = load_output(path2data)

    rlist = list(output.get('R2'))
    tlist = list(output.get('t_now'))
    phaselist = list(output.get('current_phase', as_array=False))
    isCollapse_list = list(output.get('isCollapse', as_array=False))

    phaselist[-1] = 'done'

    # --- collapse line: first time isCollapse becomes True
    collapse_mask = np.array([bool(c) for c in isCollapse_list])
    idx_collapse = np.flatnonzero(collapse_mask)
    if idx_collapse.size:
        x_collapse = tlist[idx_collapse[0]]
        axs[pp].axvline(x_collapse, color="purple", ls="--", lw=1.8, alpha=0.6, zorder=10)
        axs[pp].text(
            x_collapse, 0.95, "Collapse",
            transform=axs[pp].get_xaxis_transform(),
            ha="left", va="top",
            fontsize=8, color="purple", alpha=0.8,
            rotation=90, zorder=11
        )
    
    
    
    
    colour_map = { 'energy': 'r',
                  'implicit': 'g',
                  'transition': 'b',
                  'momentum': 'k'
                  } 
    
    
    current_phase_idx = change_points(phaselist)
    
    
    current_phase_idx = np.concatenate([[0], current_phase_idx])
    
    
        
    axs[pp].plot(tlist, rlist,
             c = 'k',
             alpha = 0.8,
             )
    
    
    textdict = {             # text string
                'fontsize':10,                     # size of the text
                'fontfamily':'sans-serif',         # readable font
                'fontweight':'medium',             # weight
                'ha':'center', 'va':'center',        # alignment
                # 'bbox':dict(facecolor='white', alpha=0.7, boxstyle='round'),
                'rotation':90,
                # 'transform':axs.get_yaxis_transform(),
              }
    
    
    
    for ii in range(len(current_phase_idx)-1):
        
        idx = current_phase_idx[ii]
        idx_1 = current_phase_idx[ii+1]
        
        phase_at_idx = phaselist[idx]
        
        phase_encompassed = phaselist[idx]
        
        # version 1
        axs[pp].axvspan(tlist[idx], tlist[idx_1], 
                    color = colour_map[phase_encompassed],
                    edgecolor = None,
                    alpha = 0.3)
        
        # plt.axvspan(0, tlist[idx_1],
        #             color = 'grey',
        #             alpha = 0.3,
        #             )
        
        # plt.axvline(tlist[idx_1], linestyle = '--', color = 'k')
        
        # if ii != 0 and ii != len(current_phase_idx)-2:
        #     plt.text(tlist[idx_1+7], 32, 'Phase I', **textdict)
        
        axs[pp].set_xlim(min(tlist), max(tlist))
        # plt.xlim(1e-3, max(tlist))
        # axs[pp].set_xscale('log')
        # plt.yscale('log')
        axs[pp].set_xlabel('$t$ [Myr]')
        axs[pp].set_ylabel('$R_2$ [pc]')
        
        pass
    

# axs[0].text(0.23, 15, 'ENERGY DRIVEN', **textdict)
# axs[0].text(2.63, 15, 'TRANSITION', **textdict)
# axs[0].text(3.53, 15, 'MOMENTUM DRIVEN', **textdict)
# # # plt.ylim(0, 35)

# axs[1].text(0.23, 50, 'ENERGY DRIVEN', **textdict)
# axs[1].text(1.63, 50, 'TRANSITION', **textdict)
# axs[1].text(4.7, 50, 'MOMENTUM DRIVEN', **textdict)

plt.tight_layout()




# Output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PDF = True

if SAVE_PDF:
    plt.savefig(FIG_DIR / 'bubblePhase.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'bubblePhase.pdf'}")
plt.show()





