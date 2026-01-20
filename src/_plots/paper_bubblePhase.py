#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:52:11 2025

@author: Jia Wei Teh
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import src._functions.unit_conversions as cvt

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from load_snapshots import load_snapshots


# Paths to data files (can be .json or .jsonl)
path2data2 = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe010_n1e4/dictionary.json'
# path2data1 = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4_BE/dictionary.json'
path2data1 = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe030_n1e4/dictionary.json'
# path2data2 = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4/dictionary.json'
# path2data = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe010_n1e2/dictionary.json'



    
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

    rlist = []
    tlist = []
    phaselist = []

    # Load snapshots (supports both JSON and JSONL)
    snaps = load_snapshots(path2data)

    for snap in snaps:
        rlist.append(snap['R2'])
        tlist.append(snap['t_now'])
        phaselist.append(snap['current_phase'])
        
    phaselist[-1] = 'done'
    
    
    
    
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
plt.show()





