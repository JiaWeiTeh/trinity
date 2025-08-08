#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:52:11 2025

@author: Jia Wei Teh
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt


path2json2 = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe030_n1e4/dictionary.json'
path2json1 = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4_BE/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe030_n1e4/dictionary.json'
# path2json2 = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe010_n1e2/dictionary.json'



    
#--------------

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True  # Show minor ticks
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 6        # Major tick size
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3        # Minor tick size
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1       # Major tick width
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["xtick.minor.width"] = 0.8     # Minor tick width
plt.rcParams["ytick.minor.width"] = 0.8




# find index at which values change.

def change_points(arr):
    arr = np.asarray(arr)
    return np.where(arr[1:] != arr[:-1])[0] + 1




#--

fig, axs = plt.subplots(2, 1, figsize = (5,5), dpi = 200)
    
    
for pp, path2json in enumerate([path2json1, path2json2]):

    import json
    
    rlist = []
    tlist = []
    phaselist = []

    
    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)
    
    for key, val in snaplists.items():
        rlist.append(val['R2'])
        tlist.append(val['t_now'])
        phaselist.append(val['current_phase'])
        
    phaselist[-1] = 'done'
    
    
    
    
    colour_map = { '1a': 'r',
                  '1b': 'g',
                  '2': 'b',
                  '3': 'k'
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
    

axs[0].text(0.23, 15, 'ENERGY DRIVEN', **textdict)
axs[0].text(2.63, 15, 'TRANSITION', **textdict)
axs[0].text(3.53, 15, 'MOMENTUM DRIVEN', **textdict)
# # plt.ylim(0, 35)

axs[1].text(0.23, 50, 'ENERGY DRIVEN', **textdict)
axs[1].text(1.63, 50, 'TRANSITION', **textdict)
axs[1].text(4.7, 50, 'MOMENTUM DRIVEN', **textdict)

plt.tight_layout()




path2fig = r'/Users/jwt/unsync/Code/Trinity/fig/'

# plt.savefig(path2fig + 'bubblePhase.pdf')
plt.show()





