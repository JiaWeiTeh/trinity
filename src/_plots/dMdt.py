#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:09:10 2024

@author: Jia Wei Teh 

This one simply plots the findings of phase_energy.py in warpfield4
"""


import numpy as np
import matplotlib.pyplot as plt

def plot(path2json):
    print('reading... dMdt')

    v3 = []
    v3res = []
    v3_t = []
    
    #--------------
    
    v3a_length = 0
    v3b_length = 0
    v3c_length = 0
    
    import json
    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        dictionary = json.load(f)

    for snapshots in dictionary.values():
        for key, val in snapshots.items():
            if key.endswith('t_now'):
                v3_t.append(val[0])
            elif key.endswith('dMdt') and 'bubble' not in key:
                v3.append(val[0])
            elif key.endswith('v0_residual'):
                v3res.append(val[0])
            elif key.endswith('current_phase'):
                if val == '1a':
                    v3a_length += 1
                elif val == '1b':
                    v3b_length += 1
                elif val == '1c':
                    v3c_length += 1
    
        

#--------------
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=12)
    fig, axs = plt.subplots(2, 1, figsize = (7,5), dpi = 150, height_ratios = [3, 1])
    
    
    v3data = np.array(v3, dtype = float)
    v3res = np.array(v3res, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    
    #--- dMdt
    
    axs[0].plot(v3_t, v3data, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[0].set_ylabel('dMdt')
    axs[0].set_xlabel('time (Myr)')
    axs[0].legend(loc = 'lower left')
    axs[0].set_xlim(0, max(v3_t))
    
    #--- velocity residual
    
    axs[1].plot(v3_t, v3res, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[1].axhline(0, c = 'k', linestyle = '--', alpha = 0.5)
    axs[1].set_ylabel('v0 residual')
    axs[1].set_xlabel('time (Myr)')    
    axs[1].set_ylim(-5e-2, 5e-2)
    
    # plt.legend()
    plt.tight_layout()  
    # path2figure = snapshots['path2output'].value + '/fig/'
    # plt.savefig(path2figure + 'dMdt_comparison.png')
    plt.show()






