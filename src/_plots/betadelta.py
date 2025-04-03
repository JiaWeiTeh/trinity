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
    print('reading... beta delta')

    v3_b = []
    v3_d = []
    v3_bres = []
    v3_dres = []
    
    v3Edot1 = []
    v3Edot2 = []
    v3T1 = []
    v3T2 = []
    
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
            elif key.endswith('beta') and 'transformation' not in key:
                v3_b.append(val[0])
            elif key.endswith('delta') and 'transformation' not in key:
                v3_d.append(val[0])
            elif key.endswith('Edot_residual'):
                v3_bres.append(val[0])
            elif key.endswith('T_residual'):
                v3_dres.append(val[0])                
            elif key.endswith('Edot1_guess'):
                v3Edot1.append(val[0])                
            elif key.endswith('Edot2_guess'):
                v3Edot2.append(val[0])                
            elif key.endswith('T1_guess'):
                v3T1.append(val[0])                
            elif key.endswith('T2_guess'):
                v3T2.append(val[0])                
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
    fig, axs = plt.subplots(3, 2, figsize = (7,7), dpi = 150, height_ratios = [3, 1, 3])
    
    v3data_b = np.array(v3_b, dtype = float)
    v3data_d = np.array(v3_d, dtype = float)
    v3data_bres = np.array(v3_bres, dtype = float)
    v3data_dres = np.array(v3_dres, dtype = float)
    v3Edot1 = np.array(v3Edot1, dtype = float)
    v3Edot2 = np.array(v3Edot2, dtype = float)
    v3T1 = np.array(v3T1, dtype = float)
    v3T2 = np.array(v3T2, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    
    
    #-- beta
    
    axs[0][0].plot(v3_t, v3data_b, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[0][0].set_ylabel('beta')
    axs[0][0].set_xlabel('time (Myr)')
    
    axs[0][0].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[0][0].axhline(1, linestyle = '--', c = 'r', alpha = 0.5)
    axs[0][0].legend()
    
    axs[0][0].set_xlim(0, max(v3_t))
    # axs[0][0].set_ylim(-0.05, 1)
    
    #-- beta residual
    
    
    axs[1][0].plot(v3_t, v3data_bres, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[1][0].set_ylabel('Edot residual')
    axs[1][0].set_xlabel('time (Myr)')
    
    axs[1][0].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[1][0].axhspan(-0.05, 0.05, color = 'b', alpha = 0.1)
    
    axs[1][0].set_xlim(0, max(v3_t))
    axs[1][0].set_ylim(-0.2, 0.2)
    
    
    #-- delta

    axs[0][1].plot(v3_t, v3data_d, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[0][1].set_ylabel('delta')
    axs[0][1].set_xlabel('time (Myr)')

    axs[0][1].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[0][1].axhline(1, linestyle = '--', c = 'r', alpha = 0.5)
    axs[0][1].legend(loc = 'lower left')
    
    axs[0][1].set_xlim(0, max(v3_t))
    # axs[0][1].set_ylim(-0.3, 0.05)
    
        
    #-- delta residual
    
    
    axs[1][1].plot(v3_t, v3data_dres, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[1][1].set_ylabel('T residual')
    axs[1][1].set_xlabel('time (Myr)')
    
    axs[1][1].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[1][1].axhspan(-0.05, 0.05, color = 'b', alpha = 0.1)
    
    axs[1][1].set_xlim(0, max(v3_t))
    axs[1][1].set_ylim(-0.2, 0.2)
    
    
    #-- Edot guesses 
    
    axs[2][0].plot(v3_t, v3Edot1, '-k', alpha = 1, linewidth = 2, label = 'Edot1 beta')
    axs[2][0].plot(v3_t, v3Edot2, '-b', alpha = 0.2, linewidth = 2, label = 'Edot2 eq')
    axs[2][0].set_ylabel('Edot guesses')
    axs[2][0].set_xlabel('time (Myr)')
    axs[2][0].set_yscale('symlog')
    
    axs[2][0].legend(loc = 'lower left')
    
    axs[2][0].set_xlim(0, max(v3_t))
    axs[2][0].set_ylim(1e6, 1e10)
    
    
    #-- T guesses 
    
    axs[2][1].plot(v3_t, v3T1, '-k', alpha = 1, linewidth = 2, label = 'Trgoal')
    axs[2][1].plot(v3_t, v3T2, '-b', alpha = 0.2, linewidth = 2, label = 'T0')
    axs[2][1].set_ylabel('T guesses')
    axs[2][1].set_xlabel('time (Myr)')
    axs[2][1].set_yscale('symlog')
    
    axs[2][1].legend(loc = 'lower left')
    
    axs[2][1].set_xlim(0, max(v3_t))
    axs[2][1].set_ylim(1e6, 1e8)
    
    #-- final
    
    plt.tight_layout()  
    # path2figure = snapshots['path2output'].value + '/fig/'
    # plt.savefig(path2figure + 'betadelta_comparison.png')
    plt.show()
    




