#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:09:10 2024

@author: Jia Wei Teh 

This one simply plots the findings of E2, v2, T, and R2.
"""


import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt
    
def plot(path2json):
    print('reading... current phase')
    
    v3_r = []
    v3_v = []
    v3_E = []
    v3_T = []
    
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
            elif key.endswith('R2'):
                v3_r.append(val[0])
            elif key.endswith('v2'):
                v3_v.append(val[0])
            elif key.endswith('Eb'):
                v3_E.append(val[0])
            elif key.endswith('T0'):
                v3_T.append(val[0])                
            elif key.endswith('current_phase'):
                if val == '1a':
                    v3a_length += 1
                elif val == '1b':
                    v3b_length += 1
                elif val == '1c':
                    v3c_length += 1
    

    # prefix of the final key
    last_key = '_' + str(key.split('_')[1]) + '_'

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=12)
    fig, axs = plt.subplots(2, 2, figsize = (5,5), dpi = 150)
    
    
    v3_r = np.array(v3_r, dtype = float)
    v3_v = np.array(v3_v, dtype = float) / cvt.v_kms2au
    v3_E = np.array(v3_E, dtype = float) / cvt.E_cgs2au
    v3_T = np.array(v3_T, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    
    #-- r
    
    axs[0][0].plot(v3_t, v3_r, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[0][0].set_ylabel('radius (pc)')
    axs[0][0].set_xlabel('time (Myr)')
    
    axs[0][0].axhline(snapshots[last_key+'r_coll'][0], linestyle = '--', c = 'k', alpha = 0.7)
    
    #-- v
    
    
    axs[1][0].plot(v3_t, np.log10(v3_v), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[1][0].set_ylabel('log velocity (km/s)')
    axs[1][0].set_xlabel('time (Myr)')
    
    axs[1][0].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    
    #-- E
    
    axs[0][1].plot(v3_t, np.log10(v3_E), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[0][1].set_ylabel('log Energy (ergs)')
    axs[0][1].set_xlabel('time (Myr)')
        
    #-- T
    
    
    axs[1][1].plot(v3_t, np.log10(v3_T), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[1][1].set_ylabel('log Temperature (K)')
    axs[1][1].set_xlabel('time (Myr)')
    
    
    #-- final
    
    plt.tight_layout()  
    # path2figure = snapshots['path2output'].value + '/fig/'
    # plt.savefig(path2figure + 'current_comparison.png')
    plt.show()
    










