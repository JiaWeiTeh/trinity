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
import os
    


# TODO: plot cloud radius and and shell radius.

def plot(path2json):
    print('reading... current phase')
    
    v3_r = []
    v3_v = []
    v3_E = []
    v3_T = []
    v3_rShell = []
    
    v3_t = []
        
    #--------------
    
    v3a_length = 0
    v3b_length = 0
    v3c_length = 0

    import json
    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)

    for key, val in snaplists.items():
        v3_t.append(val['t_now'])
        v3_r.append(val['R2'])
        try:
            v3_rShell.append(val['shell_grav_r'][-1])
        except:
            # asume if above fails that means shell is not there. 
            v3_rShell.append(val['R2'])
            print('here')
        v3_v.append(val['v2'])
        v3_E.append(val['Eb'])
        v3_T.append(val['T0'])
        
        # print(v3_rShell[-1] - v3_r[-1])
        # if val['current_phase'] == '1b':
        #     break


    # prefix of the final key
    # last_key = '_' + str(key.split('_')[1]) + '_'

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=12)
    fig, axs = plt.subplots(2, 2, figsize = (5,5), dpi = 150)
    
    fig.suptitle(f'{path2json}')
    
    v3_r = np.array(v3_r, dtype = float)
    v3_v = np.array(v3_v, dtype = float) / cvt.v_kms2au
    v3_E = np.array(v3_E, dtype = float) / cvt.E_cgs2au
    v3_T = np.array(v3_T, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    v3_rShell = np.array(v3_rShell, dtype = float)
    
    
    
    #-- r
    axs[0][0].plot(v3_t, v3_r, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[0][0].plot(v3_t[:len(v3_rShell)], v3_rShell, 'gray', alpha = 0.5, linewidth = 5, label = 'trinity')
    axs[0][0].set_ylabel('radius (pc)')
    axs[0][0].set_xlabel('time (Myr)')
    # axs[0][0].set_xlim(0, max(v3_t))
    # axs[0][0].set_xlim(0, 1)
    # axs[0][0].set_yscale('symlog')
    print(snaplists['0']['rCloud'])
    
    # print(snapshots)
    # axs[0][0].axhline(snaplists[0]['r_coll'], linestyle = '--', c = 'k', alpha = 0.7)
    axs[0][0].axhline(snaplists['0']['rCloud'], linestyle = '--', c = 'k', alpha = 0.7)
    
    # print(v3_r)
    # print(v3_rShell)
    #-- v
    
    
    axs[1][0].plot(v3_t, (v3_v), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[1][0].set_ylabel('log velocity (km/s)')
    axs[1][0].set_xlabel('time (Myr)')
    
    axs[1][0].set_yscale('symlog')
    axs[1][0].set_xlim(0, max(v3_t))
    axs[1][0].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    
    #-- E
    
    axs[0][1].plot(v3_t, (v3_E), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[0][1].set_ylabel('log Energy (ergs)')
    axs[0][1].set_xlabel('time (Myr)')
    axs[0][1].set_xlim(0, max(v3_t))
    axs[0][1].set_yscale('log')
        
    #-- T
    
    
    axs[1][1].plot(v3_t, (v3_T), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    axs[1][1].set_ylabel('log Temperature (K)')
    axs[1][1].set_xlabel('time (Myr)')
    axs[1][1].set_xlim(0, max(v3_t))
    axs[1][1].set_yscale('log')
    
    
    #-- final
    
    plt.tight_layout()  
    
    path2figure = os.path.dirname(path2json) 
    
    if not os.path.exists(path2figure):
        os.makedirs(path2figure)
    
    plt.savefig(os.path.join(path2figure, 'current_comparison.png'))
    plt.show()










