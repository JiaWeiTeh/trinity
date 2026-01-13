#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:09:10 2024

@author: Jia Wei Teh 

This one simply plots the findings of phase_energy.py in warpfield4
"""


import numpy as np
import matplotlib.pyplot as plt
import os

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
        snaplists = json.load(f)
        
    for key, val in snaplists.items():
        v3_t.append(val['t_now'])
        v3.append(val['bubble_dMdt'])
        # if val['current_phase'] == '1b':
        #     break
        # v3_d.append(val['cool_delta'])
        

    # for snapshots in snaplists:
    #     for key, val in snapshots.items():
    #         if key.endswith('t_now'):
    #             v3_t.append(val)
    #         elif key.endswith('dMdt') and 'bubble' not in key:
    #             v3.append(val)
    #         elif key.endswith('v0_residual'):
    #             v3res.append(val)
    #         elif key.endswith('current_phase'):
    #             if val == '1a':
    #                 v3a_length += 1
    #             elif val == '1b':
    #                 v3b_length += 1

#--------------
    
    plt.style.use('/home/user/trinity/src/_plots/trinity.mplstyle')
    fig, axs = plt.subplots(2, 1, figsize = (7,5), dpi = 150, height_ratios = [3, 1])
    
    fig.suptitle(f'{path2json}')

    v3data = np.array(v3, dtype = float)
    v3res = np.array(v3res, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    
    #--- dMdt
    
    axs[0].plot(v3_t, v3data, '-k', alpha = 1, linewidth = 2)
    axs[0].set_ylabel('dMdt')
    axs[0].set_xlabel('time (Myr)')
    axs[0].legend(loc = 'lower left')
    axs[0].set_xlim(0, v3_t[int(v3a_length+v3b_length-1)])
    
    # #--- velocity residual
    
    # axs[1].plot(v3_t, v3res, '-k', alpha = 1, linewidth = 2)
    # axs[1].axhline(0, c = 'k', linestyle = '--', alpha = 0.5)
    # axs[1].set_ylabel('v0 residual')
    # axs[1].set_xlabel('time (Myr)')    
    # axs[1].set_ylim(-1e-2, 1e-2)
    # axs[1].set_xlim(0, v3_t[int(v3a_length+v3b_length-1)])
    
    # plt.legend()
    plt.tight_layout()  

    path2figure = os.path.dirname(path2json) 
    
    if not os.path.exists(path2figure):
        os.makedirs(path2figure)
    
    plt.savefig(os.path.join(path2figure, 'dMdt_comparison.png'))
    plt.show()







