#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:09:10 2024

@author: Jia Wei Teh 

This one simply plots the findings of phase_energy.py in warpfield4
"""


import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt

def plot(path2json):
    # grab argument
    # path2filev3 = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_test1b_withlq.txt'
    print('reading... dMdt')
    
    # path2filev3 = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_testwithoutlq.txt'
    # path2filev3 = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_test.txt'
    # path2filev3 = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_test_dmdt.txt'
    # path2filev3a = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_test.txt'
    # path2filev3b = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_test1b_withlq.txt'
    path2filev4 = r'/Users/jwt/Documents/Code/warpfield3/testresultv4.txt'
    # open file
    # filev3 = open(path2filev3, "r")
    # filev3a = open(path2filev3a, "r")
    # filev3b = open(path2filev3b, "r")
    filev4 = open(path2filev4, "r")
    # get list of lines in the file, separated by \n
    # file_linesv3 = filev3.read().splitlines()
    # file_linesv3a = filev3a.read().splitlines()
    # file_linesv3b = filev3b.read().splitlines()
    file_linesv4 = filev4.read().splitlines()
    # file size
    
    # go through each line to search for matching string. Can of course
    # use regex but not necessary.
    # TODO: in the future, allow user to use regex expressions, too.
    # record previous line to know evolution
    
    # loop through
    
    #--------------
    
    
    v3length = 0
    v4length = 0
    
    # for ii, line in enumerate(file_linesv3):
    #     if 'completed a phase in ODE_equations in implicit_phase' in line: 
    #         v3length += 1
    # for ii, line in enumerate(file_linesv4):
    #     if 'completed a phase in ODE_equations in implicit_phase' in line: 
    #         v4length += 1
    
    v3 = []
    v3res = []
    v4 = []
    v3_t = []
    v4_t = []
    
        
    #--------------
    
    v3a_length = 0
    v3b_length = 0
    v3c_length = 0
    v4a_length = 0
    v4_inphase1a = True
    v3_inphase1a = True
    
     
    # =============================================================================
    fullv3 = True
    # =============================================================================
    
    
   
    # path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/dictionary.json'

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
    
    # # if fullv3:
    # for ii, line in enumerate(file_linesv3):
    
    #     if 'dMdt : ' in line and 'bubble' not in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3.append(part)
            
    #         if v3_inphase1a:
    #             v3a_length += 1
            
    #     if 'Phase 1a completed.' in line:
    #         v3_inphase1a = False
            
    #     if 't_now : ' in line:
    #         part = line.split()[2]
    #         v3_t.append(part)
            
    #     if 'v0_residual : ' in line:
    #         part = line.split()[2]
    #         v3res.append(part)
                
    # if we have two files for phase1a and phase 1b
    # else:
    #     for ii, line in enumerate(file_linesv3a):
            
    #         if 'dMdt : ' in line and 'bubble' not in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3.append(part)
    #             v3a_length += 1
                
    #         if 't_now : ' in line:
    #             part = line.split()[2]
    #             v3_t.append(part)
                
        
    #     for ii, line in enumerate(file_linesv3b):
            
    #         if 'dMdt : ' in line and 'bubble' not in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3.append(part)
                
    #         if 't_now : ' in line:
    #             part = line.split()[2]
    #             v3_t.append(part)
                
    #--------------
    
    
    for ii, line in enumerate(file_linesv4):
        
        # if 'guesses' in line: 
        #     # velocity
        #     v4val.append(float(line[10:10+10]))
       
        if 'dMdt_guess, dMdt' in line: 
            # Split the input string into parts
            dMdt = line.split()[-1][1:-1]
            v4.append(dMdt)
            
            if v4_inphase1a:
                v4a_length += 1
    
        if 'here are the parameters for calc_Lb.' in line: 
            time = file_linesv4[ii+7].split()[-1]
            v4_t.append(time)
            
        if 'done with' in line:
            v4_inphase1a = False
        

#--------------



    # version_dict = {'4': v4, '3': v3}
    # data = version_dict[version]
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=12)
    fig, axs = plt.subplots(2, 1, figsize = (7,5), dpi = 150, height_ratios = [3, 1])
    
    # xval = np.arange(0, len(v4))    
    
    v3data = np.array(v3, dtype = float)
    v3res = np.array(v3res, dtype = float)
    v4data = np.array(v4, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    v4_t = np.array(v4_t, dtype = float)
    
    
    #--- dMdt
    
    axs[0].plot(v3_t, v3data, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[0].plot(v4_t, v4data, '-b', alpha = 0.3, linewidth = 2, label = 'warpfield')
    axs[0].set_ylabel('dMdt')
    axs[0].set_xlabel('time (Myr)')
    axs[0].legend(loc = 'lower left')
    # axs.set_xscale('log')
    
    # axs[0].axvline(v3_t[v3a_length-1], linestyle = '--', c = 'k')
    # axs[0].axvline(v4_t[v4a_length], linestyle = '--', c = 'b', alpha = 0.5)
    
    # axs[0].set_ylim(1750, max(v3data)+100)
    axs[0].set_xlim(0, max(v3_t))
    
    # axs.set_yscale('log')
    # axs[0].axhline(0.8, c = 'k')
    # axs[0][0].set_yscale('symlog')
    
    # axs[1].plot(v4data, '-k', alpha = 1, linewidth = 2)
    # axs[1].set_ylabel('dMdt')
    # axs[1].set_xlabel('time')
    # axs[1].set_xscale('log')
    
    # axs[1].plot(data[:,1])
    # axs[1].set_ylabel('delta')
    # axs[1].axhline(-.2, c = 'k')
    # # axs[0][1].set_yscale('symlog')
    
    # for aa in range(2):
    #     axs[aa].set_xlabel('Time')
    
    #--- velocity residual
    
    axs[1].plot(v3_t, v3res, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[0].plot(v4_t, v4data, '-b', alpha = 0.3, linewidth = 2, label = 'warpfield')
    axs[1].axhline(0, c = 'k', linestyle = '--', alpha = 0.5)
    axs[1].set_ylabel('v0 residual')
    axs[1].set_xlabel('time (Myr)')    
    axs[1].set_ylim(-5e-2, 5e-2)
    # axs[1].set_yscale('symlog')
    
    # plt.legend()
    plt.tight_layout()  
    path2figure = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/graphs/'
    plt.savefig(path2figure + 'dMdt_comparison.png')
    plt.show()
    








