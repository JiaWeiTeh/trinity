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
    print('reading... beta delta')
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
    
    v3_b = []
    v3_d = []
    v3_bres = []
    v3_dres = []
    
    v3Edot1 = []
    v3Edot2 = []
    v3T1 = []
    v3T2 = []
    
    
    v4_b = []
    v4_d = []
    v4_bres = []
    v4_dres = []
    
    
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
                    
    # print(v3_bres)
    # if fullv3:
    
    # for ii, line in enumerate(file_linesv3):
    
    #     if 'beta : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_b.append(part)
            
    #         if v3_inphase1a:
    #             v3a_length += 1
            
    #     if 'delta : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_d.append(part)
            
    #     if 'beta_Edot_residual : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_bres.append(part)
            
    #     if 'delta_T_residual : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_dres.append(part)
            
    #     if 't_now : ' in line:
    #         part = line.split()[2]
    #         v3_t.append(part)
    
    #     if 'Phase 1a completed.' in line:
    #         v3_inphase1a = False
                
    # else:
    #     for ii, line in enumerate(file_linesv3a):
            
    #         if 'beta : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_b.append(part)
    #             v3a_length += 1
                
    #         if 'delta : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_d.append(part)
                
    #         if 'beta_Edot_residual : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_bres.append(part)
                
    #         if 'delta_T_residual : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_dres.append(part)
                
    #         if 't_now : ' in line:
    #             part = line.split()[2]
    #             v3_t.append(part)
    
    #     for ii, line in enumerate(file_linesv3b):
            
    #         if 'beta : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_b.append(part)
                
    #         if 'delta : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_d.append(part)
                
    #         if 'beta_Edot_residual : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_bres.append(part) 
                
    #         if 'delta_T_residual : ' in line: 
    #             # Split the input string into parts
    #             part = line.split()[2]
    #             v3_dres.append(part)  
                
    #         if 't_now : ' in line:
    #             part = line.split()[2]
    #             v3_t.append(part)
            
    #--------------
    
    
    for ii, line in enumerate(file_linesv4):
        
        # if 'guesses' in line: 
        #     # velocity
        #     v4val.append(float(line[10:10+10]))
       
        if 'here are the parameters for calc_Lb.' in line: 
            # Split the input string into parts
            beta = file_linesv4[ii+3].split()[-1]
            v4_b.append(beta)
            
            delta = file_linesv4[ii+4].split()[-1]
            v4_d.append(delta)
            
            time = file_linesv4[ii+7].split()[-1]
            v4_t.append(time)
            
            if v4_inphase1a:
                v4a_length += 1
        
        if 'done with' in line:
            v4_inphase1a = False
        

#--------------


    # print(v3_b)
    # version_dict = {'4': v4, '3': v3}
    # data = version_dict[version]
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=12)
    fig, axs = plt.subplots(3, 2, figsize = (7,7), dpi = 150, height_ratios = [3, 1, 3])
    
    # xval = np.arange(0, len(v4))    
    
    # print(v3_b)
    
    v3data_b = np.array(v3_b, dtype = float)
    v3data_d = np.array(v3_d, dtype = float)
    v3data_bres = np.array(v3_bres, dtype = float)
    v3data_dres = np.array(v3_dres, dtype = float)
    v4data_b = np.array(v4_b, dtype = float)
    v4data_d = np.array(v4_d, dtype = float)
    v3Edot1 = np.array(v3Edot1, dtype = float)
    v3Edot2 = np.array(v3Edot2, dtype = float)
    v3T1 = np.array(v3T1, dtype = float)
    v3T2 = np.array(v3T2, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    v4_t = np.array(v4_t, dtype = float)
    
    # # for limits
    # for ax in axs:
    #     for jj in ax:
    #         jj.set_xlim(0, 2)
            
    
    #-- beta
    
    axs[0][0].plot(v3_t, v3data_b, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[0][0].plot(v4_t, v4data_b, '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[0][0].set_ylabel('beta')
    axs[0][0].set_xlabel('time (Myr)')
    # axs.set_xscale('log')
    
    axs[0][0].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[0][0].axhline(1, linestyle = '--', c = 'r', alpha = 0.5)
    # axs[0][0].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    axs[0][0].legend()
    
    axs[0][0].set_xlim(0, max(v3_t))
    # axs[0][0].set_xlim(0, 0.1)
    axs[0][0].set_ylim(min(min(v3data_b), min(v4data_b)), max(max(v3data_b),max(v4data_b)))
    axs[0][0].set_ylim(-0.05, 1)
    

    
    #-- beta residual
    
    
    axs[1][0].plot(v3_t, v3data_bres, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[1].plot(v4_t, v4data_b, '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[1][0].set_ylabel('Edot residual')
    axs[1][0].set_xlabel('time (Myr)')
    # axs[1].set_yscale('log')
    
    # axs[1][0].axvline(v3_t[v3a_length], linestyle = '--', c = 'k')
    axs[1][0].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[1][0].axhspan(-0.05, 0.05, color = 'b', alpha = 0.1)
    # axs[1].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    # axs[1].legend()
    
    axs[1][0].set_xlim(0, max(v3_t))
    axs[1][0].set_ylim(-0.2, 0.2)
    
    
    #-- delta

    axs[0][1].plot(v3_t, v3data_d, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[0][1].plot(v4_t, v4data_d, '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[0][1].set_ylabel('delta')
    axs[0][1].set_xlabel('time (Myr)')
    # axs.set_xscale('log')

    axs[0][1].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[0][1].axhline(1, linestyle = '--', c = 'r', alpha = 0.5)
    axs[0][1].legend(loc = 'lower left')
    
    axs[0][1].set_xlim(0, max(v3_t))
    # axs[0][1].set_xlim(0, 0.1)
    # axs[2].set_ylim(min(min(v3data_d), min(v4data_d)), max(max(v3data_d),max(v4data_d)))
    axs[0][1].set_ylim(-0.3, 0.05)
    
        
    #-- delta residual
    
    
    axs[1][1].plot(v3_t, v3data_dres, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[1].plot(v4_t, v4data_b, '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[1][1].set_ylabel('T residual')
    axs[1][1].set_xlabel('time (Myr)')
    # axs[1].set_yscale('log')
    
    # axs[1][1].axvline(v3_t[v3a_length], linestyle = '--', c = 'k')
    axs[1][1].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    axs[1][1].axhspan(-0.05, 0.05, color = 'b', alpha = 0.1)
    # axs[1].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    # axs[3].legend()
    
    axs[1][1].set_xlim(0, max(v3_t))
    axs[1][1].set_ylim(-0.2, 0.2)
    
    
    #-- Edot guesses 
    
    axs[2][0].plot(v3_t, v3Edot1, '-k', alpha = 1, linewidth = 2, label = 'Edot1 beta')
    axs[2][0].plot(v3_t, v3Edot2, '-b', alpha = 0.2, linewidth = 2, label = 'Edot2 eq')
    axs[2][0].set_ylabel('Edot guesses')
    axs[2][0].set_xlabel('time (Myr)')
    axs[2][0].set_yscale('symlog')
    
    # axs[2][0].axvline(v3_t[v3a_length], linestyle = '-', c = 'k')
    # axs[2][0].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    axs[2][0].legend(loc = 'lower left')
    
    axs[2][0].set_xlim(0, max(v3_t))
    # axs[2][0].set_ylim(min(min(v3data_b), min(v4data_b)), max(max(v3data_b),max(v4data_b)))
    axs[2][0].set_ylim(1e6, 1e10)
    
    
    #-- T guesses 
    
    axs[2][1].plot(v3_t, v3T1, '-k', alpha = 1, linewidth = 2, label = 'Trgoal')
    axs[2][1].plot(v3_t, v3T2, '-b', alpha = 0.2, linewidth = 2, label = 'T0')
    # axs[1].plot(v4_t, v4data_b, '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[2][1].set_ylabel('T guesses')
    axs[2][1].set_xlabel('time (Myr)')
    axs[2][1].set_yscale('symlog')
    
    axs[2][1].legend(loc = 'lower left')
    
    axs[2][1].set_xlim(0, max(v3_t))
    # axs[2][0].set_ylim(min(min(v3data_b), min(v4data_b)), max(max(v3data_b),max(v4data_b)))
    axs[2][1].set_ylim(1e6, 1e8)
    
    # axs[1].set_yscale('log')
    
    # axs[2][1].axvline(v3_t[v3a_length], linestyle = '--', c = 'k')
    # axs[2][1].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    # axs[1].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    # axs[3].legend()
    
    # axs[2][1].set_xlim(0, max(v3_t))
    # axs[2][1].set_ylim(-0.5, 0.5)
    
    
    
    #-- problematic time?
    # axs[1].axvline(0.01273765, linestyle = '--', c = 'k', alpha = 0.5)
    # axs[3].axvline(0.01273765, linestyle = '--', c = 'k', alpha = 0.5)
    
    
    
    
    #-- final
    
    plt.tight_layout()  
    path2figure = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/graphs/'
    plt.savefig(path2figure + 'betadelta_comparison.png')
    plt.show()
    








