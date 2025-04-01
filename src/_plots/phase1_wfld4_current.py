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
    # grab argument
    print('reading... current phase')
    # path2filev3 = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_test.txt'
    # path2filev3 = r'/Users/jwt/Documents/Code/warpfield3/testresultv3_full_testwithoutlq.txt'
    path2filev4 = r'/Users/jwt/Documents/Code/warpfield3/testresultv4.txt'
    # open file
    # filev3 = open(path2filev3, "r")
    filev4 = open(path2filev4, "r")
    # get list of lines in the file, separated by \n
    # file_linesv3 = filev3.read().splitlines()
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
    
    v3_r = []
    v3_v = []
    v3_E = []
    v3_T = []
    
    
    v4_r = []
    v4_v = []
    v4_E = []
    v4_T = []
    
    
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
    
    
    
    # for ii, line in enumerate(file_linesv3):
    
    #     if 'R2 : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_r.append(part)
            
    #         if v3_inphase1a:
    #             v3a_length += 1
            
    #     if 'v2 : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_v.append(part)
            
    #     if 'Eb : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_E.append(part)
            
    #     if 'T0 : ' in line: 
    #         # Split the input string into parts
    #         part = line.split()[2]
    #         v3_T.append(part)
            
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
        
       
        
        if 'here are the parameters for calc_Lb.' in line: 
            # Split the input string into parts
            part = file_linesv4[ii+5].split()[-1]
            v4_E.append(part)
    
            part = file_linesv4[ii+6].split()[-1]
            v4_r.append(part)
            
            part = file_linesv4[ii+7].split()[-1]
            v4_t.append(part)
            
            part = file_linesv4[ii+8].split()[-1]
            v4_v.append(part)
            
            part = file_linesv4[ii+9].split()[-1]
            v4_T.append(part)
            
            if v4_inphase1a:
                v4a_length += 1
        
        if 'done with' in line:
            v4_inphase1a = False
        

#--------------

    # prefix of the final key
    last_key = '_' + str(key.split('_')[1]) + '_'


    # version_dict = {'4': v4, '3': v3}
    # data = version_dict[version]
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=12)
    fig, axs = plt.subplots(2, 2, figsize = (5,5), dpi = 150)
    
    # xval = np.arange(0, len(v4))    
    
    
    v3_r = np.array(v3_r, dtype = float)
    v3_v = np.array(v3_v, dtype = float) / cvt.v_kms2au
    v3_E = np.array(v3_E, dtype = float) / cvt.E_cgs2au
    v3_T = np.array(v3_T, dtype = float)
    v3_t = np.array(v3_t, dtype = float)
    
    v4_r = np.array(v4_r, dtype = float)
    v4_v = np.array(v4_v, dtype = float) / cvt.v_kms2au
    v4_E = np.array(v4_E, dtype = float) / cvt.E_cgs2au
    v4_T = np.array(v4_T, dtype = float)
    v4_t = np.array(v4_t, dtype = float)
    
    # # for limits
    # for ax in axs:
    #     for jj in ax:
    #         jj.set_xlim(0, 2)
    
    #-- r
    
    axs[0][0].plot(v3_t, v3_r, '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[0][0].plot(v4_t, v4_r, '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[0][0].set_ylabel('radius (pc)')
    axs[0][0].set_xlabel('time (Myr)')
    
    # axs.set_xscale('log')
    axs[0][0].axhline(snapshots[last_key+'r_coll'][0], linestyle = '--', c = 'k', alpha = 0.7)
    # axs[0].axvline(v3_t[v3a_length], linestyle = '--', c = 'k')
    # axs[0].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    # axs[0].legend()
    
    # axs[0][0].set_xlim(0, 2)
    # axs[0].set_ylim(min(min(v3data_b), min(v4data_b)), max(max(v3data_b),max(v4data_b)))
    # axs[0][0].set_ylim(0.6, 15)

    
    #-- v
    
    
    axs[1][0].plot(v3_t, np.log10(v3_v), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[1][0].plot(v4_t, np.log10(v4_v), '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[1][0].set_ylabel('log velocity (km/s)')
    axs[1][0].set_xlabel('time (Myr)')
    # axs[1].set_yscale('log')
    
    # axs[1].axvline(v3_t[v3a_length], linestyle = '--', c = 'k')
    axs[1][0].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    # axs[1].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    # axs[1].legend()
    
    # axs[1].set_xlim(0, max(v3_t))
    # axs[1].set_ylim(-0.5, 0.5)
    
    
    #-- E
    
    axs[0][1].plot(v3_t, np.log10(v3_E), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[0][1].plot(v4_t, np.log10(v4_E), '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[0][1].set_ylabel('log Energy (ergs)')
    axs[0][1].set_xlabel('time (Myr)')
    # axs.set_xscale('log')
    
    # axs[2].axvline(v3_t[v3a_length], linestyle = '--', c = 'k')
    # axs[2].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    # axs[2].legend()
    
    # axs[2].set_xlim(0, max(v3_t))
    # axs[2].set_ylim(min(min(v3data_d), min(v4data_d)), max(max(v3data_d),max(v4data_d)))
    # axs[2].set_ylim(-0., -0.3)
    
        
    #-- T
    
    
    axs[1][1].plot(v3_t, np.log10(v3_T), '-k', alpha = 1, linewidth = 2, label = 'trinity')
    # axs[1][1].plot(v4_t, np.log10(v4_T), '-b', alpha = 0.3, linewidth = 4, label = 'warpfield')
    axs[1][1].set_ylabel('log Temperature (K)')
    axs[1][1].set_xlabel('time (Myr)')
    # axs[1].set_yscale('log')
    
    # axs[3].axvline(v3_t[v3a_length], linestyle = '--', c = 'k')
    # axs[3].axhline(0, linestyle = '--', c = 'r', alpha = 0.5)
    # axs[1].axvline(v4_t[v4a_length], linestyle = '--', c = 'b')
    # axs[3].legend()
    
    # axs[1].set_xlim(0, max(v3_t))
    # axs[3].set_ylim(-0.5, 0.5)
    
    
    #-- problematic time?
    # axs[1].axvline(0.01273765, linestyle = '--', c = 'k', alpha = 0.5)
    # axs[3].axvline(0.01273765, linestyle = '--', c = 'k', alpha = 0.5)
    
    
    
    
    #-- final
    
    plt.tight_layout()  
    path2figure = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/graphs/'
    plt.savefig(path2figure + 'current_comparison.png')
    plt.show()
    










