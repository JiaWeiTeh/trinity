#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:21:52 2025

@author: Jia Wei Teh
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt



print('...plotting radius comparison')


mCloud_list = ['1e5', '1e6', '1e7']
ndens_list = ['1e2', '1e4']
sfe_list = ['001']



plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
fig, ax = plt.subplots(len(mCloud_list), len(ndens_list), figsize = (5, 7), dpi = 200)   
# row
for ii, mCloud in enumerate(mCloud_list):
    # column
    for jj, ndens in enumerate(ndens_list):
        # lines
        for sfe in sfe_list:
            
            tlist, r2list, v2list, Tlist, Elist, phaselist = [], [], [], [], [], []
        
            path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/' + mCloud + '_' + 'sfe' + sfe + '_n' + ndens + '/dictionary.json'

            print(f"plotting mCloud {np.log10(float(mCloud))}, sfe {sfe}, ndens {float(ndens)}")
            
            with open(path2json, 'r') as f:
                # step one is to make sure they are lists i think
                dictionary = json.load(f)        
                    
            
            for snapshots in dictionary.values():
                for key, val in snapshots.items():
                    if key.endswith('t_now'):
                        tlist.append(val[0])
                    elif key.endswith('R2'):
                        r2list.append(val[0])
                    elif key.endswith('v2'):
                        v2list.append(val[0])
                    elif key.endswith('Eb'):
                        Tlist.append(val[0])
                    elif key.endswith('T0'):
                        Elist.append(val[0])                
                    elif key.endswith('current_phase'):
                        phaselist.append(val[0])                
            
            
            ax[ii][jj].plot(tlist, r2list)

        
        
        
        
        
        
        
        
        