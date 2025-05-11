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


# IDEA: a gridmap of mass vs density to check what casues recollapse
        
def str2float(string):
    # for sfe, 010 -> 0.1
    return float(string[0] + '.' + string[1:])


print('...plotting radius comparison')


mCloud_list = ['1e5', '1e6', '1e7']
ndens_list = ['1e2', '1e4']
sfe_list = ['001', '010', '030']
colour = ['r', 'b', 'g']
scatter_list = ['d', '^', 'o', 'x']
metallicity_list = ['_Z015', '']


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)

# Set default tick styles globally
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


# mCloud = '1e7'
# sfe = '001'
# ndens = '1e4'

for metallicity in metallicity_list:
        for mCloud in mCloud_list:
            for sfe in sfe_list:
                for ndens in ndens_list:
                    
                    try:
                        path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/' + mCloud + '_' + 'sfe' + sfe + '_n' + ndens + metallicity + '/dictionary.json'
                        
                        print(f"plotting mCloud {np.log10(float(mCloud))}, sfe {sfe}, ndens {float(ndens)}, Z {metallicity}")
                        
                        with open(path2json, 'r') as f:
                            # step one is to make sure they are lists i think
                            dictionary = json.load(f)        
                            
                            
                            
                        F_gravlist = []
                        F_radlist = []
                        F_ramlist = []
                        tlist = []
                        r2list = []
                        v2list = []
                        tlist = []
                        phaselist = []
                        tlist = []
                            
                        
                        for snapshots in dictionary.values():
                            for key, val in snapshots.items():
                                if key.endswith('t_now'):
                                    tlist.append(val[0])
                                elif key.endswith('R2'):
                                    r2list.append(val[0])
                                elif key.endswith('v2'):
                                    v2list.append(val[0])
                                elif key.endswith('current_phase'):
                                    phaselist.append(val[0])  
                                elif key.endswith('rCloud_au'):
                                    rCloud = val[0]
                                elif key.endswith('completed_reason'):
                                    completed_reason = val[0]
                                elif key.endswith('F_grav'):
                                    F_gravlist.append(val[0])
                                elif key.endswith('F_rad'):
                                    F_radlist.append(val[0])
                                elif key.endswith('F_ram'):
                                    F_ramlist.append(val[0])
                        
                        
                        fig, ax = plt.subplots(1, 1, figsize = (7, 5), dpi = 200) 
                        
                        
                        
                        ax.text(0.47, 0.1, f'Mcloud = $10^{mCloud[-1]} M_\\odot$, $\\epsilon$ = {sfe}, n = $10^{ndens[-1]}$ 1/cm$^3$, Z = {metallicity}', fontsize = 10,
                            transform=ax.transAxes,
                            bbox=dict(facecolor="white", edgecolor="black", 
                            boxstyle="round,pad=0.5", alpha = 1)
                            )
                            
                            
                        for ii, grav in enumerate(F_gravlist):
                            if hasattr(grav, "__len__"):
                                F_gravlist[ii] = grav[0]
                        for ii, rad in enumerate(F_radlist):
                            if hasattr(rad, "__len__"):
                                F_radlist[ii] = rad[0]
                        for ii, ram in enumerate(F_ramlist):
                            if hasattr(ram, "__len__"):
                                F_ramlist[ii] = ram[0]
                        
                        F_gravlist = np.array(F_gravlist)
                        F_radlist = np.array(F_radlist)
                        F_ramlist = np.array(F_ramlist)
                        
                        F_total = F_gravlist + F_radlist + F_ramlist
                        
                        
                        ax.fill_between(tlist, F_gravlist/F_total, color = 'k', alpha = 0.3)
                        ax.fill_between(tlist, (F_gravlist+F_ramlist)/F_total, color = 'k', alpha = 0.3)
                        ax.fill_between(tlist, 1, color = 'k', alpha = 0.3)
                        
                        
                        ax.set_xlim(0, max(tlist))
                        ax.set_xlabel('t [Myr]')
                        ax.set_ylabel('F/Ftotal')
                        ax.set_ylim(0, 1)
                    
                        plt.show()
                    except FileNotFoundError as e: 
                        print(e)
                        pass        
                    except Exception as e:
                        print(e)
                        pass

        
        
        
        
        
        
        
        
        
        