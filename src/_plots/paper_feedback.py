#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:41:07 2025

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


# mCloud_list = ['1e5', '1e6', '1e7']
# mCloud_list = ['1e8']
# mCloud_list = ['1e7']
# mCloud_list = ['1e5']
# ndens_list = ['1e2', '1e4']
ndens_list = ['1e4']
sfe_list = ['001', '010', '030']
# sfe_list = ['001']
# sfe_list = ['001']


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

for mCloud in mCloud_list:
    for sfe in sfe_list:
        for ndens in ndens_list:
            try:
                path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/' + mCloud + '_' + 'sfe' + sfe + '_n' + ndens + '/dictionary.json'
                # path2json = path2json
                
                print(f"plotting mCloud {np.log10(float(mCloud))}, sfe {sfe}, ndens {float(ndens)}")
                
                with open(path2json, 'r') as f:
                    # step one is to make sure they are lists i think
                    snaplists = json.load(f)
            
                rlist = []
                tlist = []
                F_gravlist = []
                F_radlist = []
                F_ion_inlist = []
                F_ion_outlist = []
                F_ramlist = []
                F_ram_SNlist = []
                F_ram_windlist = []
                v2list = []
                phaselist = []
                
                #--------------
                
                for key, val in snaplists.items():
                    rlist.append(val['R2'])
                    tlist.append(val['t_now'])
                    F_gravlist.append(val['F_grav'])
                    F_radlist.append(val['F_rad'])
                    F_ion_inlist.append(val['F_ion_in'])
                    F_ion_outlist.append(val['F_ion_out'])
                    F_ramlist.append(val['F_ram'])
                    F_ram_SNlist.append(val['F_ram_SN'])
                    F_ram_windlist.append(val['F_ram_wind'])
                    v2list.append(val['v2'])
                    phaselist.append(val['current_phase'])
                
                fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 200) 
                
                
                plt.title(f'Mcloud = $10^{mCloud[-1]} M_\\odot$, $\\epsilon$ = {sfe}, n = $10^{ndens[-1]}$ 1/cm$^3$')
                
                # ax.text(0.47, 0.1, f'Mcloud = $10^{mCloud[-1]} M_\\odot$, $\\epsilon$ = {sfe}, n = $10^{ndens[-1]}$ 1/cm$^3$', fontsize = 10,
                #     transform=ax.transAxes,
                #     bbox=dict(facecolor="white", edgecolor="black", 
                #     boxstyle="round,pad=0.5", alpha = 1)
                #     )
                    
                    
                for ii, grav in enumerate(F_gravlist):
                    if hasattr(grav, "__len__"):
                        F_gravlist[ii] = grav[0]
                for ii, rad in enumerate(F_radlist):
                    if hasattr(rad, "__len__"):
                        F_radlist[ii] = rad[0]
                for ii, ion_out in enumerate(F_ion_outlist):
                    if hasattr(ion_out, "__len__"):
                        F_ion_outlist[ii] = ion_out[0]
                for ii, ion_in in enumerate(F_ion_inlist):
                    if hasattr(ion_in, "__len__"):
                        F_ion_inlist[ii] = ion_in[0]
                for ii, ram in enumerate(F_ramlist):
                    if hasattr(ram, "__len__"):
                        F_ramlist[ii] = ram[0]
                for ii, ram_SN in enumerate(F_ram_SNlist):
                    if hasattr(ram_SN, "__len__"):
                        F_ram_SNlist[ii] = ram_SN[0]
                for ii, ram_wind in enumerate(F_ram_windlist):
                    if hasattr(ram_wind, "__len__"):
                        F_ram_windlist[ii] = ram_wind[0]
                
                F_gravlist = np.array(F_gravlist)
                F_radlist = np.array(F_radlist)
                F_ion_outlist = np.array(F_ion_outlist)
                F_ion_inlist = np.array(F_ion_inlist)
                F_ramlist = np.array(F_ramlist)
                F_ram_SNlist = np.array(F_ram_SNlist)
                F_ram_windlist = np.array(F_ram_windlist)
                
                # Flist = [F_gravlist, F_radlist, F_ion_outlist, F_ram_windlist, F_ram_SNlist]
                Flist = [F_gravlist, F_ramlist, F_ion_outlist, F_radlist ]
                Flist = [F_gravlist, F_ram_windlist, F_ram_SNlist, F_ion_outlist, F_radlist ]
                # Flist = [F_gravlist, F_ramlist]
                # Flist = [F_gravlist, F_ram_windlist, F_ram_SNlist]
                
                Ftotal = np.sum(Flist, axis = 0)
                
                tlist = np.array(tlist)
                
                for jj, forces in enumerate(Flist):
                    ax.fill_between(tlist, np.sum(Flist[:(jj+1)], axis = 0)/Ftotal, color = 'k', alpha = 0.3)
                
                
                change_idx = np.flatnonzero(phaselist[1:] != phaselist[:-1]) + 1   # +1 because we compared shifted arrays
                change_t   = tlist[change_idx]
                for x in change_t:
                    ax.axvline(x, linestyle="--")   
                    
    
                
                n = 10
                
                print('F_gravlist', F_gravlist[:n:])
                print('F_radlist', F_radlist[:n:])
                print('F_ion_inlist', F_ion_inlist[:n:])
                print('F_ion_outlist', F_ion_outlist[:n:])
                print('F_ramlist', F_ramlist[:n:])
                print('F_ram_SNlist', F_ram_SNlist[:n:])
                print('F_ram_windlist', F_ram_windlist[:n:])
                
                ax.set_xlim(0, max(tlist))
                ax.set_xlabel('t [Myr]')
                ax.set_ylabel('F/Ftotal')
                ax.set_ylim(0, 1)
            
                ax.text(0.1, 0.2, 'grav')
                ax.text(0.1, 0.4, 'ram')
                ax.text(0.1, 0.6, 'ionised')
                ax.text(0.1, 0.8, 'rad')
            
                plt.show()
                plt.clf()

                # ========
            
                fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 200) 
                
                ax.set_yscale('log')
                
                c = ['k', 'b', 'g', 'c', 'r']
                
                
                for jj, forces in enumerate(Flist):
                    ax.plot(tlist, forces, c = c[jj])

                for x in change_t:
                    ax.axvline(x, linestyle="--")   
                
                ax.set_ylim(1e5, 1e10)
                ax.legend()
    

            except FileNotFoundError as e: 
                print(e)
                pass        
            except Exception as e:
                print(e)
                pass
        
                
        
        
        
        