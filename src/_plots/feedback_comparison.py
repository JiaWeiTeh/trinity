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
            
            path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/' + mCloud + '_' + 'sfe' + sfe + '_n' + ndens + '/dictionary.json'
            
            print(f"plotting mCloud {np.log10(float(mCloud))}, sfe {sfe}, ndens {float(ndens)}")
            
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
            
            
            
            ax.text(0.47, 0.1, f'Mcloud = $10^{mCloud[-1]} M_\\odot$, $\\epsilon$ = {sfe}, n = $10^{ndens[-1]}$ 1/cm$^3$', fontsize = 10,
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
        

#%%


completed_reason = ''

fig, ax = plt.subplots(len(mCloud_list), len(ndens_list), figsize = (5, 7), dpi = 200)   
# row
for ii, mCloud in enumerate(mCloud_list):
    # column
    for jj, ndens in enumerate(ndens_list):
        # lines
        
        for cc, sfe in enumerate(sfe_list):
            
            tlist, r2list, v2list, Tlist, Elist, phaselist = [], [], [], [], [], []
        
            try:
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
                        elif key.endswith('rCloud_au'):
                            rCloud = val[0]
                        elif key.endswith('completed_reason'):
                            completed_reason = val[0]
                
                ax[ii][jj].plot(tlist, r2list, label = f'sfe = {str2float(sfe)}', c = colour[cc])
                
                ax[ii][jj].axhline(rCloud, linestyle = '--', alpha = 0.8, c = colour[cc])
                
                ax[ii][jj].set_xlim(0, 15)
                ax[ii][jj].set_ylim(1, 100)
                ax[ii][jj].set_xlabel('t (Myr)')
                
                try:
                    idx = completed_reason_list.index(completed_reason)
                    print(completed_reason)
                    ax[ii][jj].scatter(tlist[-1], r2list[-1], marker = scatter_list[idx], label = f'{completed_reason}')
                except ValueError:
                    print('complete reason not found:', completed_reason)
                    pass
                except Exception as e:
                    print(e)
                
                ax[ii][jj].set_yscale('log')
                
                # if any(np.array(r2list) > 80):
                #     ax[ii][jj].axhline(100, linestyle = '--', alpha = 0.5)

            except FileNotFoundError as e:
                print(e)

# ADD CLOUD RADIUS IN THE FUTURE

# for final
ax[ii][jj].legend(fontsize = 10)


fig.text(0.0, 0.85, "$10^5 \\rm{M}_\odot$ Cloud", va='center', ha='center', rotation=90, fontsize=12)
fig.text(0.0, 0.55, "$10^6 \\rm{M}_\odot$ Cloud", va='center', ha='center', rotation=90, fontsize=12)
fig.text(0.0, 0.2, "$10^7 \\rm{M}_\odot$ Cloud", va='center', ha='center', rotation=90, fontsize=12)


fig.text(0.3, 1, "n = $100 \\rm{ cm}^3$", va='center', ha='center', fontsize=12 )
fig.text(0.75, 1, "n = $10^4 \\rm{ cm}^3$", va='center', ha='center', fontsize=12 )
        
        
plt.tight_layout()
plt.show()
    
    
        
        
        
        
        
        
        
        
        
        
        