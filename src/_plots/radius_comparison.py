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

completed_reason_list = ['Stopping time reached', 'Small radius reached', 'Large radius reached', 'Shell dissolved', 'Bubble radius larger than cloud']


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
                            Elist.append(val[0])
                        elif key.endswith('T0'):
                            Tlist.append(val[0])                
                        elif key.endswith('current_phase'):
                            phaselist.append(val[0])  
                        elif key.endswith('rCloud_au'):
                            rCloud = val[0]
                        elif key.endswith('completed_reason'):
                            completed_reason = val[0]
                
                ax[ii][jj].plot(tlist, r2list, label = f'sfe = {str2float(sfe)}', c = colour[cc])
                
                ax[ii][jj].axhline(rCloud, linestyle = '--', alpha = 0.8, c = colour[cc])
                
                ax[ii][jj].set_xlim(0, 5)
                ax[ii][jj].set_ylim(1, 100)
                ax[ii][jj].set_xlabel('t (Myr)')
                
                try:
                    idx = completed_reason_list.index(completed_reason)
                    print(completed_reason)
                    # TODO:
                    # ax[ii][jj].scatter(tlist[-1], r2list[-1], marker = scatter_list[idx], label = f'{completed_reason}')
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


fig.text(0.3, 1, "n = $100 \\rm{ cm}^{-3}$", va='center', ha='center', fontsize=12 )
fig.text(0.75, 1, "n = $10^4 \\rm{ cm}^{-3}$", va='center', ha='center', fontsize=12 )
        
        
plt.tight_layout()
plt.show()
    
    
        
        
        
        
        
        
        
        
        
        
        