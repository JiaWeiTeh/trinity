#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 10:29:13 2025

@author: Jia Wei Teh
"""

import os
import sys
import json
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt

from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets

import outputs.example_pl.example_pl_settings as trinity_params

import importlib



mCloud_list = ['1e7']
# ndens_list = ['1e4_BE']
ndens_list = ['1e4_BE']
sfe_list = ['010', '030', '001']

for mCloud in mCloud_list:
    for ndens in ndens_list:
        for sfe in sfe_list:
                    
            # try:
                
            print(f'creating gif for (sfe:{sfe}, mCloud:{mCloud}, ndens:{ndens}) ...')
                        
            path2output = r'/Users/jwt/unsync/Code/Trinity/outputs/' + mCloud + '_' + 'sfe' + sfe + '_n' + ndens
            
            path2json = path2output + '/dictionary.json' 
            
            with open(path2json, 'r') as f:
                # step one is to make sure they are lists i think
                snapshot_list = json.load(f)        
                
                
            tlist = []
            r2list =[]
            r1list =[]
            r_Tlist = []
            nlist = []
            r_nlist = []
            rCloud_list = []
            phaselist = []
            Tlist = []
            shell_rlist = []
            
            current_t = 1e-10
            completed_reason  = ''
            
            
            for key, val in snapshot_list.items():
                tlist.append(val['t_now'])
                r2list.append(val['R2'])
                r1list.append(val['R1'])
                r_Tlist.append(val['bubble_T_arr_r_arr'])
                nlist.append(val['log_bubble_n_arr'])
                r_nlist.append(val['bubble_n_arr_r_arr'])
                rCloud_list.append(val['rCloud'])
                phaselist.append(val['current_phase'])
                Tlist.append(val['log_bubble_T_arr'])
                shell_rlist.append(val['shell_grav_r'])
                                                   

                        
            plt.rc('text', usetex=True)
            plt.rc('font', family='sans-serif', size=12)
            
                                            
            fig, axs = plt.subplots(3, 1, figsize = (10, 5), dpi = 300, height_ratios=(2, 1, 1))   
            
            axs2 = axs[2].twinx()
                
            # fig.patch.set_alpha(0)  # transparent figure background
            
            
            shell_max = []
            
            
            def update(frame):
                
                if not hasattr(shell_rlist[frame], "__len__"):
                    # lazy way to make nan into 0, since nan only appears if its not a list (during initialisation)
                    if np.isnan(shell_rlist[frame]):
                        shell_rlist[frame] = r2list[frame]
                    shell_rlist[frame] = [shell_rlist[frame]]
                   
                elif len(shell_rlist[frame]) == 0:
                    shell_rlist[frame] = r2list[frame]
                    
                print(r2list[frame]) 
                
                shell_max.append(max(shell_rlist[frame]))
                
                    
                # try:
                #     n = nlist[frame]
                # except IndexError as e:
                #     print(e)
                #     nlist[frame] = np.nan
            
                for ii, ax in enumerate(axs):
                    ax.clear()
                    axs2.clear()
                    if ii == 0:
                        continue
                    # ax.set_facecolor("none")  # transparent axes background
                    # bubble
                    ax.axvspan(0, r1list[frame], color = 'yellow', label = 'bubble (free-streaming)', alpha = 0.3)
                    
                    # outer bubble
                    ax.axvspan(r1list[frame], r2list[frame], color = 'red', alpha = 0.5, label = 'bubble (ionised)')
                    
                    # shell
                    ax.axvspan(r2list[frame], max(shell_rlist[frame]), color = 'darkgrey', label = 'shell')
            
                    # cloud
                    ax.axvspan(max(shell_rlist[frame]), rCloud_list[frame], color = 'lightblue', label = 'cloud')
            
            
                # axs[0].plot(tlist[:frame], shell_max[:frame], color = 'gray')
                axs[0].plot(tlist[:frame], r2list[:frame], color = 'red')
                axs[0].set_xlabel('t [Myr]')
                axs[0].set_ylabel('Radius [pc]')
                axs[0].set_ylim(0, max(rCloud_list)) 
                axs[0].set_xlim(0, max(tlist))
                axs[0].text(0.85, 0.8, f'$t=$ {np.round(tlist[frame], 4)} Myr', fontsize = 10,
                            transform=axs[0].transAxes,
                            bbox=dict(facecolor="white", edgecolor="black", 
                            boxstyle="round,pad=0.5", alpha = 0.5)
                            )
                
                axs[1].yaxis.set_visible(False)
                axs[1].set_xlim(0, 1.1 * rCloud_list[frame])
                axs[1].legend(loc = 'lower right', frameon=True, facecolor="lightgray", edgecolor="black",
                                               framealpha = 0.5, fontsize = 6)
                axs[1].set_xlabel('Radius [pc]') 
                
                
                try:
                    unit = cvt.ndens_cgs2au
                    # axs[2].set_ylim(4, 10)
                    # axs2.set_ylim(0, 7)
                    
                    axs[2].plot(r_nlist[frame], np.log10((10**np.array(nlist[frame]))/unit), color = 'k')
                    
                    
                    unit = cvt.v_kms2au
                    axs2.plot(r_Tlist[frame], np.array(Tlist[frame])/unit, color = 'b')
                    axs2.tick_params(axis='y', which ='both', colors='b') 
                    
                except:
                    # delete axis otherwise
                    axs[2].yaxis.set_visible(False)
                    axs2.yaxis.set_visible(False)
                    
                # try:
                #     axs[2].set_xlim(0, max(shell_rlist[frame]))
                # except:
                #     axs[2].set_xlim(0, r2list[frame])
                
                axs[2].set_xlim(r1list[frame], r2list[frame])
                
                # axs2.set_ylabel('log T [K]', color = 'b')
                # axs[2].set_ylabel('log density [cm-3]', color = 'k')
                axs[2].set_xlabel('Radius (zoom-in) [pc]')
            
                return axs
             
            plt.tight_layout()
            
            # ani = animation.FuncAnimation(fig=fig, func=update, frames=int(3),
            ani = animation.FuncAnimation(fig=fig, func=update, frames=int(len(rCloud_list)-1),
                                               # interval = interval
                                              )
            
            
            ani.save(filename=path2output+'/radius_evolution.gif', writer='pillow', fps=20, dpi=200, savefig_kwargs={
                'transparent': True,
                'facecolor': 'none'
            })
            
            print(f'The following gif (sfe:{sfe}, mCloud:{mCloud}, ndens:{ndens}) is saved.')
                
            # except Exception as e:
            #     print(f'The following parameters (sfe:{sfe}, mCloud:{mCloud}, ndens:{ndens}) cannot run because {e}')
                                    



