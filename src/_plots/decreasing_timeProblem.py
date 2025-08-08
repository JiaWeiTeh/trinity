#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:07:48 2025

@author: Jia Wei Teh


investigating problems with decreasing time

"""



import json
import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
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






alist = [0.3, 0.4, 0.6]

sfelist = ['001', '010', '030']
# sfelist = ['010']

for ii, sfe in enumerate(sfelist):
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4/dictionary.json'
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe{sfe}_n1e4_BE/dictionary.json'
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_BE/dictionary.json'

    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)

    t2list = []
    tlist = []
    phaselist = []
    
    #--------------
    ii = 0
    phaseNow = '0'
    
    for key, val in snaplists.items():
        # t2list.append(val['t_next'])
        tlist.append(val['t_now'])
        if val['current_phase'] != phaseNow:
            phaseNow = val['current_phase']
            phaselist.append(ii)
        ii += 1
        
    tlist = np.array(tlist)
        
    fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 150,)
    plt.plot(tlist[1:] - tlist[:-1])
    plt.ylim(-1e-3, 1e-3)
    plt.axhline(0, c = 'k', linestyle = '--', alpha = 0.5)
    for jj in phaselist:
        plt.axvline(jj, c = 'k', linestyle = '--', alpha = 0.5)
    # plt.yscale('symlog')
    # plt.plot(t2list)
    plt.show()
    # break











