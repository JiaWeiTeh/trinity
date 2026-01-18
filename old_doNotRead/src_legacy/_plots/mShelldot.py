#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:41:55 2025

@author: Jia Wei Teh
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



#------

fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 150,)


alist = [0.2, 0.4, 0.8]
# alist = [0.8]

sfelist = ['001', '010', '030']
# sfelist = ['001']

for ii, sfe in enumerate(sfelist):
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4/dictionary.json'
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_BE/dictionary.json'
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe{sfe}_n1e4_BE/dictionary.json'
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe{sfe}_n1e2/dictionary.json'

    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)


    mlist = []
    mshell = []
    tlist = []
    phaselist = []
    
    #--------------
    
    for key, val in snaplists.items():
        mlist.append(val['shell_massDot'])
        mshell.append(val['shell_mass'])
        tlist.append(val['t_now'])

    # label = r"$P_{\mathrm{ISM}}/k_B = " + sfelist[ii] + "\ \mathrm{K}\ \mathrm{cm}^{-3}$"

    plt.plot(tlist, mlist, c = 'k', alpha = alist[ii])
    plt.plot(tlist, mshell, c = 'b', alpha = alist[ii])
    plt.yscale('log')
    # plt.xlim(0, 5)
    # plt.axhline(snaplists['1']['rCloud'], c = 'b', alpha = alist[ii])


#%%


fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 150,)


alist = [0.3, 0.4, 0.6]

sfelist = ['001', '010', '030']
# sfelist = ['010']

for ii, sfe in enumerate(sfelist):
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4/dictionary.json'
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_BE/dictionary.json'

    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)

    mlist = []
    tlist = []
    phaselist = []
    
    #--------------
    
    for key, val in snaplists.items():
        mlist.append(val['R2'])
        tlist.append(val['t_now'])

    # label = r"$P_{\mathrm{ISM}}/k_B = " + sfelist[ii] + "\ \mathrm{K}\ \mathrm{cm}^{-3}$"

    plt.plot(tlist, mlist, c = 'k', alpha = alist[ii])
    # plt.xscale('log')
    # plt.xlim(0, 5)
    plt.axhline(snaplists['1']['rCloud'], c = 'b', alpha = alist[ii])


#%%







alist = [0.3, 0.4, 0.6]

sfelist = ['001', '010', '030']
# sfelist = ['010']

for ii, sfe in enumerate(sfelist):
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4/dictionary.json'
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_BE/dictionary.json'

    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)

    t2list = []
    tlist = []
    phaselist = []
    
    #--------------
    
    for key, val in snaplists.items():
        # t2list.append(val['t_next'])
        tlist.append(val['t_now'])
        
    fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 150,)
    plt.plot(tlist)
    # plt.plot(t2list)
    plt.show()



#%%


import json
import matplotlib.pyplot as plt
import numpy as np


alist = [0.3, 0.4, 0.6]

sfelist = ['001', '010', '030']
# sfelist = ['001']

for ii, sfe in enumerate(sfelist):
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4/dictionary.json'
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_BE/dictionary.json'

    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)

    t2list = []
    tlist = []
    rlist = []
    mlist = []
    phaselist = []
    
    #--------------
    fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 200,)
    
    last_item = next(reversed(snaplists.items()))[1]
    
    # print(last_item.keys())
    
    tlist = np.array(last_item['array_t_now'])
    # tlist = np.array(last_item['array_mShell'])
    # tlist = np.array(last_item['array_R2'])
    
    tlist = tlist[1:] - tlist[:-1]
    plt.ylim(-1e-10, 1e-10)
    
    
    
    
    print(tlist)
    
    
    # plt.xlim(0, 200)
    
    plt.plot(tlist, linewidth = 1)
    
    # plt.yscale('symlog')
    plt.axhline(0, c ='k')
    # plt.plot(t2list)
    plt.show()
    break

#%%



fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 200,)


for ii, sfe in enumerate(sfelist):
    # path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4/dictionary.json'
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_BE/dictionary.json'

    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)

    t2list = []
    tlist = []
    rlist = []
    mlist = []
    phaselist = []
    
    #--------------
    
    last_item = next(reversed(snaplists.items()))[1]
    
    # print(last_item.keys())
    
    tlist = np.array(last_item['array_t_now'])
    rlist = np.array(last_item['array_mShell'])
    # rlist = np.array(last_item['array_mShellDot'])
    # rlist = np.array(last_item['array_R2'])
    
    # tlist = tlist[1:] - tlist[:-1]
    # plt.ylim(-1e-10, 1e-10)
    
    print(tlist)
    print(rlist)
    
    # plt.xlim(0, 0.05)
    # plt.ylim(0, 2)
    
    plt.plot(tlist, rlist, linewidth = 1)
    
    # plt.yscale('symlog')
    plt.axhline(0, c ='k')
    # plt.plot(t2list)
    # plt.show()
    # break








