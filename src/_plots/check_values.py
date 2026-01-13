#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:46:08 2025

@author: Jia Wei Teh
"""


    
import json
import numpy as np
import matplotlib.pyplot as plt


path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe010_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4_BE/dictionary.json'

with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

xlist = []
ylist = []
ylist2 = []
phaselist = []
    
#--------------


import json
with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)


# yvalue = 'shell_fAbsorbedIon'
# yvalue = 'shell_fAbsorbedWeightedTotal'
yvalue = 'shell_fAbsorbedNeu'
# yvalue2 = 'rShell'
xvalue = 't_now'
phase = 'current_phase'


for key, val in snaplists.items():
    xlist.append(val[xvalue])
    ylist.append(val[yvalue])
    # ylist2.append(val[yvalue2])
    phaselist.append(val[phase])


xlist = np.array(xlist)
ylist = np.array(ylist)
# ylist2 = np.array(ylist2)
phaselist = np.array(phaselist)



import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))
fig, axs = plt.subplots(2, 1, figsize = (7,7), dpi = 300)

# axs[0].set_yscale('log')
axs[0].set_xlabel(xvalue)
axs[0].set_ylabel(yvalue)
axs[0].plot(xlist, ylist)
# axs[0].plot(xlist, ylist2)

change_idx = np.flatnonzero(phaselist[1:] != phaselist[:-1]) + 1   # +1 because we compared shifted arrays
change_t   = xlist[change_idx]
for x in change_t:
    axs[0].axvline(x, linestyle="--")   


axs[1].plot(xlist[1:], ylist[1:]- ylist[:-1])
axs[1].set_ylabel('residual')
# axs[1].set_ylim(-1e-10, 1e-10)
axs[1].axhline(0, c = 'k', linestyle = '--', alpha = 0.3)
axs[1].set_yscale('symlog')

plt.show()
plt.tight_layout(pad=10)







            













