#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 17:20:50 2025

@author: Jia Wei Teh

Plots the density profile from inner bubble radius out until the outer shell radius, towards the ISM
"""

    
import json
import numpy as np
import matplotlib.pyplot as plt


path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe010_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4_BE/dictionary.json'

with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

xlist = []
ylist = []
    
#--------------


import json
with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)


bubble_r
bubble_n
shell_r
shell_n
ISM_n

bubble_T

for key, val in snaplists.items():
    xlist.append(val[xvalue])
    ylist.append(val[yvalue])


xlist = np.array(xlist)
ylist = np.array(ylist)

plt.style.use('/home/user/trinity/src/_plots/trinity.mplstyle')
fig, axs = plt.subplots(2, 1, figsize = (7,7), dpi = 300)

axs[0].set_yscale('log')
axs[0].set_xlabel(xvalue)
axs[0].set_ylabel(yvalue)
axs[0].plot(xlist, ylist)



axs[1].plot(xlist[1:], ylist[1:]- ylist[:-1])
axs[1].set_ylabel('residual')
# axs[1].set_ylim(-1e-10, 1e-10)
axs[1].axhline(0, c = 'k', linestyle = '--', alpha = 0.3)
axs[1].set_yscale('symlog')

plt.show()
plt.tight_layout(pad=10)







            

