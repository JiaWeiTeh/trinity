#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:49:38 2025

@author: Jia Wei Teh
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt


plt.style.use('/home/user/trinity/src/_plots/trinity.mplstyle')



#------

fig, ax = plt.subplots(1, 1, figsize = (5,5), dpi = 150,)


alist = [0.3, 0.4, 0.6, 0.8, 1]

pressurelist = ['0', '10^4', '10^5', '10^6']

for ii, pressure in enumerate(['0', '1e4', '1e5', '1e6']):
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe030_n1e4_PISM{pressure}/dictionary.json'

    with open(path2json, 'r') as f:
        # step one is to make sure they are lists i think
        snaplists = json.load(f)

    rlist = []
    tlist = []
    phaselist = []
    
    #--------------
    
    for key, val in snaplists.items():
        rlist.append(val['R2'])
        tlist.append(val['t_now'])

    label = r"$P_{\mathrm{ISM}}/k_B = " + pressurelist[ii] + "\ \mathrm{K}\ \mathrm{cm}^{-3}$"

    plt.plot(tlist, rlist, c = 'k', alpha = alist[ii], label = label)


plt.axhline(snaplists['1']['rCloud'], linestyle = '--', c = 'b')

print('cloud radius is', snaplists['1']['rCloud'])

plt.axhspan(0, snaplists['1']['rCloud'], color = 'lightblue', alpha = 0.8)
plt.axhspan(snaplists['1']['rCloud'], 1e3, color = 'lightblue', alpha = 0.3)
# plt.text(8, 10, 'ISM')

plt.yscale('log')
plt.xlabel('$t$ [Myr]')
plt.ylabel('$R_2$ [pc]')
plt.legend()
plt.xlim(min(tlist), 10)
plt.ylim(1e-1, 5e2)

path2fig = r'/Users/jwt/unsync/Code/Trinity/fig/'

plt.savefig(path2fig + 'PISM.pdf')
plt.show()











