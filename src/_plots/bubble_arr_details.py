#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 09:44:05 2025

@author: Jia Wei Teh
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt


path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4/dictionary.json'

with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

bubble_rT_list = []
bubble_T_list = []
bubble_rn_list = []
bubble_n_list = []
    
#--------------


import json
with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

for key, val in snaplists.items():
    try:
        bubble_rT_list.append(val['bubble_T_arr_r_arr'])
        bubble_T_list.append(val['log_bubble_T_arr'])
        bubble_rn_list.append(val['bubble_n_arr_r_arr'])
        bubble_n_list.append(val['log_bubble_n_arr'])
    except:
        pass
    

plt.style.use('/home/user/trinity/src/_plots/trinity.mplstyle')
fig, axs = plt.subplots(2, 1, figsize = (8,5), dpi = 200)

axs[0].plot(bubble_rT_list[-1], bubble_T_list[-1])
axs[1].plot(bubble_rn_list[-1], np.log10(10**np.array(bubble_n_list[-1]) * cvt.ndens_au2cgs))
plt.show()











