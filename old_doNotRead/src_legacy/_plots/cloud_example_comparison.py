#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:29:48 2025

@author: Jia Wei Teh
"""

import json
import numpy as np
import matplotlib.pyplot as plt



path2json_list = [\
                  r'/Users/jwt/unsync/Code/Trinity/outputs/cloud_example_PL/dictionary.json',
                  r'/Users/jwt/unsync/Code/Trinity/outputs/cloud_example_BE/dictionary.json',
                  r'/Users/jwt/unsync/Code/Trinity/outputs/cloud_example_homogeneous/dictionary.json',
    ]
    
    
radius_PL = []
radius_BE = []
radius_HO = []
t_PL = []
t_BE = []
t_HO = []
    
c_list = ['k', 'b', 'r']
radius_list = [radius_PL, radius_BE, radius_HO]
t_list = [t_PL, t_BE, t_HO]
label_list = ['PL', 'BE', 'Homogeneous']
rCloud_list = [0, 0, 0]


for ii, paths in enumerate(path2json_list):
    with open(paths, 'r') as f:
        snaplists = json.load(f)
        
    for key, val in snaplists.items():
        rCloud_list[ii] = val['rCloud']
        radius_list[ii].append(val['R2'])
        t_list[ii].append(val['t_now'])




plt.subplots(1, 1, figsize = (5,5), dpi = 200)

for ii, (r, t) in enumerate(zip(radius_list, t_list)):
    if ii == 0:
        continue
    plt.plot(t[:-2], r[:-2], c = c_list[ii], label = label_list[ii])
    plt.axhline(rCloud_list[ii], linestyle = '--')
    
plt.xlabel('time [Myr]')
plt.ylabel('radius [pc]')
plt.legend()