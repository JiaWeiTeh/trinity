#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:46:08 2025

@author: Jia Wei Teh
"""


    
import json
import numpy as np
import matplotlib.pyplot as plt


# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4/dictionary.json'
path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4_BE/dictionary.json'

with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

t_list = []
rShell_list = []
thickness_list = []
    
#--------------


import json
with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

for key, val in snaplists.items():
    t_list.append(val['t_now'])
    # rShell_list.append(val['rShell'])
    # thickness_list.append(val['shell_thickness'])
    thickness_list.append(val['Eb'])
    

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
fig, axs = plt.subplots(1, 1, figsize = (5,5), dpi = 200)

plt.plot(t_list, thickness_list)
plt.show()







            













