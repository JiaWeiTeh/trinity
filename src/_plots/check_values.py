#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:46:08 2025

@author: Jia Wei Teh
"""


    
import json


path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4_exceedcloud/dictionary.json'

with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)


v1, v2, v3 = [], [], []

for snapshots in snaplists:
    for key, val in snapshots.items():
        if key.endswith('T_rgoal'):
            v1.append(val)
        elif key.endswith('T_goal'):
            v2.append(val)
        elif key.endswith('r_goal'):
            v3.append(val)

#%%


print(v1[:4])
print(v1[:4])
print(v3[:4])
