#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:43:28 2025

@author: Jia Wei Teh

Just a main script that plots other things 
"""

import phase1_wfld4_betadelta as betadelta
import phase1_wfld4_current as current
import phase1_wfld4_dMdt as dMdt


# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e6_001/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e6_010/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e6_030/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e7_001/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e7_010/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e7_030/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e5_001/dictionary.json' #temperature not monotonic
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e5_010/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e5_030/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e5_001_n1e2/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e6_001_n1e2/dictionary.json'
# path2json = r'/Users/jwt/Documents/Code/warpfield3/outputs/1e7_001_n1e2/dictionary.json'

path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/example_pl/dictionary.json'

betadelta.plot(path2json)
current.plot(path2json)
dMdt.plot(path2json)
# 

