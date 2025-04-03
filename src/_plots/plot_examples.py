#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:43:28 2025

@author: Jia Wei Teh

Just a main script that plots other things 
"""

import src._plots.betadelta as betadelta
import src._plots.current_status as current_status
import src._plots.dMdt as dMdt


# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/example_pl/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe001_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe001_n1e2/dictionary.json'
path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe010_n1e4/dictionary.json'

betadelta.plot(path2json)
current_status.plot(path2json)
dMdt.plot(path2json)
