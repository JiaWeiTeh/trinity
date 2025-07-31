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
import matplotlib.pyplot as plt



def plot(path2json):
    
    # lets try this but iwht 0.1 condition for Lloss Lgain
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=12)
    
    # Set default tick styles globally
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
    
    
    betadelta.plot(path2json)
    current_status.plot(path2json)
    # dMdt.plot(path2json)
    
    
    
    
    

# # path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/example_pl/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe001_n1e2/dictionary.json'





# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe001_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe010_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe030_n1e4/dictionary.json'


# =============================================================================
# homogeneous 1e7
# =============================================================================
# for sfe in ['001', '010', '030']:
#     path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4/dictionary.json'
#     # 1e4010_1e7 seems to have the problem where v suddenly blows up to -1e45.
#     plot(path2json)
    
    
    
# =============================================================================
# homogeneous 1e5
# =============================================================================
# for sfe in ['001', '010', '030']:
#     path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe{sfe}_n1e4/dictionary.json'
#     plot(path2json)
    


    
# =============================================================================
# PISM, homogeneous
# =============================================================================
# for sfe in ['010', '030']:
#     path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_PISM5e5/dictionary.json'
#     # 1e4010_1e7 seems to have the problem where v suddenly blows up to -1e45.
#     plot(path2json)


# =============================================================================
# BE spheres
# =============================================================================
for sfe in ['001', '010', '030']:
    path2json = f'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe{sfe}_n1e4_BE/dictionary.json'
    # 1e4010_1e7 seems to have the problem where v suddenly blows up to -1e45.
    plot(path2json)











# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/test_dict_BE/dictionary.json'






# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/cloud_example_homogeneous/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/cloud_example_PL/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/cloud_example_BE/dictionary.json'






# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe030_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4_exceedcloud/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4_exceedcloud/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e6_sfe030_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e6_sfe010_n1e4/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe001_n1e2/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e2/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe010_n1e2/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e2/dictionary.json'
# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4_Z015/dictionary.json'






