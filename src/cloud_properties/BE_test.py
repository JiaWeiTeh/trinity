#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 18:57:49 2025

@author: Jia Wei Teh


another way of setting up a BE sphere, this time with focus on Pext
"""


import src._functions.unit_conversions as cvt
import astropy.constants as c
import numpy as np

path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4_BE/dictionary.json'



import json
with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

# for key, val in snaplists.items():
#     G = 
#     break
    
    


xi_out = 6.45
m_total = 1.18
M_total = 1e7 * 0.99
nCore = 1e4
gamma = 5/3

# for neutral gas
mu_atom = (14/11) * c.m_p.cgs.value
# for ionised
mu_ion = (14/23) * c.m_p.cgs.value

rhoCore = nCore * mu_ion


G = c.G.cgs.value
k_B = c.k_B.cgs.value
c_light = c.c.cgs.value



Pext_kb = 1e4
Pext = Pext_kb * k_B


# unit check
nCore *= cvt.ndens_cgs2au
rhoCore = rhoCore * cvt.ndens_cgs2au * cvt.g2Msun
k_B *= cvt.k_B_cgs2au
G *= cvt.G_cgs2au
c_light *= cvt.v_cms2au
mu_atom *= cvt.g2Msun
mu_ion *= cvt.g2Msun
Pext *= cvt.Pb_cgs2au



def cs2R(c_s):
    return np.sqrt(c_s**2 / (4 * np.pi * G * rhoCore)) * xi_out


def Pext2cs(Pext):
    return (Pext**0.5 * G**(3/2) * M_total/m_total)**(1/4)

def cs2T(c_s):
    return mu_ion * c_s**2 / gamma / k_B


print('cs is (this is in cgs)', Pext2cs(Pext) * cvt.v_au2cms)
print('temperature is (this is in cgs)', cs2T(Pext2cs(Pext)))
print('R is', cs2R(Pext2cs(Pext)))












#%%



import matplotlib.pyplot as plt

params = snaplists['1']

plt.subplots(dpi = 200, )

n_arr = params['initial_cloud_n_arr']
m_arr = params['initial_cloud_m_arr']
r_arr = params['initial_cloud_r_arr']

plt.plot(r_arr, np.array(n_arr) * cvt.ndens_au2cgs)
# plt.plot(r_arr, m_arr)
plt.yscale('log')
# plt.xscale('log')
plt.show()

sys.exit()








