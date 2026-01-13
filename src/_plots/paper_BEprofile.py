#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:16:20 2025

@author: Jia Wei Teh

plots different radius profiles for BE
"""


import json
import numpy as np
import matplotlib.pyplot as plt
import src._functions.unit_conversions as cvt


# path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e5_sfe030_n1e4/dictionary.json'
path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4_BE/dictionary.json'

with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)

nlist = []
rlist = []
mlist = []
phaselist = []
    
#--------------


import json
with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)
    
        # n_arr = params['initial_cloud_n_arr'].value
    # m_arr = params['initial_cloud_m_arr'].value
    # r_arr = params['initial_cloud_r_arr'].value
    
    # # plt.plot(r_arr, n_arr * cvt.ndens_au2cgs)
    
    

for key, val in snaplists.items():
    nlist = np.array(val['initial_cloud_n_arr'])
    rlist = np.array(val['initial_cloud_r_arr'])
    mlist = np.array(val['initial_cloud_m_arr'])
    break
    
import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


#%%


fig, axs = plt.subplots(1, 1, figsize = (5, 5), dpi = 200)



# the line for scaling
xmin = 10**(-0.5)
xmax = 60

xrange = np.logspace(xmin, xmax, 1000)

yn2 = xrange**-2



# plt.plot(xrange, 10**(6.24) * yn2 - 10**(2.), linestyle = '--', c = 'b', label = '$n\propto r^{-2}$')
plt.plot(xrange, 10**(6.21) * yn2, linestyle = '--', c = 'b', label = '$n\propto r^{-2}$')
plt.plot(xrange, np.ones_like(xrange)*1e4, linestyle = '--', c = 'r', label = '$n\propto r^0$')





plt.plot(rlist, nlist * cvt.ndens_au2cgs, c = 'k', linewidth = 3)





plt.legend()
plt.xscale('log')
plt.xlim(xmin, xmax)
plt.ylim(10**(0.9), 1e5)
plt.yscale('log')
plt.ylabel('$n$ [$\mathrm{cm}^{-3}$]')
plt.xlabel('$r$ [pc]')

path2fig = r'/Users/jwt/unsync/Code/Trinity/fig/'

plt.savefig(path2fig + 'BE_profile_dens.pdf')
plt.show()


#---- mass


#%%


fig, axs = plt.subplots(1, 1, figsize = (5, 5), dpi = 200)



# the line for scaling
xmin = -1
xmax = 8

xrange = np.logspace(xmin, xmax, 1000)

# yn2 = xrange
yn3 = xrange**3


plt.plot(xrange, 10**(2.8)*yn3, linestyle = '--', c = 'r', label = '$M\propto r^{3}$')
plt.plot(xrange, 10**(5.35)*xrange, linestyle = '--', c = 'b', label = '$M\propto r$')

# plt.plot(xrange, 10**(6.24) * yn2 - 10**(2.), linestyle = '--', c = 'b', label = '$n\propto r^{-2}$')
# plt.plot(xrange, 10**(6.22) * yn2, linestyle = '--', c = 'b', label = '$n\propto r^{-2}$')
# plt.plot(xrange, np.ones_like(xrange)*1e4, linestyle = '--', c = 'r', label = '$n\propto r^0$')


plt.plot(rlist, mlist, c = 'k', linewidth = 3)



# plt.legend()
plt.xscale('log')
plt.xlim(10*(0.3), 50)
plt.ylim(1e4, 1e8)
plt.yscale('log')
plt.ylabel('$M$ [$\mathrm{M}_\odot$]')
plt.xlabel('$r$ [pc]')





path2fig = r'/Users/jwt/unsync/Code/Trinity/fig/'

plt.savefig(path2fig + 'BE_profile_mass.pdf')
plt.show()


