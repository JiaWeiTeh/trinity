#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:16:20 2025

@author: Jia Wei Teh

Paper figure: Bonnor-Ebert density profile visualization.

USAGE:
    Configure the data path below before running, or set environment variable:
    - TRINITY_DATA_PATH: Path to simulation output (.json or .jsonl)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import src._functions.unit_conversions as cvt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_path

# Output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA PATH - Configure this before running
# =============================================================================
# Option 1: Set path directly (without extension - will search for .jsonl first, then .json)
# base_path = '/path/to/your/outputs/simulation/dictionary'

# Option 2: Use environment variable (recommended for portability)
base_path = os.environ.get('TRINITY_DATA_PATH', 'outputs/simulation/dictionary')

# Find data file (prioritizes .jsonl over .json)
path2data = find_data_path(base_path)

# Load using TrinityOutput reader
output = load_output(path2data)

#--------------

# Get initial cloud profile from first snapshot
first_snap = output[0]
nlist = np.array(first_snap.get('initial_cloud_n_arr', []))
rlist = np.array(first_snap.get('initial_cloud_r_arr', []))
mlist = np.array(first_snap.get('initial_cloud_m_arr', []))
    
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

plt.savefig(FIG_DIR / 'BE_profile_dens.pdf', bbox_inches='tight')
print(f"Saved: {FIG_DIR / 'BE_profile_dens.pdf'}")
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





plt.savefig(FIG_DIR / 'BE_profile_mass.pdf', bbox_inches='tight')
print(f"Saved: {FIG_DIR / 'BE_profile_mass.pdf'}")
plt.show()


