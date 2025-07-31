#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 12:28:45 2025

@author: Jia Wei Teh

This script shows the radius of GMC as a function of core density and cloud mass.
"""


import numpy as np
import astropy.constants as c 
import astropy.units as u
import scipy.integrate
import scipy.interpolate
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# All constants are given in cgs
# --- constants

# for neutral gas
mu_neu = (14/11) * c.m_p.cgs.value
# for ionised
mu_ion = (14/23) * c.m_p.cgs.value
# constants
G = c.G.cgs.value
kB = c.k_B.cgs.value
# conversions
g2Msun = u.g.to(u.Msun)
Msun2g = 1/g2Msun
cm2pc = u.cm.to(u.pc)
pc2cm = 1/cm2pc
# adiabatic
gamma = 5/3
# maximum density ratio
Omega = 14.1

# --- sound speed
def get_cs(T):
    if T > 1e4:
        return np.sqrt(gamma * kB * T / mu_ion)
    else:
        return np.sqrt(gamma * kB * T / mu_neu)
    
# --- lane-emden
def laneEmden(y,t):
    """ The Lane-Emden equation."""
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = du/dxi, let y = [u, dudxi],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    u, dudxi = y
    return [dudxi, np.exp(-u) - 2 * dudxi / t]

# --- solve lane-emden, returns also the interpolaltion of density contrast (rhoCore/rho(xi))
def solve_laneEmden():
    # initial values
    u0, dudxi0 = 1e-5, 1e-5
    # x values
    xi_array = np.logspace(-5, 4, 3000)
    # solve
    solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)
    # solution
    u_array = solution[:,0]
    dudxi_array = solution[:,1]
    # get density contrast
    rho_rhoc_array = np.exp(-u_array)
    # interpolation
    f_rho_rhoc = scipy.interpolate.interp1d(xi_array, rho_rhoc_array, kind='cubic', fill_value="extrapolate")
    # return
    return xi_array, u_array, dudxi_array, rho_rhoc_array, f_rho_rhoc

# --- create BE sphere, returningouter radius, outer density and sphere effective temperature
def create_BESphere(Omega, nCore, mCloud):
    
    # Omega = max(rhoCore/rho(xi)) ~ 13.8-14.1 
    
    xi_array, u_array, dudxi_array, rho_rhoc_array, f_rho_rhoc = solve_laneEmden()
    
    rhoCore = nCore * mu_ion
    
    n_outOmega = nCore / Omega
    
    # What is the effective temperature? This is the most important parameter
    # as it sets the pressure balance in the sphere
    def solve_structure(T, mCloud, rhoCore, f_rho_rhoc, n_outOmega):
        
        # sound speed
        c_s = get_cs(T)
        
        # with this sound speed, what is xi_out such that mass encompassed = mCloud?
        def solve_xi_out(xi, rhoCore, c_s, mCloud, f_rho_rhoc):
            # what is the mass at xi_out (cloud)?
            # note rhoc_rho(xi) = np.exp(-u)
            f_mass = lambda xi_out : 4 * np.pi * rhoCore * (c_s**2 / (4 * np.pi * G * rhoCore))**(3/2) * xi_out**2 * f_rho_rhoc(xi_out)
            # integrate
            mass, _ = scipy.integrate.quad(f_mass, 0, xi)
            # compare mass obtained
            return mass - mCloud
            
        xi_out = scipy.optimize.brentq(solve_xi_out, 1e-5, 1e3, args = (rhoCore, c_s, mCloud, f_rho_rhoc))
        
        rho_rhoc = f_rho_rhoc(xi_out)
        
        r_out = xi_out * (4 * np.pi * G * rhoCore / c_s**2) ** (-1/2) 
        
        n_out = nCore * rho_rhoc
        
        return xi_out, r_out, n_out
            
    # the minimum temperature is around 1e4 from trial and error. However
    # there are cases where 1e4 would lead to failure because brentq requires
    # sign change within brackets. Hence we include options for 1e2 and 1e3 as
    # the minimum guess.
    
    for Tinit in np.logspace(9, 1, 9):
        try:
            def solve_BE_Teff(T, mCloud, rhoCore, f_rho_rhoc, n_outOmega):
                _, _, n_out = solve_structure(T, mCloud, rhoCore, f_rho_rhoc, n_outOmega)
                return n_out - n_outOmega
            
            bE_Teff = scipy.optimize.brentq(solve_BE_Teff, Tinit, 1e10, args = (mCloud, rhoCore, f_rho_rhoc, n_outOmega))
            break
        
        except ValueError:
            # print(f'skip {Tinit}')
            continue
        except Exception as e:
            print(e)
            sys.exit()        
        
    # sound speed
    xi_out, r_out, n_out = solve_structure(bE_Teff, mCloud, rhoCore, f_rho_rhoc, n_outOmega)
    
    return xi_out, r_out, n_out, bE_Teff



# =============================================================================
# Now we show the figure
# =============================================================================
# range of nCore to explore (cm3)
nStart = 2.5
nEnd = 6
nNum = int((nEnd - nStart)*2+1)

nCore_list = np.logspace(nStart, nEnd, nNum)

# range of mCloud to explore (Msun)
mStart = 4
mEnd = 9
mNum = int((mEnd-mStart)*2+1)

mCloud_list = np.logspace(mStart, mEnd, mNum) * Msun2g

# create datacube
data = np.zeros(shape = (len(nCore_list)*len(mCloud_list), 3))

# loop through
idx = 0
for mCloud in mCloud_list:
    for nCore in nCore_list:
        _, r_out, _, _ = create_BESphere(Omega, nCore, mCloud)
        data[idx] = np.array([nCore, mCloud, r_out])
        idx += 1
        

# extract data
nCore_array = data[:,0]
mCloud_array = data[:,1]
rCloud_array = data[:,2] * cm2pc

# Set up colormap
cmap = cm.viridis
norm = mcolors.Normalize(vmin=mStart, vmax=mEnd)
colors = cmap(norm(np.log10(mCloud_list * g2Msun)))  # Get colors from colormap

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
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



fig, ax = plt.subplots(1, 2, figsize = (9,5), dpi = 200,)
for ii in range(mNum):

    ax[0].plot(nCore_array[ii*len(nCore_list):(ii+1)*len(nCore_list)], rCloud_array[ii*len(nCore_list):(ii+1)*len(nCore_list)], \
             label = f'M$_c$$_l$ = {np.log10(mCloud_array[ii*len(nCore_list)] * g2Msun)} M$_\\odot$',\
                 color=colors[ii]
                 )

# Create colorbar
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Needed for older versions of matplotlib
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('log$_{10}$ M$_{\\rm cl}$ (M$_\\odot$)')


ax[0].set_xscale('log')
# plt.yscale('log')
ax[0].set_ylim(0, 700)
# ax[0].set_ylabel('$r_{\\rm cl}$ (pc)')
ax[0].set_xlabel('$\\rho_c$ [cm$^{-3}$]')    
ax[1].set_xlabel('$\\rho_c$ [cm$^{-3}$]')    

# path2figure = r'/Users/jwt/unsync/Code/Trinity/fig'
# plt.savefig(os.path.join(path2figure, 'BESpheres_radius.pdf'))


# =============================================================================
# Version for homogeneous cloud
# =============================================================================

def get_homoR(mCloud, nCore):
    
    r = (3 * mCloud / 4 / np.pi / (nCore * mu_neu))**(1/3)

    return r


# loop through
idx = 0
for mCloud in mCloud_list:
    for nCore in nCore_list:
        r_out = get_homoR(mCloud, nCore)
        data[idx] = np.array([nCore, mCloud, r_out])
        idx += 1

# extract data
nCore_array = data[:,0]
mCloud_array = data[:,1]
rCloud_array = data[:,2] * cm2pc

# Set up colormap
cmap = cm.viridis
norm = mcolors.Normalize(vmin=mStart, vmax=mEnd)
colors = cmap(norm(np.log10(mCloud_list * g2Msun)))  # Get colors from colormap

for ii in range(mNum):

    ax[1].plot(nCore_array[ii*len(nCore_list):(ii+1)*len(nCore_list)], rCloud_array[ii*len(nCore_list):(ii+1)*len(nCore_list)], \
             label = f'M$_c$$_l$ = {np.log10(mCloud_array[ii*len(nCore_list)] * g2Msun)} M$_\\odot$',\
                 color=colors[ii]
                 )

# Create colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Needed for older versions of matplotlib
cbar = plt.colorbar(sm, ax=ax, pad = 0.04)
cbar.set_label('log$_{10}$ M$_{\\rm cl}$ [M$_\\odot$]')


ax[1].set_xscale('log')
ax[1].set_ylim(0, 700)
# plt.yscale('log')
ax[0].set_ylabel('$r_{\\rm cl}$ [pc]')

path2figure = r'/Users/jwt/unsync/Code/Trinity/fig'
plt.savefig(os.path.join(path2figure, 'InitialCloudRadius.pdf'))






















