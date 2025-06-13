#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:04:46 2022

@author: Jia Wei Teh

This script contains helper functions to aid bonner-ebert sphere related 
calculations. See density_profile.py and mass_profile.py for more.

This differs with the first version, in that here we set the core radius
ourselves. This has the benefit of lesser calculation, and to allow
for more other parameter exploration since the cloud can now be described
by lesser number of paramters/properties

"""

import numpy as np
import scipy.integrate
import src._functions.unit_conversions as cvt
import src._functions.operations as operations
from scipy.interpolate import interp1d
import astropy.constants as c 
import matplotlib.pyplot as plt



# --- sound speed
def get_cs(T):
    if T > 1e4:
        return np.sqrt(gamma * kB * T / mu_ion)
    else:
        return np.sqrt(gamma * kB * T / mu_neu)

# --- constants
# g 
mu_neu = 1.0181176926808696e-24 
mu_ion = 2.1287915392418182e-24
G = c.G.cgs.value
kB = c.k_B.cgs.value
# adiabatic
gamma = 5/3
# maximum density ratio
Omega = 14.1

# --- ISM parameters
# K
TISM = 1e2
# cm3
nISM = 1
# gcm3
rhoISM = nISM * mu_neu
# sound speed
c_s_ISM = get_cs(TISM)
# pressure
P_ISM = c_s_ISM**2 * rhoISM

# --- cloud parameters
# cm3
nCore = 1e4
rhoCore = nCore * mu_ion
# Msol
mCloud = 1e7
mCloud *= cvt.Msun2g


#%%

T = (mCloud * P_ISM**(1/2) * G**(3/2) / 1.18)**(1/2) * mu_ion / kB / gamma

print(f'temperature is around {T} K')


#%%


# --- lane-emden
def laneEmden(y,t):
    """ This function specifics the Lane-Emden equation."""
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = du/dxi, let y = [ u, dudxi],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    u, dudxi = y
    dydt = [
        dudxi, 
        np.exp(-u) - 2 * dudxi / t
        ]
    return dydt

#%%


# --- mass integral
def get_mass(xi_out, rhoCore, c_s):
    
    # we would want to solve for xi_out
    
    u0, dudxi0 = 1e-5, 1e-5

    xi_array =  np.logspace(-5, np.log10(xi_out), 3000)
    
    # print(xi_out)
    
    solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)

    u_out = solution[:,0][-1]
    
    return 4 * np.pi * rhoCore * (c_s**2 / (4 * np.pi * G * rhoCore))**(3/2) * xi_out**2 * np.exp(-u_out)

# --- temperature of sphere via nCore/nEdge residual

def create_BESphere(Omega, nCore, mCloud):
    
    rhoCore = nCore * mu_ion
    
    # BE sphere requires a certain sound speed (hence effective temeprature)
    # that would support itself, and satisfy the ratio Omega = 14.1 
    # At the meantime, the mass calculated should also satisfy mBE = mCloud
    
    
    def get_bErCloud_nEdge(T, rhoCore, mCloud):
        
        c_s = get_cs(T)
        
        def get_xi_out(xi_out, rhoCore, c_s, mCloud):
        
            mass, _ = scipy.integrate.quad(get_mass, 0, xi_out, args = (rhoCore, c_s))
            
            return mass - mCloud
        
        xi_out = scipy.optimize.brentq(get_xi_out, 1e-7, 1e3, args = (rhoCore, c_s, mCloud))
        
        # print(f'first solution found: xi_out = {xi_out}')
        # sys.exit()
        
        # recalculate
        xi_array =  np.logspace(-5, np.log10(xi_out), 3000)
        u0, dudxi0 = 1e-5, 1e-5
    
        solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)
        
        u_out = solution[:,0][-1]
        
        r_out = xi_out * (4 * np.pi * G * rhoCore / c_s**2) ** (-1/2) 
        n_out = nCore * np.exp(-u_out)
        
        return r_out, n_out, xi_out
    
    nEdge_Omega = nCore / Omega
    
    def get_bE_Teff(T, rhoCore, mCloud, nEdge_Omega):
        
        rCloud, nEdge, xi_out = get_bErCloud_nEdge(T, rhoCore, mCloud)
        
        return nEdge - nEdge_Omega
    
    # the minimum temperature is around 1e4 from trial and error. 
    bE_Teff = scipy.optimize.brentq(get_bE_Teff, 1e3, 1e9, args = (rhoCore, mCloud, nEdge_Omega))
    
    rCloud, nEdge, xi_out = get_bErCloud_nEdge(bE_Teff, rhoCore, mCloud)
    
    return rCloud, nEdge, bE_Teff, xi_out


import time

start_time = time.time()

rCloud, nEdge, bE_Teff, xi_out = create_BESphere(Omega, nCore, mCloud)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Version 1 Time taken: {elapsed_time:.6f} seconds")

print('rCloud', rCloud * cvt.cm2pc)
print('nCore', nCore)
print('nEdge', nEdge)
print('ratio', nCore/nEdge)
print('bE_Teff', bE_Teff)
print('xi_out', xi_out)

# Time taken: 31.264554 seconds
# rCloud 9.317802070724694
# nCore 10000.0
# nEdge 709.2198580906238
# ratio 14.100000001300309
# bE_Teff 207939.71992813936
# xi_out 0.8103454761096396

#%%





# --- mass integral
def get_massV2(xi_out, rhoCore, c_s, f_eu_array):
    
    # we would want to solve for xi_out
    
    # u0, dudxi0 = 1e-5, 1e-5

    # xi_array =  np.logspace(-5, np.log10(xi_out), 10)
    
    # # print(xi_out)
    
    # solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)

    # u_out = solution[:,0][-1]
    # eu_out = np.exp(-u_out)
    
    eu_out = f_eu_array(xi_out)
    
    return 4 * np.pi * rhoCore * (c_s**2 / (4 * np.pi * G * rhoCore))**(3/2) * xi_out**2 * eu_out


def create_BESphereV2(Omega, nCore, mCloud):
    
    rhoCore = nCore * mu_ion
    
    # BE sphere requires a certain sound speed (hence effective temeprature)
    # that would support itself, and satisfy the ratio Omega = 14.1 
    # At the meantime, the mass calculated should also satisfy mBE = mCloud
    
    # initially large array covering all possible values, but solved only once
    u0, dudxi0 = 1e-5, 1e-5
    xi_array = np.logspace(-5, 4, 3000)
    solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)
    u_array = solution[:,0]
    f_eu_array = interp1d(xi_array, np.exp(-u_array), kind='cubic', fill_value="extrapolate")
    # f_eu_array = 1

    def get_bErCloud_nEdge(T, rhoCore, mCloud, f_eu_array):
        
        c_s = get_cs(T)
        
        def get_xi_out(xi_out, rhoCore, c_s, mCloud, f_eu_array):
            
        
            mass, _ = scipy.integrate.quad(get_massV2, 0, xi_out, args = (rhoCore, c_s, f_eu_array))
            
            return mass - mCloud
        
        xi_out = scipy.optimize.brentq(get_xi_out, 1e-5, 1e3, args = (rhoCore, c_s, mCloud, f_eu_array))
        
        # print(f'first solution found: xi_out = {xi_out}')
        # sys.exit()

        # recalculate
        # xi_array =  np.logspace(-5, np.log10(xi_out), 10)
        # u0, dudxi0 = 1e-5, 1e-5
    
        # solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)
        
        # u_out = solution[:,0][-1]
        eu_out = f_eu_array(xi_out)
        
        r_out = xi_out * (4 * np.pi * G * rhoCore / c_s**2) ** (-1/2) 
        n_out = nCore * eu_out
        
        return r_out, n_out, xi_out
    
    nEdge_Omega = nCore / Omega
    
    def get_bE_Teff(T, rhoCore, mCloud, nEdge_Omega, f_eu_array):
        
        rCloud, nEdge, xi_out = get_bErCloud_nEdge(T, rhoCore, mCloud, f_eu_array)
        
        return nEdge - nEdge_Omega
    
    # the minimum temperature is around 1e4 from trial and error. 
    bE_Teff = scipy.optimize.brentq(get_bE_Teff, 1e4, 1e9, args = (rhoCore, mCloud, nEdge_Omega, f_eu_array))
    
    rCloud, nEdge, xi_out = get_bErCloud_nEdge(bE_Teff, rhoCore, mCloud, f_eu_array)
    
    return rCloud, nEdge, bE_Teff, xi_out


import time

start_time = time.time()

rCloud, nEdge, bE_Teff, xi_out = create_BESphereV2(Omega, nCore, mCloud)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Version 2 time taken: {elapsed_time:.6f} seconds")

print('rCloud', rCloud * cvt.cm2pc)
print('nCore', nCore)
print('nEdge', nEdge)
print('ratio', nCore/nEdge)
print('bE_Teff', bE_Teff)
print('xi_out', xi_out)


#%%




































