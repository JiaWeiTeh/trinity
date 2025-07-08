#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 11:47:56 2025

@author: Jia Wei Teh
"""


    # TODO: i wonder if one only has to initialise once the density profile (or to some
    # extend even the mass profile) because it is only used to calculate the density 
    # at that one point (for ion pressure) and the mass for enclosed bubble+shell mass (despite being named Msh which it kinda is because mBubble is <<< mShell)
    # This might not work on mdot profile because it relies on velocity. But then again 
    # if we have an interpolation function that means we can do that too right? since 
    # velocity is just a multiplication factor i think? I need to look more into this. 


import sys
import numpy as np
import scipy.integrate
import scipy.interpolate
from src._functions import operations
import src._functions.unit_conversions as cvt


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
    # x values spanning across large logspace.
    xi_array = np.logspace(-5, 4, 3000)
    # solve
    solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)
    # solution
    u_array = solution[:,0]
    dudxi_array = solution[:,1]
    # get density contrast
    rho_rhoc_array = np.exp(-u_array)
    # interpolation of density contrase
    f_rho_rhoc = scipy.interpolate.interp1d(xi_array, rho_rhoc_array, kind='cubic', fill_value="extrapolate")
    # return
    return xi_array, u_array, dudxi_array, rho_rhoc_array, f_rho_rhoc





def get_m():
    """
    # dimensionless mass profile. This is NOT the profile of the sphere, but
    # rather the value of m (total sphere mass), given a value or radius r (external radius). 
    """
    
    xi_array, u_array, dudxi_array, rho_rhoc_array, f_rho_rhoc = solve_laneEmden()
    
    m_array = (4 * np.pi / rho_rhoc_array)**(-1/2) * xi_array**2 * dudxi_array
    
    return xi_array, m_array













































# # --- create BE sphere, returning outer radius, outer density and sphere effective temperature
# def create_BESphereVersion2(params):
    
#     G = params['G'].value
#     mCloud = params['mCloud'].value
#     nCore = params['nCore'].value
#     densBE_Omega = params['densBE_Omega'].value
#     mu_ion = params['mu_ion'].value
    
#     m_total = 1.18
    
#     # Omega = max(rhoCore/rho(xi)) ~ 13.8-14.1 
    
#     xi_array, u_array, dudxi_array, rho_rhoc_array, f_rho_rhoc = solve_laneEmden()
    
#     params['densBE_xi_arr'].value = xi_array
#     params['densBE_u_arr'].value = u_array
#     params['densBE_dudxi_arr'].value = dudxi_array
#     params['densBE_rho_rhoc_arr'].value = rho_rhoc_array
#     params['densBE_f_rho_rhoc'].value = f_rho_rhoc
    
#     # or is it neu? because Teff is high but its just an effective temp? 
#     rhoCore = nCore * mu_ion
    
#     n_out = nCore / densBE_Omega

            
#     Pext_kb = 1e4
#     Pext = Pext_kb * params['k_B'] * cvt.k_B_au2cgs
#     Pext *= cvt.Pb_cgs2au

#     xi_out = 6.45
        
#     def cs2R(c_s):
#         return np.sqrt(c_s**2 / (4 * np.pi * G * rhoCore)) * xi_out
    
#     def Pext2cs(Pext):
#         return (Pext**0.5 * G**(3/2) * mCloud/m_total)**(1/4)
    
#     def cs2T(c_s):
#         return mu_ion * c_s**2 / params['gamma_adia'] / params['k_B']
    
#     c_s = Pext2cs(Pext)
    
#     r_out = cs2R(c_s)
    
#     bE_Teff = cs2T(c_s)
    
#     print(xi_out, r_out, n_out, bE_Teff)
    
#     import sys
#     sys.exit()
    
#     params['densBE_Teff'].value = bE_Teff
    
    
#     return xi_out, r_out, n_out, bE_Teff


        
# # --- create BE sphere, returning outer radius, outer density and sphere effective temperature
def create_BESphere(params):
    
    G = params['G'].value
    k_B = params['k_B'].value
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    densBE_Omega = params['densBE_Omega'].value
    mu_ion = params['mu_ion'].value
    
    # Omega = max(rhoCore/rho(xi)) ~ 13.8-14.1 
    
    xi_array, u_array, dudxi_array, rho_rhoc_array, f_rho_rhoc = solve_laneEmden()
    
    params['densBE_xi_arr'].value = xi_array
    params['densBE_u_arr'].value = u_array
    params['densBE_dudxi_arr'].value = dudxi_array
    params['densBE_rho_rhoc_arr'].value = rho_rhoc_array
    params['densBE_f_rho_rhoc'].value = f_rho_rhoc
    
    # or is it neu? because Teff is high but its just an effective temp? 
    rhoCore = nCore * mu_ion
    
    n_outOmega = nCore / densBE_Omega
    
    # What is the effective temperature? This is the most important parameter
    # as it sets the pressure balance in the sphere
    def solve_structure(T, mCloud, rhoCore, f_rho_rhoc, n_outOmega):
        
        # sound speed
        c_s = operations.get_soundspeed(T, params)
        
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
            
    # update
    params['densBE_Teff'].value = bE_Teff
    # rerun to get full output
    xi_out, r_out, n_out = solve_structure(bE_Teff, mCloud, rhoCore, f_rho_rhoc, n_outOmega)
    
    # # external pressure according to this sound speed
    # c_s = operations.get_soundspeed(bE_Teff, params)
    # m = 1.18
    # P_ext = c_s**8 / G**3 * m**2 / mCloud**2
    # print(f'the stable cloud is supported by an external pressure of P/k = {P_ext / k_B * cvt.Pb_au2cgs / cvt.k_B_au2cgs} K/cm3')
    
    return xi_out, r_out, n_out, bE_Teff


# define xi - r relation

def r2xi(r, params):
    c_s = operations.get_soundspeed(params['densBE_Teff'].value, params)
    return r * np.sqrt(4 * np.pi * params['G'] * params['nCore'] * params['mu_ion'] / c_s**2) 

def xi2r(xi, params):
    c_s = operations.get_soundspeed(params['densBE_Teff'].value, params)
    return xi * (4 * np.pi * params['G'] * params['nCore'] * params['mu_ion'] / c_s**2) ** (-1/2) 
































