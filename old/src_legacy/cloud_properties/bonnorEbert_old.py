#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:04:46 2022

@author: Jia Wei Teh

This script contains helper functions to aid bonner-ebert sphere related 
calculations. See density_profile.py and mass_profile.py for more.
"""

import numpy as np
import scipy.integrate
import astropy.constants as c
import astropy.units as u
import scipy.optimize
import scipy.integrate
import sys
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])


def laneEmden(y,xi):
    """
    This function specifics the Lane-Emden equation. This will then be fed
    into scipy.integrate.odeint() to be solved.
    """
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = dpsi/dxi, let y = [ psi, omega ],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    psi, omega = y
    dydxi = [
        omega, 
        np.exp(-psi) - 2 * omega / xi
        ]
    # return
    return dydxi

def massIntegral(xi, rhoCore, c_s):
    """
    A function that outputs an expression (integral) to integrate to obtain
    the mass profile M(r).
    
    Watch out units!

    Parameters
    ----------
    xi : a list of xi
        xi is dimensionless radius, where:
            xi = (4 * pi * G * rho_core / c_s^2)^(0.5) * r
    rho_core : float
        The core density (Units: kg/m3)
    c_s : float
        Sound speed. (Units: m/s)

    Returns
    -------
    The expression for integral of M(r). (Units: kg)

    """
    # Note:
        # old code: MassIntegrate3()
        
    # An array for the range of xi for solving
    xi_arr = np.linspace(1e-12, xi, 200)
    # initial condition (set to a value that is very close to zero)
    y0 = [1e-12, 1e-12]
    # integrate the ODE to get values for psi and omega.
    psi, omega = zip(*scipy.integrate.odeint(laneEmden, y0, xi_arr))
    psi = np.array(psi)
    omega = np.array(omega)
    # Evaluate at the end point of xi_array, i.e., at xi(r) such that r is of
    # our interest.
    psipoint = psi[-1] 
    # See Eq33 http://astro1.physics.utoledo.edu/~megeath/ph6820/lecture6_ph6820.pdf
    A = 4 * np.pi * rhoCore * (c_s**2 / (4 * np.pi * c.G.value * rhoCore))**(3/2)
    # return the integral
    return A * np.exp(-psipoint) * xi**2

def get_bE_soundspeed(T, mu_n, gamma):
    """
    This function computes the isothermal soundspeed, c_s, given temperature
    T and mean molecular weight mu.

    Parameters
    ----------
    T : float (Units: K)
        Temperature of the gas.
    mu_n : float (Units: g) 
        Mean molecular weight of the gas. Watch out for the units.
    gamma: float 
        Adiabatic index of gas. 

    Returns
    -------
    The isothermal soundspeed c_s (Units: m/s)

    """
    # return
    mu_n = mu_n * u.g.to(u.kg)
    return np.sqrt(gamma * c.k_B.value * T / mu_n )


def get_bE_rCloud_nEdge(nCore, bE_T, mCloud, mu_n, gamma):
    """
    This function computes the bE cloud radius and the 
    density at the edge of the cloud.

    Parameters
    ----------
    nCore : float
        Core number density. (Units: 1/cm^3)
    bE_T : float
        The temperature of the BE sphere. (Units: K). 
    mCloud : float
        Mass of cloud (Units: solar mass).
    mu_n : float
        Mean mass per nucleus (Units: cgs, i.e., g)
    gamma: float
        Adiabatic index of gas.

    Returns
    -------
    rCloud : float
        Cloud radius. (Units: pc)
    nEdge: float
        Density at edge of cloud. (Units: 1/cm3)

    """
    # Note:
        # old code:
            # FindRCBE()
            
    # sound speed 
    c_s = get_bE_soundspeed(bE_T, mu_n, gamma)
    # convert to SI units
    rhoCore = nCore * mu_n * u.g.to(u.kg) * (1/u.cm**3).to(1/u.m**3).value
    mCloud = mCloud * u.Msun.to(u.kg)
    # Solve for xi such that the mass of cloud is mCloud at xi(r).
    def solve_xi(xi, rhoCore, c_s, mCloud):
        mass, _ = scipy.integrate.quad(massIntegral, 0, xi,
                                       args=(rhoCore, c_s))
        return mass - mCloud
    sol = scipy.optimize.root_scalar(solve_xi,
                                     args=(rhoCore, c_s, mCloud),
                                     bracket=[8.530955303346797e-07, 170619106.06693593],
                                     method='brentq')
    # get xi(r)
    xiCloud = sol.root
    # get r 
    rCloud = xiCloud * np.sqrt(c_s**2/(4 * np.pi * c.G.value * rhoCore))
    # get r in pc
    rCloud = rCloud * u.m.to(u.pc)
    # An array for the range of xi for solving ODE
    xi_arr = np.linspace(1e-12, xiCloud, 200)
    # initial condition (set to a value that is very close to zero)
    y0 = [1e-12, 1e-12]
    # integrate the ODE to get values for psi and omega.
    psi, omega = zip(*scipy.integrate.odeint(laneEmden, y0, xi_arr))
    # get density at xiCloud
    nEdge = nCore * np.exp(-psi[-1])
    # return
    return rCloud, nEdge

def get_bE_T(mCloud, nCore, g, mu_n, gamma):
    """
    This function returns the temperature of bE sphere. The temperature is
    determined such that the density at cloud edge, nEdge, is the same
    when obtained via parameter g and via density at rCloud.

    Parameters
    ----------
    mCloud : float
        Mass of cloud (Units: solar mass).
    nCore : float
        Core number density. (Units: 1/cm^3)
    g : float
        The ratio given as g = rho_core/rho_edge. The default is 14.1.
        This will only be considered if `bE_prof` is selected.
    mu_n : float
        Mean mass per nucleus (Units: cgs, i.e., g)
    gamma: float
        Adiabatic index of gas.

    Returns
    -------
    bE_T: float
        The temperature of the bE sphere.

    """
    
    # Note:
        # old code:
            # AutoT()
    
    # nEdge obtained from g = nCore/nEdge
    nEdge_g = nCore/g
    # balance between nEdge = nCore/g and nEdge obtained from get_bE_rCloud.
    def solve_T(T, mCloud, nCore, mu_n, gamma, nEdge_g):
        # old code:
            # Root()
        _, nEdge = get_bE_rCloud_nEdge(nCore, T, mCloud, mu_n, gamma)
        return nEdge - nEdge_g  # nEdge1 = nEdge2
    try:
        sol = scipy.optimize.root_scalar(solve_T,
                                         args=(mCloud, nCore, mu_n, gamma, nEdge_g),
                                         bracket=[2e+02, 2e+10], 
                                         method='brentq')
    except: 
        sys.exit("Solver could not find solution for the temperature of BE sphere.")
    # temperature of the bE sphere
    bE_T = sol.root
    # return
    return bE_T

#%%

# temperature = get_bE_T(1000000.0, 1000.0, 14.1, 2.1287915392418182e-24, 1.6666666666666667)
# print(temperature)







