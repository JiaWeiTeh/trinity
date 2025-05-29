#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:44:16 2025

@author: Jia Wei Teh
"""



import numpy as np
import constants as c
import auxiliary_functions as aux
import init as i
import astropy.constants as cons
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import sys
from scipy.integrate import quad
from scipy import optimize


#%%

def laneEmden(y,t):
    """ This function specifics the Lane-Emden equation."""
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = dpsi/dxi, let y = [ psi, omega ],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    psi, omega = y
    dydt = [
        omega, 
        np.exp(-psi) - 2 * omega / t
        ]
    return dydt




def get_densityprofile(r, cloudMass, nCloud):
    
    
    
    
    
    
    
    
    
    return










#%%



def f_densBE(r, n0, T , n_intercl, rcloud):
    """
    :param r: list of radii assume pc
    :param n0: core density (namb) 1/cm3
    :param n_intercl: intercluster density
    :param rcloud: cloud radius pc
    :param T: temperature of BEsphere K
    :return: number density n(r)
    """
    # input density function (as a function of radius r)

    if type(r) is not np.ndarray:
        r = np.array([r])

    # print("\n\n\n\ndebug\n\n\n\n")
    # print("This is r array")
    # print(r)
    # print("n_intercl", n_intercl, "rcloud", rcloud, 'T', T, 'n0', n0)
    # n_intercl 10 rcloud 355.8658723191992 T 451690.2638133162 n0 1000
    dens = np.nan*r
    rcloud= rcloud * c.pcSI #rcloud in m
    r = r * c.pcSI # r in m
    rho_0= n0*i.muiSI*10**6# num dens in vol dens (SI)
    
    
    munum=i.mui/c.mp
    cs= aux.sound_speed_BE(T)
    zet=r*(((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2))
    #t=np.linspace(0.0001*10**(-9),b,endpoint=True,num=int(100000/500))
    y0=[0.0001*10**(-9), 0.0001*10**(-9)]
    sol = odeint(laneEmden, y0, zet)
    #psipoints=sol[:, 0][len(sol[:, 0])-1]
    psipoints=sol[:, 0]

    rho_gas= rho_0*np.exp(-psipoints)
    
    inambient = r > rcloud

    dens = rho_gas/(i.muiSI*10**6)
    dens[inambient] = n_intercl
    return dens


def MassIntegrate3(xi,rho_c,c_s):
    """This function creates a function to solve for the integral of M(xi).
    This will then be fed into scipy.integrate.quad()."""
    # rho_c = central density
    G = cons.G.value
    # array of times
    t = np.linspace(0.0001e-9, xi, 200)
    # t=np.linspace(0.0001*10**(-9),s,endpoint=True,num=200)
    # TOASK: the vector of initial conditions (set close to zero?)
    y0 = [0.0001e-9, 0.0001e-9]
    # y0=[0.0001*10**(-9), 0.0001*10**(-9)]
    psi, omega = zip(*odeint(laneEmden, y0, t))
    psi = np.array(psi)
    omega = np.array(omega)
    # ASK: why only use one point for psi?
    # Ans: To solve the lane Emden equation in 1-D function
    psipoints = psi[-1]
    # psipoints = sol[:, 0][-1]
    # See Eq33 http://astro1.physics.utoledo.edu/~megeath/ph6820/lecture6_ph6820.pdf
    A = 4 * np.pi * rho_c * (c_s**2 / (4 * np.pi * G * rho_c))**(3/2)
    
    return A*np.exp(-psipoints)*xi**2

def FindRCBE(n0, T, mCloud, plint=True):
    """
    :param n0: core density (namb) 1/cm3
    :param T: temperature of BEsphere K
    :param M: cloud mass in solar masses
    :return: cloud radius (Rc) in pc and density at Rc in 1/ccm
    """
    
    #t=np.linspace(10**(-5),2*(10**(9)),num=80)*c.pcSI
    mCloud = mCloud* c.MsunSI
    rho_0= n0*i.muiSI*10**6
    munum=14/11
    cs= aux.sound_speed_BE(T)
    #zet=t*(((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2))
    def Root(zet,rho_0,cs,Mcloud):
        return quad(MassIntegrate3,0,zet,args=(rho_0,cs))[0]-Mcloud
    #h=0
    #print(Mcloud,rho_0,cs)
    #while h < len(t):
    # These are results after many calculations
    sol = optimize.root_scalar(Root,args=(rho_0,cs,mCloud),bracket=[8.530955303346797e-07, 170619106.06693593], method='brentq')
    zetsol=sol.root
    rsol=zetsol/((((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2)))
    b=rsol
    rs=rsol/c.pcSI
    
    
    zeta=b*(((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2))
    w=np.linspace(0.0001*10**(-9),zeta,endpoint=True,num=int(100000/500))
    y0=[0.0001*10**(-9), 0.0001*10**(-9)]
    sol = odeint(laneEmden, y0, w)
    # psipoints=sol[:, 0][len(sol[:, 0])-1]
    psipoints=sol[:, 0][len(sol[:, 0])-1]
    nedge=n0*np.exp(-psipoints)
    
    if plint == True:
        
        rhoavg=(3*mCloud)/(4*np.pi*(b**3))
        navg=rhoavg/(i.muiSI*10**6)
        print('Cloud radius in pc=',rs)
        print('nedge in 1/ccm=',nedge)
        print('g after=',n0/nedge)
        print('navg=',navg)
    
    return rs,nedge


def AutoT(M,ncore,g):
    # TOASK: What are the T, g params?
    # T is basically for sound speed for the equation of state. 
    # g = BonnerEbert param
    nend=ncore/g
    # print("nedge in autoT here", nend)
    def Root(T,M,ncore,nend):
        rs, nedge = FindRCBE(ncore, T, M, plint=False)
        # print("rs = ",rs)
        # print("nedge = ",nedge)
        # so that root_scalar can solve for nedge - nend = 0, in other words
        # solve for x such that nedge(x)  = nend
        return nedge - nend
    sol = optimize.root_scalar(Root,args=(M,ncore,nend),bracket=[2e+02, 2e+10], method='brentq')
    Tsol=sol.root
    
    # print('Tsol = ', Tsol)
    return Tsol

#%%


# AutoT(1000000, 1000, 14.1)












