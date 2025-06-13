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


# creating the bonnor ebert sphere

# begin with calculting the pressure of the outer ISM

# ISM temperature in K
TISM = 100
# TISM = 1e4
# TCloud = 10
TCloud = 2900
# 
mu = 1.0181176926808696e-24 # g 
# mu = 3.9e-24 # g 

# number density to denstiy
def gcm2cm(dens):
    return dens / mu 
# density to number density
def cm2gcm(dens):
    return dens * mu

# this is a value from Rahner+17
print(f'According to Rahner17 nISM should be set to {np.round(gcm2cm(1.67e-25), 3)} cm-3')

# boltzmann 
k = c.k_B.cgs.value
G = c.G.cgs.value

# ISM density in cm-3
nISM = 1
# cloud density in cm-3
nCloud = 1e2
# core density in cm-3
nCore = 1e5

# ISM number denstiy
rhoISM = nISM * mu
# Core number density
rhoCore = nCore * mu
# Cloud number density
rhoCloud = nCloud * mu

# adiabatic index
gamma = 5/3

# cloud mass in Msol
M_real = 1e6
# in g
M_real *= cvt.Msun2g

# CORE sound speed in cgs
c_s_Core = np.sqrt(gamma * k * TCloud / mu) 
# ISM sound speed in cgs
c_s_ISM = np.sqrt(gamma * k * TISM / mu) 
# comparison
print(f'c_s_Core is {c_s_Core} and c_s_ISM is {c_s_ISM} and the ratio is {c_s_Core/c_s_ISM}')

# ISM pressure
P_ISM = rhoISM * c_s_ISM**2
# cloud edge pressure
P_Cloud = rhoCloud * c_s_Core**2
# P_Cloud = P_ISM
# or
# P_Cloud = 

print(c_s_Core, f'm/s for T = {TCloud}K (core sound speed)')
print(np.round(P_ISM / k, 4), 'P/kB')

# OPTION 1
# One we have P0, calculate m, the total mass of the sphere
m_total = P_Cloud ** (0.5) * G **(3/2) * M_real / c_s_Core**4
print(np.round(m_total, 4), 'dimensionless mass')




# T = (M_real * P_ISM**(1/2) * G**(3/2) / 1.18)**(1/2) * mu / k / gamma

# print(f'temperature is around {T} K')


# def m2r(m, c_s):
#     return 1 / (2.4 * c_s**2 / G)

# rBE = m2r(M_real, c_s_Core) * cvt.cm2pc

# print(f'rBE from mBE is {rBE}')
    

#%%



# define xi - r relation

def r2xi(r, rho_c, c_s):
    return r * np.sqrt(4 * np.pi * G * rho_c / c_s**2) 

def xi2r(xi, rho_c, c_s):
    return xi * (4 * np.pi * G * rho_c / c_s**2) ** (-1/2) 

def rho_rhoCore2u(rhoISM):
    return - np.ln(rhoISM)

def u2rho_rhoCore(u):
    return np.exp(-u)



# =============================================================================
# solving the lane-emden equation
# =============================================================================

def laneEmden(y,t):
    """ This function specifics the Lane-Emden equation."""
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = dpsi/dxi, let y = [ psi, dpsidxi],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    u, dudxi = y
    dydt = [
        dudxi, 
        np.exp(-u) - 2 * dudxi / t
        ]
    return dydt

u0, dudxi0 = 0, 0

xi_array =  np.logspace(-2, 5, 3000)


# =============================================================================
# Run the integration
# =============================================================================
solution  = scipy.integrate.odeint(laneEmden, [u0, dudxi0], xi_array)
# solution
u_array = solution[:,0]
dudxi_array = solution[:,1]


# convert 
rho_rhoCore_array = u2rho_rhoCore(u_array)
rhoCore_rho_array = 1 / rho_rhoCore_array

plt.plot(xi_array, rho_rhoCore_array)

plt.ylabel('$\\rho / \\rho_c$')
plt.xlabel('$\\xi$')
plt.xlim(min(xi_array), max(xi_array))
plt.ylim(1e-9, 1e1)
plt.yscale('log')
plt.xscale('log')





#%%



def get_nedge(T):
    
    c_s_Core = np.sqrt(gamma * k * T / mu) 

    RHS = 4 * np.pi * G**3 * M_real**2 * P_Cloud / c_s_Core**8
    
    LHS = np.exp(-u_array) * (xi_array**2 * dudxi_array)**2
    
    f = interp1d(LHS, xi_array, kind='cubic', fill_value="extrapolate")
    
    xi_out = f(RHS)
    
    # print(f'xi_out according to RHS and LHS is {xi_out}')
    
    r_out = xi2r(xi_out, rhoCore, c_s_Core) * cvt.cm2pc
    
    # print(f'r_out according to RHS and LHS is {r_out} pc')
    
    # find the right u
    
    f = interp1d(xi_array, u_array, kind='cubic', fill_value="extrapolate")
    
    u_out = f(xi_out)
    
    nEdge = nCore * np.exp(-u_out) + 1e-10
    
    # print(f'nEdge is {nEdge} giving ratio of {nCore/nEdge}')
    
    return nCore/nEdge - 14.1


Teff = scipy.optimize.brentq(get_nedge, 1e3, 1e8)

print(f'Teff is {Teff} K')

c_s_Core = np.sqrt(gamma * k * Teff / mu) 

RHS = 4 * np.pi * G**3 * M_real**2 * P_Cloud / c_s_Core**8

LHS = np.exp(-u_array) * (xi_array**2 * dudxi_array)**2

f = interp1d(LHS, xi_array, kind='cubic', fill_value="extrapolate")

xi_out = f(RHS)

print(f'xi_out according to RHS and LHS is {xi_out}')

r_out = xi2r(xi_out, rhoCore, c_s_Core) * cvt.cm2pc

print(f'r_out according to RHS and LHS is {r_out} pc')

# find the right u

f = interp1d(xi_array, u_array, kind='cubic', fill_value="extrapolate")

u_out = f(xi_out)

nEdge = nCore * np.exp(-u_out) + 1e-10

print(f'nEdge is {nEdge} giving ratio of {nCore/nEdge}')

    




#%%



n_array = np.ones(shape = len(xi_array)) * rhoISM / mu


n_array[xi_array < xi_out] = rho_rhoCore_array[xi_array < xi_out] * rhoCore / mu

r_array = xi2r(xi_array, rhoCore, c_s_Core) * cvt.cm2pc


# plt.plot(xi_array, rho_rhoCore_array)
plt.plot(r_array, n_array)

plt.ylabel('$\\rho / \\rho_c$')
plt.xlabel('$\\xi$')
# plt.xlim(min(r_array), 1e2)
# plt.ylim(1e-9, 1e1)
plt.yscale('log')
plt.xscale('log')




#%%













plt.plot(xi_array, rhoISM_rho_array)

plt.ylabel('$\\rho_c / \\rho$')
plt.xlabel('$\\xi$')
plt.xlim(1e-1, 10)
plt.ylim(1, 14.1)
# plt.yscale('log')
# plt.xscale('log')


#%%


# xi-squared vs xi term

plt.plot(xi_array, xi_array**2 * dudxi_array)
plt.ylabel('$\\xi^2(du/d\\xi)$')
plt.xlabel('$\\xi$')
plt.xlim(min(xi_array), max(xi_array))
plt.ylim(1e-4, 1e4)
plt.yscale('log')
plt.xscale('log')


#%%

def get_m(rhoISM_rho_array, xi_array, dudxi_array):
    return (4 * np.pi * rhoISM_rho_array)**(-1/2) * xi_array**2 * dudxi_array


m_array = get_m(rhoISM_rho_array, xi_array, dudxi_array)


plt.plot(rhoISM_rho_array, m_array)


# OPTION 1
max_rhoc_rho0 = rhoISM_rho_array[np.where(m_array == max(m_array))[0]][0]


plt.axvline(max_rhoc_rho0, linestyle = '--')

plt.xlabel('$\\rho_c/\\rho_o$')
plt.ylabel('m (dimless)')
plt.ylim(0, 1.2)
plt.xlim(1, 1e5)
plt.xscale('log')


#%%


# given 14.1, find xi0, then map and find xi2dudx, then plug into find m, then 

# Find xi0
# in cgs
f = interp1d(rhoISM_rho_array, xi_array, kind='cubic', fill_value="extrapolate")
# in cgs
xi_out = f(max_rhoc_rho0)


print(f'xi_out if g = 14.1 is {np.round(xi_out, 3)}')

r_out = xi2r(xi_out, rhoCore, c_s_Core)

print(f'this corresponds to radius of {np.round(r_out * cvt.cm2pc, 3)} pc')


# Find xi2dudx
# in cgs
f = interp1d(xi_array, xi_array**2 * dudxi_array, kind='cubic', fill_value="extrapolate")
# in cgs
xi2dudxi_out = f(xi_out)

print(f'xi2dudxi_out if g = 14.1 is {np.round(xi2dudxi_out, 3)}')


# m value
# use M =1e6 to solve for rhoCore?
m = (4 * np.pi * max_rhoc_rho0)**(-1/2) * xi2dudxi_out

print(f'dimensionless mass is {np.round(m, 3)}')


plt.plot()




#%%



#%%

# what is the mass array?
def get_M(rho_c, c_s, xi_array, dudxi_array):
    
    return 4 * np.pi * rho_c * (c_s**2 / (4 * np.pi * c.G.cgs.value * rho_c))**(3/2) * xi_array**2 * dudxi_array


M_array = get_M(rhoCore, c_s_Core, xi_array, dudxi_array) * cvt.g2Msun
        

plt.plot(xi_array, M_array)
plt.ylabel('$M_\\odot (\\xi)$')
plt.xlabel('$\\xi$')
# plt.xlim(min(xi_array), max(xi_array))
# plt.ylim(1e-4, 1e4)
plt.yscale('log')
plt.xscale('log')











#%%


# =============================================================================
# version 1
# =============================================================================


def get_m(rhoISM_rho_array, xi_array, dudxi_array):
    return (4 * np.pi * rhoISM_rho_array)**(-1/2) * xi_array**2 * dudxi_array


m_array = get_m(rhoISM_rho_array, xi_array, dudxi_array)


plt.plot(rhoISM_rho_array, m_array)


# OPTION 1
max_rhoc_rho0 = rhoISM_rho_array[np.where(m_array == max(m_array))[0]][0]
print('maximum at', max_rhoc_rho0, 'with value of m =', m_array[np.where(m_array == max(m_array))[0][0]])
# OPTION 2
# max_rhoc_rho0 = rhoc_rho_array[np.where(m_array == m_total)[0]][0]
# print('maximum at', max_rhoc_rho0, 'with value of m =', m_array[np.where(m_array == max(m_array))[0][0]])



plt.axvline(max_rhoc_rho0, linestyle = '--')

plt.xlabel('$\\rho_c/\\rho_o$')
plt.ylabel('m (dimless)')
plt.ylim(0, 1.2)
plt.xlim(1, 1e5)
plt.xscale('log')


#%%


rho0 = rhoCore / max_rhoc_rho0

print(f'the number density at cloud edge according to 14.1 is {np.round(rho0/mu, 3)} cm-3')





# get xi0 from rhoCore/rhoCor0 ratio


from scipy.interpolate import interp1d

# in cgs
f = interp1d(M_total, xi_array, kind='cubic', fill_value="extrapolate")
# in cgs
xi_out = f(M_real)


















# therefore the radius xi

def get_M(rho_c, c_s, xi_array, dudxi_array):
    
    return 4 * np.pi * rho_c * (c_s**2 / (4 * np.pi * c.G.cgs.value * rho_c))**(3/2) * xi_array**2 * dudxi_array

if overwriteNCORE:
    print(f'nCore is being overwritten to {nCore_OVERWRITE}')
    nCore = nCore_OVERWRITE
    rho_c = nCore * mu

M_total = get_M(rho_c, c_s_Core, xi_array, dudxi_array)


# therefore this gives xi_out

from scipy.interpolate import interp1d

# in cgs
f = interp1d(M_total, xi_array, kind='cubic', fill_value="extrapolate")
# in cgs
xi_out = f(M_real)

print(f'xi_out is found at xi = {xi_out}')

plt.plot(xi_array, M_total * cvt.g2Msun)
# plt.plot(rhoc_rho_array, M_real * cvt.g2Msun)
plt.xlabel('$\\xi$')
plt.ylabel('M (M$_\\odot$)')
plt.axvline(xi_out, linestyle = '--')
plt.axhline(M_real * cvt.g2Msun, linestyle = '--')
plt.xscale('symlog')
plt.yscale('log')
plt.xlim(0, 1e3)


r_out = xi2r(xi_out, rho_c, c_s_Core)

print(f'r_out is found at r = {np.round(r_out * cvt.cm2pc, 3)} pc')



























#%%

# # =============================================================================
# # version 2
# # =============================================================================


# shortened_xi_array = xi_array[::50]
# shortened_dudxi_array = dudxi_array[::50]
# shortened_rhoISM_rho_array = rhoISM_rho_array[::5]


# for xx, yy in zip(shortened_xi_array, shortened_dudxi_array):

#     shortened_m_array = get_m(shortened_rhoISM_rho_array, xx, yy)

#     plt.plot(shortened_rhoISM_rho_array, shortened_m_array)

# plt.yscale('log')



# %%


# therefore this gives rho_c

# since nEdge is not exactly nISM?

pressure_factor = c_s_ISM**2 / c_s_Core**2 

rho_c = rhoISM * pressure_factor * max_rhoc_rho0 

nCore = rho_c / mu

print('nISM:', nISM, 'nCore calculated:', nCore)

# option 2

# core density
rho_c_test = 1.18 ** 2 *  c_s_Core ** 6 / M_real ** 2 / c.G.cgs.value ** 3

print(f'nCore according to Krumholz is {rho_c_test / mu}')


# therefore the radius xi

def get_M(rho_c, c_s, xi_array, dudxi_array):
    
    return 4 * np.pi * rho_c * (c_s**2 / (4 * np.pi * c.G.cgs.value * rho_c))**(3/2) * xi_array**2 * dudxi_array



if overwriteNCORE:
    print(f'nCore is being overwritten to {nCore_OVERWRITE}')
    nCore = nCore_OVERWRITE
    rho_c = nCore * mu



M_total = get_M(rho_c, c_s_Core, xi_array, dudxi_array)


# therefore this gives xi_out

from scipy.interpolate import interp1d

# in cgs
f = interp1d(M_total, xi_array, kind='cubic', fill_value="extrapolate")
# in cgs
xi_out = f(M_real)

print(f'xi_out is found at xi = {xi_out}')

plt.plot(xi_array, M_total * cvt.g2Msun)
# plt.plot(rhoc_rho_array, M_real * cvt.g2Msun)
plt.xlabel('$\\xi$')
plt.ylabel('M (M$_\\odot$)')
plt.axvline(xi_out, linestyle = '--')
plt.axhline(M_real * cvt.g2Msun, linestyle = '--')
plt.xscale('symlog')
plt.yscale('log')
plt.xlim(0, 1e3)


r_out = xi2r(xi_out, rho_c, c_s_Core)

print(f'r_out is found at r = {np.round(r_out * cvt.cm2pc, 3)} pc')


#%%

# rho_c = 1e4 * mu
# r_out = xi2r(xi_out, rho_c, c_s)

plt.plot(xi2r(xi_array, rho_c, c_s_Core) * cvt.cm2pc, rho_rhoISM_array * rho_c / mu)
plt.axvline(r_out * cvt.cm2pc, linestyle = '--')

plt.ylabel('$\\rho(r) (\\rm{cm}^{-3})$')
plt.xlabel('$r$ (pc)')
plt.xlim(0, 50)
# plt.xlim(min(xi_array), max(xi_array))
plt.yscale('log')
# plt.xscale('log')








#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%







plt.plot(xi_array, M_total / M_real, c = 'k', label = 'mass')
plt.plot(xi_array, 1/rhoc_rho_array, linewidth = 5, c = 'g', alpha = 0.5, label = 'density')
# plt.plot(rhoc_rho_array, M_real * cvt.g2Msun)
plt.xlabel('$\\xi$')
plt.ylabel('M (M$_\\odot$)')
plt.axvline(xi_out, linestyle = '--')
plt.axhline(1, linestyle = '--')
plt.ylim(0, 1.1)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 1e3)
plt.xlim(0, 15)
plt.legend()






#%%




















#%%






import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Physical constants
G = 6.67430e-8          # cm^3 g^-1 s^-2
k_B = 1.380649e-16      # erg K^-1
m_H = 1.6735575e-24     # g
mu = 2.33               # mean molecular weight (typical for molecular cloud)

def sound_speed(T):
    return np.sqrt(k_B * T / (mu * m_H))

def lane_emden_rhs(xi, y):
    psi, dpsi_dxi = y
    d2psi_dxi2 = - (2/xi)*dpsi_dxi - np.exp(-psi) if xi != 0 else -np.exp(-psi)
    return [dpsi_dxi, d2psi_dxi2]

def solve_lane_emden(xi_max=10, n_points=1000):
    xi = np.linspace(1e-6, xi_max, n_points)
    y0 = [0, 0]  # psi(0) = 0, dpsi/dxi(0) = 0
    sol = solve_ivp(lane_emden_rhs, [xi[0], xi[-1]], y0, t_eval=xi, method='RK45')
    return sol.t, sol.y[0]

def compute_density_profile(M, P_ext, T=10, xi_max=6.5):
    # Solve Lane-Emden
    xi, psi = solve_lane_emden(xi_max)
    e_psi = np.exp(-psi)
    # Guess central density from mass and external pressure
    c_s = sound_speed(T)
    rho_c = (P_ext) / (c_s**2)  # crude estimate
    alpha = np.sqrt(c_s**2 / (4 * np.pi * G * rho_c))
    r = xi * alpha
    rho = rho_c * e_psi

    # Normalize to match total mass
    dr = np.gradient(r)
    shell_mass = 4 * np.pi * r**2 * rho * dr
    total_mass = np.sum(shell_mass)
    rho *= M / total_mass  # scale to match desired mass

    return r, rho

# Example usage
M = 1.0 * 1.989e33  # 1 solar mass in grams
P_ext = 1e-12       # external pressure in dyne/cm^2
T = 10              # K

r, rho = compute_density_profile(M, P_ext, T)

# Plot
plt.figure(figsize=(6,4))
plt.loglog(r / 3.086e18, rho)  # convert r to parsecs
plt.xlabel("Radius (pc)")
plt.ylabel("Density (g/cm³)")
plt.title("Bonnor-Ebert Sphere Density Profile")
plt.grid(True)
plt.tight_layout()
plt.show()










#%%

















#%%











#%%


plt.loglog(x, rho_rhoc_array, 'k-') 
plt.loglog(x, 1.7*x**(-2), 'b--', label=r'$\propto x^{-2}$')

plt.axhline(y = 1, c='r', ls='--', label=r'$\propto x^0$')

plt.legend(loc=1,fontsize=18)
plt.xlabel(r'$r/r_c$', size=18)
plt.ylabel(r'$\rho / \rho_c$', size=18)






#%%








def laneEmden(y,t):
    """ This function specifics the Lane-Emden equation."""
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = dpsi/dxi, let y = [ psi, dpsidxi],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    psi, dpsidxi = y
    dydt = [
        dpsidxi, 
        np.exp(-psi) - 2 * dpsidxi / t
        ]
    return dydt


BCs = [0,0]
x = np.logspace(-3,3,1000)

#solution to ODE
solution = scipy.integrate.odeint(laneEmden, BCs, x)
z1 = solution[:, 0]
z2 = solution[:, 1]


#set in terms of what we want
y = z1
rho_rhoc = np.exp(-y)
r_rc = x

'''MAKE SURE UNITS WORK OUT'''
########
###a)###
########
plt.rc('font', size=15)
plt.rc('lines', linewidth=2)

#figure rho/rho_c vs. r/r_c
plt.loglog(x,rho_rhoc,'k-')
plt.loglog(x,1.7*x**(-2),'b--', label=r'$\propto x^{-2}$')

plt.axhline(y = 1, c='r', ls='--', label=r'$\propto x^0$')

plt.legend(loc=1,fontsize=18)
plt.xlabel(r'$r/r_c$', size=18)
plt.ylabel(r'$\rho / \rho_c$', size=18)
plt.show()
plt.close()

#%%


plt.plot(x, rho_rhoc)
plt.xlim(0, 10)

#%%

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




def M(rho,r):
#    return 4*np.pi*rho*r**3/3.0
    return rho*r**3/3.

# masslist = M(rho_rhoc,x)

masslist = []

for xi in x:
    mass = scipy.integrate.quad(massIntegral, 0, xi, (100, 100))
    masslist.append(mass)



#%%


#figure M(r) vs. x
plt.loglog(x,mass, 'k-')
plt.loglog(x,4./(4*np.pi)*x**(3),'b--', label=r'$\propto x^{3}$')
plt.loglog(x,7./(4*np.pi)*x**(1),'r--', label=r'$\propto x^{1}$')

plt.legend(loc=2,fontsize=18)
# pl.xlim(1E-1,50)
# pl.ylim(1E-3,1E3)
plt.ylabel(r'$m_r$ $[c_s^3 \rho_c^{-1/2} G^{3/2}] $', size=18)
plt.xlabel(r'$r/r_c$', size=18)
# pl.savefig('mass.pdf', bbox_inches='tight')
plt.show()
plt.close()




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


import astropy.constants as c




path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe001_n1e4/dictionary.json'

import json
with open(path2json, 'r') as f:
    # step one is to make sure they are lists i think
    snaplists = json.load(f)


params = snaplists[10]


def get_soundspeed(T, params):
    if T > 1e4:
        mu = params['mu_p_au'] / cvt.g2Msun
    else:
        mu = params['mu_n_au'] / cvt.g2Msun 
    
    return  np.sqrt(params['gamma_adia'] * (params['k_B_au'] / cvt.k_B_cgs2au) * T / mu  / 1e4 / 1e4) * 1e4# this is in cms


def get_soundspeedv2(T, params):
    if T > 1e4:
        mu = params['mu_p_au']
    else:
        mu = params['mu_n_au'] 
    
    return  np.sqrt(params['gamma_adia'] * (params['k_B_au']) * T / mu) # this is in cms





T = params['T0']
G = params['G_au'] / cvt.G_cgs2au
c_s = get_soundspeed(T, params) 
vs_v2 = get_soundspeedv2(T, params)
print('difference', c_s - vs_v2)
rho_c = params['nCore_au'] / cvt.ndens_cgs2au * params['mu_n_au'] / cvt.g2Msun
rCloud = params['rCloud_au'] / cvt.cm2pc
nISM = params['nISM_au'] / cvt.ndens_cgs2au


print(T, c_s, rCloud * cvt.cm2pc, nISM, G, rho_c)

def get_densityprofile(r):
    
    
    
    chi = np.sqrt(4 * np.pi * G * rho_c /c_s**2) * r 
    
    # print(chi)
    
    
    y0 = [1e-9, 1e-9]


    sol = scipy.integrate.odeint(laneEmden, y0, r)
    
    psi = sol[:,0]
    omega = sol[:,1]
    
    # print(psi)
    # print(omega)

    rho_gas= rho_c * np.exp(-psi)
    
    # print(np.exp(-psi))
    
    inambient = r > rCloud

    dens_arr = rho_gas / (params['mu_n_au'] / cvt.g2Msun)
    # this correct? would this not affect the pressure and hydrostatic balance?
    dens_arr[inambient] = nISM 
    
    return dens_arr




#%%




r_arr = np.linspace(0, 20 , 200) / cvt.cm2pc
dens_arr = get_densityprofile(r_arr)


import matplotlib.pyplot as plt


plt.plot(r_arr * cvt.cm2pc, dens_arr )
plt.yscale('log')



#%%















#%%


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


def f_dens(r, n0, n_intercl, rcloud, nalpha = 0., rcore = 0.1):
    """
    :param r: list of radii
    :param n0: core density (namb)
    :param n_intercl: intercluster density
    :param rcloud: cloud radius
    :param nalpha: exponent (with correct sign) for density power law
    :param rcore: core radius (where the density is n0 = constant)
    :return: number density n(r)
    """
    if type(r) is not np.ndarray:
        r = np.array([r])

    # be careful that rcloud is not < rcore
    if rcore > rcloud:
        rcore = rcloud

    incore = r < rcore
    # ingrad = (r <= rcloud) & (r >= rcore)
    inambient = r > rcloud

    # input density function (as a function of radius r)
    dens = n0*(r/rcore)**(nalpha)
    dens[incore] = n0
    dens[inambient] = n_intercl

    # print('Checking initial density')
    # print(dens)
    # sys.exit()
    
    return dens


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













#!/usr/bin/env python
from __future__ import division
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from scipy import integrate


"""Info: Code solves lane-emden differential equation for following boundary conditions:
        psi(0) = 0; dpsi/dzi(zi=0) = 0. 

        The external pressure is determine from the following expression for the
        critical mass: 1.18*sigma^4/sqrt(G^3*P_o).
        A temperature of 20K is assume for the medium.
        The sound of speed is found from: sqrt(RT/mu); where R is the gas constant and mu is the
        molecular weight.

        After solving the ODE, the code will solve for dimensionless mass 'm'
        given an M and P_o. The value of rho_c is estimated. 
        Then, the code solves for zi_o for a given rho_c until the desired 
        TOTAL critical mass is found. This will give us the value of the BE boundary.   

By Aisha Mahmoud-Perez
Star Formation
"""

#----Functions-----#
def get_ext_pressure(T, m_crit):
        k = 1.38E-23
        p = 79.4 * T**4 * k / m_crit**2 #in SI units.
        p = p *10 #convert to bar.
        return p

def get_dmnless_mass(m_crit, p_out, c_sound):
        G = 6.67E-11
        p_out = p_out/10.0 #Pascals
        c_sound = c_sound / 100.0 #SI units
        m = (np.sqrt(p_out) * G**(1.5) * m_crit) / c_sound**4
        return m

def get_zi(m_dmnless,rho_out, rho_in, psi_diff):
        con = rho_out / (4 * np.pi * rho_in)
        zi_squared = m_dmnless / (con**(0.5) * psi_diff)
        return zi_squared

def get_mass(rho_in, c_sound, zi_sq, psi_diff):
        G = 6.67E-8 #cgs units
        con1 = 4 * np.pi * rho_in
        con2 = c_sound**2 / (4 * np.pi * G *rho_in)
        m = con1 * con2**(1.5) * zi_sq * psi_diff
        m = m / 1000.0 #change to kg
        m_s = m / 1.98E30 #chanege to solar masses
        return m_s
        
#----Constants----#
mu = 1.5 #any value between 1-2
T = 10 #temperature of gas in K
R = 8.31E7 # gas constant in cgs units
cs = np.sqrt(R*T/mu) #cgs units
P_o = get_ext_pressure(T, 1)
rho_o = P_o/(cs**2)

#----Solve Lane-Emden----#
def laneEmden(y,t):
    """ This function specifics the Lane-Emden equation."""
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = dpsi/dxi, let y = [ psi, dpsidxi],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    psi, dpsidxi = y
    dydt = [
        dpsidxi, 
        np.exp(-psi) - 2 * dpsidxi / t
        ]
    return dydt


y_init=[0, 0] #initial conditions
t = sp.linspace(0.0001, 8, 5000) #create array of values. Equidistant integration steps.
solution = sp.integrate.odeint(laneEmden, y_init, t)
psi = solution[:,0] #dump psi values here
dpsi = solution[:,1] #dump dpsi values here
rho_frac = sp.exp(-psi) #rho(r)/rho_o = exp(-psi)

#----Plot basic Lane-Emden----#
font = {'family' : 'ubuntu',
                        'size'   : 13}

plt.rc('font', **font)
plt.figure(1)
plt.plot(t, psi, color='LightSeaGreen', linewidth = 2, label='$\psi$')
plt.plot(t, rho_frac, color='Plum', linewidth = 2,label='$\\rho$/$\\rho_c$')
plt.xlabel('Nondimentsional radius $\\xi$')
plt.legend()

plt.rc('font', **font)
plt.figure(2)
plt.plot(-0.5, 0, color='b', linewidth = 2,label='log($\\rho$/$\\rho_c$)')
plt.xlim(-0.3,4.3)
plt.ylim(-1.2, 0)
plt.xlabel('Radius (AU)')
plt.legend(loc='lower left')


plt.show()

#----Find zi_o----#
rho_c = np.linspace(1.5E-20, 2.9E-18, 2000) #find the array of values of rho_c. Guesstimate!
rho_c_len = len(rho_c)
dpsi_len = len(dpsi)
# Loop trough each value of rho_c and find a value of zi^2. Use that zi^2 to find
# the total enclosed mass. If the mass desired is reached, save the
# rho_c and zi_sq (and their indexes)  and exit the loop.  
# Convert the zi_sq into a physical radius.
dimensionless_mass = get_dmnless_mass(0.5*1.98E30, P_o, cs) #do not use solar units
for i in range(rho_c_len):
        final_mass_computed = 0.48
        for n in range(dpsi_len):
                dimensionless_radius_squared = get_zi(dimensionless_mass,rho_o, rho_c[i], dpsi[n])
                if dimensionless_radius_squared < 6.4:
                        mass_cloud = get_mass(rho_c[i], cs, dimensionless_radius_squared, dpsi[n])
                        if mass_cloud > 0.4999 and mass_cloud < 0.5001:
                                if mass_cloud > final_mass_computed:
                                        final_mass_computed=mass_cloud
                                        total_mass = mass_cloud
                                        index_rho_c = i
                                        index_dpsi = n
                                        z_real = np.sqrt(dimensionless_radius_squared)
                                        rho_c_real = rho_c[i]

#----Get physical values for plotting----#
radius_au = physical_r(z_real, rho_c_real, cs)
radius_cutoff = []
rho_frac_cutoff = []
rho_c_m = rho_c_real*1E-4 #convert to kg/m^3
for m in range(len(t)):
        radius_list = physical_r(t, rho_c_real, cs)
        if radius_list[m] < radius_au:
                rho_frac_cutoff.append(rho_c_m*rho_frac[m])
                radius_cutoff.append(radius_list[m])


#----Plot results----#
rho_frac_cutoff = np.array(rho_frac_cutoff)
radius_cutoff = np.array(radius_cutoff)
font = {'family' : 'ubuntu',
                                        'size'   : 13}

plt.rc('font', **font)
plt.figure(3)
plt.plot(radius_cutoff, rho_frac_cutoff, color='b', linewidth = 2,label='$\\rho(r)$')
plt.xlabel('Radius (AU)')
plt.ylabel('$\\rho$(r) $kg/m^3$')
plt.legend()

plt.rc('font', **font)
plt.figure(4)
plt.plot(np.log10(radius_cutoff), np.log10(rho_frac_cutoff/rho_c_m), color='b', linewidth = 2,label='log($\\rho$/$\\rho_c$)')
plt.xlabel('Radius (AU)')
plt.legend(loc='lower left')
plt.show()




#%%


import numpy as np
import pylab as pl
from scipy.integrate import odeint, quad

#x = r/r_c
#y = ln(rho/rho_c)
#z1 = y
#z2 = dz1/dx
#dz2/dx = -2/x*z2-np.exp(z1)

#define Bonnor-Ebert equilibria
def BE(boundaries, x):
	z1 = boundaries[0]
	z2 = boundaries[1]
	dz1dx = z2
	dz2dx = -2.0/x*z2 - np.exp(z1)
	return [dz1dx, dz2dx]

#boundary conditions and x-range
BCs = [0,0]
x = np.logspace(-3,7,1000)

#solution to ODE
solution = odeint(BE, BCs, x)
z1 = solution[:, 0]
z2 = solution[:, 1]

#set in terms of what we want
y = z1
rho_rhoc = np.exp(y)
r_rc = x


'''MAKE SURE UNITS WORK OUT'''
########
###a)###
########
pl.rc('font', size=15)
pl.rc('lines', linewidth=2)

#figure rho/rho_c vs. r/r_c
pl.loglog(x,rho_rhoc,'k-')
pl.loglog(x,1.7*x**(-2),'b--', label=r'$\propto x^{-2}$')

pl.axhline(y = 1, c='r', ls='--', label=r'$\propto x^0$')
#pl.axvline(x = 30, c='r', ls='--')

pl.legend(loc=1,fontsize=18)
# pl.ylim(1E-4,10)
# pl.xlim(1E-1,50)
pl.xlabel(r'$r/r_c$', size=18)
pl.ylabel(r'$\rho / \rho_c$', size=18)
# pl.savefig('density.pdf', bbox_inches='tight')
pl.show()
pl.close()



pl.loglog(x, rho_rhoc)
pl.show()
pl.close()


#%%



import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate 



def laneEmden(y,t):
    """ This function specifics the Lane-Emden equation."""
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = dpsi/dxi, let y = [ psi, dpsidxi],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    psi, dpsidxi = y
    dydt = [
        dpsidxi, 
        np.exp(-psi) - 2 * dpsidxi / t
        ]
    return dydt


BCs = [0,0]
x = np.logspace(-3,7,1000)

#solution to ODE
solution = scipy.integrate.odeint(laneEmden, BCs, x)
z1 = solution[:, 0]
z2 = solution[:, 1]


#set in terms of what we want
y = z1
rho_rhoc = np.exp(-y)
r_rc = x

'''MAKE SURE UNITS WORK OUT'''
########
###a)###
########
plt.rc('font', size=15)
plt.rc('lines', linewidth=2)

#figure rho/rho_c vs. r/r_c
plt.loglog(x,rho_rhoc,'k-')
plt.loglog(x,1.7*x**(-2),'b--', label=r'$\propto x^{-2}$')

plt.axhline(y = 1, c='r', ls='--', label=r'$\propto x^0$')

plt.legend(loc=1,fontsize=18)
plt.xlabel(r'$r/r_c$', size=18)
plt.ylabel(r'$\rho / \rho_c$', size=18)
plt.show()
plt.close()




plt.plot(x, rho_rhoc)
plt.xlim(0, 10)





#%% 



########
###b)###
########

##define mass of sphere
def M(rho,r):
#    return 4*np.pi*rho*r**3/3.0
    return rho*r**3/3.

mass = M(rho_rhoc,x)









#%%



#figure M(r) vs. x
plt.loglog(x,mass, 'k-')
plt.loglog(x,4./(4*np.pi)*x**(3),'b--', label=r'$\propto x^{3}$')
plt.loglog(x,7./(4*np.pi)*x**(1),'r--', label=r'$\propto x^{1}$')

plt.legend(loc=2,fontsize=18)
# pl.xlim(1E-1,50)
# pl.ylim(1E-3,1E3)
plt.ylabel(r'$m_r$ $[c_s^3 \rho_c^{-1/2} G^{3/2}] $', size=18)
plt.xlabel(r'$r/r_c$', size=18)
# pl.savefig('mass.pdf', bbox_inches='tight')
plt.show()
plt.close()


#%%


########
###c)###
########

dimless_mass = mass*np.sqrt(rho_rhoc)/np.sqrt(2.9246796896)#/(np.pi**2*2)/1.0887942136 ####GET RID OF THESE NORMALIZATION FACTORS####

print("max m(r_0) resides at x = ", round(x[np.where(dimless_mass == max(dimless_mass))[0][0]],3))

#figure of dimensionless mass
pl.semilogx(x,dimless_mass, 'k-')
pl.xlim(1E-1,50)
pl.ylabel(r'$m(r_0)$ $[M(r) / (c_s^3 \rho_0^{-1/2} G^{3/2})] $', size=18)
pl.xlabel(r'$r_0/r_c$', size=18)
pl.rc('font', size=15)
# pl.savefig('dimless_mass.pdf', bbox_inches='tight')
pl.show()
pl.close()

########
###d)###
########

pressure = dimless_mass**2

print(max(pressure))
print("max p(r_0) resides at x = ", round(x[np.where(pressure == max(pressure))[0][0]],3))

#figure of pressure
pl.semilogx(x,pressure, 'k-')
#pl.semilogx(x,0.03*x**(4),'b--', label=r'$\propto x^{4}$')
#pl.loglog(x,475*x**(-3)+0.16,'r--', label=r'$\propto x^{-3}$')

pl.xlim(1E-1,50)
pl.ylim(0,1.4)
pl.ylabel(r'$P_0(x)$ $[c_s^8 G^3 M^2] $', size=18)
pl.xlabel(r'$r_0/r_c$', size=18)
#pl.legend(loc=2,fontsize=18)
# pl.savefig('pressure.pdf', bbox_inches='tight')
pl.show()
pl.close()


#max mass
print("the maximun mass of a BE sphere is max(m(r)) * c_s^4/P_0^{1/2} G^{3/2}")
print("max mass = ", round(max(dimless_mass),3)," c_s^4/P_0^{1/2} G^{3/2}")

rho_rhoc/=2.7777

#figure of pressure vs. y
pl.semilogx(rho_rhoc,pressure, 'k-')
#pl.semilogx(x,0.03*x**(4),'b--', label=r'$\propto x^{4}$')
#pl.loglog(x,475*x**(-3)+0.16,'r--', label=r'$\propto x^{-3}$')
pl.semilogx(rho_rhoc[np.where(pressure == max(pressure))[0][0]], max(pressure), 'bo')
pl.axvline(x=rho_rhoc[np.where(pressure == max(pressure))[0][0]],c='b',ls='--')
pl.text(0.1,1.3, r'$(\rho / \rho_c)_{max} \sim 1/14.1$', size=18)
pl.xlim(1E-3,2)
pl.ylim(0,1.4)
pl.xlabel(r'$\rho / \rho_c$', size=18)
pl.ylabel(r'$P_0(x)$ $[c_s^8 G^3 M^2] $', size=18)
#pl.legend(loc=2,fontsize=18)
# pl.savefig('pressure_density.pdf', bbox_inches='tight')
pl.show()
pl.close()

print(round(rho_rhoc[np.where(pressure == max(pressure))[0][0]],3))















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







