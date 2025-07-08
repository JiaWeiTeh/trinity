#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:01:22 2025

@author: Jia Wei Teh
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
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
plt.rc('font', size=15)
plt.rc('lines', linewidth=2)
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
    
    
fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 200)
fig.patch.set_alpha(0.0)      # Figure background
ax.patch.set_alpha(0.0)       # Axes background

#figure rho/rho_c vs. r/r_c
plt.loglog(x,rho_rhoc,'k-')
plt.loglog(x,1.7*x**(-2),'b--', label=r'$\propto x^{-2}$')
plt.axhline(y = 1, c='r', ls='--', label=r'$\propto x^0$')

plt.legend(loc=1,fontsize=18)
plt.ylim(1E-4,10)
plt.xlim(1E-1,50)
plt.xlabel(r'$r/r_c$', size=18)
plt.ylabel(r'$\rho / \rho_c$', size=18)
# plt.savefig('density.pdf', bbox_inches='tight')
plt.show()
# plt.close()

########
###b)###
########



#%%

fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 200)
fig.patch.set_alpha(0.0)      # Figure background
ax.patch.set_alpha(0.0)       # Axes background



##define mass of sphere
def M(rho,r):
#    return 4*np.pi*rho*r**3/3.0
    return rho*r**3/3.0

mass = M(rho_rhoc,x)

#figure M(r) vs. x
plt.loglog(x,mass, 'k-')
plt.loglog(x,4./(4*np.pi)*x**(3),'b--', label=r'$\propto x^{3}$')
plt.loglog(x,7./(4*np.pi)*x**(1),'r--', label=r'$\propto x^{1}$')

plt.legend(loc=2,fontsize=18)
plt.xlim(1E-1,50)
plt.ylim(1E-3,1E3)
plt.ylabel(r'$m_r$ $[c_s^3 \rho_c^{-1/2} G^{3/2}] $', size=18)
plt.xlabel(r'$r/r_c$', size=18)
# plt.savefig('mass.pdf', bbox_inches='tight')
plt.show()
# plt.close()




#%%



########
###c)###
########

dimless_mass = mass*np.sqrt(rho_rhoc)/np.sqrt(2.9246796896)#/(np.pi**2*2)/1.0887942136 ####GET RID OF THESE NORMALIZATION FACTORS####

print("max m(r_0) resides at x = ", round(x[np.where(dimless_mass == max(dimless_mass))[0][0]],3))

#figure of dimensionless mass
plt.semilogx(x,dimless_mass, 'k-')
plt.xlim(1E-1,50)
plt.ylabel(r'$m(r_0)$ $[M(r) / (c_s^3 \rho_0^{-1/2} G^{3/2})] $', size=18)
plt.xlabel(r'$r_0/r_c$', size=18)
plt.rc('font', size=15)
# plt.savefig('dimless_mass.pdf', bbox_inches='tight')
plt.show()
# plt.close()




#%%


########
###d)###
########

pressure = dimless_mass**2

print(max(pressure))
print("max p(r_0) resides at x = ", round(x[np.where(pressure == max(pressure))[0][0]],3))

#figure of pressure
plt.semilogx(x,pressure, 'k-')
#plt.semilogx(x,0.03*x**(4),'b--', label=r'$\propto x^{4}$')
#plt.loglog(x,475*x**(-3)+0.16,'r--', label=r'$\propto x^{-3}$')

plt.xlim(1E-1,50)
plt.ylim(0,1.4)
plt.ylabel(r'$P_0(x)$ $[c_s^8 G^3 M^2] $', size=18)
plt.xlabel(r'$r_0/r_c$', size=18)
#plt.legend(loc=2,fontsize=18)
# plt.savefig('pressure.pdf', bbox_inches='tight')
plt.show()
plt.close()


#%%

#max mass
print("the maximun mass of a BE sphere is max(m(r)) * c_s^4/P_0^{1/2} G^{3/2}")
print("max mass = ", round(max(dimless_mass),3)," c_s^4/P_0^{1/2} G^{3/2}")

rho_rhoc/=2.7777

#figure of pressure vs. y
plt.semilogx(rho_rhoc,pressure, 'k-')
#plt.semilogx(x,0.03*x**(4),'b--', label=r'$\propto x^{4}$')
#plt.loglog(x,475*x**(-3)+0.16,'r--', label=r'$\propto x^{-3}$')
plt.semilogx(rho_rhoc[np.where(pressure == max(pressure))[0][0]], max(pressure), 'bo')
plt.axvline(x=rho_rhoc[np.where(pressure == max(pressure))[0][0]],c='b',ls='--')
plt.text(0.1,1.3, r'$(\rho / \rho_c)_{max} \sim 1/14.1$', size=18)
plt.xlim(1E-3,2)
plt.ylim(0,1.4)
plt.xlabel(r'$\rho / \rho_c$', size=18)
plt.ylabel(r'$P_0(x)$ $[c_s^8 G^3 M^2] $', size=18)
#plt.legend(loc=2,fontsize=18)
# plt.savefig('pressure_density.pdf', bbox_inches='tight')
plt.show()
plt.close()

print(round(rho_rhoc[np.where(pressure == max(pressure))[0][0]],3))
