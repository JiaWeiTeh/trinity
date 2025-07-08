#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 12:36:36 2025

@author: Jia Wei Teh
"""



import numpy as np

rho = 2
G = 3 


t = np.array([1, 10, 20, 30, 40, 60, 70])
r = t**2


def get_m(r):
    return 4 * np.pi / 3  * rho * r**3


def get_mdot(r, rdot):
    return 4 * np.pi * rho * r**2 * rdot




from scipy.interpolate import CubicSpline

# Cubic spline with extrapolation
cs = CubicSpline(t, r, extrapolate=True)





t_now = 90


v = np.gradient(r, t)
print(v)



t = np.concatenate([t, np.array([t_now])])
r = np.concatenate([r, np.array([cs(t_now)])])




v = np.gradient(r, t)




numeric = np.gradient(get_m(r), t)

# numeric = np.gr



import matplotlib.pyplot as plt

plt.plot(t, numeric)
# plt.plot(t, analytic)
plt.yscale('log')




















#%%

# import numpy as np
# import matplotlib.pyplot as plt

# # Given discrete data
# t = np.linspace(0, 10, 1000)
# r = np.sqrt(t + 1)  # assume some unknown behavior; you don’t use this formula directly

# # Physical constant
# rho = 1000  # density in kg/m^3

# # Compute M(t)
# M = (4/3) * np.pi * rho * r**3

# # Numerical derivatives
# dt = np.gradient(t)
# rdot = np.gradient(r, t)
# Mdot_numerical = np.gradient(M, t)

# # Analytical expression from chain rule
# Mdot_analytical = 4 * np.pi * rho * r**2 * rdot

# # Plotting
# plt.plot(t, Mdot_numerical, label='Numerical $\dot{M}$', linestyle='--')
# plt.plot(t, Mdot_analytical, label='Analytical $\dot{M}$', alpha=0.7)
# plt.xlabel('Time (s)')
# plt.ylabel('Mass flow rate (kg/s)')
# plt.legend()
# plt.title('Validation of $\dot{M} = 4 \pi \rho r^2 \dot{r}$ from Data')
# plt.grid(True)
# plt.show()