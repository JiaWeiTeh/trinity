#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:46:23 2024

@author: Jia Wei Teh
"""

import re

# some cgs to astronomical units (au) conversions
# cm2pc = u.cm.to(u.pc)
cm2pc = 3.240779289444365e-19
pc2cm = 1/cm2pc

# s2yr = u.s.to(u.Myr)
s2Myr = 3.168808781402895e-14
Myr2s = 1/s2Myr

# dens_cgs2au
# nens_cgs2au = (1/u.cm**3).to(1/u.pc**3).value
ndens_cgs2au = 2.937998946096347e+55
ndens_au2cgs = 1/ndens_cgs2au

# phi_cgs2au
# phi_cgs2au = (1/u.cm**2/u.s).to(1/u.pc**2/u.Myr).value
phi_cgs2au = 3.0047272630641653e+50
phi_au2cgs = 1/phi_cgs2au

# E_cgs2au = u.erg.to(u.M_sun*u.pc**2/u.Myr**2)
E_cgs2au = 5.260183968837699e-44
E_au2cgs = 1/E_cgs2au

# Luminosity
# L_cgs2au = (u.erg/u.s).to(u.M_sun*u.pc**2/u.Myr**3)
L_cgs2au = 1.6599878161499254e-30
L_au2cgs = 1/L_cgs2au

# d/dt of momentum
# pdot_cgs2au = (u.g * u.cm / u.s**2).to(u.M_sun*u.pc/u.Myr**2)
pdot_cgs2au = 1.623123174716277e-25
pdot_au2cgs = 1/pdot_cgs2au

# d/dt2 of momentum
# pdotdot_cgs2au = (u.g * u.cm / u.s**3).to(u.M_sun*u.pc/u.Myr**3)
pdotdot_cgs2au = 5.122187189842638e-12
pdotdot_au2cgs = 1/pdotdot_cgs2au

# Gravitational constant
# G_cgs2au = (u.cm**3/u.g/u.s**2).to(u.pc**3/u.M_sun/u.Myr**2)
G_cgs2au = 67400.3588611473
G_au2cgs = 1/G_cgs2au

# velocity
# v_kms2au = (u.km/u.s).to(u.pc/u.Myr)
v_kms2au = 1.022712165045695
v_au2kms = 1/v_kms2au

# velocity
# v_cms2au = (u.cm/u.s).to(u.pc/u.Myr)
v_cms2au = 1.022712165045695e-05
v_au2cms = 1/v_cms2au

# Mass
# g2Msun = u.g.to(u.M_sun)
g2Msun = 5.029144215870041e-34
Msun2g = 1/g2Msun

# Gravitational force
# F_cgs2au = (u.g*u.cm/u.s**2).to(u.M_sun*u.pc/u.Myr**2)
F_cgs2au = 1.623123174716277e-25
F_au2cgs = 1/F_cgs2au

# Bubble pressure (or any presure unit for that matter)
# Pb_cgs2au = (u.g/u.cm/u.s**2).to(u.M_sun/u.pc/u.Myr**2)
Pb_cgs2au = 1545441495671.806
Pb_au2cgs = 1/Pb_cgs2au

# boltzman constant
# k_B_cgs2au = (u.g*u.cm**2/u.s**2/u.K).to(u.M_sun * u.pc**2 / u.Myr**2 / u.K)
k_B_cgs2au = 5.260183968837699e-44
k_B_au2cgs = 1/k_B_cgs2au

# thermal coefficient
# c_therm_cgs2au = (u.g*u.cm/u.s**3 / u.K**(7/2)).to(u.M_sun*u.pc/u.Myr**3 / u.K**(7/2))
c_therm_cgs2au = 5.122187189842638e-12
c_therm_au2cgs = 1/c_therm_cgs2au

# dudt
# dudt_cgs2au = (u.erg/u.cm**3/u.s).to(u.M_sun/u.pc/u.Myr**3)  
dudt_cgs2au = 4.877042454381257e+25
dudt_au2cgs = 1/dudt_cgs2au

# Lambda
# Lambda_cgs2au = (u.erg*u.cm**3/u.s).to(u.M_sun*u.pc**5/u.Myr**3)
Lambda_cgs2au = 5.650062667161655e-86
Lambda_au2cgs = 1/Lambda_cgs2au

# tau_kappa_IR
# tau_cgs2au = (u.g/u.cm**2).to(u.M_sun/u.pc**2)
tau_cgs2au = 4788.452460043275
tau_au2cgs = 1/tau_cgs2au

# grav stuffs
# gravPhi_cgs2au = (u.cm**2/u.s**2).to(u.pc**2/u.Myr**2)
gravPhi_cgs2au = 1.045940172532453e-10
gravPhi_au2cgs = 1/gravPhi_cgs2au

# grav_force_m_cgs2au = (u.cm/u.s**2).to(u.pc/u.Myr**2)
grav_force_m_cgs2au = 322743414.19646025
grav_force_m_au2cgs = 1/grav_force_m_cgs2au





# =============================================================================
# Here we include unit conversion for dictionary
# =============================================================================


def convert2au(unit_string):
    """ 
    Given string (e.g., from .param file) of [km * s**2 * g**(3/2)] calculate the extra
    factor to convert to [pc * Myr**2 * Msun**(3/2)].
    """
    
    # if there is no unit
    if unit_string is None:
        return 1
    
    # Step 1: Clean and split units safely (excluding '**')
    # whitespace
    unit_string = re.sub(r'\s+', '', unit_string)
    units = [uu.strip() for uu in re.split(r'(?<!\*)\*(?!\*)', unit_string)]

    # Step 2: Mapping for replacements
    unit_map = {
        'g': str(g2Msun),
        's': str(s2Myr),
        'cm': str(cm2pc),
        'km': str(cm2pc/1e-5),
        'K': str(1), # *1 because there is no additional factor (no unit change required)
        'Zsun': str(1),
        'Msun': str(1),
        'pc': str(1),
        'Myr': str(1),
        'erg': str(E_cgs2au),
    }

    # Step 3: Replace base units while preserving exponents
    mapped_units = []
    for uu in units:
        match = re.match(r'^([a-zA-Z]+)(\*\*.*)?$', uu)
        if match:
            base, exp = match.groups()
            mapped_base = unit_map.get(base, base)
            new_unit = mapped_base + (exp if exp else '')
            mapped_units.append(new_unit)
        else:
            mapped_units.append(uu)  # fallback

    # a function that only allows string including 0-9, ., +-/*, e, and ().
    # this is a small check for safer eval().
    def is_valid_expression(s):
        pattern = r'^[0-9e+\-*/().]*$'
        return bool(re.fullmatch(pattern, s))

    # evaluate
    factor = 1
    for element in mapped_units:
        if is_valid_expression(element):
            factor *= eval(element)
        else:
            raise Exception(f'Unit contains odd expression: \'{element}\'. If the expression is right, include it in unit_map.')

    return factor



