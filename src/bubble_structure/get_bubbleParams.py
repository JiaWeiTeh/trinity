#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:36:10 2022

@author: Jia Wei Teh

This script contains useful functions that help compute properties and parameters
of the bubble. grep "Section" so jump between different sections.
"""
# libraries
import numpy as np
import os
import sys
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c
from astropy.table import Table
import src._functions.unit_conversions as cvt

#--
import src._functions.operations as operations
# get parameter
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

# =============================================================================
# This section contains function which computes the ODEs that dictate the 
# structure (e.g., temperature, velocity) of the bubble. 
# =============================================================================

def delta2dTdt(t, T, delta):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        Temperature at xi = r/R2.

    Returns
    -------
    dTdt : float
    """
    dTdt = (T/t) * delta

    return dTdt


def dTdt2delta(t, T, dTdt):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        DESCRIPTION.

    Returns
    -------
    delta : float
    """
    
    delta = (t/T) * dTdt
    
    return delta



def beta2Edot(params
              ):
    # old code: beta_to_Edot()
    """
    see pg 80, A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf 


    my_params:: contains
        
    Parameters
    ----------
    bubble_P : float
        Bubble pressure.
    bubble_E : float
        Bubble energy.
    r1 : float
        Inner bubble radius.
    r2 : float
        Outer bubble radius (aka rShell).
    beta : float
        dbubble_P/dt.
    t_now : float
        time.
    pwdot : float
        dPw/dt.
    pwdotdot : float
        dPw/dt/dt.
    r2dot : float
        Outer bubble velocity.

    Returns
    -------
    bubble_Edot : float
        dE/dt.

    """
    # dp/dt pressure 
    press_dot = - params['Pb'].value * params['beta'].value / params['t_now'].value
    # define terms
    # print('pwdot', pwdot)
    # print('r1', r1)
    a = np.sqrt(params['pwdot'].value/2)
    b = 1.5 * a**2 * params['R1'].value
    d = params['R2'].value**3 - params['R1'].value**3
    adot = 0.25 * params['pwdot_dot'].value / a
    # print('b', b)
    # print('bubble_E', bubble_E)
    e = b / ( b + params['Eb'].value )
    # main equation
    bubble_Edot = (2 * np.pi * press_dot * d**2 + 3 * params['Eb'].value * params['v2'].value * params['R2'].value**2 * (1 - e) -\
                    3 * (adot / a) * params['R1'].value**3 * params['Eb'].value**2 / (params['Eb'].value + b)) / (d * (1 - e))
    
    # return 
    return bubble_Edot


    # converts beta to dE/dt
    # :param Pb: pressure of bubble
    # :param R1: inner radius of bubble
    # :param beta: -(t/Pb)*(dPb/dt), see Weaver+77, eq. 40
    # :param my_params:
    # :return: 
    # """
    # R2 = my_params['R2']
    # v2 = my_params["v2"]
    # E = my_params['Eb']
    # Pdot = -Pb*beta/my_params["t_now"]

    # pwdot = my_params['pwdot'] # pwdot = 2.*Lw/vw

    # A = np.sqrt(pwdot/2.)
    # A2 = A**2
    # C = 1.5*A2*R1
    # D = R2**3 - R1**3
    # #Adot = (my_params['Lw_dot']*vw - Lw*my_params['vw_dot'])/(2.*A*vw**2)
    # Adot = 0.25*my_params['pwdot_dot']/A

    # F = C / (C + E)

    # #Edot = ( 3.*v2 * R2**2 * E + 2.*np.pi*Pdot*D**2 ) / D # does not take into account R1dot
    # #Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) ) / (D * (1.-F)) # takes into account R1dot but not time derivative of A
    # Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) - 3.*(Adot/A)*R1**3*E**2/(E+C) ) / (D * (1.-F)) # takes everything into account

    # #print "term1", "%.5e"%(2.*np.pi*Pdot*D**2), "term2", "%.5e"%(3.*E*v2*R2**2 * (1.-F)), "term3", "%.5e"%(3.*(Adot/A)*R1**3*E**2/(E+C))

    # #print "Edot", "%.5e"%Edot, "%.5e"%Edot_exact

    # return Edot


def Edot2beta(bubble_P, r1, bubble_Edot, my_params
              ):
    # old code: Edot_to_beta()
    """
    see pg 80, A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf 
    
    Parameters
    ----------
    bubble_P : float
        Bubble pressure.
    bubble_E : float
        Bubble energy.
    r1 : float
        Inner bubble radius.
    r2 : float
        Outer bubble radius (aka rShell).
    bubble_Edot : float
        dE/dt.
    t_now : float
        time.
    pwdot : float
        dPw/dt.
    pwdotdot : float
        dPw/dt/dt.
    r2dot : float
        Outer bubble velocity.

    Returns
    -------
    beta : float
        dbubble_P/dt.

    """
    t_now = my_params["t_now"]
    pwdot = my_params["pwdot"]
    pwdotdot = my_params["pwdotdot"]
    r2 = my_params["R2"]
    r2dot = my_params["v2"]
    bubble_E = my_params["Eb"]
    # define terms
    a = np.sqrt(pwdot/2)
    b = 1.5 * a**2 * r1
    d = r2**3 - r1**3
    adot = 0.25 * pwdotdot / a
    e = b / ( b + bubble_E ) 
    # main equation
    pdot = 1 / (2 * np.pi * d**2 ) *\
        ( d * (1 - e) * bubble_Edot - 3 * bubble_E * r2dot * r2**2 * (1 - e) + 3 * adot / a * r1**3 * bubble_E**2 / (bubble_E + b))
    beta = - pdot * t_now / bubble_P
    # return
    return beta



# =============================================================================
# Section: conversion between bubble energy and pressure. Calculation of ram pressure.
# =============================================================================

def bubble_E2P(Eb, r2, r1, gamma = warpfield_params.gamma_adia):
    """
    This function relates bubble energy to buble pressure

    Parameters 
    ----------
    Eb : float 
        Bubble energy.
    r1 : float 
        Inner radius of bubble (outer radius of wind cavity).
    r2 (aka rShell.rBubble) : float 
        Outer radius of bubble (inner radius of ionised shell).

    Returns
    -------
    bubble_P : float 
        Bubble pressure.

    # Note:
        # old code: PfromE()
    """
    
    # Make sure units are in cgs
    r1 /= cvt.cm2pc
    r2 /= cvt.cm2pc
    Eb /= cvt.E_cgs2au
    # avoid division by zero
    r2 += 1e-10 
    
    # pressure, see https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf
    # pg71 Eq 6.
    Pb = (gamma - 1) * Eb / (r2**3 - r1**3) / (4 * np.pi / 3)
    # return back in au
    return Pb * cvt.Pb_cgs2au
    
def bubble_P2E(Pb, r2, r1, gamma = warpfield_params.gamma_adia):
    """
    This function relates bubble pressure to buble energy 

    Parameters [cgs]
    ----------
    Pb : float
        Bubble pressure.
    r1 : float
        Inner radius of bubble (outer radius of wind cavity).
    r2 (aka rShell): float
        Outer radius of bubble (inner radius of ionised shell).

    Returns
    -------
    Eb : float
        Bubble energy.

    """
    # Note:
        # old code: EfromP()
    # see bubble_E2P()
    # Make sure units are in cgs
    r2 = r2.to(u.cm)
    r1 = r1.to(u.cm)
    Pb = Pb.to(u.g/u.cm/u.s**2)
    Eb = 4 * np.pi / 3 / (gamma - 1) * (r2**3 - r1**3) * Pb
    
    return Eb.to(u.erg)

def pRam(r, Lwind, vWind):
    """
    This function calculates the ram pressure.
    
    returns in [au].
    
    Parameters
    ----------
    r : float
        Radius of outer edge of bubble.
    Lwind : float
        Mechanical wind luminosity.
    vWind : float 
        terminal velocity of wind.

    Returns 
    -------
    Ram pressure. 
    """
    # Note:
        # old code: Pram()
        
    return Lwind / (2 * np.pi * r**2 * vWind)


# =============================================================================
# Find inner discontinuity
# R1 = interface separating inner bubble radius and outer solar wind
# =============================================================================

def get_r1(r1, params):
    """
    Root of this equation sets r1 (see Rahners thesis, eq 1.25).
    This is derived by balancing pressure.
    
    units of au
    
    Parameters
    ----------
    r1 : variable for solving the equation 
        The inner radius of the bubble.

    Returns
    -------
    equation : equation to be solved for r1.

    """
    # Note
    # old code: R1_zero()
    
    Lw, Ebubble, vw, r2 = params
    
    # set minimum energy to avoid zero
    if Ebubble < 1e-30:
        Ebubble = 1e-30
    # the equation to solve
    equation = np.sqrt( Lw / vw / Ebubble * (r2**3 - r1**3) ) - r1
    # return
    return equation



# =============================================================================
# Function that computes the bubble luminosity and mass loss dt
# This function should be removed, because they are re-written in bubble_luminosity.get_bubbleproperties()
# =============================================================================


# Warning: this part is deleted and is moved into new section and its own script.


def get_bubbleLuminosity(Data_struc,
                cool_struc,
                counter = 999, 
                xtol = 1e-6
        ):
    
    #  note: rgoal_f, verbose, plot, no_calc, error_exit was removed here. 
    #  xtol=1e-6 by default in old code.
    """
    calculate luminosity lost to cooling, and bubble temperature at radius rgoal_f*R2
    whole routine assumes units are Myr, Msun, pc and also returns result (Lb) in those units
    :param data_struc: for alpha, beta, delta, see definitions in Weaver+77, eq. 39, 40, 41
    :param rgoal_f: optional, sets location where temperature of bubble is reported: r = rgoal_f * R2; R1/R2 < rgoal_f < 1.
    :return: cooling luminity Lb, temperature at certain radius T_rgoal

    Parameters
    ----------
    Data_struc : TYPE
        List of parameters. See delta_new_root(). Includes the following:
            {'alpha',
             'beta',
             'Eb',
             'R2',
             't_now',
             'Lw',
             'vw',
             'dMdt_factor',
             'Qi',
             'mypath'}
    cool_struc : TYPE
        DESCRIPTION.
    warpfield_params : TYPE
        DESCRIPTION.

    Returns
    -------
    Lb : TYPE
        DESCRIPTION.
    T_rgoal : TYPE
        DESCRIPTION.
    Lb_b : TYPE
        DESCRIPTION.
    Lb_cz : TYPE
        DESCRIPTION.
    Lb3 : TYPE
        DESCRIPTION.
    dMdt_factor_out : TYPE
        DESCRIPTION.
    Tavg : TYPE
        DESCRIPTION.

    """
    
    
    # from get_bubbleStructure(), which is from run_energy_phase()
    
    # Note
    # old code: calc_Lb()
    # data_struc and cool_struc obtained from bubble_wrap()
    
    # TODO: double check units and functions before merging
    # Unpack input data
    # cgs units unless otherwise stated!!!
    # parameters for ODEs
    alpha = Data_struc['alpha']
    beta = Data_struc['beta']
    delta = Data_struc['delta']
    # Bubble energy
    Eb = Data_struc['Eb']
    # shell radius in pc (or outer radius of bubble)
    R2 = Data_struc['R2'] 
    # current time in Myr
    t_now = Data_struc['t_now'] 
    # mechanical luminosity
    Lw = Data_struc['Lw'] 
    # wind luminosity (and SNe ejecta)
    vw = Data_struc['vw'] 
    # guess for dMdt_factor (classical Weaver is 1.646; this is given as the 
    # constant 'A' in Eq 33, Weaver+77)
    dMdt_factor = Data_struc['dMdt_factor'] 
    # current photon flux of ionizing photons
    Qi = Data_struc['Qi'] 
    # velocity at r --> 0.
    v0 = 0.0 

    # solve for inner radius of bubble
    R1 = scipy.optimize.brentq(get_r1,
                               1e-3 * R2, R2, 
                               args=([Lw, Eb, vw, R2]), 
                               xtol=1e-18) # can go to high precision because computationally cheap (2e-5 s)
    
    # get bubble pressure
    press = bubble_E2P(Eb, R2, R1, warpfield_params.gamma_adia)
    
    # thermal coefficient in astronomical units
    c_therm = warpfield_params.c_therm * u.cm.to(u.pc) * u.g.to(u.Msun) / (u.s.to(u.Myr))**3
    # boltzmann constant in astronomical units 
    k_B = c.k_B.cgs.value * u.g.to(u.Msun) * u.cm.to(u.pc)**2 / u.s.to(u.Myr)**2

    # These constants maps to system of ODEs for bubble structure (see Weaver+77, eqs. 42 and 43).
    cons = calc_cons(alpha, beta, delta, t_now, press, c_therm)
    cons["Qi"] = Qi
    
    # print("cons", cons)
    
    # find dMdt -- 
    
    # See eq. 33, Weaver+77
    # get guess value
    dMdt_guess = float(os.environ["DMDT"])
    # if not given, then set it 
    if dMdt_guess == 0:
        dMdt_guess = 4. / 25. * dMdt_factor * 4. * np.pi * R2 ** 3. / t_now\
            * 0.5 * c.m_p.cgs.value * u.g.to(u.Msun) / k_B * (t_now * c_therm / R2 ** 2.) ** (2. / 7.) * press ** (5. / 7.)

    # initiate integration at radius R2_prime slightly less than R2 
    # (we define R2_prime by T(R2_prime) = TR2_prime
    # this is the temperature at R2_prime (important: must be > 1e4 K)
    # This thing here sets Tgoal. 
    TR2_prime = 3e4 
    
    # path to bubble strucutre file
    path2bubble = os.environ["Bstrpath"]
    # load r1/r2, r2prime/r2
    R1R2, R2pR2 = np.loadtxt(path2bubble, skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
    # what xi = r/R2 should we measure the bubble temperature?
    # old code: rgoal_f. This was the previous variable.
    xi_goal = get_xi_Tb(R1R2, R2pR2, warpfield_params)
    # print('xi_goal', xi_goal, R2)
    r_goal = xi_goal * R2
    # update 
    R1R2 = np.append(R1R2, R1/R2)
    R2pR2 = np.append(R2pR2, 0)
    # save file
    np.savetxt(path2bubble, np.c_[R1R2,R2pR2],delimiter='\t',header='R1/R2'+'\t'+'R2p/R2')  

    # sanity check. r_goal should be greater than the inner radius R1. 
    if r_goal <  R1:
        # for now, stop code. However, could also say that energy-driven phase is over if xi_goal < R1/R2, 
        # for a reasonable xi_goal (0.99 or so)
        sys.exit('The computed value for rgoal is too small. Please increase xi_Tb in your .param file!')

    # for the purpose of solving initially, try to satisfy BC v0 = 0 at some small radius R_small
    # R_small = np.min([R1,0.015]) #0.011 # in an old version, R_small = R1
    R_small = R1 # I presume Weaver meant: v -> 0 for R -> R1 (since in that chapter he talks about R1 being very small)
    bubble_params = {"v0": v0, "cons": cons, "rgoal": r_goal,
              "Tgoal": TR2_prime, "R2": R2, "R_small": R_small,
              "press": press, "Cool_Struc": cool_struc, "path": path2bubble}

    # prepare wrapper (to skip 2 superflous calls in fsolve)
    # Question: meaning this can actually be skipped and included in the main dMdt call to avoid confusion?
    bubble_params["dMdtx0"] = dMdt_guess
    bubble_params["dMdty0"] = compare_boundaryValues(dMdt_guess, bubble_params, warpfield_params)

    # print("bubble_params[\"dMdty0\"]", bubble_params["dMdty0"])
    # print(bubble_params)
    
    # 1. < factor_fsolve < 100.; if factor_fsolve is chose large, the rootfinder usually finds the solution faster
    # however, the rootfinder may then also enter a regime where the ODE soultion becomes unphysical
    # low factor_fsolve: slow but robust, high factor_fsolve: fast but less robust
    factor_fsolve = 50. #50
    
    # TODO: Add output_verbosity  
    # if i.output_verbosity <= 0:
    #     with stdout_redirected():
    #         dMdt = find_dMdt(dMdt_guess, params, factor_fsolve=factor_fsolve, xtol=1e-3)
            
    # compute the mass loss rate to find out how much of that is loss 
    # from shell into the shocked region.
    dMdt = get_dMdt(dMdt_guess, bubble_params, warpfield_params, factor_fsolve = factor_fsolve, xtol = 1e-3)
    
    
    # -- dMdt found
    
    # print("dMdt", dMdt_guess, dMdt)
        
    # if output is an array, make it a float (this is here because some
    # scipy.integrate solver returns float and some an array).
    if hasattr(dMdt, "__len__"): 
        dMdt = dMdt[0]
        
    ################################################      ######################
    
    # Here, two kinds of problem can occur:
    #   Problem 1 (for very high beta): dMdt becomes negative, the cooling luminosity diverges towards infinity
    #   Problem 2 (for very low beta): the velocity profile has negative velocities
    
    
    # CHECK 1: negative dMdt must not happen! (negative velocities neither, check later)
    # new factor for dMdt (used in next time step to get a good initial guess for dMdt)
    dMdt_factor_out = dMdt_factor * dMdt/dMdt_guess

    # get initial values
    R2_prime, y0 = get_start_bstruc(dMdt, bubble_params, warpfield_params)
    [vR2_prime, TR2_prime, dTdrR2_prime] = y0
    
    # print("R2_prime, y0", R2_prime, y0)

    # now we know the correct dMdt, but since we did not store the solution for T(r), calculate the solution once again (IMPROVE?)
    # figure out at which positions to calculate solution
    n_extra = 0  # number of extra points
    deltaT_min = 5000. # resolve at least temperature differences of deltaT_min
    dx0 = min(np.abs(deltaT_min / dTdrR2_prime), (R2_prime-R1)/1e6)
    r, top, bot, dxlist = get_r_list(R2_prime, bubble_params["R_small"], dx0, n_extra=n_extra)
    # find index where r is closest to rgoal
    r_goal_idx = np.argmin(np.abs(r - r_goal)) 
    # replace that entry in r with rgoal (with that we can ensure that we get the correct Tgoal
    r[r_goal_idx] = r_goal 

    # Create dictionary to feed into another ODE
    data_struc = {"cons": cons, "Cool_Struc": cool_struc}

    # TODO: if output verbosity is low, do not show warnings
    # if i.output_verbosity <= 0:
    #     with stdout_redirected():
    #         psoln = scipy.integrate.odeint(bubble_struct, y0, r, args=(Data_Struc,), tfirst=True)
    
    psoln = scipy.integrate.odeint(get_bubbleODEs, y0, r, args=(data_struc, warpfield_params.metallicity), tfirst=True)
    v = psoln[:,0]
    T = psoln[:,1]
    dTdr = psoln[:,2]
    # electron density ( = proton density), assume astro units (Msun, pc, Myr)
    n_e = press/((warpfield_params.mu_n/warpfield_params.mu_p) * k_B * T) 
    
    # print('v', v)
    # print('T', T)
    # print('dTdr', dTdr)
    # print('n_e', n_e)
    

    # CHECK 2: negative velocities must not happen! (??) [removed]

    # CHECK 3: temperatures lower than 1e4K should not happen
    min_T = np.min(T)
    if (min_T < 1e4):
        print("data_struc in bubble_structure2:", data_struc)
        sys.exit("could not find correct dMdt in bubble_structure.py")

    ######################################################################
    # Here, we deal with heating and cooling
    # heating and cooling (log10)
    onlyCoolfunc = cool_struc['Cfunc']
    onlyHeatfunc = cool_struc['Hfunc']
    
    # interpolation range (currently repeated in bubble_struct --> merge?)
    log_T_interd = 0.1
    log_T_noeqmax = cool_struc["log_T"]["max"] - 1.01 * log_T_interd
    
    # find 1st index where temperature is above Tborder ~ 3e5K 
    # (above this T, cooling becomes less efficient and less resolution is ok)

    # at Tborder we will switch between usage of CIE and non-CIE cooling curves
    Tborder = 10 ** log_T_noeqmax

    # find index of radius at which T is closest (and higher) to Tborder
    idx_6 = operations.find_nearest_higher(T, Tborder)

    # find index of radius at which T is closest (and higher) to 1e4K (no cooling below!), needed later
    idx_4 = operations.find_nearest_higher(T, 1e4)

    # interpolate and insert, so that we do have an entry with exactly Tborder
    if (idx_4 != idx_6):
        iplus = 20
        r46_interp = r[:idx_6+iplus]
        fT46 = interp1d(r46_interp, T[:idx_6+iplus] - Tborder, kind='cubic') # zero-function for T=Tborder
        fdTdr46 = interp1d(r46_interp, dTdr[:idx_6 + iplus], kind='linear')

        # calculate radius where T==Tborder
        rborder = scipy.optimize.brentq(fT46, np.min(r46_interp), np.max(r46_interp), xtol=1e-14)
        nborder = press/((warpfield_params.mu_n/warpfield_params.mu_p) * k_B * Tborder)
        dTdrborder = fdTdr46(rborder)

        # insert quantities at rborder to the full vectors
        dTdr = np.insert(dTdr,idx_6,dTdrborder)
        T = np.insert(T, idx_6, Tborder)
        r = np.insert(r, idx_6, rborder)
        n_e = np.insert(n_e, idx_6, nborder)

    ######################## 1) low resolution region (bubble) (> 3e5 K) ##############################
    r_b = r[idx_6:] # certain that we need -1?
    T_b = T[idx_6:]
    dTdr_b = dTdr[idx_6:]
    n_b = n_e[idx_6:] # electron density (=proton density)

    # at temperatures above 3e5 K assumption of CIE is valid. Assuming CIE, the cooling can be calculated much faster than if CIE is not valid.
    # extract the interpolation function for cooling
    f_logLambdaCIE = get_coolingFunction.create_coolCIE(warpfield_params.metallicity)
    
    Lambda_b = 10.**(f_logLambdaCIE(np.log10(T_b))) / (c.M_sun.cgs.value * c.pc.cgs.value**5 / u.Myr.to(u.s)**3)
    #Lambda_b = cool.coolfunc_arr(T_b) / myc.Lambda_cgs  # assume astro units (Msun, pc, Myr) # old (slow) version
    integrand = n_b ** 2 * Lambda_b * 4. * np.pi * r_b ** 2

    # power lost to cooling in bubble without conduction zone (where 1e4K < T < 3e5K)
    Lb_b = np.abs(np.trapz(integrand, x=r_b))

    # intermediate result for calculation of average temperature
    Tavg_tmp_b = np.abs(np.trapz(r_b**2*T_b, x=r_b))

    ######################### 2) high resolution region (CONDUCTION ZONE, 1e4K - 3e5K) #######################

    # there are 2 possibilities:
    # 1. the conduction zone extends to temperatures below 1e4K (unphysical, photoionization!)
    # 2. the conduction zone extends to temperatures above 1e4K
    # in any case, take the index where temperature is just above 1e4K

    # it could happen that idx_4 == idx_6 == 0 if the shock front is very, very steep
    if (idx_4 != idx_6): 
        # if this zone is not well resolved, solve ODE again with high resolution (IMPROVE BY ALWAYS INTERPOLATING)
        if idx_6 - idx_4 < 100: 
            # want high resolution here
            dx = (r[idx_4]-r[idx_6])/1e4 
            top = r[idx_4]
            bot = np.max([r[idx_6]-dx,dx])
            r_cz = np.arange(top, bot, -dx)

            # since we are taking very small steps in r, the solver might bitch around --> shut it up
            # with stdout_redirected():
            psoln = scipy.integrate.odeint(get_bubbleODEs, [v[idx_4],T[idx_4],dTdr[idx_4]], r_cz, 
                                           args=(data_struc, warpfield_params.metallicity), tfirst=True) # solve ODE again, there should be a better way (event finder!)

            T_cz = psoln[:,1]
            dTdr_cz = psoln[:,2]
            dTdR_4 = dTdr_cz[0]
        ######################
        else:
            r_cz = r[:idx_6+1]
            T_cz = T[:idx_6+1]
            dTdr_cz = dTdr[:idx_6+1]
            dTdR_4 = dTdr_cz[0]

        # TO DO: include interpolation
        # electron density (=proton density), assume astro units (Msun, pc, Myr)
        n_cz = press/( (warpfield_params.mu_n/warpfield_params.mu_p) * k_B *T_cz)
        Phi_cz = (Qi / u.Myr.to(u.s)) / (4. * np.pi * (r_cz * u.pc.to(u.cm)) ** 2)
        # cooling and heating
        mycool = 10. ** onlyCoolfunc(np.transpose(np.log10([n_cz / u.pc.to(u.cm) ** 3, T_cz, Phi_cz])))
        myheat = 10. ** onlyHeatfunc(np.transpose(np.log10([n_cz / u.pc.to(u.cm) ** 3, T_cz, Phi_cz])))
        dudt_cz = (myheat - mycool) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
        integrand = dudt_cz * 4. * np.pi * r_cz ** 2

        # power lost to cooling in conduction zone
        Lb_cz = np.abs(np.trapz(integrand, x=r_cz))

        # intermediate result for calculation of average temperature
        Tavg_tmp_cz = np.abs(np.trapz(r_cz ** 2 * T_cz, x=r_cz))

    elif ((idx_4 == idx_6) and (idx_4 == 0)):
        dTdR_4 = dTdr_b[0]
        Lb_cz = 0.

    ############################## 3) region between 1e4K and T[idx_4] #######################################

    #(use triangle approximation, i.e.interpolate)
    # If R2_prime was very close to R2 (where the temperature should be 1e4K), then this region is tiny (or non-existent)

    # find radius where temperature would be 1e4 using extrapolation from the measured T just above 1e4K, i.e. T[idx_4]
    T4 = 1e4
    R2_1e4 = (T4-T[idx_4])/dTdR_4 + r[idx_4]
    # this radius should be larger than r[idx_4] since temperature is monotonically decreasing towards larger radii
    #print(dTdR_4)
    #print(r_b[0], T_b[0])
    if R2_1e4 < r[idx_4]:
        sys.exit("Something went wrong in the calculation of radius at which T=1e4K in bubble_structure.py")

    # interpolate between R2_prime and R2_1e4 (triangle)
    # it's important to interpolate because the cooling function varies a lot between 1e4 and 1e5K
    f3 = interp1d(np.array([r[idx_4],R2_1e4]), np.array([T[idx_4],T4]), kind = 'linear')
    #f = interp1d(np.append(r_cz[5:0:-1], [R2_1e4]), np.append(T_cz[5:0:-1], [T4]), kind='quadratic')
    r3 = np.linspace(r[idx_4], R2_1e4, num = 1000, endpoint = True)
    T3 = f3(r3)
    # electron density (=proton density), assume astro units (Msun, pc, Myr)
    n3 = press/( (warpfield_params.mu_n/warpfield_params.mu_p) * k_B * T3) 
    Phi3 = (Qi/ u.Myr.to(u.s)) / (4. * np.pi * (r3 * u.pc.to(u.cm)) ** 2)

    mask = {'loT': T3 < Tborder, 'hiT': T3 >= Tborder}
    Lb3 = {}
    for mask_key in ['loT', 'hiT']:
        
        msk = mask[mask_key]

        if mask_key == "loT":
            mycool = 10. ** onlyCoolfunc(np.transpose(np.log10([n3[msk] / u.pc.to(u.cm) ** 3, T3[msk], Phi3[msk]])))
            myheat = 10. ** onlyHeatfunc(np.transpose(np.log10([n3[msk] / u.pc.to(u.cm) ** 3, T3[msk], Phi3[msk]])))
            dudt3 = (myheat - mycool) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
            integrand = dudt3 * 4. * np.pi * r3[msk] ** 2
        elif mask_key == "hiT":
            Lambda_b = 10. ** (f_logLambdaCIE(np.log10(T3[msk]))) / (c.M_sun.cgs.value * c.pc.cgs.value**5 / u.Myr.to(u.s)**3)
            integrand = n3[msk] ** 2 * Lambda_b * 4. * np.pi * r3[msk] ** 2

        Lb3[mask_key] = np.abs(np.trapz(integrand, x=r3[msk]))

    Lb3 = Lb3['loT'] + Lb3['hiT']

    # intermediate result for calculation of average temperature
    Tavg_tmp_3 = np.abs(np.trapz(r3 ** 2 * T3, x=r3))

    ######################################################################

    # add up cooling luminosity from the 3 regions
    Lb = Lb_b + Lb_cz + Lb3
    
    # print("Lb_b + Lb_cz + Lb3", Lb_b, Lb_cz,Lb3)
    # old check 
    # return 

    if (idx_4 != idx_6):
        Tavg = 3.* (Tavg_tmp_b/(r_b[0]**3. - r_b[-1]**3.) + Tavg_tmp_cz/(r_cz[0]**3. - r_cz[-1]**3.) + Tavg_tmp_3/(r3[0]**3. - r3[-1]**3.))
    else:
        Tavg = 3. * (Tavg_tmp_b / (r_b[0] ** 3. - r_b[-1] ** 3.) + Tavg_tmp_3 / (r3[0] ** 3. - r3[-1] ** 3.))


    # get temperature inside bubble at fixed scaled radius
    if r_goal > r[idx_4]: # assumes that r_cz runs from high to low values (so in fact I am looking for the highest element in r_cz)
        T_rgoal = f3(r_goal)
    elif r_goal > r[idx_6]: # assumes that r_cz runs from high to low values (so in fact I am looking for the smallest element in r_cz)
        idx = operations.find_nearest(r_cz, r_goal)
        T_rgoal = T_cz[idx] + dTdr_cz[idx]*(r_goal - r_cz[idx])
    else:
        idx = operations.find_nearest(r_b, r_goal)
        T_rgoal = T_b[idx] + dTdr_b[idx]*(r_goal - r_b[idx])

    # TODO: These should not be constants?
    # r_Phi = np.array([r[0]])
    # Phi_grav_r0b = np.array([5.0])
    # f_grav = np.array([5.0])
    # Mbubble = 10.
    
    # get graviational potential (in cgs units)
    # first we need to flip the r and n vectors (otherwise the cumulative mass will be wrong)
    # now r is monotonically increasing
    r_Phi_tmp = np.flip(r,0) * u.pc.to(u.cm) 
    # mass density (monotonically increasing)
    rho_tmp =  (np.flip(n_e,0)/(u.pc.to(u.cm)**3)) * c.m_p.cgs.value
    dx = np.flip(dxlist,0)
    # mass per bin (number density n was in 1/pc**3)
    m_r_tmp = rho_tmp * 4.*np.pi*r_Phi_tmp**2 * dx * u.pc.to(u.cm)  
    # cumulative mass
    Mcum_tmp = np.cumsum(m_r_tmp) 
    Phi_grav_r0b = -4.*np.pi* c.G.cgs.value * scipy.integrate.simps(r_Phi_tmp*rho_tmp,x=r_Phi_tmp)
    # gravitational force per unit mass
    f_grav_tmp = c.G.cgs.value * Mcum_tmp/r_Phi_tmp**2. 

    # skip some entries, so that length becomes 100, then concatenate the last 10 entries (potential varies a lot there)
    potentialFile_internalLength = 10000
    len_r = len(r_Phi_tmp)
    skip = max(int(float(len_r) / float(potentialFile_internalLength)),1)
    r_Phi = np.concatenate([r_Phi_tmp[0:-10:skip], r_Phi_tmp[-10:]]) # flip lists (r was monotonically decreasing)
    #Phi_grav = np.concatenate([Phi_grav_tmp[0:-10:skip], Phi_grav_tmp[-10:]])
    f_grav = np.concatenate([f_grav_tmp[0:-10:skip], f_grav_tmp[-10:]])

    # mass of material inside bubble (in solar masses)
    Mbubble = Mcum_tmp[-1] / c.M_sun.cgs.value
        
    # save bubble structure as .txt file (radius, density, temperature)?
    if warpfield_params.write_bubble == True:
        # only save Ndat entries (equally spaced in index, skip others)
        Ndat = 450
        len_r_b = len(r_b)
        Nskip = int(max(1, len_r_b/Ndat))

        rsave = np.append(r_b[-1:Nskip:-Nskip],r_b[0])
        nsave = np.append(n_b[-1:Nskip:-Nskip],n_b[0])
        Tsave = np.append(T_b[-1:Nskip:-Nskip],T_b[0])
        if idx_6 != idx_4: # conduction zone is resolved
            Ndat = 50
            len_r_cz = len(r_cz)
            Nskip = int(max(1, len_r_cz / Ndat))
            # start at -2 to make sure not to have the same radius as in r_b again
            rsave = np.append(rsave,r_cz[-Nskip-1:Nskip:-Nskip]) 
            nsave = np.append(nsave,n_cz[-Nskip-1:Nskip:-Nskip])
            Tsave = np.append(Tsave,T_cz[-Nskip-1:Nskip:-Nskip])

            rsave = np.append(rsave,r_cz[0])
            nsave = np.append(nsave, n_cz[0])
            Tsave = np.append(Tsave, T_cz[0])
            
        # convert units to cgs for CLOUDY
        rsave *= c.pc.cgs.value
        nsave *= c.pc.cgs.value**(-3.)

        bub_savedata = {"r_cm": rsave, "n_cm-3":nsave, "T_K":Tsave}
        name_list = ["r_cm", "n_cm-3", "T_K"]
        tab = Table(bub_savedata, names=name_list)
        mypath = data_struc["mypath"]
        # age in years (factor 1e7 hardcoded), naming convention matches naming convention for cloudy files
        age1e7_str = ('{:0=5.7f}e+07'.format(t_now / 10.)) 
        outname = os.path.join(mypath, "bubble/bubble_SB99age_"+age1e7_str+".dat")
        formats = {'r_cm': '%1.9e', 'n_cm-3': '%1.5e', 'T_K': '%1.5e'}
        tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)

        # create cloudy bubble.in file?
        # TODO
        # if warpfield_params.write_bubble_CLOUDY == True:
            # __cloudy_bubble__.write_bubble(outname, Z = warpfield_params.metallicity)

        # Should I uncomment this and add to thefinal return?
        # # some random debug values
        # r_Phi = np.array([r[0]])
        # Phi_grav_r0b = np.array([5.0])
        # f_grav = np.array([5.0])
        # Mbubble = 10.
        
        # The original return:
            # (but looks like most valeus are useless.)
        # return Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav
        
    return Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav



def get_xi_Tb(l1, l2, warpfield_params):
    """
    This function extracts the relative radius xi = r/R2 at which to measure 
    the bubble temperature. Unless the bubble structure file already has an 
    input, it will assume the default value given in the initial .param file.

    Parameters
    ----------
    l1 : ratio
        R1R2.
    l2 : ratio
        R2pR2.

    """
    
    xi_Tb = warpfield_params.xi_Tb
    # check if there are any existing values in the bubble file.
    try:
        if len(l1) > 2:
            l1 = l1[l1!=0]
            l2 = l2[l2!=0]
            if len(l1) > 1:
                a = np.max(l1)
                b = np.min(l2)
                xi_Tb = b - 0.2 * (b - a)
                if np.isnan(xi_Tb):
                    xi_Tb = warpfield_params.xi_Tb
    except:
        pass
    # return
    return xi_Tb

def get_r_list(r_upper, r_lower, 
               r_step0, 
               n_extra):
    """
    This function creates a list of r values where bubble structure 
    will be claculated. 
    The output is monotonically decreasing. 

    Parameters  (these are all in pc)
    ----------
    r_upper : float
        Upper limit of r (first entry in output).
    r_lower : float
        Lower limit of r (usually last entry in output).
    r_step0 : float
        Initial step size.
    n_extra : float
        DESCRIPTION.

    Returns
    -------
    r : array
        An array of r.
    top : float
        upper limit in r.
    bot : float
        lower limit in r.
    dxlist : array
        An array of step sizes.

    """
    
    # figure out at which position we would want to calculate the solution
    top = r_upper
    bot = np.max([r_lower, r_step0])
    
    clog = 2.  # max increase in dx (in log) that is allowed, e.g. 2.0 -> dxmax = 100.*dx0 (clog = 2.0 seems to be a good compromise between speed an precision)
    dxmean = (10. ** clog - 1.) / (clog * np.log(10.)) * r_step0  # average step size
    Ndx = int((top - bot) / dxmean)  # number of steps with chosen step size
    dxlist = np.logspace(0., clog, num=Ndx) * r_step0  # list of step sizes
    r = top + r_step0 - np.cumsum(dxlist)  # position vector on which ODE will be solved
    r = r[r > bot]  # for safety reasons
    r = np.append(r, bot)  # make sure the last element is exactly the bottom entry

    return r, top, bot, dxlist   


def get_dMdt(dMdt_guess, bubble_params, warpfield_params, factor_fsolve = 50., xtol = 1e-6):
    """
    This function employs root finder to get correct dMdt, i.e., the 
    mass loss rate dM/dt from shell into shocked region.

    Parameters
    ----------
    dMdt_guess : float
        initial guess for dMdt.
    bubble_params : dict
        A temporary dictionary made to store necessary information of the bubble.
        This is defined in bubble_structure.bubble_structure()
        includes: [v0 = v[Rsmall], cons = [a,b,c,d,e, Phi], R2_prime, R2, Rsmall, dR2, press
            Rsmall: some very small radius (nearly 0)
            R2_prime: radius very slightly smaller than shell radius R2
            R2: shell radius
            dR2: R2 - R2_prime
            press: pressure inside bubble
    warpfield_params : object
        Object containing WARPFIELD parameters.
    factor_fsolve : float, optional
        scipy.optimize.fsolve parameter. The default is 50..
    xtol : float, optional
        scipy.optimize.fsolve parameter. The default is 1e-6.

    Returns
    -------
    dMdt : float
        mass loss rate dM/dt from shell into shocked region.

    """
    
    # Note
    # old code: find_dMdt()

    # retrieve data
    countl = float(os.environ["COUNT"])
    dmdt_0l = float(os.environ["DMDT"])
    
    # solve for dMdt
    dMdt = scipy.optimize.fsolve(compare_boundaryValues_wrapper, dMdt_guess, args=(bubble_params, warpfield_params), 
                                 factor = factor_fsolve, xtol = xtol, epsfcn = 0.1 * xtol)
    if dMdt < 0:
        print('rootfinder of dMdt gives unphysical result...trying to solve again with smaller step size')
        dMdt = scipy.optimize.fsolve(compare_boundaryValues_wrapper, dMdt_guess, args=(bubble_params, warpfield_params), 
                                     factor=15, xtol=xtol, epsfcn=0.1*xtol)
        if dMdt < 0:
            #if its unphysical again, take last dmdt and change it slightly for next timestep
            dMdt = dmdt_0l+ dmdt_0l*1e-3 
            # count how often you did this crude approximation
            countl += 1 
            if countl >3:
                sys.exit("Unable to find correct dMdt, have to abort WARPFIELD")
    # dmdt
    try:
        os.environ["DMDT"] = str(dMdt[0])
    except:
        os.environ["DMDT"] = str(dMdt)
    # counter for fsolve
    os.environ["COUNT"] = str(countl)
    # return
    return dMdt

def compare_boundaryValues_wrapper(dMdt, bubble_params, warpfield_params):
    """A mini wrapper which do initial check before running the 
    compare_boundaryValues function"""
    if dMdt == bubble_params["dMdtx0"] and bubble_params["dMdty0"] is not None:
        return bubble_params["dMdty0"]
    else:
        return compare_boundaryValues(dMdt, bubble_params, warpfield_params)
    
def compare_boundaryValues(dMdt, bubble_params, warpfield_params):
    """
    This function compares boundary value calculated from dMdt guesses with 
    true boundary conditions. This routine is repeatedly called with different
    dMdt intil the true v0 and estimated v0 from this dMdt agree.
    Finally, this yields a residual dMdt, which is nearly zero, and that 
    is what we are looking for.

    Parameters
    ----------
    dMdt : float
        Guess for mass loss rate.
    bubble_params : dict
        A temporary dictionary made to store necessary information of the bubble.
        This is defined in bubble_structure.bubble_structure()
        includes: [v0 = v[Rsmall], cons = [a,b,c,d,e, Phi], R2_prime, R2, Rsmall, dR2, press
            Rsmall: some very small radius (nearly 0)
            R2_prime: radius very slightly smaller than shell radius R2
            R2: shell radius
            dR2: R2 - R2_prime
            press: pressure inside bubble
    warpfield_params : object
        Object containing WARPFIELD parameters.

    Returns
    -------
    residual : float
        residual of true v(Rsmall)=v0 (usually 0) and estimated v(Rsmall).

    """
    
    # Notes:
    # old code: comp_bv_au()
    # bubble_params is defined in calc_Lb()
    
    # if dMdt is given as a length one list
    if hasattr(dMdt, "__len__"): 
        dMdt = dMdt[0]
    
    # get initial values
    R2_prime, y0 = get_start_bstruc(dMdt, bubble_params, warpfield_params)
    # unravel
    [vR2_prime, TR2_prime, dTdrR2_prime] = y0
    # define data structure to feed into ODE
    Data_Struc = {"cons" : bubble_params["cons"], "Cool_Struc" : bubble_params["Cool_Struc"]}

    # figure out at which postions to calculate solution
    # number of extra points
    n_extra = 0 
    # some initial step size in pc
    dx0 = (R2_prime - bubble_params["R_small"]) / 1e6  
    # An array of r
    r, _, _, _ = get_r_list(R2_prime, bubble_params["R_small"], dx0, n_extra=n_extra)
    
    # print('comp values', vR2_prime, TR2_prime, dTdrR2_prime, r, warpfield_params.metallicity)
    # try to solve the ODE (might not have a solution)
    try:
        psoln = scipy.integrate.odeint(get_bubbleODEs, y0, r, args=(Data_Struc, warpfield_params.metallicity), tfirst=True)
        # get
        v = psoln[:, 0]
        T = psoln[:, 1]
    
        # this are the calculated boundary value (velocity at r=R_small)
        v_bot = v[-(n_extra+1)]
        # compare these to correct calues!
        residual = (bubble_params["v0"] - v_bot)/v[0]
    
        # very low temperatures are not allowed! 
        # This check is also necessary to prohibit rare fast (and unphysical) oscillations in the temperature profile
        min_T = np.min(T)
        if min_T < 3e3:
            residual *= (3e4/min_T)**2
    # should the ODE solver fail
    except:
        # this is the case when the ODE has no solution with chosen inital values
        print("Giving a wrong residual here; unable to solve the ODE. Suggest to set xi_Tb to default value of 0.9.")
        if dTdrR2_prime < 0.:
            residual = -1e30
        else:
            residual = 1e30

    return residual



def get_start_bstruc(dMdt, bubble_params, warpfield_params):
    """
    This function computes starting values for the bubble structure
    measured at r2_prime (upper limit of integration, but slightly lesser than r2).
    
    This uses a shooting method to find what is the upper boundary condition. 

    Parameters
    ----------
    dMdt : float
        Mass flow rate into bubble.
    bubble_params : dict
        A temporary dictionary made to store necessary information of the bubble.
        This is defined in bubble_structure.bubble_structure()
        includes: [v0 = v[Rsmall], cons = [a,b,c,d,e, Phi], R2_prime, R2, Rsmall, dR2, press
            Rsmall: some very small radius (nearly 0)
            R2_prime: radius very slightly smaller than shell radius R2
            R2: shell radius
            dR2: R2 - R2_prime
            press: pressure inside bubble
    warpfield_params : object
        Object containing WARPFIELD parameters.

    Returns
    -------
    R2_prime : float
        upper limit of integration.
    y0 : list
        [velocity, temperature, dT/dr].
    """
    # Notes:
    # old code: calc_bstruc_start()
    
    # thermal coefficient in astronomical units
    c_therm = warpfield_params.c_therm * u.cm.to(u.pc) * u.g.to(u.Msun) / (u.s.to(u.Myr))**3
    # boltzmann constant in astronomical units 
    k_B = c.k_B.cgs.value * u.g.to(u.Msun) * u.cm.to(u.pc)**2 / u.s.to(u.Myr)**2
    # coefficient for temperature calculations (see Weaver+77, Eq 44)
    # https://articles.adsabs.harvard.edu/pdf/1977ApJ...218..377W
    coeff_T = (25./4.) * k_B / (0.5 * c.m_p.cgs.value * u.g.to(u.Msun) * c_therm) 
    # here dR2 is R2-r in Eq 44
    # Tgoal is the target temperature. It is set as 3e4K.
    # spatial separation between R2 and the point where the ODE solver is initialized (cannot be exactly at R2)
    dR2 = (bubble_params["Tgoal"]**2.5) / (coeff_T * dMdt / (4. * np.pi * bubble_params["R2"] ** 2.))
    
    
    # Question: What does this do? This just records the value. Is this important for future?
    # path2bubble structure
    path2bubble = os.environ["Bstrpath"]
    # load data; the file is empty, and was initialised in initialise_bstruc().
    R1R2 , R2pR2 = np.loadtxt(path2bubble, skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
    
    
    # IMPORTANT: this number might have to be higher in case of very strong winds (clusters above 1e7 Msol)! 
    # TODO: figure out, what to set this number to...
    dR2min = 1.0e-7 
    mCloud = float(os.environ["Mcl_aux"])
    sfe = float(os.environ["SF_aux"])
    mCluster = mCloud * sfe
    if mCluster > 1.0e7:
        dR2min = 1.0e-14 * mCluster + 1.0e-7
    if dR2 < dR2min: 
        dR2 = np.sign(dR2)*dR2min # prevent super small dR2
        
        
    # radius at which ODE solver is initialized. At this radius the temperature is Tgoal
    # these primes are analogous to dR2 from above.
    R2_prime = bubble_params["R2"] - dR2 
    # should be Tgoal (usually set to 30,000 K)
    TR2_prime = (coeff_T * dMdt * dR2/ (4. * np.pi * bubble_params["R2"] ** 2.)) ** 0.4  
    # append values for r1/r2, r2prime/r2
    R1R2 = np.append(R1R2, 0)
    R2pR2 = np.append(R2pR2 ,R2_prime/bubble_params["R2"])
    # save data
    np.savetxt(path2bubble, np.c_[R1R2,R2pR2],delimiter='\t',header='R1/R2'+'\t'+'R2p/R2')
    # sanity check
    if (bubble_params["rgoal"] > R2_prime):
        sys.exit("rgoal_f is outside allowed range in bubble_structure.py (too large). Decrease r_Tb in .param (<1.0)!")
    # temperature gradient at R2_prime, this is not the correct boundary condition (only a guess). We will determine the correct value using a shooting method
    dTdrR2_prime = -2. / 5. * TR2_prime / dR2  
    # velocity at R2_prime
    vR2_prime = bubble_params["cons"]["a"] * bubble_params["R2"] - dMdt * k_B *\
        TR2_prime / (4. * np.pi * bubble_params["R2"] ** 2. * 0.5 * c.m_p.cgs.value * u.g.to(u.Msun) * bubble_params["press"]) 
    # y0: initial conditions for bubble structure
    y0 = [vR2_prime, TR2_prime, dTdrR2_prime]
    # return
    
    
    return R2_prime, y0

# =============================================================================
# Functions that computes delta
# =============================================================================

def get_delta_residual(delta_input, params):
    """
    This function takes in a predictor and outputs a better estimate of delta

    Parameters
    ----------
    delta_input : float
        delta; (see Weaver+77, eqs. 39-41)
    params : list
        list of useful parameters:
            [Data_struc, Cool_Struc, t_10list, T_10list, fit_len, warpfield_params]
            See bubble_wrap() for corresponding inputs.
    """
    # Notes:
    # old code: delta_zero(), cdelta()
    
    # unravel
    Data_struc, Cool_Struc, t_10list, T_10list, fit_len, warpfield_params = params
    # copy to change item to add extra parameter
    data_struc_temp = dict.copy(Data_struc)
    data_struc_temp['delta'] = delta_input

    # get structure
    T_rgoal = get_bubbleLuminosity(data_struc_temp, Cool_Struc, 999)[1]

    # use temperature of the bubble T_temp which has been calculated with a slightly wrong delta 
    # (delta_old, the predictor) to calculate a better estimate of delta
    # Appends value to end of list and removes first element of list if
    # list would become longer than a given maximum length.
    T_10list_temp = np.append(T_10list, T_rgoal)
    while (len(T_10list_temp) > fit_len):
        T_10list_temp = np.delete(T_10list_temp, 0)
    log_t = np.log(t_10list)
    log_T = np.log(T_10list_temp)
    c_guess = np.round(log_T[0], decimals=2)
    m_guess = np.round(data_struc_temp['old_delta'], decimals = 2)
    my_fscale = np.std(log_T)
    # function for linear regression
    def f_lin(x,t,y):
        return x[0] + x[1]*t - y
    # robust residual
    res_robust = scipy.optimize.least_squares(f_lin, [c_guess, m_guess], loss='soft_l1', f_scale=my_fscale,
                                              args=(log_t, log_T))
    delta_output = res_robust.x[1]
    # find residual
    residual = delta_input - delta_output
    # return
    return residual
    


# param1 = {'alpha': alpha, 'beta': beta, 'Eb': E0, 'R2': r0, 't_now': t0, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr, 'mypath': mypath}
# param0 = {'alpha': alpha, 'beta': beta, 'Eb': E0m1, 'R2': r0m1, 't_now': t0m1, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr, 'mypath': mypath}
# dzero_params = [param0, param1, Cool_Struc]
 
    
 
def get_delta_new(delta_old, params):

    
    # Notes:
    # old code: delta_new_root()
        

    def get_delta_residual_new(delta_in, params):
        
        # Notes:
        # old code: new_zero_delta()
        
        Cool_Struc = params[2]
        data_struc0 = dict.copy(params[0])
        data_struc1 = dict.copy(params[1])
        t0 = data_struc0['t_now']
        t1 = data_struc1['t_now']
        data_struc0['delta'] = delta_in
        data_struc1['delta'] = delta_in
        Lb_temp0, T_rgoal0, dMdt_factor_out0 = bstrux([data_struc0, Cool_Struc])        # get output
        Lb_temp1, T_rgoal1, dMdt_factor_out1 = bstrux([data_struc1, Cool_Struc])
        delta_out = (T_rgoal1 - T_rgoal0)/(t1-t0) * t1/T_rgoal1
        # calculate residual
        residual = delta_out - delta_in
        # return
        return residual

    bubbleFailed = False
    
    # try once with some boundary values and hope there is a zero point in between
    try:
        # this might fail if no fixpoint exists in the given range. 
        # If so, try with a larger range
        delta = scipy.optimize.brentq(get_delta_residual_new, delta_old - 0.1 , delta_old + 0.1,
                                      args = (params), xtol = 1e-9, rtol = 1e-8) 
    except:
        # it seems either the boundary values were too far off (and bubble_structure crashed) or there was no zero point in between the boundary values
        # try to figure out what limits in delta are allowed

        worked_last_time_lo = True
        worked_last_time_hi = True

        iic = 0
        n_trymax = 30 # maximum number of tries before we give up
        sgn_vec = np.zeros(2*n_trymax+1) # list containing the signs of the residual (if there is a sign flip between two values, there must be a zero point in between!)
        delta_in_vec = np.zeros(2*n_trymax+1) # list containing all tried input deltas
        ii_lo = np.nan
        ii_hi = np.nan
        # list which contains the number 2.0 where a sign flip ocurred
        diff_sgn_vec = abs(sgn_vec[1:]-sgn_vec[:-1]) 

        # stay in loop as long as sign has not flipped
        while all(diff_sgn_vec < 2.):

            res_0 = get_delta_residual_new(delta_old, params)
             # is probably not 0 (because of small numerical noise) but ensure it is not 0 further down
            sgn_vec[n_trymax] = np.sign(res_0)
            delta_in_vec[n_trymax] = delta_old

            if worked_last_time_lo:
                try:
                    delta_in_lo = delta_old - 0.02 - float(iic) * 0.05
                    res_lo = get_delta_residual_new(delta_in_lo, params)
                    ii_lo = n_trymax-iic-1
                    sgn_vec[ii_lo] = np.sign(res_lo)
                    delta_in_vec[ii_lo] = delta_in_lo
                    if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax]= sgn_vec[n_trymax-1] # make sure 0 does not ocurr
                except:
                    worked_last_time_lo = False

            if worked_last_time_hi:
                try:
                    delta_in_hi = delta_old + 0.02 + float(iic) * 0.05
                    res_hi = get_delta_residual_new(delta_in_hi, params)
                    ii_hi = n_trymax+iic+1
                    sgn_vec[ii_hi] = np.sign(res_hi)
                    delta_in_vec[ii_hi] = delta_in_hi
                    if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax] = sgn_vec[n_trymax + 1] # make sure 0 does not ocurr
                except:
                    worked_last_time_hi = False

            if iic > n_trymax / 2:
                print("I am having a hard time finding delta...")
                if iic >= n_trymax - 1:
                    sys.exit("Could not find delta.")
                    
            # this list contains a 2.0 where the sign flip ocurred (take abs, so that -2.0 becomes +2.0)
            diff_sgn_vec = abs(sgn_vec[1:] - sgn_vec[:-1]) 

            iic += 1

        # find the index where the sign flip ocurred (where the diff list has the element 2.0)
        idx_zero0 = np.argmax(diff_sgn_vec) # we could also look for number 2.0 but because there are no higher number, finding the maximum is equivalent
        delta_in_lo = delta_in_vec[idx_zero0]
        delta_in_hi = delta_in_vec[idx_zero0+1]
        
        
    #########################################
        try:
            # this might fail if no fixpoint exists in the given range
            delta = scipy.optimize.brentq(get_delta_residual, delta_in_lo , delta_in_hi, args=(params),
                                          xtol=0.1 * 1e-9, rtol=1e-9) 
        except:
            delta = delta_old
            bubbleFailed = True # something went wrong

    return delta, bubbleFailed


def get_fitSlope(x, y, old_guess = np.nan, c_guess=0.):
        """
        calculate slope of linear fit
        neglect outliers for the fits
        :param x: e.g. time list (np array)
        :param y: e.g. temperature list (np array)
        :param loss: correction function for increasing robustness: 'linear' gives you normal least_squares (not robust), 'soft_l1' and 'huber' have medium robustness, 'cauchy' and 'arctan' have high robustness
                    (for more info, see http://scipy-cookbook.readthedocs.io/items/robust_regression.html)
        :return: slope m
        """
        
        # old code: calc_linfit()

        # IT SEEMS SAFER NOT TO USE GUESSED VALUES PROVIDED BY THE USER. 
        # WE ARE JUST USING OUR OWN ONES (but keep in mind that this means, 
        # this routine only works to calculate alpha and beta)
        my_c_guess = c_guess
        my_m_guess = 0.7
        
        # we need to guess what the soft threshold between inliners and outliers is
        # very rough order of magnitude approximation: use standard deviation
        # (better: use standard deviation from fit curve, but for that we would need to know the fit beforehand)
        my_fscale = np.std(y) # my_fscale = 0.1
        
        # linear regression
        def f_lin(x,t,y):
            return x[0] + x[1]*t - y
        # get residual
        res_robust = scipy.optimize.least_squares(f_lin, [my_c_guess, my_m_guess], loss='soft_l1', f_scale=my_fscale, args=(x, y))
        m_temp1 = res_robust.x[1]

        # maybe we picked the wrong guess for m?
        if ((not np.isnan(old_guess)) and abs(m_temp1-old_guess) > 0.05):
            my_m_guess = 0.0
            res_robust = scipy.optimize.least_squares(f_lin, [my_c_guess, my_m_guess], loss='soft_l1', f_scale=my_fscale, args=(x, y))
            m_temp2 = res_robust.x[1]

            my_m_guess = 2.0
            res_robust = scipy.optimize.least_squares(f_lin, [my_c_guess, my_m_guess], loss='soft_l1', f_scale=my_fscale, args=(x, y))
            m_temp3 = res_robust.x[1]

            m_temp_list = np.array([m_temp1, m_temp2, m_temp3])
            idx = np.argmin(abs(m_temp_list-old_guess))
            m = m_temp_list[idx]
        else:
            m = m_temp1

        return m


def bstrux(full_params):
    # A more simplified version of get_bubbleLuminosity()
    # only returns Lb, Trgoal, dMdt_factor, Tavg.
    Cool_Struc = full_params[1]
    my_params = dict.copy(full_params[0])
    # yo why is this such high value Daniel
    counter = 789

    # call calc_Lb with or without set xtol?
    if 'xtolbstrux' in my_params:
        # this is always a constant and is always 1e-6. Remove in update.
        xtol = my_params['xtolbstrux']
        [Lb, Trgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav] = get_bubbleLuminosity(my_params, Cool_Struc, counter, xtol=xtol)
    else:
        [Lb, Trgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav] = get_bubbleLuminosity(my_params, Cool_Struc, counter)

    bstrux_result = {'Lb':Lb, 'Trgoal':Trgoal, 'dMdt_factor': dMdt_factor_out, 'Tavg': Tavg}

    return bstrux_result


