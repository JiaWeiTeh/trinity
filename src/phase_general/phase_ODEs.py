#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:28:20 2023

@author: Jia Wei Teh

"""


import numpy as np
import os
import scipy.optimize
import astropy.constants as c
import astropy.units as u
#--
import src.cloud_properties.mass_profile as mass_profile
import src.cloud_properties.density_profile as density_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.shell_structure.shell_structure as shell_structure
import src._functions.unit_conversions as cvt

import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()



'''
Actually, the meaning of this function is that this calculates
part 1 of the fE_tot that is in phase_energy1. Meaning, 
that this only returns vd (importantly), since the whole fE_tot should
return vd, rd, Ed, and Td.

Interestingly, this function is also being used in warp_reconstruct() in warp_writedata.py, 
to calculate values and to 


'''

# TODO: this is old function: remember to add what cf_construct etc do

# def get_vdot(t, y, 
#                  params, SB99f, 
#                  Eb0=1, cfs=False, cf_reconstruct=1 #these three are unused, because we assume cf = 1 now.
#                  ):
    
    
def get_vdot(t, y, 
                 params, SB99f):
    
    """
    units:
        t [yr]
        R2 [pc]
        v2 [pc/yr]
        Eb [u.M_sun*u.pc**2/u.yr**2]
        
    returns: 
        vdot: pc/yr2
    """
    
    
    # Note:
        # old code: ODE_tot_aux.fE_tot_part1

    # unpack current values of y (r, rdot, E, T)
    R2, v2, Eb, T0 = y  
    
    tSF = params['tSF'].value
    
    
    
    # # Add units
    # rShell *= u.pc
    # vShell *= u.km/u.s
    # Ebubble *= u.erg
    # TBubble *= u.K
    
    # # unpack 'ODEpar' parameters
    # mCloud = ODEpar['mCloud']
    # mCluster = warpfield_params.mCluster
    # rCloud = ODEpar['rCloud']
    # tSF = 0.0 * u.Myr


    # Interpolate SB99 to get feedback parameters
    # can't quite use what's in the dictionary because it could be changed.
    # TODO: in future just add them into dictionary
    # mechanical luminosity at time t (erg)
    L_wind = SB99f['fLw_cgs'](t) * cvt.L_cgs2au
    # momentum of stellar winds at time t (cgs)
    pdot_wind = SB99f['fpdot_cgs'](t) * cvt.pdot_cgs2au
    # other luminosities
    Lbol = SB99f['fLbol_cgs'](t) * cvt.L_cgs2au
    Ln = SB99f['fLn_cgs'](t) * cvt.L_cgs2au
    Li = SB99f['fLi_cgs'](t) * cvt.L_cgs2au
    Qi = SB99f['fQi_cgs'](t) / cvt.s2Myr
    
    # velocity from luminosity and change of momentum (au)
    v_wind = (2.*L_wind/pdot_wind)
    
    # I think there is a big problem with this side of the code.
    # The problem is that mShell calculated here is using the mass profile, 
    # which actually is not correct. The profile is the intiial mass profile of the 
    # entire cloud. However, we are talking about the shell here. 
    # We shopudl therefore use the shell mass parameter in the shell prop function.
    # Also, what exactly do we really need here? Is it mShell or mBubble for vd?
    # This needs to be check properly.
    
    
    
    # TODO!! check in the old version, which script updates [Rsh_max] and add it in.
    # It might have been removed cause I originally thought it was a useless additon.


    # check max extent of shell radius ever (important for recollapse, since the shell mass will be kept constant)
    # We only want to set this parameter when the shell has reached its maximum extent and is about to collapse
    # (reason: solver might overshoot and 'check out' a state with a large radius, only to then realize that it should take smaller time steps)
    # This is hard to do since we can define an event (velocity == 0) but I don't know how to set this value only in the case when the event occurs
    # workaround: only set this value when the velocity is close to 0. (and via an event make sure that solver gets close to v==0)
    # ---- verions1 TODO:
    # if v2 <= 0.0:
    #     print('v2, Rsh_max, R2', v2, params['Rsh_max'].value, R2)
    #     params['Rsh_max'].value = max(R2, params['Rsh_max'].value)


    # ADD SHELL THING HERE

    # =============================================================================
    # Shell mass, where radius = maximum extent of shell.
    # =============================================================================


    # If there is a collapse event, ODEpar['Rsh_max'] could be smaller than r. 
    # this is because ODEpar['Rsh_max'] is only updated after the completion of event.
    # straightforward solution: take max(ODEpar['Rsh_max'], r) to ensure we are using the right shell radius.
    # ---- verions1 TODO:
    # mShell, mShell_dot = mass_profile.get_mass_profile(max(params['Rsh_max'].value, R2),
    
    if params['isCollapse'].value == True:
        # stays constant during collapse
        mShell = params['mShell'].value
        mShell_dot = 0
    else:
        mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                       return_mdot = True, rdot_arr = v2)
    
    
    # However this is not so straightforward. During recollapse we have to make sure that the 
    # shell mass stays constant, i.e., Msh_dot = 0.
    # Define recollapse event such that the radius is smaller than the maximum radius from previous step (ODEpar['Rsh_max']).
    # ---- verions1 TODO:
    # simply say that if velocity is negative we are collapsing
    # if v2 < 0:
    #     print('v2, Rsh_max, R2', v2, params['Rsh_max'].value, R2)
    #     mShell_dot = 0  
    # if (R2 < params['Rsh_max'].value):
    #     print('v2, Rsh_max, R2', v2, params['Rsh_max'].value, R2)
    #     mShell_dot = 0  
        
        
        
    params['mShell'].value = mShell
        
    # ----TODO: in the future, uncomment this part and deal with cover fractions.
    # for now, we assume no cover fraction, i.e., cf = 1
    cf = 1
    
    # # ----
    # # TODO future: this section should be used. However, im a lil too scared to change this cause it might break everything
    # # and then shii hits the fan and all hell break loose.
    
    
    # def calc_coveringf(t,tFRAG,ts):
    #     """
    #     estimate covering fraction cf (assume that after fragmentation, during 1 sound crossing time cf goes from 1 to 0)
    #     if the shell covers the whole sphere: cf = 1
    #     if there is no shell: cf = 0
        
    #     Note: I think, since we set tFRAG ultra high, that means cf will almost always be 1. 
    #     """
    #     cfmin = 0.4
    #     # simple slope
    #     cf = 1. - ((t - tFRAG) / ts)**1.
    #     cf[cf>1.0] = 1.0
    #     cf[cf<cfmin] = cfmin
    #     # return
    #     return cf

    # # calculate covering fraction
    # # cf = calc_coveringf(np.array([t.value])*u.s,tFRAG,tSCR)
    
    
    # #----
    # # OLD version
    # # If frag_cf is enabled, what is the final cover fraction at the end
    # # of the fragmentation process?

    # def coverfrac(E,E0,cfe):
    #     if int(os.environ["Coverfrac?"])==1:
    #         if (1-cfe)*(E/E0)+cfe < cfe:    # just to be safe, that 'overshooting' is not happening. 
    #             return cfe
    #         else:
    #             return (1-cfe)*(E/E0)+cfe
    #     else:
    #         return 1
    
    # if cfs == True:
    #     cf = coverfrac(Ebubble,Eb0,warpfield_params.frag_cf_end)
    #     try:
    #         tcf,cfv=np.loadtxt(ODEpar['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODEpar['Mcluster_list']))+".txt", skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
    #         tcf=np.append(tcf, t)
    #         cfv=np.append(cfv, cf)
    #         np.savetxt(ODEpar['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODEpar['Mcluster_list']))+".txt", np.c_[tcf,cfv],delimiter='\t',header='Time'+'\t'+'Coverfraction (=1 for t<Time[0])')
    #     except:
    #         pass
    # elif cfs=='recon':     ##### coverfraction has to be considered again in the reconstruction of the data. 
    #     cf = cf_reconstruct
    # else:
    #     cf=1
        
    # #----
        
        
    # gravity correction (self-gravity and gravity between shell and star cluster)
    # if you don't want gravity, set .inc_grav to zero
    
    # F_grav = (params['G_au'].value * mShell / R2**2 * (params['mCluster_au'].value + mShell/2)  * 1) 
    F_grav = (params['G_au'].value * mShell / R2**2 * (params['mCluster_au'].value + mShell/2)  * warpfield_params.inc_grav) 
  
    # get pressure from energy. 
    if Eb > 1.1 * warpfield_params.phase_Emin.value * cvt.E_cgs2au:
        # calculate radius of inner discontinuity (inner radius of bubble)
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, R2,
                                   args=([L_wind, Eb, v_wind, R2])) 
        # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
        # IDEA: move R1 gradually outwards
        dt_switchon = 1e-3 # in Myr, gradually switch on things during this time period
        tmin = dt_switchon
        # remember t is in year
        if (t > (tmin + tSF)):
            # equation of state PB is in cgs
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1)
        elif (t <= (tmin + tSF)):
            R1_tmp = (t-tSF)/tmin * R1
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp)
    else: # energy is very small: case of pure momentum driving
        import sys
        # sys.exit('here is energy. ')
        R1 = R2 # there is no bubble --> inner bubble radius R1 equals shell radius r
        # ram pressure from winds
        press_bubble = get_bubbleParams.pRam(R2, L_wind, v_wind)
            
        
    # =============================================================================
    # Shell structure
    # =============================================================================
    
    # calculate simplified shell structure (warpfield-internal shell structure, not cloudy)
    # We are setting mBubble = 0 here, since we are not interested in the potential. This can skip some calculations.
    # TODO: right now just lazily add units. future fix also shell_structure
    shell_prop = shell_structure.shell_structure(R2 * u.pc, 
                                                 press_bubble * (u.M_sun/u.pc/u.Myr**2), 
                                        0 * u.M_sun, 
                                        Ln * (u.M_sun*u.pc**2/u.Myr**3),
                                        Li * (u.M_sun*u.pc**2/u.Myr**3),
                                        Qi / u.Myr,
                                        mShell * u.M_sun,
                                        1,
                                        params,
                                        )
    
    # clarity
    f_absorbed_ion, f_absorbed_neu, f_absorbed, f_ionised_dust, is_fullyIonised,\
       shellThickness, nShellInner, nShell_max, tau_kappa_IR, grav_r, grav_phi, grav_force_m = shell_prop
   

    params['shell_f_absorbed_ion'].value = f_absorbed_ion
    params['shell_f_absorbed_neu'].value = f_absorbed_neu
    params['shell_f_absorbed'].value = f_absorbed
    params['shell_f_ionised_dust'].value = f_ionised_dust
    params['shell_thickness'].value = shellThickness  
    params['shell_nShellInner'].value = nShellInner  
    params['shell_nShell_max'].value = nShell_max  
    params['shell_tau_kappa_IR'].value = tau_kappa_IR
    
    params['shell_grav_r'].value = grav_r
    params['shell_grav_phi'].value = grav_phi
    params['shell_grav_force_m'].value = grav_force_m
    
    
    # units right
    
    # radiation pressure coupled to the shell
    fRad = f_absorbed * Lbol / (params['c_au'].value)
    params['shell_f_rad'].value = fRad

    isLowdense = nShell_max < params['stop_n_diss'].value

    if isLowdense != params['isLowdense'].value:
        # update if it is low dense now
        if isLowdense == True:
            params['t_Lowdense'].value = params['t_now'].value
            params['isLowdense'].value = True
        else:
            # set so that t_lowdense - t_now in check_event() will never be positive (stop_t_diss)
            params['t_Lowdense'].value = 1e30
            params['isLowdense'].value = False    

    
    def get_press_ion(r, ion_dict):
        """
        calculates pressure from photoionized part of cloud at radius r
        :return: pressure of ionized gas outside shell
        """
        # old code: ODE.calc_ionpress()
        
        # n_r: total number density of particles (H+, He++, electrons)
        try:
            r = np.array([r])
        except:
            pass
        
        n_r = density_profile.get_density_profile(r, ion_dict)
        
        P_ion = n_r * ion_dict['k_B_au'].value * ion_dict['t_ion'].value
        
        return P_ion
    

    # calculate inward pressure from photoionized gas outside the shell 
    # (is zero if no ionizing radiation escapes the shell)
    if f_absorbed_ion < 1.0:
        press_HII = get_press_ion(R2, params)
    else:
        press_HII = 0.0
        
    # =============================================================================
    # calculate the ODE part: Acceleration 
    # =============================================================================
    vd = (cf * 4 * np.pi * R2**2 * (press_bubble - press_HII)\
            - mShell_dot * v2\
                - F_grav + cf * fRad) / mShell
    
    print('vd calculations')
    print('press_bubble', press_bubble)
    print('press_HII', press_HII)
    print('F_grav', F_grav)
    print('fRad', fRad)
    print('mShell', mShell)
    print('v2', v2)
    print('mShell_dot', mShell_dot)
    print('mShell_dot * v2', mShell_dot * v2)
    
    # force calculation
    params['F_grav'].value = F_grav
    params['F_rad'].value = fRad
    params['F_ram'].value = 4 * np.pi * R2**2 * (press_bubble - press_HII)
    # params['F_wind'].value = fRad
    # params['F_SN'].value = fRad
    
    
    # honestly not sure why this dictionary is here. Its a relic of the previous
    # code, and there is no reason why we need yet another dictionary structure.
    # TODO: add or merge this into the main dictionary.

    # TODO: update these dictionary names in the future, and include them in dictionary.
    # return vd (main result of this function), and additional parameters for recording the evolution.
    evolution_data = {'Msh': mShell, 'fabs_i': f_absorbed_ion, 'fabs_n': f_absorbed_neu,
              'fabs': f_absorbed, 'Pb': press_bubble, 'R1': R1, 'n0':nShellInner, 'nmax':nShell_max}
    
    # update!! 
    # TODO
    # params['R1'].value = R1
    

    return vd, evolution_data
