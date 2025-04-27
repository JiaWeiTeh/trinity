#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 23:16:58 2022

@author: Jia Wei Teh

"""
# libraries
import numpy as np
import astropy.units as u
import astropy.constants as c
import scipy.interpolate
import sys
import scipy.optimize
import os
#--
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.shell_structure.shell_structure as shell_structure
import src.cloud_properties.mass_profile as mass_profile
import src.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
from src._output import terminal_prints
from src.cooling.non_CIE import read_cloudy
import src.bubble_structure.bubble_luminosity as bubble_luminosity
import src.cooling.CIE.read_coolingcurve as CIE
import src.cooling.non_CIE.read_cloudy as non_CIE
import src._functions.operations as operations
from src._input.dictionary import updateDict
import src._functions.unit_conversions as cvt

import os
import importlib
trinity_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

def run_energy( main_dict
        # t0, y0, #r0, v0, E0, T0
        #        ODEpar,
        #         tcoll, coll_counter,
        #         shell_dissolved, t_shelldiss,
        #         stellar_outputs, # old code: SB99_data
        #         SB99f, # interpolation functions from starburst99
        #         # TODO: change tfinal to depend on warpfield_param
        #         # not only tfinal, but all others too.
        #         tfinal = 50 * u.Myr,
        #         Tarr = [], Larr = [], 
        
    # Note:
    # old code: Weaver_phase()
        
        
      ):
    """In this function, rCloud and mCloud is called assuming au units."""
    
    # TODO: remember double check with old files to make sure
    # that the cloudy business are taken care of. This is becaus
    # in the original file, write_cloudy is set to False. 
    # But we have to be prepared for if people wanted to check out
    # write_cloudy = True.
    
    # the energy-driven phase
    # winds hit the shell --> reverse shock --> thermalization
    # shell is driven mostly by the high thermal pressure by the shocked ISM, 
    # also (though weaker) by the radiation pressure, at late times also SNe

    # -----------
    # Describing the free-expanding phase
    # We consider first region (c) of swept-up interstellar
    # gas, whose outer boundary, at R2, is a shock separating
    # it from the ambient interstellar gas (d), and whose
    # inner boundary, at Rc, is a contact discontinuity
    # separating it from the shocked stellar wind (b). The
    # structure of this region can be described by a similarity
    # solution (Avedisova 1972). Our calculation parallels
    # the theory of the adiabatic blast wave given by Taylor
    # (1950); the only substantive difference in the case at
    # hand is that the energy is fed into the system at a
    # constant rate instead of in an initial blast.
    # -----------

    # get cooling cube
    # _timer.begin('heating data')
    # TODO: remember to use this!!!
    # cooling_data, heating_data = read_cloudy.get_coolingStructure(t0.to(u.yr).value)
    # _timer.end()


    # extract dictionary infos
    t_now = main_dict['t_now'].value
    R2 = main_dict['R2'].value
    v2 = main_dict['v2'].value
    Eb = main_dict['Eb'].value
    T0 = main_dict['T0'].value
    
    rCloud = main_dict['rCloud_au'].value
    mCloud = main_dict['mCloud_au'].value
    t_neu = main_dict['t_neu'].value
    t_ion = main_dict['t_ion'].value
    
    
    SB99f = main_dict['SB99f'].value
    SB99_data = main_dict['SB99_data'].value
    
    # print(main_dict)






    # =============================================================================
    # Now, we begin Energy-driven calculations (Phase 1)
    # =============================================================================
    # header
    
    # test only
    # verbosity.test()
    # sys.exit('Demo done.')
    # mypath = warpfield_params.out_dir

    # -----------
    # Step1: Obtain initial values
    # -----------
        
    # # get data from stellar evolution code output
    # # unit of t_evo is Myr, the other units are cgs
    # # See read_SB99.read_SB99 for documentation.
    # t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo = stellar_outputs 
    
    # # Question: isnt this already handelled in main.py in read_SB99?
    # # Answer: yes, but in future there might be another expansion and this needs to be calculated again.
    # # Also, why is this linear instead?
    # # interpolation functions for SB99 values
    # fQi_evo = scipy.interpolate.interp1d(t_evo, Qi_evo, kind = 'cubic')
    # fLi_evo = scipy.interpolate.interp1d(t_evo, Li_evo, kind = 'cubic')
    # fLn_evo = scipy.interpolate.interp1d(t_evo, Ln_evo, kind = 'cubic')
    # fLbol_evo = scipy.interpolate.interp1d(t_evo, Lbol_evo, kind = 'cubic')
    # fLw_evo = scipy.interpolate.interp1d(t_evo, Lw_evo, kind = 'cubic')
    # fpdot_evo = scipy.interpolate.interp1d(t_evo, pdot_evo, kind = 'cubic')


    # mechanical luminosity at time t0 
    L_wind = SB99f['fLw_cgs'](t_now) * cvt.L_cgs2au
    # momentum of stellar winds at time t0
    pdot_wind = SB99f['fpdot_cgs'](t_now) * cvt.pdot_cgs2au
    # velocity from luminosity and change of momentum (au)
    v_wind = (2 * L_wind / pdot_wind) 


    
    # Identify potentially troublesome timestep; i.e., when change in mechanical luminosity is morre than 300% per Myr
    def Lw_slope(x, y):
        dydx = (y[1:] - y[0:-1])/(x[1:] - x[0:-1])
        dydx = np.concatenate([dydx,[0.]])
        return dydx
    
    # the slope
    t_evo = SB99_data[0]
    Lw_evo = SB99_data[5]
    dLwdt = np.abs(Lw_slope(t_evo, Lw_evo))
    # problematic time (which needs small timesteps)  [Myr]
    t_problem = t_evo[ (dLwdt / Lw_evo).value > 3].value
    

    # -----------
    # Solve equation for inner radius of the inner shock.
    # -----------
    
    # print('\n\nvalues to solve r1')
    # print(                       r0,  Lw0, 
    #                                   E0, 
    #                                   vterminal0, 
    #                                   r0)
                                      
    # initial radius of inner discontinuity [pc]
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                       a = 1e-3 * R2, b = R2, 
                       args=([L_wind, Eb, v_wind, R2]))
    
    
    # initial energy derivative
    # Question: why?
    # I suspec this is for the delta_new_root routine, where one compares to another.
    Ebd0 = 0. 
    E0m1 = 0.9*Eb
    t0m1 = 0.9*t_now
    r0m1 = 0.9*R2
    
    
    # declare variables
    # dens_a_pL = warpfield_params.dens_a_pL
    # rCore = warpfield_params.rCore 
    # nCore = warpfield_params.nCore
    # mu_n = warpfield_params.mu_n
    # nAvg = warpfield_params.dens_navg_pL
    # nISM = warpfield_params.nISM
    # t_ion = warpfield_params.t_ion
    # t_neu = warpfield_params.t_neu
    # gamma_adia = warpfield_params.gamma_adia
    # inc_grav = warpfield_params.inc_grav
    # mCluster = warpfield_params.mCluster
    
    # -----------
    # Solve equation for mass and pressure within bubble (r0)
    # -----------
    
    # The initial mass [Msol]
    Msh0 = mass_profile.get_mass_profile(R2, main_dict,
                                         return_mdot = False)[0]
    # The initial pressure [au]
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1)
    
    
    # How long to stay in Weaver phase? Until what radius?
    if trinity_params.density_gradient == True:
        rfinal = np.min([main_dict['rCloud_au'], main_dict['rCore']])
    else:
        rfinal = rCloud
    
    print(f'Inner discontinuity: {R1}.')
    print(f'Initial bubble mass: {Msh0}')
    print(f'Initial bubble pressure: {Pb}')
    print(f'rfinal: {rfinal}')
    print(f'L_wind: {L_wind}')
    print(f'pdot_wind: {pdot_wind}')
    print(f'v_wind: {v_wind}')
    
    updateDict(main_dict, ['R1', 'L_wind', 'Pb', 'pwdot'], [R1, L_wind, Pb, pdot_wind])
    
    # Calculate bubble structure
    # preliminary - to be tested
    # This should be something in bubble_structure.bubble_wrap(), which is being called in phase_solver2. 
    
    # Initialise constants. What is here really do not matter that much - becasuse
    # they will be changed a lot in the loop.
    
    # first stopping time (will be incremented at beginning of while loop)
    # start time t0 will be incremented at end of while loop
    tStop_i = t_now
    
    # Now, we define some start values. In this phase, these three will stay constant throughout.
    alpha = main_dict['alpha'].value # 3/5
    beta = main_dict['beta'].value # 4/5
    delta = main_dict['delta'].value # ~ -0.17
    
    # old: tscr
    # sound crossing time
    t_sound = 1e99 #* u.Myr
    # time takes to fragmentation
    t_frag = 1e99 #* u.Myr
    
    # minimum separation in this phase?
    dt_Emin = 1e-5 #* u.Myr
    # maximum separation in this phase?
    dt_Emax = 0.04166 #* u.Myr
    dt_Estart = 1e-4 #* u.Myr
    
    dt_L = dt_Estart
    
    # this will be the proper dt, timestep
    dt_real  = 1e-4 #* u.Myr
    
    # record the initial Lw0. This value will be changed in the loop. 
    # old code: Lw_old
    Lw_previous = L_wind
    delta_previous = delta
    Lres0 = 1.0 # au
    
    # idk what is this. has to do with fitting?
    fit_len_max = 13
    
    # old code: mom_phase
    immediately_to_momentumphase = False
    
    # factor for calculation of dMdt. See .params file for more explanation.
    dMdt_factor = trinity_params.dMdt_factor
    
    # when to switch on cooling. This should be in parameter file
    dt_switchon = 0.001 #* u.Myr
    
    condition_to_reduce_timestep = False
    
    # according to main.py code, tfinal = t0 + 30. * i.dt_Estart
    
    # Question: isnt this all redundant? Only continueWeaver is relevant?
    
    continueWeaver = True
    # how many times had the main loop being ran?
    # old code: temp_counter
    loop_count = 0
    
    
    # =============================================================================
    # Initialise arrays to record values
    # =============================================================================
    
    # tSweaver, rSweaver, vSweaver, ESweaver
    # time, inner shell radius, shell velocity, shell energy
    weaver_tShell = []; weaver_rShell = []; weaver_vShell = []; weaver_EShell = []; 
    
    # bubble temperature
    weaver_Tbubble = [] 
    
    # shellmass
    weaver_mShell = [] 
    
    # Lbweaver, Lbbweaver, Lbczweaver, Lb3weaver
    weaver_L_total = []; weaver_L_bubble = []; weaver_L_conduction = []; weaver_L_intermediate = []
    
    # fraction of absorbed photons
    weaver_f_absorbed_ion = []; weaver_f_absorbed_neu = []; weaver_f_absorbed = []; weaver_f_ionised_dust = []
    
    # parameters used in ODE functions
    weaver_alpha = []; weaver_beta = []; weaver_delta = []
    
    # fabsweaver = []; fabs_i_weaver = []; fabs_n_weaver = []; ionshweaver = []; Mshell_weaver = []
    # FSgrav_weaver = []; FSwind_weaver = []; FSradp_weaver = []; FSsne_weaver = []; FSIR_weaver = []; dRs_weaver = []; nmax_weaver = []
    # n0weaver = []; n0_cloudyweaver = []; logMcluster_weaver = []; logMcloud_weaver = []; phase_weaver = []; R1weaver = []; Ebweaver = []; Pbweaver = []
    # Lbweaver = []; Lwweaver = []; Tbweaver = []; alphaweaver = []; betaweaver = []; deltaweaver = []
    # fragweaver = [];

    # Lets make this phase at max 16 Myr according to Eq4, Rahner thesis pg44.
    # actually lets make it less than 1e4 yr (sedov taylor cooling time i think)
    # in previous code this is 3e-3 (~3000 yr)
    # Myr
    # tfinal = 1e-2
    tfinal = 3e-3

    while all([R2 < rfinal, (tfinal - t_now) > dt_Emin, continueWeaver]):
        
        
        # =============================================================================
        # Prelude: prepare cooling structures so that it doesnt have to run every loop.
        # Tip: Get cooling structure every 50k years (or 1e5?) or so. 
        # =============================================================================
        
        if np.abs(main_dict['time_last_cooling_update'].value - main_dict['t_now'].value) > 5e-2:
            # recalculate non-CIE
            cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(main_dict['t_now'].value * 1e6)
            # save
            main_dict['cStruc_cooling_nonCIE'].value = cooling_nonCIE
            main_dict['cStruc_heating_nonCIE'].value = heating_nonCIE
            main_dict['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
            # update current value
            main_dict['time_last_cooling_update'].value = main_dict['t_now'].value
        
        
        
        # calculate bubble structure and shell structure?
        # no need to calculate them at very early times, since we say that no bubble or shell is being
        # created at this time. Old code: structure_switch
        calculate_bubble_shell = loop_count > 0
        
        # an initially small value if it is the first loop.
        if loop_count == 0:
            dt_Emin = dt_Estart
            
        
        # =============================================================================
        # identify time step size and end-time value
        # =============================================================================
        
        # in the very beginning (and around first SNe) decrease tCheck because fabs varys a lot
        # increment stopping time
        ii_dLwdt = operations.find_nearest_lower(t_evo.value, t_now)
        if dLwdt[ii_dLwdt] > 0:
            dt_Lw = 0.94 * (0.005 * Lw_previous) / dLwdt[ii_dLwdt]
        else:
            dt_Lw = dt_Emax
        # consistent unit

        
        # check stoptime: it should be minimum between tfinal and tadaptive, 
        # where tadaptive = min (t0 + dt_total or dt_emax or dt_wind)
        # [Myr]
        tStop_i = np.min([tfinal, t_now + np.min([dt_L, dt_Emax, dt_Lw])]) #* u.Myr
        # find nearest problematic neighbor of current stop time
        t_problem_hi = t_problem[t_problem > (t_now + 0.001 * dt_Emin)]
        # the next problematic time
        t_problem_nnhi = t_problem_hi[0]
        # restate [Myr]
        tStop_i = np.min([tStop_i, (t_problem_nnhi - float(fit_len_max + 1) * dt_Emin)]) #* u.Myr
 
        # ensure that time step becomes not too small or negative
        if tStop_i < (t_now + dt_Emin):
            tStop_i = t_now + dt_Emin
            
        # t0 changes at the end of this loop.
        dt_real = tStop_i - t_now    
        
        
        
        print(f'\n\nloop {loop_count}, R2: {R2}.\n\n')
        print(f'conditions: R2 < rfinal: {R2}:{rfinal}')
        print(f'conditions: tfinal - t_now: {tfinal}:{t_now}')
        print(f'dt_Emin: {dt_Emin}')
        print(f'tStop_i: {tStop_i}')
        print(f'dt_real: {dt_real}') 
        
        # t0 will increase bit by bit in this main loop

        # In this while loop, we calculate the bubble structure.
        # before this time step is accepted, check how the mechanical luminosity would change
        # if it would change too much, reduce time step
        while True:
            
            # save a snapshot here
            main_dict.save_snapShot()
            
            # TODO
            # remember to add units!
            
            
            if condition_to_reduce_timestep:
                # continue the while loop, with smaller timestep otherwise, reduce timestep in next iteratin of loop
                # prepare for the next bubble loop
                dt_real = 0.5 * dt_real
                tStop_i = t_now + dt_real
            
            
            # =============================================================================
            # Here, we create the array for timesteps for bubble calculation. 
            # =============================================================================
            
            t_inc = dt_real / 1e3
            # time vector for this time step
            # make sure to include tStop_i+t_inc here, i.e. go one small dt further
            t_temp = np.arange(t_now, tStop_i + t_inc, t_inc) #* u.Myr
            
            # insert a value close to t[-1]
            # I don't know why specifically this value, but it is in the original code. 
            t_smallfinal = t_temp[-1] -  9.9147e-6 #* u.Myr
            # if this value is smaller than the beginning, don't bother adding
            if t_smallfinal > t_temp[0]:
                idx_t_smallfinal = np.searchsorted(t_temp, t_smallfinal)
                # old code: t
                t_arr = np.insert(t_temp,idx_t_smallfinal,t_smallfinal)
            else:
                # initialise index anyway for future use
                idx_t_smallfinal = 0
                t_arr = t_temp
            
            # We then evaluate the feedback parameters at these midpoint timesteps.
            # old code: thalf
            t_midpoint = 0.5 * (t_arr[0] + t_arr[-1])
            # mechanical luminosity at time t_midpoint (erg)
            Lw = SB99f['fLw_cgs'](t_midpoint) * cvt.L_cgs2au 
            Lbol = SB99f['fLbol_cgs'](t_midpoint) * cvt.L_cgs2au 
            Ln = SB99f['fLn_cgs'](t_midpoint) * cvt.L_cgs2au  
            Li = SB99f['fLi_cgs'](t_midpoint) * cvt.L_cgs2au  
            # momentum of stellar winds at time t0 (cgs)
            pdot = SB99f['fpdot_cgs'](t_midpoint) * cvt.pdot_cgs2au 
            # terminal wind velocity at time t0 (pc/Myr)
            v_wind = (2. * Lw / pdot) 
            # ionizing
            Qi = SB99f['fQi_cgs'](t_midpoint) / cvt.s2Myr   
            # old code: Lw_temp, evaluated at tStop_i
            Lw_tStop = SB99f['fLw_cgs'](tStop_i) * cvt.L_cgs2au  
    
            # if mechanical luminosity would change too much, run through this while = True loop again with reduced time step
            # check that the difference with previous loop is not more than 0.5% of the original value.
            # if these conditions are not met, no need to calculate bubble - save computation
            # timestep needed to be reduced.
            condition_to_reduce_timestep1 = abs(Lw_tStop - Lw_previous) < (0.005 * Lw_previous)
            condition_to_reduce_timestep2 = abs(Lw - Lw_previous) < (0.005 * Lw_previous)
            condition_to_reduce_timestep3 = (dt_real < 2 * dt_Emin)
            
            # print(condition_to_reduce_timestep1, condition_to_reduce_timestep2, condition_to_reduce_timestep3)
            # sys.exit()
            
            # combined
            condition_to_reduce_timestep = (condition_to_reduce_timestep1 and condition_to_reduce_timestep2) or condition_to_reduce_timestep3
            
            # update
            updateDict(main_dict,
                       ['R2', 'Qi', 'v_wind', 'Eb', 'alpha', 'beta', 'delta', 'L_wind', 'v_wind', 'Ln', 'Li', 'Qi'],
                       [R2, Qi, v_wind, Eb, alpha, beta, delta, Lw, v_wind, Ln, Li, Qi])
            
            updateDict(main_dict, ['R1', 'L_wind', 'Pb', 'pwdot', 'mShell'], [R1, L_wind, Pb, pdot_wind, Msh0])
            
            # TODO: make all dictionary extraction at the beginning, so taht we know
            # which values need to be updated.
            
            if condition_to_reduce_timestep:
                # TODO: remove after debug
            # if True:
                # should we calculate the bubble structure?
                
                # =============================================================================
                # Calculate bubble structure
                # =============================================================================
                
                if calculate_bubble_shell:
                # if True:
                    
                    print('\nCalculate bubble and shell\n')
                    
                    # output = bubble_luminosity.get_bubbleproperties(t0 - tcoll[coll_counter],
                    #                                                 # T_goal, rgoal,
                    #                                                 # R2 is r0 in old code
                    #                                                 r0,
                    #                                                 Qi, alpha, beta, delta,
                    #                                                 Lw, E0, vterminal,
                    #                                                 dMdt_factor
                    #                                                 )
                    output = bubble_luminosity.get_bubbleproperties(main_dict
                                                                    )
                    # restate just for clarity
                    # T_rgoal here is T0 in original code.
                    # here, dMdt_factor is also being updated.
                    # L_total, T0, L_bubble, L_conduction, L_intermediate, Tavg, mBubble = output
                    
                    # update this here instead of in bubble_luminosity so that 
                    # T0 will not be overwrite when we are dealing with phase1b.
                    main_dict['T0'].value = main_dict['bubble_T_rgoal'].value

                    T0 = main_dict['bubble_T_rgoal'].value
                    L_total = main_dict['bubble_L_total'].value
                    L_conduction = main_dict['bubble_L_conduction'].value
                    L_intermediate = main_dict['bubble_L_intermediate'].value
                    L_bubble = main_dict['bubble_L_bubble'].value
                    Tavg = main_dict['bubble_Tavg'].value
                    mBubble = main_dict['bubble_mBubble'].value
                    
                    print('\n\nFinish bubble\n\n')
                    print('L_total', main_dict['bubble_L_total'].value)
                    print('T_rgoal', main_dict['bubble_T_rgoal'].value)
                    print('L_bubble', main_dict['bubble_L_bubble'].value)
                    print('L_conduction', main_dict['bubble_L_conduction'].value)
                    print('L_intermediate', main_dict['bubble_L_intermediate'].value)
                    print('bubble_Tavg', main_dict['bubble_Tavg'].value)
                    print('bubble_mBubble', main_dict['bubble_mBubble'].value)
                    # sys.exit('round2')
                        
                elif not calculate_bubble_shell:
                    L_total = 0
                    L_bubble =  0 
                    L_conduction = 0 
                    L_intermediate = 0 
                    dMdt_factor = 1.646 # as in classical Weaver
                    Tavg = T0
                    
                    mBubble = np.nan
                    r_Phi = np.nan
                    Phi_grav_r0b = np.nan
                    f_grav = np.nan
                    dt_L = dt_L
                
                # average sound speed in bubble
                cs_avg = operations.get_soundspeed(Tavg, main_dict)

        
            # leave the while = True loop, if:   1) the bubble is not calculated, OR
            #                                    2) if it is SUCCESSFULLY calculated AND delta did not change by large margin, OR
            #                                    3) time step is already close to the lower limit
            if not calculate_bubble_shell: 
                break
            if abs(delta - delta_previous) < 0.03:
                break
            if dt_real < 2 * dt_Emin:
                break

            condition_to_reduce_timestep = True
            # put this here to signify ending of loop
            pass

    
        # gradually switch on cooling (i.e. reduce the amount of cooling at very early times)
        # TODO: add this later for recollapse
        # if (t0-tcoll[coll_counter] < dt_switchon):
        if (t_now < dt_switchon):
            reduce_factor = np.max([0.,np.min([1.,t_now/dt_switchon])])
            L_total *= reduce_factor # scale and make sure it is between 0 and 1
            L_bubble *= reduce_factor
            L_conduction *= reduce_factor
            L_intermediate *= reduce_factor
        
        
        # =============================================================================
        # Calculate shell structure
        # =============================================================================
        
        # TODO: Where is Mbubble in previous code? Why does it only appear once in else case?
        
        
        if calculate_bubble_shell:
            
            
            print('\n\nhere calculate_bubble_shell\n\n')
            
            
            # print(r0)
            # print(P0)
            # print(Mbubble)
            # print(Ln, Li, Qi,)
            # print(Msh0)
            
            # print('debugging here')
            # r0 = 0.24900205367057132 * u.pc
            # P0 = 4.329788040892236e-06 * u.g / (u.cm * u.s**2)
            # Mbubble = np.nan * u.M_sun
            # Ln = 1.5150154294119439e41 * u.erg / u.s 
            # Li = 1.9364219639465926e+41 *  u.erg / u.s 
            # Qi = 5.395106225151268e+51 / u.s
            # Msh0 = 20.341185347035363 * u.M_sun

            # shell_prop = shell_structure.shell_structure(R2 * u.pc, 
            #                                             Pb * (u.M_sun/u.pc/u.Myr**2), 
            #                                             mBubble * u.M_sun, 
            #                                             Ln * (u.M_sun*u.pc**2/u.Myr**3),
            #                                             Li * (u.M_sun*u.pc**2/u.Myr**3),
            #                                             Qi / u.Myr,
            #                                             Msh0 * u.M_sun,
            #                                             1,
            #                                             main_dict,
            #                                             )
            
            shell_prop = shell_structure.shell_structure(
                                                        main_dict,
                                                        # R2, 
                                                        # Pb, 
                                                        # mBubble, 
                                                        # Ln * (u.M_sun*u.pc**2/u.Myr**3),
                                                        # Li * (u.M_sun*u.pc**2/u.Myr**3),
                                                        # Qi / u.Myr,
                                                        # Msh0 * u.M_sun,
                                                        # 1,
                                                        # main_dict,
                                                        )
            
            f_absorbed_ion, f_absorbed_neu, f_absorbed,\
                f_ionised_dust, is_fullyIonised, shell_thickness,\
                    nShellInner, nShell_max, tau_kappa_IR, grav_r, grav_phi, grav_force_m = shell_prop


        elif not calculate_bubble_shell:
            # TODO: redefine these values so that they are more physically similar to the environments
            f_absorbed_ion = 0.0
            f_absorbed_neu = 0.0
            f_absorbed = 0.0
            f_ionised_dust = 0
            is_fullyIonised = False
            shell_thickness = 0.0
            nShellInner = 0
            nShell_max = 1e5  
            tau_kappa_IR = 0
            grav_r = 0 
            grav_phi = 0 
            grav_force_m = 0
            
        # update
        
        main_dict['shell_f_absorbed_ion'].value = f_absorbed_ion
        main_dict['shell_f_absorbed_neu'].value = f_absorbed_neu
        main_dict['shell_f_absorbed'].value = f_absorbed
        main_dict['shell_f_ionised_dust'].value = f_ionised_dust
        main_dict['shell_thickness'].value = shell_thickness 
        main_dict['shell_nShellInner'].value = nShellInner
        main_dict['shell_nShell_max'].value = nShell_max
        main_dict['shell_tau_kappa_IR'].value = tau_kappa_IR
        
        # TODO: here requires unit conversion becsuse these are in cgs i th
        main_dict['shell_grav_r'].value = grav_r 
        main_dict['shell_grav_phi'].value = grav_phi
        main_dict['shell_grav_force_m'].value = grav_force_m
        main_dict['shell_f_rad'].value = f_absorbed_ion * Lbol / main_dict['c_au'].value

        # =============================================================================
        # call ODE solver to solve for equation of motion (r, v (rdot), Eb). 
        # =============================================================================
        # radiation pressure coupled to the shell
        # f_absorbed_ion calculated from shell_structure.
        # F_rad = f_absorbed_ion * Lbol / main_dict['c_au'].value
        
        # params = [Lw, pdot, mCloud, rCore, mCluster, L_total, F_rad, f_absorbed_ion, rCloud, tcoll, t_frag, t_sound, cs_avg,
        #           dens_a_pL, nAvg, nISM, nCore, t_ion,
        #           gamma_adia, inc_grav, mu_n]
        
        
        # y0 = [r0.to(u.cm).value, v0.to(u.cm/u.s).value, E0.to(u.erg).value]
        y0 = [R2, v2, Eb]
        
        # print('y0', y0)
        # print('t_arr', t_arr)
        
        
        # update these
        # sbparams
        # L_wind
        # pdot
        # R2
        
        
        
        # call ODE solver
        # remember that the output is in cgs
        psoln = scipy.integrate.odeint(energy_phase_ODEs.get_ODE_Edot, y0, t_arr, args=(main_dict,))
        
        
        # [pc]
        r_arr = psoln[:,0] 
        # rd in old code.
        
        v_arr = psoln[:, 1]
        # [au]
        Eb_arr = psoln[:, 2] 

        # print('\n\nhere are the results for the ODE\n\n')
        # print(r_arr)
        # print(v_arr)
        # print(Eb_arr)
        # sys.exit()

        # =============================================================================
        # calculate mass
        # =============================================================================

        # get shell mass
        mShell_arr = mass_profile.get_mass_profile(r_arr, main_dict,
                                                    return_mdot = False)
        
        # print(mShell_arr)
    
        # print(mShell_arr)
        # sys.exit()
        
        
            
            
        # =============================================================================
        # Here, we perform checks to see if we should continue the branch (i.e., increasing steps)
        # =============================================================================
        # TODO: 
        #----------------------------
        # 1. Stop calculating when radius threshold is reached
        #----------------------------
        # TODO: old code contains adabaticOnlyCore
        # TODO: 
        # if r_arr[-1].to(u.pc).value > rCloud.to(u.pc).value:
        #     # if r values reach beyond the cloud radius, mask away, and stop the branch.
        #     continue_branch = False
        #     mask = r_arr.to(u.pc).value < rCloud.to(u.pc).value
        #     t_arr = t_arr[mask]
        #     r_arr = r_arr[mask]
        #     v_arr = v_arr[mask]
        #     Eb_arr = Eb_arr[mask]
        #     mShell_arr = mShell_arr[mask]
            
        # else:
        #     print('\n\nhere in else\n\n')
        #     print(warpfield_params.rCore)
        #     print(r_arr.to(u.pc).value)
        #     print(r_arr[r_arr.to(u.pc).value >= warpfield_params.rCore.to(u.pc).value][0])
        #     # if r values are still within the cloud, 
        #     continue_branch = True
        #     mask = r_arr.to(u.pc).value <= r_arr[r_arr.to(u.pc).value >= warpfield_params.rCore.to(u.pc).value][0]
        #     t_arr = t_arr[mask]
        #     r_arr = r_arr[mask]
        #     v_arr = v_arr[mask]
        #     Eb_arr = Eb_arr[mask]
        #     mShell_arr = mShell_arr[mask]
            
        #----------------------------
        # 2. When does fragmentation occur?
        #----------------------------
        # -----------
        # Option1 : Gravitational instability
        # -----------
        # which temperature?
        if f_absorbed_ion < 0.99:
            T_shell = t_neu
        else:
            T_shell = t_ion
        # sound speed
        c_sound = operations.get_soundspeed(T_shell, main_dict)
        main_dict['cs_avg'].value = c_sound
        
        # TODO: add this in the future.
        # Unstable
        # if warpfield_params.frag_enabled:
        #     # fragmentation value array, due to gravitational instabiliey.
        #     frag_arr = (warpfield_params.frag_grav_coeff * c.G * 3 / 4 * mShell_arr / (4 * np.pi * v_arr * c_sound * r_arr)).decompose()
        #     # fragmentation occurs when this value is greater than 1.
        #     frag_value_threshold = frag_arr[-1]
            
        #     # frag_value_final can jump from positive to negative (if velocity becomes negative) if time resolution is too coarse
        #     # Thus at v=0, fragmentation always occurs
        #     if frag_value_threshold <= 0:
        #         frag_value_threshold = 2
                
        #     # check if this time it is closer to fragmentation, if it wasn't already.
        #     # if not close_to_frag:
        #     #     frag_threshold = 1 = float()
            
            
            
            # TODO
            # -----------
            # Option2 : Rayleigh-Taylor isntability (not yet implemented)
            # -----------    
            
            
            # TODO: Below is not implemented yet. Specifically, I think the cover fraction
            # will always be one, since tfrag is very large. The slope will always end up at 1. I think.
            # OPTION 2 for switching to mom-driving: if i.immediate_leak is set to False, when covering fraction drops below 50%, switch to momentum driving
    
        # ---- this seems to be redundant now, since we have dictionary.flush().
        # # record values
        # # Idea: for each t0, record also all r or n values in r0. Need shell/bubble to return them
        # # so we can make movies.
        # weaver_tShell = np.concatenate([weaver_tShell, [t_now]])
        # weaver_rShell = np.concatenate([weaver_rShell, [R2]])
        # weaver_vShell = np.concatenate([weaver_vShell, [v2]])
        # weaver_EShell = np.concatenate([weaver_EShell, [Eb]])
        # # bubble temperature
        # weaver_Tbubble = np.concatenate([weaver_Tbubble, [T0]])
        # # mass
        # weaver_mShell = np.concatenate([weaver_mShell, [Msh0]])
        # # luminosity properties
        # weaver_L_total = np.concatenate([weaver_L_total, [L_total]])
        # weaver_L_bubble = np.concatenate([weaver_L_bubble, [L_bubble]])
        # weaver_L_conduction = np.concatenate([weaver_L_conduction, [L_conduction]])
        # weaver_L_intermediate = np.concatenate([weaver_L_intermediate, [L_intermediate]])
        # # absorbed photons 
        # weaver_f_absorbed_ion = np.concatenate([weaver_f_absorbed_ion, [f_absorbed_ion]])
        # weaver_f_absorbed_neu = np.concatenate([weaver_f_absorbed_neu, [f_absorbed_neu]])
        # weaver_f_absorbed = np.concatenate([weaver_f_absorbed, [f_absorbed]])
        # # ODE parameters
        # weaver_alpha = np.concatenate([weaver_alpha, [alpha]])
        # weaver_beta = np.concatenate([weaver_beta, [beta]])
        # weaver_delta = np.concatenate([weaver_delta, [delta]])
    
        # # remember to change names once this is done. Names are quite misleading and i would love to change them.
        # # This is doable because they are referred not many times. 
        
        # # TODO: remember to also update the script in write_outputs.py, which saves data into a fits file.
        # weaver_data = {'t':weaver_tShell, 'r':weaver_rShell, 'v':weaver_vShell, 'E':weaver_EShell, 
        #           'logMshell':np.log10(weaver_mShell),
        #           'Tb': weaver_Tbubble,
        #           # this was dMdt_factor_end
        #           'dMdt_factor': dMdt_factor,
        #           'alpha': weaver_alpha, 'beta': weaver_beta, 'delta': weaver_delta, 
        #           # I think these are not important for the code, but still worth tracking.
        #           'fabs':weaver_f_absorbed, 'fabs_n': weaver_f_absorbed_neu, 'fabs_i':weaver_f_absorbed_ion,
        #           'Lcool':weaver_L_total, 'Lbb':weaver_L_bubble, 'Lbcz':weaver_L_conduction, 'Lb3':weaver_L_intermediate,
        #           }
          
        #   # 't_end': weaver_tShell[-1], 'r_end':weaver_rShell[-1], 'v_end':weaver_vShell[-1], 'E_end':weaver_EShell[-1],
        #   # 'Fgrav':FSgrav_weaver, 'Fwind':FSwind_weaver, 'Fradp_dir':FSradp_weaver, 'FSN':FSsne_weaver, 'Fradp_IR':FSIR_weaver,
        #   # 'dRs':dRs_weaver, 'logMshell':np.log10(Mshell_weaver), 'nmax':nmax_weaver, 'logMcluster':logMcluster_weaver, 'logMcloud':logMcloud_weaver,
        #   # 'phase': phase_weaver, 'R1':R1weaver, 'Eb':Ebweaver, 'Pb':Pbweaver, 'Lmech':Lwweaver, 'Lcool':Lbweaver, 'Tb':Tbweaver,
        #   # 'alpha':alphaweaver, 'beta':betaweaver,'delta':deltaweaver, 'Lbb':Lbbweaver, 'Lbcz':Lbczweaver, 'Lb3':Lb3weaver, 'frag':fragweaver, 'dMdt_factor_end': dMdt_factor}


    
        # =============================================================================
        # Prepare for next loop
        # =============================================================================

        # new initial values
        # time
        t_now = t_arr[-1]
        # shell radius
        R2 = r_arr[-1]
        # shell velocity
        v2 = v_arr[-1]
        # bubble energy
        Eb = Eb_arr[-1]
        # wind velocity = 2 * wind Luminosity / pdot
        v_wind = (2 * (SB99f['fLw_cgs'](t_now) * cvt.L_cgs2au) / (SB99f['fpdot_cgs'](t_now) * cvt.pdot_cgs2au))
        
        # if we are going to the momentum phase next, do not have to 
        # calculate the discontinuity for the next loop
        if immediately_to_momentumphase:
            R1 = R2 # why?
            # bubble pressure
            Pb = get_bubbleParams.pRam(R2, L_wind, v_wind)
        # else, if we are continuing this loop and staying in energy
        else:
            R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                           1e-3 * R2, R2, 
                           args=([L_wind, Eb, v_wind, R2]))
            # bubble pressure
            Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1)
            
            
        Msh0 = mShell_arr[-1] # shell mass
        
        
        # renew constants
        Lres0 = Lw - L_total
        Lw_previous = Lw
        # delta_previous = delta
        
        
        # if warpfield_params.frag_enabled:
        #     pass
            # dfragdt = (frag_value_threshold - frag_value0)/(t_arr[-1]-t_arr[0])
            # frag_value0 = frag_value_threshold
        
        
        updateDict(main_dict, 
                    ['R1', 'R2', 'v2', 'Eb', 'v_wind', 't_now', 'Pb', 'mShell'], 
                    [R1, R2, v2, Eb, v_wind, t_now, Pb, Msh0])
        
        
        # update loop counter
        loop_count += 1
        
        # verbosity.print_parameter(weaver_data)
    
        # sys.exit('one loop done')
    
        # break
        pass


    # TODO: debug.

    # Problem also: why is time not increasing, and only ever so slightly until 0.002Myr?
# solution: line 200!! Need to be incremented!?


    # Problem that seems to behappening: the velocity drops way too quickly. 
    # it goes from 3656 to 400, then quickly to 150. 
    # Also, the M was negative at first. The radius then is stuck at 0.249pc. Why?


    # Another problem: now it seems that the dMdt factor drops from 2k Msol/Myr to 200 Msol/Myr, then 
    # suddenly turn to -200 Msol/Myr. What could be the problem?
    # This seems to have to do with the ODE solver which switches the guesses automatically.
    # Perhaps the values of other parameters caused this? i.e., why did dMdt drop to 200 in the first place?

    return 
    # return weaver_data
    
    
    # here is testing region.
    #------





#%%









#     alpha = 0.6
#     beta = 0.8
#     delta = -0.17142857142857143
#     Eb = 94346.55799234606 * u.M_sun * u.pc**2 / u.Myr**2
#     # Why is this 0.2?
#     # change back? in old code R2 = r0.
#     # R2 = 90.0207551764992493 * u.pc
#     R2 = 0.07083553197734 * u.pc
#     # R2 = r0
#     # print('r0', r0)
#     # 
#     t_now = 0.00010205763664239359 * u.Myr
#     Lw =  2016488677.477017 * u.M_sun * u.pc**2 / u.Myr**3
#     vw = 3810.21965323859 * u.km / u.s
#     dMdt_factor = 1.646
#     Qi = 1.6994584609226495e+65 / u.Myr
#     v0 = 0.0 * u.km / u.s 
#     T_goal = 3e4 * u.K
#     # r_inner = R1
#     r_inner = 0.04032342117274968 * u.pc
#     rgoal = 0.06375197877960599 * u.pc

#     # L_total, T_rgoal, L_bubble, L_conduction, L_intermediate, dMdt_factor_out, Tavg = bubble_luminosity.get_bubbleproperties(t_now, T_goal, rgoal,
#     #                                                                                      r_inner, R2,
#     #                                                                                      Qi, alpha, beta, delta,
#     #                                                                                      Lw, Eb, vw, v0,
#     #                                                                                      )
    
    
    
    
    
    
#     # Then, turn on cooling gradually. I.e., reduce the amount of cooling at very early times. 
    
#     r0 = 0.07083553197734 * u.pc
#     P0 = 51802552.6048532 * u.M_sun / u.pc / u.Myr**2
#     Ln = 1.5150154294119439e41 * u.erg / u.s
#     Li = 1.9364219639465926e+41 * u.erg / u.s
#     Qi = 5.395106225151268e+51 / u.s
#     Msh0 = 0.46740340439142747 * u.M_sun
#     Mbubble = 2.641734919254874e+32  * u.g
#     countbubble = 1
#     thalf = 0.00015205763664239482 * u.Myr
#     f_cover = 1

#     # header
#     terminal_prints.shell()
    
    
#     # Calculate shell structure.
#     # preliminary - to be tested
#     shell_prop = shell_structure.shell_structure(r0, 
#                                                 P0,
#                                                 Mbubble, 
#                                                 Ln, Li, Qi,
#                                                 Msh0,
#                                                 f_cover,
#                                                 )
    
    
#     # return f_absorbed_ion, f_absorbed_neu, f_absorbed, f_ionised_dust, is_fullyIonised, shellThickness, nShell_max, tau_kappa_IR, grav_r, grav_phi, grav_force_m
    
#     print('Shell structure calculated.')
    
    
#    # shell.shell_structure2(r0, P0, fLn_evo(thalf), fLi_evo(thalf),
#                # fQi_evo(thalf), Msh0, 1, ploton = make_cloudy_model, 
#                # plotpath = filename_shell, Minterior = Mbubble*c.Msun)


    
#     # Calculate bubble mass
#     # bubbble_mass = mass_profile.calc_mass()
    
    
    
    
#     # Get new values for next loop.
    
#     # These will be used. For now, we comment to debug.
#     # now solve eq. of motion
#     # this part of the routine assumes astro-units

#     # PWDOT = pdot
#     # GAM = c.gamma

#     # RCORE = rcore_au
#     # MSTAR = Mcluster_au
#     # LB = Lb # astro units!!
#     # FRAD = fabs * Lbol/c.clight_au # astro units
#     # CS = cs_avg



#     # y0 = [r0, v0, E0]

#     # bundle parameters for ODE solver
#     # params = [Lw, PWDOT, Mcloud_au, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, fabs_i, rcloud_au, phase0, tcoll[coll_counter], t_frag, tscr, CS, SFE]
#     # print('\n\n\n')
#     # print("y0", y0)
#     # print('\n\n\n')
#     # aux.printl(("params", params), verbose=1)
#     # print('\n\n\n')
#     # print('t',t)


#     r0 = (0.07083553197734 * u.pc)
#     v0 = (412.7226637916362 * u.km / u.s)
#     E0 = (94346.55799234606 * u.M_sun * u.pc**2 / u.Myr**2)
#     y0 = [r0.to(u.cm).value, v0.to(u.cm/u.s).value, E0.to(u.erg).value]

#     Lw = 2016488677.477017 * u.M_sun * u.pc**2 / u.Myr**3
#     PWDOT = 1058463.2178688452 * u.M_sun * u.pc / u.Myr**2 
#     GAM = 1.6666666666666667
#     Mcloud_au = 9900000.0 * u.M_sun
#     RHOA = 313.94226159698525 * u.M_sun / u.pc**2
#     RCORE = 0.099 * u.pc
#     A_EXP = 0
#     MSTAR = 100000 * u.M_sun
#     LB = 31084257.266749237 * u.M_sun * u.pc**2 / u.Myr**3
#     FRAD = 1431661.1440950811 * u.M_sun * u.pc / u.Myr**2
#     fabs_i = 0.8540091033365051
#     f_absorbed_ion = 0.8540091033365051
#     rcloud_au = 19.59892574924841 * u.pc
#     phase0 = 1
#     tcoll = 0
#     t_frag = 1e99 * u.yr
#     tscr = 1e99 * u.yr
#     # I think this is pc/Myr
#     CS = 968.3051163156159 
#     SFE = 0.01
#     t = (np.arange(0.00010206, 0.00020206, 0.0000001) * u.Myr).to(u.s) # think this is in Myr
    
#     params = [Lw, PWDOT, Mcloud_au, RCORE, MSTAR, LB, FRAD, fabs_i, rcloud_au, tcoll, t_frag, tscr, CS]

#     # call ODE solver
#     psoln = scipy.integrate.odeint(energy_phase_ODEs.get_ODE_Edot, y0, t.value, args=(params,))
        
#     # get r, rdot and Edot
#     r_arr = psoln[:,0] * u.cm
#     r_dot_arr = psoln[:, 1] * u.cm/u.s
#     Ebubble_arr = psoln[:, 2] * u.erg
    
#     # get shell mass
#     mShell_arr = mass_profile.get_mass_profile(r_arr, rCloud, mCloud, return_mdot = False)
    
    # # =============================================================================
    # # Here, we perform checks to see if we should continue the branch (i.e., increasing steps)
    # # =============================================================================
    
    # #----------------------------
    # # 1. Stop calculating when radius threshold is reached
    # #----------------------------
    # # TODO: old code contains adabaticOnlyCore
    # # TODO: 
    # if r_arr[-1].value > rCloud.value:
    #     continue_branch = False
    #     mask = r_arr.value < rCloud.value
    #     t = t[mask]
    #     r_arr = r_arr[mask]
    #     r_dot_arr = r_dot_arr[mask]
    #     Ebubble_arr = Ebubble_arr[mask]
    #     mShell_arr = mShell_arr[mask]
        
    # else:
    #     continue_branch = True
    #     mask = r_arr.value < r_arr[r_arr.value >= warpfield_params.rCore.to(u.pc).value][0]
    #     t = t[mask]
    #     r_arr = r_arr[mask]
    #     r_dot_arr = r_dot_arr[mask]
    #     Ebubble_arr = Ebubble_arr[mask]
    #     mShell_arr = mShell_arr[mask]
        
        
    # # THese are TBD
    # # #----------------------------
    # # # 2. When does fragmentation occur?
    # # #----------------------------
    # # # -----------
    # # # Option1 : Gravitational instability
    # # # -----------
    # # # which temperature?
    # # if f_absorbed_ion < 0.99:
    # #     T_shell = warpfield_params.t_neu
    # # else:
    # #     T_shell = warpfield_params.t_ion
    # # # sound speed
    # # c_sound = operations.get_soundspeed(T_shell).to(u.km/u.s)
    # # # fragmentation value array, due to gravitational instabiliey.
    # # frag_arr = (warpfield_params.frag_grav_coeff * c.G * 3 / 4 * mShell_arr / (4 * np.pi * r_dot_arr * c_sound * r_arr)).decompose()
    # # # fragmentation occurs when this value is greater than 1.
    # # frag_value_final = frag_arr[-1]
    
    # # # frag_value_final can jump from positive to negative (if velocity becomes negative) if time resolution is too coarse
    # # # Thus at v=0, fragmentation always occurs
    # # if frag_value_final <= 0:
    # #     frag_value_final = 2
        
    # # # check if this time it is closer to fragmentation, if it wasn't already.
    # # # if not close_to_frag:
    # # #     frag_threshold = 1 = float()
    
    
    
    # # TODO
    # # -----------
    # # Option2 : Rayleigh-Taylor isntability (not yet implemented)
    # # -----------    
    
    
    # #     # Now, we define some start values
    # # alpha = 0.6 # 3/5
    # # beta = 0.8 # 4/5
    
    
    # # # taking more negative value seems to be better for massive clusters (start further out of equilibrium?)
    # # if fLw_evo(t0) < 1e40: 
    # #     delta = -6./35. 
    # #     # -6./35.=-0.171 
    # # else: delta = -6./35. # -0.5
    # # # TOASK: Double confirm all these parameters
    # # temp_counter = 0
    # # dt_L = 1e-4
    # # frag_value = 0.
    # # t_frag = 1e99
    # # tscr = 1e99
    # # dMdt_factor = 1.646
    # # mom_phase = False
    # # first_frag = True
    # # Lw_old = Lw0
    # # delta_old = delta
    # # cf0 = 1.0
    # # frag_value0 = 0.
    # # dfragdt = 0.
    # # was_close_to_frag = False
    # # frag_stop = 1.0
    # # fit_len = 13
    # # dt_real = 1e-4
    # # Lres0 = 1.0
    
    
    # # dt_Emax = 0.04166
    # # dt_Emin = 1e-5
    # # dt_Estart = 1e-4
    
    # # fit_len_max = 13
    # # fit_len_min = 7
    # # lum_error = 0.005
    # # lum_error2 = 0.005
    # # delta_error = 0.03
    # # dt_switchon = 0.001
    
    
    
    
    
    
    # # TODO: Below is not implemented yet. Specifically, I think the cover fraction
    # # will always be one, since tfrag is very large. The slope will always end up at 1. I think.
    # # # OPTION 2 for switching to mom-driving: if i.immediate_leak is set to False, when covering fraction drops below 50%, switch to momentum driving

    
    # sys.exit('check done!')
    
    # return
    
 










# #%%





#         # # check whether shell fragments or similar

#         # # if a certain radius has been exceeded, stop branch
#         # if (r[-1] > rfinal or (r0 < rcore_au and r[-1] >= rcore_au and i.density_gradient)):
#         #     if r[-1] > rfinal:
#         #         continue_branch = False
#         #         include_list = r<rfinal
#         #     else:
#         #         continue_branch = True
#         #         rtmp = r[r>=rcore_au][0] # find first radius larger than core radius
#         #         include_list = r <= rtmp
#         #     t = t[include_list]
#         #     r = r[include_list]
#         #     rd = rd[include_list]
#         #     Eb = Eb[include_list]
#         #     Msh = Msh[include_list]


#         # # calculate fragmentation time
#         # if fabs_i < 0.999:
#         #     Tsh = i.Tn  # 100 K or so
#         # else:
#         #     Tsh = i.Ti  # 1e4 K or so
#         # cs = aux.sound_speed(Tsh, unit="kms")  # sound speed in shell (if a part is neutral, take the lower sound speed)
#         # frag_list = i.frag_c * c.Grav_au * 3. * Msh / (4. * np.pi * rd * cs * r) # compare McCray and Kafatos 1987
#         # frag_value = frag_list[-1] # fragmentation occurs when this number is larger than 1.

#         # # frag value can jump from positive value directly to negative value (if velocity becomes negative) if time resolution is too coarse
#         # # however, before the velocity becomes negative, it would become 0 first. At v=0, fragmentation always occurs
#         # if frag_value < 0.:
#         #     frag_value = 2.0
#         # # another way to estimate when fragmentation occurs: Raylor-Taylor instabilities, see Baumgartner 2013, eq. (48)
#         # # not implemented yet

#         # if (was_close_to_frag is False):
#         #     frag_stop = 1.0 - float(fit_len)*dt_Emin*dfragdt # if close to fragmentation: take small time steps
#         #     if (frag_value >= frag_stop):
#         #         aux.printl("close to fragmentation", verbose=1)
#         #         ii_fragstop = aux.find_nearest_higher(frag_list, frag_stop)
#         #         if (ii_fragstop == 0): ii_fragstop = 1
#         #         t = t[:ii_fragstop]
#         #         r = r[:ii_fragstop]
#         #         rd = rd[:ii_fragstop]
#         #         Eb = Eb[:ii_fragstop]
#         #         Msh = Msh[:ii_fragstop]
#         #         frag_list = frag_list[:ii_fragstop]
#         #         frag_value = frag_list[-1]
#         #         dt_L = dt_Emin # reduce time step
#         #         was_close_to_frag = True


#         # if ((frag_value > 1.0) and (first_frag is True)):
#         #     #print frag_value
#         #     # fragmentation occurs
#         #     aux.printl("shell fragments", verbose=1)
#         #     ii_frag = aux.find_nearest_higher(frag_list, 1.0)  # index when fragmentation starts #debugging
#         #     if (ii_frag == 0): ii_frag = 1
#         #     if frag_list[ii_frag] < 1.0:
#         #         print(ii_frag, frag_list[0], frag_list[ii_frag], frag_list[-1])
#         #         sys.exit("Fragmentation value does not match criterion!")
#         #     t_frag = t[ii_frag]  # time when shell fragmentation starts
#         #     tscr = r[ii_frag]/cs_avg # sound crossing time
#         #     # if shell has just fragemented we need to delete the solution at times later than t_frag, since for those it was assumed that the shell had not fragmented
#         #     t = t[:ii_frag]
#         #     r = r[:ii_frag]
#         #     rd = rd[:ii_frag]
#         #     Eb = Eb[:ii_frag]
#         #     Msh = Msh[:ii_frag]
#         #     frag_list = frag_list[:ii_frag]
#         #     frag_value = frag_list[-1]
#         #     if i.output_verbosity >= 1: print(t_frag)
#         #     first_frag = False
#         #     # OPTION 1 for switching to mom-driving: if i.immediate_leak is set to True, we will immeadiately enter the momentum-driven phase now
#         #     if i.immediate_leak is True:
#         #         mom_phase = True
#         #         continue_branch = False
#         #         Eb[-1] = 0.

#         # # OPTION 2 for switching to mom-driving: if i.immediate_leak is set to False, when covering fraction drops below 50%, switch to momentum driving
#         # # (THIS APPROACH IS NOT STABLE YET)
#         # cf = ODEs.calc_coveringf(t, t_frag, tscr)
#         # if cf[-1] < 0.5:
#         #     ii_cov50 = aux.find_nearest_lower(cf, 0.5)
#         #     if (ii_cov50 == 0): ii_cov50 = 1
#         #     if i.output_verbosity >= 1: print(cf[ii_cov50])
#         #     t = t[:ii_cov50]
#         #     r = r[:ii_cov50]
#         #     rd = rd[:ii_cov50]
#         #     Eb = Eb[:ii_cov50]
#         #     Msh = Msh[:ii_cov50]
#         #     mom_phase = True
#         #     continue_branch = False
#         #     Eb[-1] = 0.



