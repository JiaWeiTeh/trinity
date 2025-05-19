#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 23:16:58 2022

@author: Jia Wei Teh

"""
# libraries
import numpy as np
import scipy.interpolate
#--
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.shell_structure.shell_structure as shell_structure
import src.cloud_properties.mass_profile as mass_profile
import src.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import src.bubble_structure.bubble_luminosity as bubble_luminosity
import src.cooling.non_CIE.read_cloudy as non_CIE
import src._functions.operations as operations
from src._input.dictionary import updateDict
import src._functions.unit_conversions as cvt

def run_energy(params):
    
    # TODO: add CLOUDY
    
    # the energy-driven phase
    # winds hit the shell --> reverse shock --> thermalization
    # shell is driven mostly by the high thermal pressure by the shocked ISM, 
    # also (though weaker) by the radiation pressure, at late times also SNe

    # extract dictionary infos
    t_now = params['t_now'].value
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    
    rCloud = params['rCloud_au'].value
    t_neu = params['t_neu'].value
    t_ion = params['t_ion'].value
    
    
    SB99f = params['SB99f'].value
    SB99_data = params['SB99_data'].value
    
    # =============================================================================
    # Now, we begin Energy-driven calculations (Phase 1)
    # =============================================================================
    # header
    
    # -----------
    # Step1: Obtain initial values
    # -----------
        
    # mechanical luminosity at time t0 
    # [()] is required because otherwise scipy interpolate returns size 0 ndarray, which will cause issues.
    L_wind = SB99f['fLw'](t_now)[()]
    # momentum of stellar winds at time t0
    pdot_wind = SB99f['fpdot'](t_now)[()] 
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
    t_problem = t_evo[ (dLwdt / Lw_evo) > 3]
    

    # -----------
    # Solve equation for inner radius of the inner shock.
    # -----------
    
    # initial radius of inner discontinuity [pc]
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                       a = 1e-3 * R2, b = R2, 
                       args=([L_wind, Eb, v_wind, R2]))
    
    
    # -----------
    # Solve equation for mass and pressure within bubble (r0)
    # -----------
    
    # The initial mass [Msol]
    Msh0 = mass_profile.get_mass_profile(R2, params,
                                         return_mdot = False)[0]
    # The initial pressure [au]
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
    
    
    # How long to stay in Weaver phase? Until what radius?
    if (params['dens_profile'].value == 'pL_prof') and (params['alpha_pL'].value != 0):
        rfinal = np.min([params['rCloud_au'], params['rCore']])
    else:
        rfinal = rCloud
    
    
    print(f'Inner discontinuity: {R1}.')
    print(f'Initial bubble mass: {Msh0}')
    print(f'Initial bubble pressure: {Pb}')
    print(f'rfinal: {rfinal}')
    print(f'L_wind: {L_wind}')
    print(f'pdot_wind: {pdot_wind}')
    print(f'v_wind: {v_wind}')
    
    updateDict(params, ['R1', 'L_wind', 'Pb', 'pwdot'], [R1, L_wind, Pb, pdot_wind])
    
    # first stopping time (will be incremented at beginning of while loop)
    # start time t0 will be incremented at end of while loop
    tStop_i = t_now
    
    # Now, we define some start values. In this phase, these three will stay constant throughout.
    
    # minimum separation in this phase?
    # dt_Emin = 1e-5 #* u.Myr
    # maximum separation in this phase?
    dt_Emax = 0.04166 #* u.Myr
    dt_Estart = 1e-4 #* u.Myr
    
    dt_L = dt_Estart
    
    # this will be the proper dt, timestep
    dt_real  = 1e-4 #* u.Myr
    
    # record the initial Lw0. This value will be changed in the loop. 
    # old code: Lw_old
    Lw_previous = L_wind
    
    # idk what is this. has to do with fitting?
    fit_len_max = 13
    
    # old code: mom_phase
    immediately_to_momentumphase = False
    
    condition_to_reduce_timestep = False
    
    continueWeaver = True
    # how many times had the main loop being ran?
    loop_count = 0

    # Lets make this phase at max 16 Myr according to Eq4, Rahner thesis pg44.
    # actually lets make it less than 1e4 yr (sedov taylor cooling time i think)
    # in previous code this is 3e-3 (~3000 yr)
    # Myr
    # tfinal = 1e-2
    tfinal = 3e-3


    dt_Emin = 1e-5

    while all([R2 < rfinal, (tfinal - t_now) > dt_Emin, continueWeaver]):
        
        
        # =============================================================================
        # Prelude: prepare cooling structures so that it doesnt have to run every loop.
        # Tip: Get cooling structure every 50k years (or 1e5?) or so. 
        # =============================================================================
        
        if np.abs(params['time_last_cooling_update'].value - params['t_now'].value) > 5e-2:
            # recalculate non-CIE
            cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
            # save
            params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
            params['cStruc_heating_nonCIE'].value = heating_nonCIE
            params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
            # update current value
            params['time_last_cooling_update'].value = params['t_now'].value
        
        
        
        # calculate bubble structure and shell structure?
        # no need to calculate them at very early times, since we say that no bubble or shell is being
        # created at this time. Old code: structure_switch
        calculate_bubble_shell = loop_count > 0
        
        # an initially small value if it is the first loop.
        if loop_count == 0:
            dt_Emin = 1e-5
        else:
            dt_Emin = 1e-4
            
        
        # =============================================================================
        # identify time step size and end-time value
        # =============================================================================
        
        # in the very beginning (and around first SNe) decrease tCheck because fabs varys a lot
        # increment stopping time
        ii_dLwdt = operations.find_nearest_lower(t_evo, t_now)
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
            Lw = SB99f['fLw'](t_midpoint)[()]  
            Lbol = SB99f['fLbol'](t_midpoint)[()]
            Ln = SB99f['fLn'](t_midpoint)[()]
            Li = SB99f['fLi'](t_midpoint)[()]
            # momentum of stellar winds at time t0 (cgs)
            pdot = SB99f['fpdot'](t_midpoint)[()]
            # terminal wind velocity at time t0 (pc/Myr)
            v_wind = (2. * Lw / pdot)[()]
            # ionizing
            Qi = SB99f['fQi'](t_midpoint)[()] 
            # old code: Lw_temp, evaluated at tStop_i
            Lw_tStop = SB99f['fLw'](tStop_i)[()]
    
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
            updateDict(params,
                       ['R2', 'Qi', 'v_wind', 'Eb', 'L_wind', 'Ln', 'Li', 'Lbol'],
                       [R2, Qi, v_wind, Eb, Lw, Ln, Li, Lbol])
            
            updateDict(params, ['R1', 'Pb', 'pwdot', 'mShell'], [R1, Pb, pdot_wind, Msh0])
            
            # TODO: make all dictionary extraction at the beginning, so taht we know
            # which values need to be updated.
            
            if condition_to_reduce_timestep:
                # TODO: remove after debug
                # should we calculate the bubble structure?
                
                # =============================================================================
                # Calculate bubble structure
                # =============================================================================
                
                if calculate_bubble_shell:
                    
                    print('\nCalculate bubble and shell\n')
                    _ = bubble_luminosity.get_bubbleproperties(params)
                    
                    # update this here instead of in bubble_luminosity so that 
                    # T0 will not be overwrite when we are dealing with phase1b.
                    params['T0'].value = params['bubble_T_rgoal'].value

                    T0 = params['bubble_T_rgoal'].value
                    Tavg = params['bubble_Tavg'].value
                    
                    print('\n\nFinish bubble\n\n')
                    print('L_total', params['bubble_L_total'].value)
                    print('T_rgoal', params['bubble_T_rgoal'].value)
                    print('L_bubble', params['bubble_L_bubble'].value)
                    print('L_conduction', params['bubble_L_conduction'].value)
                    print('L_intermediate', params['bubble_L_intermediate'].value)
                    print('bubble_Tavg', params['bubble_Tavg'].value)
                    print('bubble_mBubble', params['bubble_mBubble'].value)
                        
                elif not calculate_bubble_shell:
                    Tavg = T0
                    dt_L = dt_L
                
                # average sound speed in bubble
                cs_avg = operations.get_soundspeed(Tavg, params)

        
            # leave the while = True loop, if:   1) the bubble is not calculated, OR
            #                                    2) if it is SUCCESSFULLY calculated AND delta did not change by large margin, OR
            #                                    3) time step is already close to the lower limit
            if not calculate_bubble_shell: 
                break
            if dt_real < 2 * dt_Emin:
                break

            condition_to_reduce_timestep = True
            # put this here to signify ending of loop
            pass

    
        params['cs_avg'].value = cs_avg

        # =============================================================================
        # Calculate shell structure
        # =============================================================================
        
        # TODO: Where is Mbubble in previous code? Why does it only appear once in else case?
        
        
        if calculate_bubble_shell:
            
            print('\n\nhere calculate_bubble_shell\n\n')

            shell_structure.shell_structure(params)
            
        elif not calculate_bubble_shell:
            # TODO: redefine these values so that they are more physically similar to the environments
            # since we already have dictionary, this part should be empty and it should be fine.
            pass
            
        # update
        
        # =============================================================================
        # call ODE solver to solve for equation of motion (r, v (rdot), Eb). 
        # =============================================================================
        # radiation pressure coupled to the shell
        # f_absorbed_ion calculated from shell_structure.
        # F_rad = f_absorbed_ion * Lbol / params['c_au'].value
        
        y0 = [R2, v2, Eb]
        
        # call ODE solver
        psoln = scipy.integrate.odeint(energy_phase_ODEs.get_ODE_Edot, y0, t_arr, args=(params,))
        
        # [pc]
        r_arr = psoln[:,0] 
        v_arr = psoln[:, 1]
        Eb_arr = psoln[:, 2] 

        # get shell mass
        mShell_arr = mass_profile.get_mass_profile(r_arr, params,
                                                    return_mdot = False)
            
        # =============================================================================
        # Here, we perform checks to see if we should continue the branch (i.e., increasing steps)
        # =============================================================================
            
        #----------------------------
        # 2. When does fragmentation occur?
        #----------------------------
            # -----------
            # Option1 : Gravitational instability
            # -----------
            
            
            # TODO
            # -----------
            # Option2 : Rayleigh-Taylor isntability (not yet implemented)
            # -----------    
            
            
        # which temperature?
        # this is obtained from shell_structure
        if params['shell_f_absorbed_ion'].value < 0.99:
            T_shell = t_neu
        else:
            T_shell = t_ion
        # sound speed
        c_sound = operations.get_soundspeed(T_shell, params)
        params['cs_avg'].value = c_sound
    
    
        # save here
        params.save_snapShot()
        # debug
        # params.flush()
    
    
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
        v_wind = 2 * SB99f['fLw'](t_now) / SB99f['fpdot'](t_now)
        
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
            Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
            
            
        Msh0 = mShell_arr[-1] # shell mass
        
        
        # renew constants
        Lw_previous = Lw
        
        # if warpfield_params.frag_enabled:
        #     pass
            # dfragdt = (frag_value_threshold - frag_value0)/(t_arr[-1]-t_arr[0])
            # frag_value0 = frag_value_threshold
        
        
        updateDict(params, 
                    ['R1', 'R2', 'v2', 'Eb', 'v_wind', 't_now', 'Pb', 'mShell'], 
                    [R1, R2, v2, Eb, v_wind, t_now, Pb, Msh0])
        
        
        # update loop counter
        loop_count += 1
        
        pass

    return 
    
    
#%%




def check_events(params, dt_params):
    
    [dt, rd, vd, Ed, Td] = dt_params
    
    t_next = params['t_now'].value + dt
    R2_next = params['R2'].value + rd * dt
    v2_next = params['v2'].value + vd * dt
    Eb_next = params['Eb'].value + Ed * dt
    T0_next = params['T0'].value + Td * dt
        
    # =============================================================================
    # Non terminating events
    # =============================================================================
        
    # check if there is a change in sign 
    if np.sign(v2_next) != np.sign(params['v2'].value):
        if np.sign(v2_next) == -1:
            print(f'Bubble currently collapsing because the next velocity is {v2_next / cvt.v_kms2au} km/s.')
            params['isCollapse'].value = True
        else:
            params['isCollapse'].value = False
            
    # =============================================================================
    # Terminating events
    # =============================================================================
    # TODO add this percent thing into params as well
    
    # Main event: when Lcool approaches 10(?) percent of Lgain.
    if (params['Lgain'].value - params['Lloss'].value)/params['Lgain'].value < 0.05:
        print(f"Phase ended because Lloss: {params['Lloss'].value} is within {(params['Lgain'].value - params['Lloss'].value)/params['Lgain'].value * 100}% of Lgain: {params['Lgain'].value}")
        
        return True
    
    #--- 1) Stopping time reached
    if t_next > params['tStop'].value:
        print(f"Phase ended because t reaches {t_next} Myr (> tStop: {params['tStop'].value}) in the next iteration.")
        params['completed_reason'].value = 'Stopping time reached'
        return True
    
    #--- 2) Small radius reached during collapse.
    if params['isCollapse'].value == True and R2_next < params['r_coll'].value:
        print(f"Phase ended because collapse is {params['isCollapse'].value} and r reaches {R2_next} pc (< r_coll: {params['r_coll'].value} pc)")
        params['completed_reason'].value = 'Small radius reached'
        return True
    
    #--- 3) Large radius reached during expansion.
    if R2_next > params['stop_r'].value:
        print(f"Phase ended because r reaches {R2_next} pc (> stop_r: {params['stop_r'].value} pc)")
        params['completed_reason'].value = 'Large radius reached'
        return True
        
    #--- 4) dissolution after certain period of low density
    if params['t_now'].value - params['t_Lowdense'].value > params['stop_t_diss'].value:
        print(f"Phase ended because {params['t_now'].value - params['t_Lowdense'].value} Myr passed since low density of {params['shell_nShell_max'].value/cvt.ndens_cgs2au} /cm3")
        params['completed_reason'].value = 'Shell dissolved'
        return True
    
    #--- 5) exceeds cloud radius
    if params['R2'].value > params['rCloud_au'].value:
        print(f"Bubble radius ({params['R2'].value} pc) exceeds cloud radius ({params['rCloud_au'].value} pc)")
        params['completed_reason'].value = 'Bubble radius larger than cloud'
        return True
    
    return False




#%%







    
    
    
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



