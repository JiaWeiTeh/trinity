#TO DO: FIX SHELL STRUCTURE AND COUPLE TO EXPANSION

import constants as c
import parameters as p
import init as i
import numpy as np
from timestep import *
import sys
import shell_structure as shell
from __cloudy__ import *
from ODEs import *
import os
from auxiliary_functions import *
import phase_lookup as ph
import bubble_structure2 as bubble_structure
import state_eq
from scipy.interpolate import interp1d
import scipy.optimize
import scipy.integrate
import time
from scipy import spatial
import mass_profile

# import matplotlib.pyplot as plt

###################################################################################################################################
################################################# Weaver (phase I) ################################################################
###################################################################################################################################

def Weaver_phase(t0, y0, rcloud_au, SB99_data, Mcloud_au, SFE, mypath, cloudypath, shell_dissolved, t_shdis, rcore_au, nalpha, Mcluster_au, nedge, Cool_Struc, coll_counter=0, tcoll=[0.0], Tarr=[], Larr=[], tfinal = i.tStop):

    # the energy-driven phase
    # winds hit the shell --> reverse shock --> thermalization
    # shell is driven mostly by the high thermal pressure by the shocked ISM, also (though weaker) by the radiation pressure, at late times also SNe

    printl("############################################################")
    printl("entering phase I (energy-driven)...")
    printl("############################################################")
    
    if i.dens_profile == "BonnorEbert":
        rcore_old = 0.1   # no core radius for BE-spheres
    else:
        rcore_old = rcore_au
        

    start_time = time.time()

    continue_branch = True
    reset = True
    Mcore_au = 4.*np.pi/3.*rcore_old**3.*i.rhoa_au

    # first stopping time (will be incremented at beginning of while loop)
    tStop_i = t0

    # start time t0 will be incremented at end of while loop

    # initial time steps
    tInc_tmp = i.tInc_small
    tCheck_tmp = i.tCheck_small


    t_10list = np.array([])
    r_10list = np.array([])
    P_10list = np.array([])
    T_10list = np.array([])

    # get Starburst99 data
    t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo = SB99_data # unit of t_evo is Myr, the other units are cgs

    # interpolation functions for SB99 values
    fQi_evo = interp1d(t_evo, Qi_evo, kind = 'linear')
    fLi_evo = interp1d(t_evo, Li_evo, kind = 'linear')
    fLn_evo = interp1d(t_evo, Ln_evo, kind = 'linear')
    fLbol_evo = interp1d(t_evo, Lbol_evo, kind = 'linear')
    fLw_evo = interp1d(t_evo, Lw_evo, kind = 'linear')
    fpdot_evo = interp1d(t_evo, pdot_evo, kind = 'linear')
    fpdot_SNe_evo = interp1d(t_evo, pdot_SNe_evo, kind='linear')


    # identify potentially problematic time steps (where feedback parameters change by a large percentage)
    dLwdt = my_differentiate(t_evo, Lw_evo)
    abs_dLwdt = np.abs(dLwdt)
    t_problem = t_evo[abs_dLwdt / Lw_evo > 3.] # problematic time (which needs small timesteps) is defined by relative change in mechanical luminosity being more than 300% per Myr (if number on RHS is 3.0)
    #import matplotlib.pyplot as plt
    #plt.semilogy(t_evo,(np.abs(dLwdt))/Lw_evo)
    #plt.semilogy(t_evo,(Lw_evo)/Lw_evo[0])
    #plt.semilogy(t_problem,1.*t_problem/t_problem,'x')
    #plt.show()


    Lw0 = fLw_evo(t0)        /c.L_cgs # initial mechanical luminosity in astro units
    pdot0 = fpdot_evo(t0) * c.Myr / (c.Msun * c.kms) # initial momentum of stellar winds in astro units
    vterminal0 = 2. * Lw0/pdot0 # initial terminal wind velocity


    # initial values (radius, velocity, energy, temperature)
    if len(y0) == 3:
        r0, v0, E0 = y0
        # initial temperature of bubble
        T0 = 1.51e6 * (Lw_evo[0]/1e36)**(8./35.) * i.namb**(2./35.) * (t0-tcoll[coll_counter])**(-6./35.) * (1.-i.r_Tb)**0.4 # see Weaver+77, eq. (37)
    elif len(y0) == 4:
        r0, v0, E0, T0 = y0
    else:
        sys.exit("y0 has wrong length in phase_solver2.py")

    R1 = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3 * r0, r0, args=([Lw0, E0, vterminal0, r0]),maxiter=1000) # initial radius of inner discontinuity
    Ebd0 = 0. # initial energy derivative
    E0m1 = 0.9*E0
    t0m1 = 0.9*t0
    r0m1 = 0.9*r0

    # calculate swept mass (depends on used density profile)
    if i.dens_profile == "powerlaw":
        #print('##############*************************#############r0=',r0)
        Msh0 = mass_profile.calc_mass(np.array([r0]),rcore_au,rcloud_au,i.rhoa_au,i.rho_intercl_au, i.nalpha, Mcloud_au)[0] # initial mass of shell (swept up from cloud)
        #print('##############*************************#############Msh0=',Msh0)
    elif i.dens_profile == "BonnorEbert":
        T_BE=rcore_au
        #print('##############*************************#############r0=',r0)
        Msh0 = mass_profile.calc_mass_BE(r0, i.rhoa_au, T_BE, i.rho_intercl_au, rcloud_au, Mcloud_au)
        #print('##############*************************#############MshzuR0=',Msh0)
    
    # initial bubble pressure
    print('phase_solver rcore',rcore_old)    
    P0 = state_eq.PfromE(E0,r0, R1)

    Lres0 = 1.0

    # initialize r and derivatives
    rweaver = []; vweaver = []; Eweaver = []; tweaver = []
    tSweaver = []; rSweaver = []; vSweaver = []; ESweaver = []
    fabsweaver = []; fabs_i_weaver = []; fabs_n_weaver = []; ionshweaver = []; Mshell_weaver = []
    FSgrav_weaver = []; FSwind_weaver = []; FSradp_weaver = []; FSsne_weaver = []; FSIR_weaver = []; dRs_weaver = []; nmax_weaver = []
    n0weaver = []; n0_cloudyweaver = []; logMcluster_weaver = []; logMcloud_weaver = []; phase_weaver = []; R1weaver = []; Ebweaver = []; Pbweaver = []
    Lbweaver = []; Lwweaver = []; Tbweaver = []; alphaweaver = []; betaweaver = []; deltaweaver = []
    Lbbweaver = []; Lbczweaver = []; Lb3weaver = []; fragweaver = [];

    # check how long to stay in Weaver phase
    if i.adiabatic_core_only:
        rfinal = rcore_old
    else:
        rfinal = rcloud_au
    #rfinal = 100.*rfinal # factor 100 for DEBUGGING

    # only stay in Weaver phase while in core (if density gradient with core)
    #if i.density_gradient:
    #    rfinal = np.min([rcore_au, rcloud_au]) # only stay until shell breaks out off core (or cloud edge if no density gradient)
    #else:
    #    rfinal = rcloud_au

    # start values
    alpha = 0.6 # 3/5
    beta = 0.8 # 4/5
    if fLw_evo(t0) < 1e40: delta = -6./35. # -6./35.=-0.171 # taking more negative value seems to be better for massive clusters (start further out of equilibrium?)
    else: delta = -6./35. # -0.5
    temp_counter = 0
    dt_L = i.dt_Estart
    frag_value = 0.
    t_frag = 1e99
    tscr = 1e99
    dMdt_factor = 1.646
    mom_phase = False
    first_frag = True
    Lw_old = Lw0
    delta_old = delta
    cf0 = 1.0
    frag_value0 = 0.
    dfragdt = 0.
    was_close_to_frag = False
    frag_stop = 1.0
    fit_len = i.fit_len_max
    dt_real = i.dt_Estart

    printl("i, phase, t,        Rs,      v,       Eb,      R1,   fabs, fabsi, rel dR, Msh,    tau_IR, nmax,    Lw,     Lb,      Tb,         Tavg,      cs_avg, cf,  alpha, beta, delta, frag, dMdt_f,   dt_Lw,   dt_real, fit, t_prob_nn, realtime", verbose=1)

    while active_branch(r0,t0, rfinal, tfinal, active = continue_branch):

        # calculate bubble structure and shell structure?
        structure_switch = (temp_counter > 0) # r0 > 0.1, Do I need only temp_counter > 0?
        if structure_switch is True:
            dt_Emin = i.dt_Emin #1e-5
        else:
            dt_Emin = i.dt_Estart #1e-4

        fit_len_interp = i.fit_len_max + (np.log(dt_real) - np.log(dt_Emin)) * (i.fit_len_min - i.fit_len_max) / (np.log(i.dt_Emax) - np.log(dt_Emin))
        fit_len = int(np.round(np.min([i.fit_len_max, fit_len + 1, np.max([i.fit_len_min, fit_len_interp])])))  # used for delta, and in the beginning also for alpha, beta (min 3, max 10 or so)
        # time list and temperature list (coarse grid) for calculation of delta
        t_10list = aux.del_append(t_10list, t0-tcoll[coll_counter], maxlen=fit_len)
        r_10list = aux.del_append(r_10list, r0, maxlen=fit_len)
        P_10list = aux.del_append(P_10list, P0, maxlen=fit_len)
        if temp_counter >= 1:
            T_10list = aux.del_append(T_10list, T0, maxlen=fit_len - 1)  # is shifted 1 time step

        # for early time (1st Myr) use a small cloudy dt
        if (t0 - tcoll[coll_counter]) < i.cloudy_t_switch:
            my_cloudy_dt = i.small_cloudy_dt
        else:
            my_cloudy_dt = i.cloudy_dt
        t_cloudy = np.ceil(t0/my_cloudy_dt)*my_cloudy_dt
        #t_cloudy = ii_cloudy*my_cloudy_dt
        # set tmax for this time step according to cloudy dump time
        #tmax = set_tmax(i.write_cloudy, t0, t_cloudy, my_cloudy_dt,tInc_tmp)
        tmax = tfinal
        # calculate time step
        # in the very beginning (and around first SNe) decrease tCheck because fabs varys a lot
        # increment stopping time

        ii_dLwdt = find_nearest_lower(t_evo,t0)
        if abs_dLwdt[ii_dLwdt] > 0.:
            dt_Lw = 0.94*i.lum_error*Lw_old*c.L_cgs/abs_dLwdt[ii_dLwdt]
        else:
            dt_Lw = i.dt_Emax

        #tStop_i, tInc_tmp, tCheck_tmp = update_dt(r0, v0, tInc_tmp, t0, tCheck_tmp, tmax = tmax)
        # make sure, that only 1 cloudy output is written per time step (if necessary, decrease time step)
        if (i.write_cloudy is True):
            tStop_i = np.min([tmax,t0+np.min([dt_L,my_cloudy_dt,i.dt_Emax,dt_Lw])]) # this is the stop time for this time step
        else:
            tStop_i = np.min([tmax,t0+np.min([dt_L,i.dt_Emax,dt_Lw])])

        # find nearest problematic neighbor of current stop time
        t_problem_hi = t_problem[t_problem > (t0 + 0.001*dt_Emin)]
        t_problem_nnhi = t_problem_hi[0] # the next problematic time
        tStop_i = np.min([tStop_i, (t_problem_nnhi - float(i.fit_len_max+1)*dt_Emin)])
        # ensure that time step becomes not too small or negative
        if tStop_i < (t0 + dt_Emin):
            tStop_i = t0 + dt_Emin

        if ((frag_value > frag_stop) and (frag_value < 1.0)):
            tStop_i = t0 + dt_Emin

        dt_real = tStop_i - t0

        if (temp_counter == 2. * fit_len): dt_real = dt_Emin

        # correct (?) calculation of beta
        beta_temp = -t0 / P0 * 1. / (2. * np.pi) * (Ebd0 / (r0 ** 3 - R1 ** 3) - 3. * E0 * v0 * r0 ** 2 / (r0 ** 3 - R1 ** 3) ** 2.)
        alpha_temp = t0 / r0 * v0
        """
        if temp_counter >= 2.*fit_len:
            alpha = alpha_temp
            beta = beta_temp
            # temporary ###########
            param1 = {'alpha': alpha, 'beta': beta, 'delta': delta, 'Eb': E0, 'R2': r0, 't_now': t0, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr}
            param0 = {'alpha': alpha, 'beta': beta, 'delta': delta, 'Eb': E0m1, 'R2': r0m1, 't_now': t0m1, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr}
            dzero_params = [param0, param1, Cool_Struc]
            #delta_temp = scipy.optimize.brentq(bubble_structure.new_zero_delta, delta - 0.1, delta + 0.1, args=(dzero_params))
            delta_temp, bubble_check_temp = bubble_structure.delta_new_root(delta, dzero_params)
            param1["delta"] = delta_temp
            Lb_temp, T_rgoal_temp, Lb_b_temp, Lb_cz_temp, Lb3_temp, dMdt_factor_out_temp, Tavg_temp, Mbubble_temp, r_Phi_temp, Phi_grav_r0b_temp, f_grav_temp = bubble_structure.calc_Lb(param1,Cool_Struc, temp_counter)
            print Lb_temp, T_rgoal_temp, delta_temp
            #######################
        #print beta_temp, alpha_temp
        """

        # before this time step is accepted, check how the mechanical luminosity would change
        # if it would change too much, reduce time step
        reduce_dt = False # when first entering the following loop, do not reduce time step
        while (True):
            if (reduce_dt is True):
                #print("reducing time step...")
                dt_real = 0.5*dt_real
                tStop_i = t0 + dt_real
                printl("halving time step...", verbose=1)

            t_inc = dt_real/1000.

            # time vector for this time step
            t_temp = np.arange(t0, tStop_i + t_inc, t_inc)  # make sure to include tStop_i here, i.e. go one small dt farther
            #t = np.insert(t_temp,-1,t_temp[-1] - min(0.5*(t_temp[-1]-t_temp[-2]),1e-6)) # make sure the last time step is very small (import for derivatives in bubble_structure)

            # insert a value close to t[-1]
            val = t_temp[-1] - 9.9147e-6
            if val > t_temp[0]:
                idx_val = np.searchsorted(t_temp, val)
                t = np.insert(t_temp,idx_val,val)
            else:
                idx_val = 0
                t = t_temp
            # midpoint time, evaluate feedback parameters here
            thalf = 0.5 * (t[0]+t[-1])

            # get SB99 values
            if i.SB99 == True:
                if (thalf > t_evo[-1]):
                    print("Warning: End of SB99 file reached")
                    sys.exit("Stop Code: Feedback parameters not provided")

                # convert cgs to astro units (Myr, Msun, pc)
                Lw = fLw_evo(thalf)         /c.L_cgs
                Lw_temp = fLw_evo(tStop_i)         /c.L_cgs
                Lbol = fLbol_evo(thalf)      /c.L_cgs
                Ln = fLn_evo(thalf)         /c.L_cgs
                Li = fLi_evo(thalf)         /c.L_cgs
                pdot= fpdot_evo(thalf)         * c.Myr / (c.Msun * c.kms)
                pdot_SNe = fpdot_SNe_evo(thalf) * c.Myr / (c.Msun * c.kms)

                vterminal_evo = 2. * fLw_evo(thalf) / fpdot_evo(thalf) # cgs units
                vterminal = 2. * Lw / pdot

            # if mechanical luminosity would change too much, run through this loop again with reduced time step
            # in that case, the following condition is not fulfilled

            if (((abs(Lw_temp - Lw_old) < i.lum_error*Lw_old) and (abs(Lw - Lw_old) < i.lum_error*Lw_old)) or (dt_real < 2*dt_Emin)):
                # if mechanical luminosity does not change much, check how much delta would change

                if temp_counter < 1e30*fit_len: # 2.*fit_len
                    printl("entering old bubble_wrap...", verbose=1)
                    structure_bubble = (structure_switch and not mom_phase)
                    bubble_wrap_struc = {'structure_switch': structure_bubble, 'alpha': alpha, 'beta': beta, 'delta': delta,
                                         'Lres0': Lres0, 't_10list': t_10list, 'r_10list': r_10list, 'P_10list': P_10list,
                                         'T_10list': T_10list, 'Lw': Lw, 'vterminal': vterminal, 'r0': r0,
                                         't0': t0 - tcoll[coll_counter], 'E0': E0,
                                         'T0': T0, 'dt_L': dt_real, 'temp_counter': temp_counter, 'dMdt_factor': dMdt_factor,
                                         'Qi': fQi_evo(thalf)*c.Myr, 'mypath': mypath}

                    printl(bubble_wrap_struc, verbose=1)
                    # it's important to use the time since the last restarting expansion, not the time since the start of the simulation

                    #print {'structure_switch': structure_bubble, 'alpha': alpha, 'beta': beta, 'delta': delta, 'Lres0': Lres0, 'Lw': Lw, 'vterminal': vterminal, 'r0': r0,'t0': t0 - tcoll[coll_counter], 'E0': E0,'T0': T0, 'dt_L': dt_real, 'temp_counter': temp_counter, 'dMdt_factor': dMdt_factor,'Qi': fQi_evo(thalf)*c.Myr}

                    # calculate bubble structure
                    [bubble_check, Lb, T0, alpha, beta, delta, dt_L, Lb_b, Lb_cz, Lb3, dMdt_factor, Tavg, Mbubble, r_Phi_b, Phi_grav_r0b, f_grav_b] = bubble_structure.bubble_wrap(bubble_wrap_struc, Cool_Struc, fit_len=fit_len, fit_len_short=fit_len, verbose=0)
                else:
                    printl("entering delta_new_root...", verbose=1)
                    alpha = alpha_temp
                    beta = beta_temp
                    # temporary ###########
                    param1 = {'alpha': alpha, 'beta': beta, 'Eb': E0, 'R2': r0, 't_now': t0, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr, 'mypath': mypath}
                    param0 = {'alpha': alpha, 'beta': beta, 'Eb': E0m1, 'R2': r0m1, 't_now': t0m1, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr, 'mypath': mypath}
                    dzero_params = [param0, param1, Cool_Struc]
                    # delta_temp = scipy.optimize.brentq(bubble_structure.new_zero_delta, delta - 0.1, delta + 0.1, args=(dzero_params))
                    delta, bubble_check = bubble_structure.delta_new_root(delta, dzero_params)
                    param1["delta"] = delta
                    Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav = bubble_structure.calc_Lb(param1, Cool_Struc, temp_counter)




                    # new time step
                    Lres_temp = Lw - Lb
                    fac = np.max([np.min([i.lum_error2 / (np.abs(Lres_temp - Lres0) / Lw), 1.42]), 0.1])
                    dt_L = fac * dt_L  # 3 per cent change
                    #print Lb_temp, T_rgoal_temp, delta_temp

                    param1['dt_L'] = dt_L
                    param1['T0'] = T_rgoal
                    param1['temp_counter'] = temp_counter
                    param1['Lres0'] = Lw - Lb

                    print("param1:", param1)


                # average sound speed in bubble
                cs_avg = np.sqrt(2.*c.gamma*c.kboltz_au*Tavg*c.Msun/c.mp)

                # in the very early energy-phase, it is not important to get it right
                # only leave loop if bubble_structure did not throw errors (as measured by bubble_check) and delta did not change a lot
                # OR if time step is already close to the lower limit
                if (((bubble_check == 0) and ((temp_counter < 2*fit_len) or (abs(delta-delta_old) < i.delta_error))) or (dt_real < 2*dt_Emin)):
                    break # everything is good, don't change time step any more, leave "while(True)"-loop

            reduce_dt = True # not everything is good --> reduce time step in next iteration of loop

        if bubble_check != 0:
            printl("Could not figure out the correct delta. Worrisome!", verbose=-1)

        # gradually switch on cooling (i.e. reduce the amount of cooling at very early times)
        if (t0-tcoll[coll_counter] < i.dt_switchon):
            reduce_factor = np.max([0.,np.min([1.,(t0-tcoll[coll_counter])/i.dt_switchon])])
            Lb *= reduce_factor # scale and make sure it is between 0 and 1
            Lb_b *= reduce_factor
            Lb_cz *= reduce_factor
            Lb3 *= reduce_factor

        ##############################################################################################################








        ##############################################################################################################


        ################ CLOUDY #############################################################################################
        make_cloudy_model = (i.write_cloudy and t0 <= t_cloudy and tStop_i >= t_cloudy)
        #####################################################################################################################

        if structure_switch:
            # shell structure routine wants cgs units
            np.seterr(all='warn')
            #fname = "shellstruct_1e" + str(round(np.log10(Mcloud_au), ndigits=2)) + "(SB99)_SFE=" + str(round(100.*SFE)) + "_n=" + str(int(i.namb)) + "_Z=" + str(i.Zism) + "_P1_t="+str(round(t0,ndigits=2))+".png"
            age1e7_str = ('{:0=5.7f}e+07'.format(t0 / 10.))
            fname = "shell_" + age1e7_str + ".dat"
            filename_shell = mypath + '/shellstruct/' + fname
            check_outdir(mypath + '/shellstruct/')
            [fabs_i, fabs_n, fabs, fion_dust, ionsh, dRs, n0, nmax, rhodr, n0_cloudy, r_Phi_sh, Phi_grav_r0s, f_grav_sh] = shell.shell_structure2(r0, P0, fLn_evo(thalf), fLi_evo(thalf), fQi_evo(thalf), Msh0, 1, ploton = make_cloudy_model, plotpath = filename_shell, Minterior = Mbubble*c.Msun)
            #[fabs_i, fabs_n, fabs, fion_dust, ionsh, dRs, n0, nmax, rhodr, n0_cloudy] = shell.shell_structure(r0, fLn_evo(thalf),fLi_evo(thalf),fQi_evo(thalf),fLw_evo(thalf),fpdot_evo(thalf),Msh0,ploton=make_cloudy_model, plotpath=filename_shell, phase="Weaver")
            tau_IR = rhodr * i.kIR

            r_Phi = np.concatenate([r_Phi_b, r_Phi_sh])
            Phi_grav_r0 = Phi_grav_r0b + Phi_grav_r0s # this is the integral from r0 to rsh_out over r*rho(r); the integral part from rsh_out to very far out will be done in write_pot_to_file
            f_grav = np.concatenate([f_grav_b, f_grav_sh])
            if i.write_potential is True:
                write_pot_to_file(mypath, t0, r_Phi, Phi_grav_r0, f_grav, rcloud_au, rcore_old, nalpha, Mcloud_au, Mcluster_au, SFE)

        else:
            # If it is very early, no shell has yet been swept. Set some default values!
            fabs = 0.0
            fabs_i = 0.0
            fabs_n = 0.0
            ionsh = False
            dRs = 0.0
            nmax = 1e5
            tau_IR = 0.
            n0 = i.namb
            n0_cloudy = n0
            Lb = 0.
        if shell_dissolved == False and nmax < i.n_intercl:
            shell_dissolved = True
            t_shdis = t0

        # now solve eq. of motion
        # this part of the routine assumes astro-units

        LW = Lw
        PWDOT = pdot
        GAM = c.gamma

        RHOA = i.rhoa_au
        RCORE = rcore_old
        A_EXP = nalpha
        MSTAR = Mcluster_au
        LB = Lb # astro units!!
        FRAD = fabs * Lbol/c.clight_au # astro units
        CS = cs_avg

        # inside core or inside density slope?
        if i.dens_profile == "BonnorEbert":
            phase0 = ph.Egrad
            r_core_old = 0
        elif (r0 < rcore_old or not i.density_gradient):
            phase0 = ph.weaver
        else:
            phase0 = ph.Egrad

        if i.output_verbosity >= 1:
            print('%d' % temp_counter, '%.1f' % phase0, '%.4e' % t0, "%.2e" % r0, "%.4e" % v0, "%.7e" % E0, "%.7e" % R1, "%.3f" % fabs, "%.4f" % fabs_i, "%.4f" % (dRs / r0), "%.2e" % Msh0, "%.4f" % tau_IR, "%.2e" % nmax, "%.3e" % (Lw*c.L_cgs), "%.3e" % (Lb*c.L_cgs), "%.3e" % (T0), "%.3e" % (Tavg), "%.2e" % (cs_avg), "%.2f" % (cf0), "%.4f" % (alpha), "%.4f" % (beta), "%.4f" % (delta), "%.1e" % (frag_value), "%.3e" % (dMdt_factor), "%.3e" % (dt_Lw), "%.3e" % (dt_real), "%d" % (float(fit_len)), "%.3e" %(t_problem_nnhi), "%.3e" %(time.time()-start_time))



        # Bundle initial conditions for ODE solver
        y0 = [r0, v0, E0]

        # bundle parameters for ODE solver
        params = [LW, PWDOT, GAM, Mcloud_au, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, fabs_i, rcloud_au, phase0, tcoll[coll_counter], t_frag, tscr, CS, SFE]
        printl(("params", params), verbose=1)

        # call ODE solver
        try:
            psoln = scipy.integrate.odeint(fE_gen, y0, t, args=(params,))
        except:
            sys.exit("EoM")

        # get r, rdot and rdotdot
        r = psoln[:,0]
        rd = psoln[:, 1]
        Eb = psoln[:, 2]
        if i.dens_profile == "powerlaw":
            #print('##############*************************#############r=',r)
            Msh = mass_profile.calc_mass(r, rcore_au, rcloud_au, i.rhoa_au, i.rho_intercl_au, nalpha, Mcloud_au)
            #print('##############*************************#############MshzuR=',Msh)
        
        elif i.dens_profile == "BonnorEbert":
            T_BE=rcore_au
            #print('##############*************************#############r=',r)
            Msh = mass_profile.calc_mass_BE(r, i.rhoa_au, T_BE, i.rho_intercl_au, rcloud_au, Mcloud_au)
            #print('##############*************************#############MshzuR=',Msh)

        """
        ################ CLOUDY #############################################################################################
        # if this is the active branch store data in cloudy input file
        if make_cloudy_model:
            #print("write cloudy model at t=", t_cloudy)
            jj_cloudy = find_nearest_higher(t, t_cloudy)
            r_cloudy = r[jj_cloudy]
            v_cloudy = rd[jj_cloudy]
            Msh_cloudy = Msh[jj_cloudy]
            E_cloudy = Eb[jj_cloudy]
            Lw_cloudy = fLw_evo(t_cloudy)         /c.L_cgs
            pdot_cloudy= fpdot_evo(t_cloudy)         * c.Myr / (c.Msun * c.kms)
            vterminal_cloudy = 2. * Lw_cloudy / pdot_cloudy
            R1_cloudy = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3 * r0, r0, args=([Lw_cloudy, E_cloudy, vterminal_cloudy, r_cloudy]))
            P_cloudy = state_eq.PfromE(E_cloudy, r_cloudy, R1_cloudy)
            [n0_temp, n0_cloudy] = n_from_press(P_cloudy, i.Ti, B_cloudy=i.B_cloudy)
            create_model(cloudypath, SFE, Mcloud_au, i.namb, i.Zism, n0_cloudy, r_cloudy, v_cloudy, Msh_cloudy,
                         np.log10(fLbol_evo(t_cloudy)), t_cloudy, rcloud_au, nedge,
                         SB99model=i.SB99cloudy_file, shell=i.cloudy_stopmass_shell, turb=i.cloudy_turb,
                         coll_counter=coll_counter, Tarr=Tarr, Larr=Larr, Li=Li*c.L_cgs, Qi=fQi_evo(thalf), Mcluster=Mcluster_au, phase=phase0)
        #####################################################################################################################
        """

        # check whether shell fragments or similar

        # if a certain radius has been exceeded, stop branch
        if (r[-1] > rfinal or (r0 < rcore_old and r[-1] >= rcore_old and i.density_gradient)):
            if r[-1] > rfinal:
                continue_branch = False
                include_list = r<rfinal
            else:
                continue_branch = True
                rtmp = r[r>=rcore_old][0] # find first radius larger than core radius
                include_list = r <= rtmp
            t = t[include_list]
            r = r[include_list]
            rd = rd[include_list]
            Eb = Eb[include_list]
            Msh = Msh[include_list]


        # calculate fragmentation time
        if fabs_i < 0.999:
            Tsh = i.Tn  # 100 K or so
        else:
            Tsh = i.Ti  # 1e4 K or so
        cs = sound_speed(Tsh, unit="kms")  # sound speed in shell (if a part is neutral, take the lower sound speed)
        frag_list = i.frag_c * c.Grav_au * 3. * Msh / (4. * np.pi * rd * cs * r) # compare McCray and Kafatos 1987
        frag_value = frag_list[-1] # fragmentation occurs when this number is larger than 1.

        # frag value can jump from positive value directly to negative value (if velocity becomes negative) if time resolution is too coarse
        # however, before the velocity becomes negative, it would become 0 first. At v=0, fragmentation always occurs
        if frag_value < 0.:
            frag_value = 2.0
        # another way to estimate when fragmentation occurs: Raylor-Taylor instabilities, see Baumgartner 2013, eq. (48)
        # not implemented yet

        if (was_close_to_frag is False):
            frag_stop = 1.0 - float(fit_len)*dt_Emin*dfragdt # if close to fragmentation: take small time steps
            if (frag_value >= frag_stop):
                printl("close to fragmentation", verbose=1)
                ii_fragstop = find_nearest_higher(frag_list, frag_stop)
                if (ii_fragstop == 0): ii_fragstop = 1
                t = t[:ii_fragstop]
                r = r[:ii_fragstop]
                rd = rd[:ii_fragstop]
                Eb = Eb[:ii_fragstop]
                Msh = Msh[:ii_fragstop]
                frag_list = frag_list[:ii_fragstop]
                frag_value = frag_list[-1]
                dt_L = dt_Emin # reduce time step
                was_close_to_frag = True


        if ((frag_value > 1.0) and (first_frag is True)):
            #print frag_value
            # fragmentation occurs
            printl("shell fragments", verbose=1)
            ii_frag = find_nearest_higher(frag_list, 1.0)  # index when fragmentation starts #debugging
            if (ii_frag == 0): ii_frag = 1
            if frag_list[ii_frag] < 1.0:
                print(ii_frag, frag_list[0], frag_list[ii_frag], frag_list[-1])
                sys.exit("Fragmentation value does not match criterion!")
            t_frag = t[ii_frag]  # time when shell fragmentation starts
            tscr = r[ii_frag]/cs_avg # sound crossing time
            # if shell has just fragemented we need to delete the solution at times later than t_frag, since for those it was assumed that the shell had not fragmented
            t = t[:ii_frag]
            r = r[:ii_frag]
            rd = rd[:ii_frag]
            Eb = Eb[:ii_frag]
            Msh = Msh[:ii_frag]
            frag_list = frag_list[:ii_frag]
            frag_value = frag_list[-1]
            if i.output_verbosity >= 1: print(t_frag)
            first_frag = False
            # OPTION 1 for switching to mom-driving: if i.immediate_leak is set to True, we will immeadiately enter the momentum-driven phase now
            if i.immediate_leak is True:
                mom_phase = True
                continue_branch = False
                Eb[-1] = 0.

        # OPTION 2 for switching to mom-driving: if i.immediate_leak is set to False, when covering fraction drops below 50%, switch to momentum driving
        # (THIS APPROACH IS NOT STABLE YET)
        cf = calc_coveringf(t, t_frag, tscr)
        if cf[-1] < 0.5:
            ii_cov50 = find_nearest_lower(cf, 0.5)
            if (ii_cov50 == 0): ii_cov50 = 1
            if i.output_verbosity >= 1: print(cf[ii_cov50])
            t = t[:ii_cov50]
            r = r[:ii_cov50]
            rd = rd[:ii_cov50]
            Eb = Eb[:ii_cov50]
            Msh = Msh[:ii_cov50]
            mom_phase = True
            continue_branch = False
            Eb[-1] = 0.

        # store data

        # fine spacing
        rweaver_end = r[-1]
        vweaver_end = rd[-1]
        Eweaver_end = Eb[-1]
        tweaver_end = t[-1]

        #rweaver = np.concatenate([rweaver, r])
        #vweaver = np.concatenate([vweaver, rd])
        #Eweaver = np.concatenate([Eweaver, Eb])
        #tweaver = np.concatenate([tweaver, t])

        # coarse spacing
        tSweaver = np.concatenate([tSweaver, [t0]])
        rSweaver = np.concatenate([rSweaver, [r0]])
        vSweaver = np.concatenate([vSweaver, [v0]])
        ESweaver = np.concatenate([ESweaver, [E0]])
        fabsweaver = np.concatenate([fabsweaver, [fabs]])
        fabs_i_weaver = np.concatenate([fabs_i_weaver, [fabs_i]])
        fabs_n_weaver = np.concatenate([fabs_n_weaver, [fabs_n]])
        ionshweaver = np.concatenate([ionshweaver, [ionsh]])
        Mshell_weaver = np.concatenate([Mshell_weaver, [Msh0]])
        dRs_weaver = np.concatenate([dRs_weaver, [dRs]])
        n0weaver = np.concatenate([n0weaver, [n0]])
        n0_cloudyweaver = np.concatenate([n0_cloudyweaver, [n0_cloudy]])
        FSgrav_weaver = np.concatenate([FSgrav_weaver, [1e-30]])
        FSwind_weaver = np.concatenate([FSwind_weaver, [1e-30]])
        FSradp_weaver = np.concatenate([FSradp_weaver, [1e-30]])
        FSsne_weaver = np.concatenate([FSsne_weaver, [1e-30]])
        FSIR_weaver = np.concatenate([FSIR_weaver, [1e-30]])
        nmax_weaver = np.concatenate([nmax_weaver, [nmax]])
        logMcluster_weaver = np.concatenate([logMcluster_weaver, [np.log10(Mcluster_au)]])
        logMcloud_weaver = np.concatenate([logMcloud_weaver, [np.log10(Mcloud_au)]])
        phase_weaver = np.concatenate([phase_weaver, [phase0]]) # set by hand (dangerous)
        R1weaver = np.concatenate([R1weaver, [R1]])
        Ebweaver = np.concatenate([Ebweaver, [E0]])
        Pbweaver = np.concatenate([Pbweaver, [P0]])
        Lwweaver = np.concatenate([Lwweaver, [Lw]])
        Lbweaver = np.concatenate([Lbweaver, [Lb]])
        Tbweaver = np.concatenate([Tbweaver, [T0]])
        alphaweaver = np.concatenate([alphaweaver, [alpha]])
        betaweaver = np.concatenate([betaweaver, [beta]])
        deltaweaver = np.concatenate([deltaweaver, [delta]])
        fragweaver = np.concatenate([fragweaver, [frag_value]])

        Lbbweaver = np.concatenate([Lbbweaver, [Lb_b]])
        Lbczweaver = np.concatenate([Lbczweaver, [Lb_cz]])
        Lb3weaver = np.concatenate([Lb3weaver, [Lb3]])

        Data_w = {'t':tSweaver, 'r':rSweaver, 'v':vSweaver, 'E':ESweaver, 't_end': tweaver_end, 'r_end':rweaver_end, 'v_end':vweaver_end, 'E_end':Eweaver_end,
                  'fabs':fabsweaver, 'fabs_n': fabs_n_weaver, 'Fgrav':FSgrav_weaver, 'Fwind':FSwind_weaver, 'Fradp_dir':FSradp_weaver, 'FSN':FSsne_weaver, 'Fradp_IR':FSIR_weaver,
                  'fabs_i':fabs_i_weaver, 'dRs':dRs_weaver, 'logMshell':np.log10(Mshell_weaver), 'nmax':nmax_weaver, 'logMcluster':logMcluster_weaver, 'logMcloud':logMcloud_weaver,
                  'phase': phase_weaver, 'R1':R1weaver, 'Eb':Ebweaver, 'Pb':Pbweaver, 'Lmech':Lwweaver, 'Lcool':Lbweaver, 'Tb':Tbweaver,
                  'alpha':alphaweaver, 'beta':betaweaver,'delta':deltaweaver, 'Lbb':Lbbweaver, 'Lbcz':Lbczweaver, 'Lb3':Lb3weaver, 'frag':fragweaver, 'dMdt_factor_end': dMdt_factor}

        Flux_Phi = fQi_evo(thalf)/(4.*np.pi*(r0*c.pc)**2)
        Pb_cgs = P0 /(c.pc*c.Myr**2/(c.Msun))
        min_Pb = np.min([1e-16*Flux_Phi**0.667, 1e-21*Flux_Phi])
        max_Pb = np.max([1e-16*Flux_Phi, 2e-11*Flux_Phi**0.667])

        #print '%.4e' % Flux_Phi,  '%.4e' % min_Pb, '%.4e' % Pb_cgs, '%.4e' % max_Pb
        if ((Pb_cgs < min_Pb) or (Pb_cgs > max_Pb)):
            print("Warning: might not have necessary cooling values tabulated. Let's see...")
            print('%.4e' % Flux_Phi, '%.4e' % min_Pb, '%.4e' % Pb_cgs, '%.4e' % max_Pb)

        # get new initial values
        t0 = t[-1]
        r0 = r[-1] # shell radius
        v0 = rd[-1] # shell velocity
        E0 = Eb[-1] # bubble energy
        Lw_new = (fLw_evo(t0) / c.L_cgs)
        vterminal0 = 2.*Lw_new / (fpdot_evo(t0) * c.Myr / c.Msun / c.kms)
        if (not mom_phase):
            R1 = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3*r0, r0, args=([Lw_new, E0, vterminal0, r0]))
            P0 = state_eq.PfromE(E0, r0, R1) # bubble pressure
        else:
            R1 = r0
            P0 = state_eq.Pram(r0,Lw_new,vterminal0)
        Msh0 = Msh[-1] # shell mass
        Lres0 = LW - LB
        Lw_old = Lw
        delta_old = delta
        cf0 = cf[-1]
        dfragdt = (frag_value - frag_value0)/(t[-1]-t[0])
        frag_value0 = frag_value

        if idx_val >= len(t): idx_val = -2

        t0m1 = t[idx_val]
        r0m1 = r[idx_val]
        E0m1 = Eb[idx_val]
        Ebd0 = (Eb[-1] - Eb[idx_val]) / (t[-1] - t[idx_val])  # last time derivative of Eb


        temp_counter += 1


    return Data_w,shell_dissolved, t_shdis

###################################################################################################################################
################################################# Momentum (phase II) #############################################################
###################################################################################################################################

# wrapper which stars Mom phase
def Mom_check_and_start(t0, r0, v0, SB99_data, Mcloud_au, SFE, mypath, cloudypath, shell_dissolved, t_shdis, Mcluster_au, rcloud_au, rcore_au, nalpha, nedge, tcoll=[0.0], coll_counter=0, Tarr=[], Larr=[]):

    # initial radius and velocity
    y0 = [r0, v0]

    [Data_mom, shell_dissolved, t_shdis, ionshmom] = Mom_phase(t0, y0,
                                                                            SB99_data, Mcloud_au,
                                                                            SFE, mypath,
                                                                            cloudypath, shell_dissolved,
                                                                            t_shdis, Mcluster_au, rcloud_au, rcore_au,
                                                                            nalpha, nedge, tcoll=tcoll, coll_counter=coll_counter, Tarr=Tarr, Larr=Larr)

    return Data_mom, shell_dissolved, t_shdis, ionshmom


def Mom_phase(t0, y0, SB99_data, Mcloud_au, SFE, mypath, cloudypath,
              shell_dissolved, t_shdis, Mcluster_au, rcloud_au, rcore_au, nalpha, nedge, coll_counter=0, tcoll=[0.0], Tarr=[], Larr=[]):
    printl("############################################################")
    printl("entering phase II (momentum-driven)... t = " + str(t0) + " Myr")
    printl("############################################################")

    start_time = time.time()

    # initialize r and derivatives
    rmom = [];
    vmom = [];
    tmom = [];

    tSmom = [];
    rSmom = [];
    vSmom = [];
    fabsmom = [];
    ionshmom = [];
    Mshell_mom = []
    FSgrav_mom = [];
    FSwind_mom = [];
    FSradp_mom = [];
    FSsne_mom = [];
    FSIR_mom = [];
    fabs_i_mom = [];
    fabs_n_mom = [];
    dRs_mom = [];
    nmax_mom = [];
    logMcloud_mom = [];
    logMcluster_mom = []
    phase_mom = []
    Lwmom = []

    n0mom = []
    n0_cloudymom = []

    reset = True

    #core mass in solar masses
    Mcore_au = 4. * np.pi / 3. * rcore_au ** 3. * i.rhoa_au

    # shell is not yet dissolved
    first_dissolve = False

    # first stopping time (will be incremented at beginning of while loop)
    tStop_i = t0

    # start time t0 will be incremented at end of while loop

    # reset time step
    tInc_tmp = i.tInc
    tCheck_tmp = i.tCheck

    r0, v0 = y0
    if i.dens_profile == "powerlaw":
        #print('##############*************************#############r0=',r0)
        Msh0 = mass_profile.calc_mass(np.array([r0]), rcore_au, rcloud_au, i.rhoa_au,i. rho_intercl_au, nalpha, Mcloud_au)[0]
        #print('##############*************************#############MshzuR0=',Msh0)
    
    elif i.dens_profile == "BonnorEbert":
        T_BE=rcore_au
        #print('##############*************************#############r0=',r0)
        Msh0 = mass_profile.calc_mass_BE(r0, i.rhoa_au, T_BE, i.rho_intercl_au, rcloud_au, Mcloud_au)
        #print('##############*************************#############MshzuR0=',Msh0)

    # unpack SB99 data
    t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo = SB99_data

    # interpolation functions for SB99 values
    fQi_evo = interp1d(t_evo, Qi_evo, kind = 'linear')
    fLi_evo = interp1d(t_evo, Li_evo, kind = 'linear')
    fLn_evo = interp1d(t_evo, Ln_evo, kind = 'linear')
    fLbol_evo = interp1d(t_evo, Lbol_evo, kind = 'linear')
    fLw_evo = interp1d(t_evo, Lw_evo, kind = 'linear')
    fpdot_evo = interp1d(t_evo, pdot_evo, kind = 'linear')
    fpdot_SNe_evo = interp1d(t_evo, pdot_SNe_evo, kind='linear')

    printl("t,     Rs,   v,    fabs, fabsi, rel dR, Msh,   tau_IR, nmax, real time", verbose=1)

    while tStop_i < i.tStop and v0 > i.vstop and (r0 > i.rcoll or v0 > -1.0) and r0 < i.rstop:
        # increment stopping time (start time has already been increemented at end of loop)
        # for early time (1st Myr) use a small cloudy dt

        if (t0 - tcoll[coll_counter]) < i.cloudy_t_switch:
            my_cloudy_dt = i.small_cloudy_dt
        else:
            my_cloudy_dt = i.cloudy_dt
        t_cloudy = np.ceil(t0/my_cloudy_dt)*my_cloudy_dt


        #if tStop_i > 1.0:
        #    my_cloudy_dt = i.cloudy_dt
        #    # need to reset cloudy counter when first switching to new cloudy dt
        #    if (tStop_i < 1.0 + my_cloudy_dt):
        #        ii_cloudy, reset = reset_ii_cloudy(ii_cloudy, 1.0, my_cloudy_dt, reset = reset)
        #else:
        #    my_cloudy_dt = i.small_cloudy_dt
        #t_cloudy = ii_cloudy*my_cloudy_dt
        # set tmax for this time step according to cloudy dump time
        #tmax = set_tmax(i.write_cloudy, t0, t_cloudy, my_cloudy_dt,tInc_tmp)
        tmax = i.tStop
        # calculate time step
        tStop_i, tInc_tmp, tCheck_tmp = update_dt(r0, v0, tInc_tmp, t0, tCheck_tmp, tmax= tmax)

        dt_real = tStop_i - t0
        # make sure, that only 1 cloudy output is written per time step (if necessary, decrease time step)
        if ((i.write_cloudy is True) and (dt_real > my_cloudy_dt)):
            dt_real = my_cloudy_dt
            tStop_i = t0 + dt_real
        t_inc = dt_real/1000.

        # time vector for this time step
        t = np.arange(t0, tStop_i+t_inc, t_inc) # make sure to include tStop_i here, i.e. go one small dt farther
        #t = np.arange(t0, tStop_i+tInc_tmp, tInc_tmp)

        # get initial values
        y0 = [r0, v0]

        # midpoint time, evaluate feedback parameters here
        thalf = 0.5 * (t[0]+t[-1])

        # get SB99 values
        if i.SB99 == True:
            if (thalf > t_evo[-1]):
                print("Warning: End of SB99 file reached")
                sys.exit("Stop Code: Feedback parameters not provided")

            # convert cgs to astro units (Myr, Msun, pc)
            Lw = fLw_evo(thalf)         /c.L_cgs
            Lbol = fLbol_evo(thalf)      /c.L_cgs
            Ln = fLn_evo(thalf)         /c.L_cgs
            Li = fLi_evo(thalf)         /c.L_cgs
            pdot= fpdot_evo(thalf)         * c.Myr / (c.Msun * c.kms)
            pdot_SNe = fpdot_SNe_evo(thalf) * c.Myr / c.Msun / c.kms

            vterminal_evo = 2. * fLw_evo(thalf) / fpdot_evo(thalf) # cgs units
            vterminal = 2. * Lw / pdot

        ##############################################################################################################

        ################ CLOUDY #############################################################################################
        make_cloudy_model = (i.write_cloudy and t0 <= t_cloudy and tStop_i >= t_cloudy)
        #####################################################################################################################

        P0 = state_eq.Pram(r0, Lw, vterminal)

        # calculate shell structure
        figtitle_shell = "shellstruct_1e" + str(round(np.log10(Mcloud_au),ndigits=2)) + "(SB99)_SFE=" + str(
            round(100. * SFE)) + "_n=" + str(int(i.namb)) + "_Z=" + str(i.Zism) + "_P3_t=" + str(
            round(t0, ndigits=2)) + ".png"
        figpath_shell = mypath + '/shellstruct/' + figtitle_shell
        print('*******cf in phase_solver2 2nd eq')
        [fabs_i, fabs_n, fabs, fion_dust, ionsh, dRs, n0, nmax, rhodr, n0_cloudy, r_Phi, Phi_grav_r0, f_grav] = shell.shell_structure2(r0, P0, fLn_evo(thalf), fLi_evo(thalf), fQi_evo(thalf), Msh0, 1, ploton = make_cloudy_model, plotpath = figpath_shell, Minterior=0.)
        tau_IR = rhodr * i.kIR

        if i.write_potential is True:
            write_pot_to_file(mypath, t0, r_Phi, Phi_grav_r0, f_grav, rcloud_au, rcore_au, nalpha, Mcloud_au, Mcluster_au, SFE)



        if i.output_verbosity >= 1:
            print('%.4f' % t0, "%.2f" % r0, "%.2f" % v0, "%.3f" % fabs, "%.4f" % fabs_i, "%.4f" % (dRs / r0), "%.2e" % Msh0, "%.2e" % tau_IR, "%.2e" % nmax, "%.3e" %(time.time()-start_time))

        if shell_dissolved == False and nmax < i.ndiss:
            shell_dissolved = True
            t_shdis = t0

        if (tStop_i > 3.0 and nmax < 1.0):
            # plot the shell structure, if density is very low to see whether something goes wrong #    
            shell_ploton = True
        else:
            shell_ploton = False

        # What phase is the expansion in? see phase_lookup
        # -1.0: collapse,
        # 2.1: expansion core (or uniform cloud)
        # 2.2: density gradient
        # 3.0: ambient medium
        phase0 = set_phase(r0, rcore_au, rcloud_au, t0, v0, dens_grad=i.density_gradient)[0]

        LBOL_ABS = Lbol * fabs
        PW = pdot
        MSTAR = Mcluster_au

        # case no collapse
        if phase0 > 0.:
            # temporary mass from which mass of the shell can be easily calculated on the fly in the ODE solver
            # Msh = M0T + 4.*pi/3. * rho_intercl_au * R**3 (see eq. in sec. phase III in paper)

            A_EXP = nalpha
            M0 = Msh0  # get mass from time step before (if shell had collapsed a bit before but is expanding now, this is important)
            RCORE = rcore_au
            RHOA = i.rhoa_au
            # Bundle parameters for ODE solver
            params = [M0, RHOA, LBOL_ABS, tau_IR, PW, MSTAR, RCORE, A_EXP, rcloud_au, Mcloud_au, SFE]

            psoln = scipy.integrate.odeint(f_mom_grad, y0, t, args=(params,))


        # case collapse
        elif phase0 == ph.collapse:
            # now solve eq. of motion for a shell of constant mass (assume that all the ambient material has been swept-up)
            # this part of the routine assumes astro-units
            M0 = Msh0

            # Bundle parameters for ODE solver
            params = [M0, LBOL_ABS, tau_IR, PW, MSTAR]

            psoln = scipy.integrate.odeint(f_collapse, y0, t, args=(params,))

        # get r, rdot and rdotdot
        r = psoln[:, 0]
        rd = psoln[:, 1]

        # check whether phase switch occures
        # if so, only take result up to phase switch
        phase1 = set_phase(r[-1], rcore_au, rcloud_au, tStop_i,  rd[-1], dens_grad=i.density_gradient)[0]
        if phase0 != phase1:
            phase = set_phase(r, rcore_au, rcloud_au, t0,  rd, dens_grad=i.density_gradient)
            in_phase0 = (r <= r[phase != phase0][0]) # find index of phase switch
            r = r[in_phase0]
            rd = rd[in_phase0]
            t = t[in_phase0]

        # only change mass if not collapsing
        if phase0 > 0.:
            if i.dens_profile == "powerlaw":
                #print('##############*************************#############r=',r)
                Msh = mass_profile.calc_mass(r, rcore_au, rcloud_au, i.rhoa_au, i.rho_intercl_au, nalpha, Mcloud_au)
                #print('##############*************************#############MshzuR=',Msh)
            
            elif i.dens_profile == "BonnorEbert":
                T_BE=rcore_au
                #print('##############*************************#############r=',r)
                Msh = mass_profile.calc_mass_BE(r, i.rhoa_au, T_BE, i.rho_intercl_au, rcloud_au, Mcloud_au)
                #print('##############*************************#############MshzuR=',Msh)
        elif phase0 == ph.collapse: #collapse
            Msh = Msh0 * r / r

        ################## CLOUDY ###########################################################################################
        if make_cloudy_model:
            jj_cloudy = find_nearest_higher(t, t_cloudy)
            r_cloudy = r[jj_cloudy]
            v_cloudy = rd[jj_cloudy]
            Msh_cloudy = Msh[jj_cloudy]
            # if density high enough or cloud not fully swept, make normal shell (+static) cloudy model
            if ((nmax > i.ndiss) or (max(np.concatenate([rSmom,r])) < rcloud_au)):
                # print("write cloudy model at t=", t_cloudy)
                Lw_cloudy = fLw_evo(t_cloudy) / c.L_cgs
                pdot_cloudy = fpdot_evo(t_cloudy) * c.Myr / (c.Msun * c.kms)
                vterminal_cloudy = 2. * Lw_cloudy / pdot_cloudy
                P_cloudy = state_eq.Pram(r_cloudy, Lw_cloudy, vterminal_cloudy)
                [n0_temp, n0_cloudy] = n_from_press(P_cloudy, i.Ti, B_cloudy=i.B_cloudy)
                # shell = True
                create_model(cloudypath, SFE, Mcloud_au, i.namb, i.Zism, n0_cloudy, r_cloudy, v_cloudy, Msh_cloudy,
                             np.log10(fLbol_evo(t_cloudy)), t_cloudy, rcloud_au, nedge,
                             SB99model=i.SB99cloudy_file, shell=i.cloudy_stopmass_shell, turb=i.cloudy_turb,
                             coll_counter=coll_counter, Tarr=Tarr, Larr=Larr, Li=fLi_evo(thalf) , Qi=fQi_evo(thalf),
                             pdot_tot=fpdot_evo(thalf), Lw_tot=fLw_evo(thalf), Mcluster=Mcluster_au, phase=phase0)
            else:
                # shell = False
                create_model(cloudypath, SFE, Mcloud_au, np.nan, i.Zism, np.nan, r_cloudy, v_cloudy, Msh_cloudy,
                             np.log10(fLbol_evo(t_cloudy)), t_cloudy, rcloud_au, nedge,
                             SB99model=i.SB99cloudy_file, shell=False, turb=i.cloudy_turb,
                             coll_counter=coll_counter, Tarr=Tarr, Larr=Larr, Li=fLi_evo(thalf) , Qi=fQi_evo(thalf),
                             pdot_tot=fpdot_evo(thalf), Lw_tot=fLw_evo(thalf), Mcluster=Mcluster_au, phase=ph.dissolve)
                #printl("Due to a very low shell density, I am not writing any cloudy input files, nmax = {}".format(nmax), verbose=1)
        #####################################################################################################################


        rmom_end = r[-1]
        vmom_end = rd[-1]
        tmom_end = t[-1]

        tSmom = np.concatenate([tSmom, [t0]])
        rSmom = np.concatenate([rSmom, [r0]])
        vSmom = np.concatenate([vSmom, [v0]])
        fabsmom = np.concatenate([fabsmom, [fabs]]) # TO DO: if there was a phase switch I actually need to do an interpolation for quantities marked with X
        fabs_i_mom = np.concatenate([fabs_i_mom, [fabs_i]]) # X
        fabs_n_mom = np.concatenate([fabs_n_mom, [fabs_n]])  # X
        dRs_mom = np.concatenate([dRs_mom, [dRs]]) # X
        ionshmom = np.concatenate([ionshmom, [ionsh]]) # X
        Mshell_mom = np.concatenate([Mshell_mom, [Msh0]])
        phase_mom = np.concatenate([phase_mom, [phase0]])

        n0mom = np.concatenate([n0mom, [n0]])
        n0_cloudymom = np.concatenate([n0_cloudymom, [n0_cloudy]])

        FSgrav_mom = np.concatenate([FSgrav_mom, [c.Grav_au * Msh0 / r0 ** 2 * (Mcluster_au + Msh0 / 2.)]])
        FSwind_mom = np.concatenate([FSwind_mom, [pdot - pdot_SNe]])
        FSradp_mom = np.concatenate([FSradp_mom, [fabs * Lbol / clight_au]]) #  X
        FSsne_mom = np.concatenate([FSsne_mom, [pdot_SNe]])
        FSIR_mom = np.concatenate([FSIR_mom, [fabs * Lbol / clight_au * tau_IR]]) # X
        nmax_mom = np.concatenate([nmax_mom, [nmax]]) # X
        logMcluster_mom = np.concatenate([logMcluster_mom, [np.log10(Mcluster_au)]])
        logMcloud_mom = np.concatenate([logMcloud_mom, [np.log10(Mcloud_au)]])
        Lwmom = np.concatenate([Lwmom, [Lw]])

        # increment stopping time: new start time is old stopping time
        #if phase0 == phase1:
        #    t0 = tStop_i
        #else: t0 = t[-1]
        t0 = t[-1]

        # if abs(t0-5.0)<0.1: shell_ploton = True
        # else: shell_ploton = False

        r0 = r[-1]
        v0 = rd[-1]
        Msh0 = Msh[-1]

        if (nmax < i.ndiss):
            # is this the first time the shell density drops very low? If so, set first_dissolve to True and mark the time
            if first_dissolve is False:
                first_dissolve = True
                tdis = tStop_i
            # If this is not the first time, i.e. the density has been low for some time, check how long ago the density dropped low. If long enough in the past (e.g. 1 c.Myr), stop simulation
            elif (tStop_i - tdis) >= i.dt_min_diss:
                printl("Shell has dissolved! Max. density of shell %.2f is below the threshold of %.2f." %(nmax, i.ndiss))
                break
        # if the density is high or the simulation is young, first_dissolve is reset to False
        # for break, continous low density is necessary
        else:
            first_dissolve = False

    Data_m = {'t': tSmom, 'r': rSmom, 'v': vSmom, 't_end': tmom_end, 'r_end': rmom_end, 'v_end': vmom_end,
              'fabs': fabsmom, 'Fgrav': FSgrav_mom, 'Fwind': FSwind_mom,
              'Fradp_dir': FSradp_mom, 'FSN': FSsne_mom, 'Fradp_IR': FSIR_mom,
              'fabs_i': fabs_i_mom, 'fabs_n': fabs_n_mom, 'dRs': dRs_mom, 'nmax': nmax_mom,
              'logMcluster': logMcluster_mom, 'logMcloud': logMcloud_mom, 'logMshell': np.log10(Mshell_mom),
              'phase': phase_mom, 'Lmech': Lwmom}

    #plt.plot(tSmom, n0mom,'b')
    #plt.plot(tSmom, n0_cloudymom,'r')
    #plt.show()

    return Data_m, shell_dissolved, t_shdis, ionshmom
