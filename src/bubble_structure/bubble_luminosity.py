 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:43:02 2023

@author: Jia Wei Teh

This is the newest version (after bubble_luminosity_neewithoutdict, because this has dict)
"""




# libraries
import numpy as np
import sys
import os
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
#--
import src._functions.operations as operations
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.cooling import net_coolingcurve
import src._functions.unit_conversions as cvt
from src._input.dictionary import (updateDict, DescribedItem)


# get parameter
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])


def get_bubbleproperties(b_params,
                         record = False,
        ):
    """
    Used in nearly all phases to calculate bubble luminosity.

    Print out the dictionary to check for their items and associated units via print(b_params)

    old code: calc_Lb(); i.e., get_bubbleLuminosity
    """
    
    print('entering get_bubbleproperties')    
    
    # =============================================================================
    # Step 1: Get necessary parameters, such as inner bubble radius R1 
    #           and pressure Pb
    # =============================================================================
    
    # velocity at r ---> 0.
    v0 = 0.0
    
    
    # initial radius of discontinuity [pc] (inner bubble radius)
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                               1e-3 * b_params['R2'].value, b_params['R2'].value, 
                               args=([b_params['L_wind'].value, b_params['Eb'].value, 
                                      b_params['v_wind'].value, b_params['R2'].value,
                                      ]))

    # The bubble Pbure [cgs - g/cm/s2, or dyn/cm2]
    Pb = get_bubbleParams.bubble_E2P(b_params['Eb'].value,
                                    b_params['R2'].value, 
                                    b_params['R1'].value)
    
    # update
    b_params['R1'].value = R1
    b_params['Pb'].value = Pb
    
    # =============================================================================
    # Step 2: Calculate dMdt, the mass flux from the shell back into the hot region
    # =============================================================================
    
    # ----------- prepare for calculation of dMdt
    
    # The mass flux from the shell back into the hot region (b, hot stellar wind)
    # if it isn't yet computed, set it via estimation from Equation 33 in Weaver+77.
    # Question: should this be mu_n instead?
 
    if np.isnan(b_params['dMdt'].value):
        # if not calculated, i.e., default np.nan, calculate it.
        dMdt_init = get_init_dMdt(b_params)
        # record
        b_params['dMdt'] = DescribedItem(dMdt_init, 'M_sun/Myr. mass loss from region c (shell) into region b (shocked winds) due to thermal conduction.')
        # output
        print(f'The initial guess for dMdt is {dMdt_init}.')
    else:
        # otherwise just use previous value as guess
        dMdt_init = b_params['dMdt'].value
    
    # This function evaluates dMdt guesses with the boundary conditions. 
    # Goes into loop and keeps evaluating until v0 (both estimated and true) agree.
    # This will yield a residual dMdt (which is nearly zero).
    # old code: compare_boundaryValues()
    # I presume Weaver meant: v -> 0 for R -> R1 (since in that chapter he talks about R1 being very small)
    # TODO: this shouldn't be 4e5. In the old code of bubble_structure2 line 310, T prime is set awayfrom Tgoal, because of dR2. 
    # old code: R_small
    r_inner = R1
    # change dR2 < _dR2min: too.
    T_goal = 3e4  
    
    
    # -- REMOVED a huge section of calculating xi_Tb. 
    xi_Tb = warpfield_params.xi_Tb
    # calculate rgoal
    rgoal = xi_Tb * b_params['R2'].value
    
    # sanity check: rgoal cannot be smaller than the inner bubble radius R1
    if rgoal < R1:
        # for now, stop code. 
        # However, could also say that energy-driven phase is over if rgoal_f < R1/R2, 
        # for a reasonable rgoal_f (0.99 or so)
        sys.exit(f'rgoal ({rgoal}) is smaller than the inner bubble radius {R1}. Consider increasing xi_Tb.')
    
    # record
    b_params['r_goal'].value = rgoal
    b_params['T_goal'].value = T_goal
    b_params['r_inner'].value = r_inner
    
    # update
    updateDict(b_params, ['r_goal', 'T_goal', 'r_inner'],
               [rgoal, T_goal, r_inner])
    
    # ----------- calculation of dMdt
    # This dictionary takes parameters in astronomical units; e.g., pc, yr, M_sun.

    # TODO: move this to further out of the file
    #here we follow the original code and use mu_p, but maybe we should use mu_n since the region is ionised?
    b_params['mu_au'] = DescribedItem(b_params['mu_p_au'].value, 'Msun')

    # print('b_params in bubble lum before dMdt', b_params)

    # While evaluating for dMdt, we create a global variable to also store the values for
    # temperature and velocity. 
    # This is so that we don't have to run the function again for the final values. 
    # We note here that T_array, the temperature structure, is in increasing order.
    # This means r_array, the radius, is in DECREASING order. 
    
    # Now we dont need global array, since we have dictionaries
    # global v_array, T_array, dTdr_array, r_array
    # v_array = np.array([])
    # T_array = np.array([])
    # dTdr_array = np.array([])
    # r_array = np.array([])
     
    # in u.M_sun/u.yr
    print('\n\n---dMdt_init befrore get_velocity_residuals---', dMdt_init)
    
    # ----------- calculation of structures, such as T_array, v_array, dTdr_array
    
    # begin a new solver here
    # dMdt_params = copy.deepcopy(b_params)
    
    # Because sometimes dMdt guesses are WILD (they sometimes go +- 500), and this
    # breaks the whole simulation, lets impose some limits here, utilising lq method.
    # initially dMdt will spike before it goes to equilibrium (before momentum phase). 
    # Lets set an if/else structure for this. 
    
    if b_params['current_phase'].value == '1a':
        print('first phase dMdt')
        dMdt_predict = scipy.optimize.least_squares(get_velocity_residuals,  dMdt_init, args = (b_params,), 
                                              # dMdt will increase.
                                              bounds = [0, np.inf],
                                              xtol = 1e-3,
                                              # ftol = 1e-4
                                              ).x[0] # u.M_sun/u.Myr
        
    elif b_params['current_phase'].value == '1b':
        # dMdt_predict = scipy.optimize.least_squares(get_velocity_residuals,  dMdt_init, args = (b_params,), 
        #                                       # at this point it will start to decrease.
        #                                       # should we set maximum limit as dMdt_init * 1.05?
        #                                       bounds = [dMdt_init * 0.70, dMdt_init * 1.05],
        #                                       xtol = 1e-4,
        #                                       # ftol = 1e-4
        #                                       ).x[0] # u.M_sun/Mu.yr
        dMdt_predict = scipy.optimize.fsolve(get_velocity_residuals,  dMdt_init, args = (b_params,), 
                                              # at this point it will start to decrease.
                                              # should we set maximum limit as dMdt_init * 1.05?
                                              # bounds = [dMdt_init * 0.70, dMdt_init * 1.05],
                                              xtol = 1e-4,
                                              factor = 50,
                                              epsfcn = 1e-4,
                                              # ftol = 1e-4
                                              )[0] # u.M_sun/u.Myr
        
        print('dMdt_predict', dMdt_predict)
        
    # ---- Old method
    # # problem: using old dMdt also implies using old varray and rarray, which causes problem
    # # since in the newer iterations R2 would have been expanded and thus larger, however
    # # the rarray still stays small since it still stays in the previous calculatin.
    # # What shouold we do?
    
    # # get value to avoid calling many times, also this skips terrible runs
    # # grab the recent most successful runs. These values are recorded in get_velocity_residuals().
    # v_array = b_params['bubble_v_arr'].value
    # T_array = b_params['bubble_T_arr'].value
    # dTdr_array = b_params['bubble_dTdr_arr'].value
    # r_array = b_params['bubble_r_arr'].value
    # dMdt = b_params['bubble_dMdt'].value
    # b_params['dMdt'].value = dMdt
    
    # There is a problem with this. 27.01.25. Which is that if nothing works,
    # the r_array will be called from previous runs, i.e., the radious of previous runds.
    # This meaans that the radius will be smaller since the bubble has been expanding,
    # and because rgoal is calculated here early in the function, we have situations such 
    # that rgoal > max(r_array), which obviously will not make sense. 
    # How should we fix this?
    # ---- New method
    b_params['dMdt'].value = dMdt_predict
    
    # here we instead use the whole function from get_velocity residuals and recalculate the arrays
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = get_bubble_ODE_initial_conditions(b_params['dMdt'].value, b_params)
    r_array = ( (r2Prime + b_params['r_inner'].value)\
               -  np.logspace(np.log10(b_params['r_inner'].value),\
              np.log10(r2Prime), int(2e4))) 
    r_improve_resolution_high_r = np.logspace(np.log10(r_array[0]), np.log10(r_array[2]), int(2e4))
    r_array = np.insert(r_array[3:], 0, r_improve_resolution_high_r)
    r_further_resolution = (r_array[-1] + r_array[-5]) - np.logspace(np.log10(r_array[-1]), np.log10(r_array[-5]), int(2e4))
    r_array = np.insert(r_array[:-5], len(r_array[:-5]), r_further_resolution)
    # Now we run the ODE solver.
    psoln = scipy.integrate.odeint(get_bubble_ODE, [v_r2Prime, T_r2Prime, dTdr_r2Prime], r_array, 
                                    args=(b_params,), tfirst=True,
                                    )
    # record arrays. 
    v_array = psoln[:, 0] 
    T_array = psoln[:, 1]  
    dTdr_array = psoln[:, 2] 
    
    b_params['bubble_v_arr'].value = v_array
    b_params['bubble_T_arr'].value = T_array
    b_params['bubble_dTdr_arr'].value = dTdr_array
    b_params['bubble_r_arr'].value = r_array
    
    # ---- end of New method
    
    
    # grab the most successful runs
    print('final array selection for bubble_luminosity:')
    print('final v_array:', v_array)
    print('final T_array:', T_array)
    print('final dTdr_array:', dTdr_array)
    print('final r_array:', r_array)
    # print('post dmdt:', dMdt_predict)
    # print('final dmdt:', dMdt)
    
    # calculate densit
    n_array = b_params['Pb'].value/(2 * b_params['k_B_au'].value * T_array)
    
    # update
    b_params['bubble_n_arr'].value = n_array
    

    # =============================================================================
    # Step 3: we identify which index in the temperature array has cooling and which doesn't. 
    # The bubble will have these regions:
    #   1. Low resolution (bubble) region. This is the CIE region, where T > 10**5.5 K.
    #   2. High resolution (conduction zone) region. This is the non-CIE region.
    #   3. Intermediate region. This is between 1e4 and T[index_cooling_switch].
    # -----
    # Goal: calculate power-loss (luminosity) in these regions due to cooling. To do this
    #       for each zone we calculate T, dTdr and n. 
    # Remember r is monotonically decreasing, so temperature increases!
    # 
    # 
    # Two things to make sure in this section:
    #   1. All cooling calculations take in au values, but the inner operations and outputs are cgs.
    #      The exception is get_dudt(), which takes in au and returns in au.
    #   2. Dobule check units when using interpolation functions; some of them also only take log10 or 10^.
    # 
    # =============================================================================

    # Temperature at which any lower will have no cooling
    _coolingswitch = 10**4  
    # Temperature of switching between CIE and non-CIE. Temperatures higher than 
    # _CIEswitch results in switching on CIE.
    _CIEswitch = 10**5.5  

    #---------------- 0. Prep: insert entry at exactly _CIEswitch via interpolation
    
    # index of radius array at which T is closest (and higher) to _CIEswitch
    index_CIE_switch = operations.find_nearest_higher(T_array, _CIEswitch)
    # index of radius array at which T is closest (and higher) to _coolingswitch
    index_cooling_switch = operations.find_nearest_higher(T_array, _coolingswitch)
    
    if any(b_params['bubble_T_arr'].value < 0):
        sys.exit('negative temperature detected.')
  
    
    # interpolate so that we have an entry at exactly _CIEswitch
    if index_cooling_switch != index_CIE_switch:
        # extra points to add
        _xtra = 20
        # array sliced from beginning until somewhere after _CIEswitch
        r_interpolation_bubble = r_array[:index_CIE_switch+_xtra]
        # interpolation function for T and dTdr.
        fdTdr_interp_bubble = interp1d(r_interpolation_bubble, dTdr_array[:index_CIE_switch+_xtra], kind='linear')
        # subtract so that it is zero at _CIEswitch
        fT_interp_bubble = interp1d(r_interpolation_bubble, (T_array[:index_CIE_switch+_xtra] - _CIEswitch), kind='cubic')
        
        # calculate quantities
        r_CIEswitch = scipy.optimize.brentq(fT_interp_bubble, np.min(r_interpolation_bubble), np.max(r_interpolation_bubble), xtol=1e-8) 
        n_CIEswitch = b_params['Pb'].value/(2 * b_params['k_B_au'].value * _CIEswitch)
        dTdr_CIEswitch = fdTdr_interp_bubble(r_CIEswitch)
        
        # insert into array
        T_array = np.insert(T_array, index_CIE_switch, _CIEswitch)
        r_array = np.insert(r_array, index_CIE_switch, r_CIEswitch)
        n_array = np.insert(n_array, index_CIE_switch, n_CIEswitch)
        dTdr_array = np.insert(dTdr_array, index_CIE_switch, dTdr_CIEswitch)


    #---------------- 1. Bubble. Low resolution region, T > 10**5.5 K. CIE is used. 
    
    # r is monotonically decreasing, so temperature increases
    T_bubble = T_array[index_CIE_switch:]
    r_bubble = r_array[index_CIE_switch:]
    n_bubble = n_array[index_CIE_switch:]
    dTdr_bubble = dTdr_array[index_CIE_switch:]

    # import values from two cooling curves
    cooling_CIE_interpolation = b_params['cStruc_cooling_CIE_interpolation'].value
    # cooling rate [au]
    Lambda_bubble = 10**(cooling_CIE_interpolation(np.log10(T_bubble))) * cvt.Lambda_cgs2au
    
    integrand_bubble = n_bubble**2 * Lambda_bubble * 4 * np.pi * r_bubble**2
    # calculate power loss due to cooling
    L_bubble = np.abs(np.trapz(integrand_bubble, x = r_bubble))
    # intermediate result for calculation of average temperature [K pc3]
    Tavg_bubble = np.abs(np.trapz(r_bubble**2 * T_bubble, x = r_bubble))
    

    #---------------- 2. Conduction zone. High resolution region, 10**4 < T < 10**5.5 K. 
    
    # it is possible that index_cooling_switch = index_CIE_switch = 0 if the shock front is very steep.
    if index_cooling_switch != index_CIE_switch:
        # if this zone is not well resolved, solve ODE again with high resolution (IMPROVE BY ALWAYS INTERPOLATING)
        if index_CIE_switch - index_cooling_switch < 100:
            
            # print('inside cz, not well-resolved')
            
            # This is the original array that is too short
            lowres_r_conduction = r_array[:index_CIE_switch+1]
            # print(f'lowres_r_conduction: {lowres_r_conduction}')
            # Find the minimum and maximum value
            original_rmax = max(lowres_r_conduction)
            original_rmin = min(lowres_r_conduction)
            
            # how many intervales in high-res version? [::-1] included because r is reversed.
            _highres = 1e2
            r_conduction = np.arange(original_rmin, original_rmax,
                                       (original_rmax - original_rmin)/_highres
                                       )[::-1] # * u.pc
            
            # rerun structure with greater precision
            # solve ODE again, though there should be a better way (event finder!)
            psoln = scipy.integrate.odeint(get_bubble_ODE, [v_array[index_cooling_switch],T_array[index_cooling_switch],dTdr_array[index_cooling_switch]], r_conduction,
                                            args = (b_params,), tfirst=True) 
            
            # solutions
            v_conduction = psoln[:,0] 
            T_conduction = psoln[:,1] 
            dTdr_conduction = psoln[:,2]        
            
            # Here, something needs to be done. Because of the precision of the solver, 
            # it may return temperature with values > 10**5.5K eventhough that was the maximum limit (i.e., 10**5.500001).
            # This will crash the interpolator. To fix this, we simple shave away values in the array where T > 10**5.5, 
            # and concatenate to the low-rez limit. 
            
            # Actually, the final value may not be required; the value is already included
            # in the first zone, so we don't have to worry about them here.
            
            _Tmask = T_conduction < (10**5.5)
            # apply mask
            r_conduction = r_conduction[_Tmask]
            v_conduction = v_conduction[_Tmask] 
            T_conduction = T_conduction[_Tmask]  
            dTdr_conduction = dTdr_conduction[_Tmask]  
            
            # value at 1e4 K
            dTdR_coolingswitch = dTdr_conduction[0]

        else:
            r_conduction = r_array[:index_CIE_switch+1] 
            T_conduction = T_array[:index_CIE_switch+1]
            dTdr_conduction = dTdr_array[:index_CIE_switch+1]
            # value at 1e4 K
            dTdR_coolingswitch = dTdr_conduction[0]            
            
            
        #--begin unit check here
        # non-CIE is required here
        # import values from two cooling curves
        cooling_nonCIE = b_params['cStruc_cooling_nonCIE'].value 
        heating_nonCIE = b_params['cStruc_heating_nonCIE'].value 
        # calculate array [au]
        n_conduction = b_params['Pb'].value/(2 * b_params['k_B_au'].value * T_conduction)
        phi_conduction = b_params['Qi'].value / (4 * np.pi * r_conduction**2)
        
        # cooling rate [cgs]
        cooling_conduction = 10 ** cooling_nonCIE.interp(np.transpose(np.log10([n_conduction / cvt.ndens_cgs2au, T_conduction, phi_conduction / cvt.phi_cgs2au])))
        heating_conduction = 10 ** heating_nonCIE.interp(np.transpose(np.log10([n_conduction / cvt.ndens_cgs2au, T_conduction, phi_conduction / cvt.phi_cgs2au])))
        # net cooling rate [au]
        dudt_conduction = (heating_conduction - cooling_conduction) * cvt.dudt_cgs2au
        # integrand [au]
        integrand_conduction = (dudt_conduction * 4 * np.pi * r_conduction**2)
        # calculate power loss due to cooling [au]
        L_conduction = np.abs(np.trapz(integrand_conduction, x = r_conduction))
        # intermediate result for calculation of average temperature
        Tavg_conduction = np.abs(np.trapz(r_conduction**2 * T_conduction, x = r_conduction))
    # if there is no conduction; i.e., the shock front is very steep. 
    elif index_cooling_switch == 0 and index_CIE_switch == 0:
        # the power loss due to cooling in this region will simply be zero. 
        L_conduction = 0 
        dTdR_coolingswitch = dTdr_bubble[0]
        
        
    #---------------- 3. Region between 1e4 K and T_array[index_cooling_switch]
    
    # If R2_prime is very close to R2 (i.e., where T ~ 1e4K), then this region is tiny (or non-existent)
    R2_coolingswitch = (_coolingswitch - T_array[index_cooling_switch])/dTdR_coolingswitch + r_array[index_cooling_switch]
    # assert R2_coolingswitch < r_array[index_cooling_switch], "Hmm? in region 3 of bubble_luminosity"
    # interpolate between R2_prime and R2_1e4, important because the cooling function varies a lot between 1e4 and 1e5K (R2_prime is above 1e4)
    fT_interp_intermediate = interp1d(np.array([r_array[index_cooling_switch], R2_coolingswitch]), 
                                      np.array([T_array[index_cooling_switch], _coolingswitch]), kind = 'linear')
    # get values
    r_intermediate = np.linspace(r_array[index_cooling_switch], R2_coolingswitch, num = 1000, endpoint=True) 
    T_intermediate = fT_interp_intermediate(r_intermediate) 
    n_intermediate =  b_params['Pb'].value/(2 * b_params['k_B_au'].value * T_intermediate)
    phi_intermediate = b_params['Qi'].value / (4 * np.pi * r_intermediate**2)
    # get cooling, taking into account for both CIE and non-CIE regimes
    regime_mask = {'non-CIE': T_intermediate < _CIEswitch, 'CIE': T_intermediate >= _CIEswitch}
    L_intermediate = {}
    for regime in ['non-CIE', 'CIE']:
        # masks
        mask = regime_mask[regime]
        
        if regime == 'non-CIE':
            # import values from cooling curves
            cooling_nonCIE = b_params['cStruc_cooling_nonCIE'].value 
            heating_nonCIE = b_params['cStruc_heating_nonCIE'].value 
            # cooling rate
            cooling_intermediate = 10 ** cooling_nonCIE.interp(np.transpose(np.log10([n_intermediate[mask] / cvt.ndens_cgs2au, T_intermediate[mask], phi_intermediate[mask] / cvt.phi_cgs2au])))
            heating_intermediate = 10 ** heating_nonCIE.interp(np.transpose(np.log10([n_intermediate[mask] / cvt.ndens_cgs2au, T_intermediate[mask], phi_intermediate[mask] / cvt.phi_cgs2au])))
            # [au]
            dudt_intermediate = (heating_intermediate - cooling_intermediate)  * cvt.dudt_cgs2au
            integrand_intermediate = dudt_intermediate * 4 * np.pi * r_intermediate[mask]**2
        elif regime == 'CIE':
            # import values from cooling curves
            cooling_CIE_interpolation = b_params['cStruc_cooling_CIE_interpolation'].value
            # [au]
            Lambda_intermediate = 10**(cooling_CIE_interpolation(np.log10(T_intermediate[mask]))) * cvt.Lambda_cgs2au
            integrand_intermediate = n_intermediate[mask]**2 * Lambda_intermediate * 4 * np.pi * r_intermediate[mask]**2
        # calculate power loss due to cooling
        L_intermediate[regime] = np.abs(np.trapz(integrand_intermediate, x = r_intermediate[mask]))
        
    # sum for both regions
    L_intermediate = L_intermediate['non-CIE'] + L_intermediate['CIE']
    # intermediate result for calculation of average temperature
    Tavg_intermediate =  np.abs(np.trapz(r_intermediate**2 * T_intermediate,  x = r_intermediate))

    #---------------- 4. Finally, sum up across all regions. Calculate the average temeprature.
    # this was Lb in old code
    L_total = L_bubble + L_conduction + L_intermediate
    
    # calculate temperature
    # with conduction zone
    if index_cooling_switch != index_CIE_switch:
        Tavg = 3 * ( Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +\
                    Tavg_conduction / (r_conduction[0]**3 - r_conduction[-1]**3) +\
                    Tavg_intermediate / (r_intermediate[0]**3 - r_intermediate[-1]**3))
    # without conduction zone
    else:
        Tavg = 3. * ( Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +\
                     Tavg_intermediate / (r_intermediate[0]**3 - r_intermediate[-1]**3))

    # Remember that r_array is in decreasing order. 
    # If rgoal is smaller than the radius of cooling threshold, i.e., larger than the index,
    if rgoal > r_array[index_cooling_switch]: # looking for the smallest value in r_cz
        # take interpolation
        T_rgoal = fT_interp_intermediate(rgoal)
    # if rgoal is instead at the point of CIE-nonCIE switch,
    elif rgoal > r_array[index_CIE_switch]: # looking for the largest value in r_cz
        idx = operations.find_nearest(r_conduction, rgoal)
        T_rgoal = T_conduction[idx] + dTdr_conduction[idx]*(rgoal - r_conduction[idx])
    # otherwise, interpolate. 
    else:
        idx = operations.find_nearest(r_bubble, rgoal)
        T_rgoal = T_bubble[idx] + dTdr_bubble[idx]*(rgoal - r_bubble[idx])
    
    
    # =============================================================================
    # Step 4: Mass/gravitational potential
    # =============================================================================
    
    def get_mass_and_grav(n, r):
        # again: r and n is monotonically decreasing. We need to flip it here to avoid problems with np.cumsum.
        
        # r is now monotonically increasing
        r_new = r[::-1] #.to(u.cm)
        # so is n (rho) now
        # old code says * mp, but it should be mu. 
        rho_new = n[::-1] * b_params['mu_au'].value
        rho_new = rho_new #.to(u.g/u.cm**3)å
        # get mass 
        m_new = 4 * np.pi * scipy.integrate.simps(rho_new * r_new**2, x = r_new)  
        # cumulative mass 
        m_cumulative = np.cumsum(m_new)
        # gravitational potential [Msun/pc]
        grav_phi = - 4 * np.pi * b_params['G_au'].value * scipy.integrate.simps(r_new * rho_new, x = r_new)  
        # gravitational force per mass
        grav_force_pmass = b_params['G_au'].value * m_cumulative / r_new**2
        
        return m_cumulative, grav_phi, grav_force_pmass
    
    # gettemåå
    m_cumulative, grav_phi, grav_force = get_mass_and_grav(n_array, r_array)
    
    # TODO:
    # bubble mass
    mBubble = m_cumulative[-1]
    # here is what was in the old code. This was wrong. 
    
    # update dictionary here
    # T_rgoal will be T0 in the output in run_energy_phase.py
    b_params['bubble_L_total'].value = L_total
    b_params['bubble_T_rgoal'].value = T_rgoal
    # should this be here?
    # b_params['T0'].value = T_rgoal
    b_params['bubble_L_bubble'].value = L_bubble
    b_params['bubble_L_conduction'].value = L_conduction
    b_params['bubble_L_intermediate'].value = L_intermediate
    b_params['bubble_Tavg'].value = Tavg
    b_params['bubble_mBubble'].value = mBubble
    
    
    return b_params


# =============================================================================
# Initial guess of dMdt
# =============================================================================

def get_init_dMdt(b_params):
    
    """
    This function provides an initial guess for dMdt, dMdt is    
    the mass flux from the shell back into the hot region (b, hot stellar wind)
    
    ref: Equation 33 in Weaver+77.
    """
    
    dMdt_init = 12 / 75 * b_params['dMdt_factor'].value**(5/2) * 4 * np.pi * b_params['R2'].value**3 / b_params['t_now'].value\
        * b_params['mu_p_au'].value / b_params['k_B_au'].value * (b_params['t_now'].value * b_params['c_therm_au'].value / b_params['R2'].value**2)**(2/7) * b_params['Pb'].value**(5/7)
     
    # Msol/Myr
    return dMdt_init


def get_velocity_residuals(dMdt_init, dMdt_params_au):
    """
    This routine calculates the value for dMdt, by comparing velocities at boundary condition.
    Check out get_bubble_ODE_initial_conditions() below, for full description.

    Parameters
    ----------
    dMdt_init: Guesses of dMdt [Msun/Myr]
    
    dMdt_params_au : dictionary

    # old code: find_dMdt()

    If this function crashes, checkout yourrun_dictionary.json for the profiles.
    """
    
   
    # =============================================================================
    # Get initial bubble values for integration  
    # =============================================================================
    # parameter input is in cgs, calculation is done in cgs
    # Watch out! these results are unitless (in units of pc, K, K/pc, and pc/Myr)
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = get_bubble_ODE_initial_conditions(dMdt_init, dMdt_params_au)
    
    # =============================================================================
    # radius array at which bubble structure is being evaluated.
    # =============================================================================
    
    # ----- radius creation -----

    # array is monotonically decreasing, and these values are all in pc.
    
    # Step 1: create array sampled at higher density at larger radius i.e., more datapoints near bubble's outer edge (sort of a reverse logspace).
    # [10, 9,.7, 9.3, 8.8, 8.2, 7.4, 6.4, 5, 3, 1]
    r_array = ( (r2Prime + dMdt_params_au['r_inner'].value)\
               -  np.logspace(np.log10(dMdt_params_au['r_inner'].value),\
              np.log10(r2Prime[0]), int(2e4))) 
        
    # Step 2: add front-heavy, i.e., 1, 1.2, 1.6, 2, 3, 4, 5, 7, 10.
    # new r_array with improved resolution at high r.
    r_improve_resolution_high_r = np.logspace(np.log10(r_array[0]), np.log10(r_array[2]), int(2e4))
    r_array = np.insert(r_array[3:], 0, r_improve_resolution_high_r)
    
    # Step 3: further front-heavy for end of array
    r_further_resolution = (r_array[-1] + r_array[-5]) - np.logspace(np.log10(r_array[-1]), np.log10(r_array[-5]), int(2e4))
    r_array = np.insert(r_array[:-5], len(r_array[:-5]), r_further_resolution)
        
    # ----- radius completed -----

    # =============================================================================
    # Now we run the ODE solver.
    # =============================================================================

    # tfirst = True because get_bubble_ODE() is defined as f(t, y). 
    psoln = scipy.integrate.odeint(get_bubble_ODE, [v_r2Prime[0], T_r2Prime[0], dTdr_r2Prime[0]], r_array, 
                                    args=(dMdt_params_au,), tfirst=True,
                                    )
    
    # record arrays. 
    v_array = psoln[:, 0] 
    T_array = psoln[:, 1]  
    dTdr_array = psoln[:, 2] 

    # --- v0 is the velocity at r -> 0.
    # v_array[-1] is the estimate of v0, given the guess of dMdt. 
    # add 1e-4 to avoid division by zero.
    residual = (v_array[-1] - dMdt_params_au['v0'].value) / (v_array[0] + 1e-4)
    
    
    
    # TODO: make more physically or numerically motivated values for residuals. 
    
    # check to avoid having very low temperature
    min_T = np.min(T_array)
    if min_T < 3e3:
        print('Rejected. minimum temperature:', min_T)
        residual *= (3e4/(min_T+1e-1))**2 # in case min_T is zero.
        return residual
    
    if np.isnan(min_T):
        print('Rejected. minimum temperature:', min_T)
        return -1e3
    
    if not operations.monotonic(T_array):
        print('temperature not monotonic')
        return 1e2

    # is this necessary? does velocity have to be monotonic?
    
    # if not operations.kindof_decreasing(v_array):
    #     import matplotlib.pyplot as plt
    #     plt.plot(r_array, v_array)
    #     plt.show()
    #     print('velocity not decreasing')
    #     return residual
    

    print('record, and min temp is', min_T)
    
    # otherwise record
    # TODO: maybe this part need to change during recollapse.
    # v_array[v_array < 0] = 0
    
    # dMdt_params_au['bubble_v_arr'].value = v_array
    # dMdt_params_au['bubble_T_arr'].value = T_array
    # dMdt_params_au['bubble_dTdr_arr'].value = dTdr_array
    # dMdt_params_au['bubble_r_arr'].value = r_array
    # dMdt_params_au['bubble_dMdt'].value = dMdt_init[0]
    # dMdt_params_au['v0_residual'].value = residual
    
    # return
    return residual
    
    
    
def get_bubble_ODE_initial_conditions(dMdt, dMdt_params_au):
    """
    dMdt_init (see above) can be used as an initial estimate of dMdt, 
    which is then adjusted until the velocity found by numerical integration (see get_velocity_residuals()) 
    remains positive and less than alpha*r/t at some chosen small radius. 
    
    For each value of dMdt, the integration of equations (42) and (43) - in get_bubbleODEs() - 
    can be initiated at a <<radius r>> slightly less than R2 by using these
    three relations for:
        T, dTdr, and v. 
        
    old code: r2_prime is R2_prime in old code.
    old code: get_start_bstruc (calc_bstruc_start)

    Params:
    -------
    dMdt : [M_sun/Myr]
        mass loss from region c (shell) into region b (shocked winds) due to thermal conduction.

    Returns
    -------
    r2_prime [pc]: the small radius (slightly smaller than R2) at which these values are evaluated.
    T [K]: T(r), ODE.
    dTdr [K/pc]: ODE.
    v [pc/Myr]: ODE.
    """
    
    
    # Important question: what is mu?
    # here we follow the original code and use mu_p, but maybe we should use mu_n since the region is ionised?
    
    # -----
    # r has to be calculated, via a temperature goal (usually 3e4 K). 
    # dR2 = (R2 - r), in Equation 44
    # -----
    
    constant = 25/4 * dMdt_params_au['k_B_au'].value / dMdt_params_au['mu_au'].value /  dMdt_params_au['c_therm_au'].value
    # old code: r is R2_prime, i.e., radius slightly smaller than R2. 
    # TODO: For very strong winds (clusters above 1e7 Msol), this number heeds to be higher!
    # if np.isnan(dMdt_params_au['bubble_T_rgoal'].value):
    #     dR2 = dMdt_params_au['T_goal'].value**(5/2) / (constant * dMdt / (4 * np.pi * dMdt_params_au['R2'].value**2) )
    # else:
    #     dR2 = dMdt_params_au['bubble_T_rgoal'].value**(5/2) / (constant * dMdt / (4 * np.pi * dMdt_params_au['R2'].value**2) )
    dR2 = dMdt_params_au['T_goal'].value**(5/2) / (constant * dMdt / (4 * np.pi * dMdt_params_au['R2'].value**2) )
    
    print('dMdt_params_au["Tgoal"] in get_bubble_ODE_initial_conditions to check when it switches away from 3e4:', dMdt_params_au["T_goal"])

    # Possibility: extremely small dR2. I.e., at mCluster~1e7, dR2 ~1e-11pc. 
    # What is the minimum dR2? Set a number here
    # _dR2min = 1e-7 # pc 
    # print('dr', dR2)
    # if dR2 < _dR2min:
    #     dR2 = _dR2min * np.sign(dR2)
    
    
    # -----
    # Now, write out the estimation equations for initial conditions for the ODE (Eq 42/43)
    # -----
    # Question: I think mu here should point to ionised region
    # T(r)
    
    # temperature
    T = (constant * dMdt * dR2/ (4 * np.pi * dMdt_params_au['R2'].value**2))**(2/5)  #* dR2**(2/5) 

    # v(r) (u.pc/u.Myr)
    # TODO: what is mu_au? in old code its 0.5 * m_p
    v = dMdt_params_au['alpha'].value * dMdt_params_au['R2'].value / dMdt_params_au['t_now'].value - dMdt / (4 * np.pi * dMdt_params_au['R2'].value**2) * dMdt_params_au['k_B_au'].value * T / dMdt_params_au['mu_au'].value / dMdt_params_au['Pb'].value 
    # T'(r) (u.K / u.pc)
    dTdr = (- 2 / 5 * T / dR2) 
    # Finally, calculate r for future use (u.pc)
    r2_prime = (dMdt_params_au['R2'].value - dR2) 
    
    # return values without units.
    # with units they would be r2_prime.to(u.pc).value, T, dTdr.to(u.K/u.pc).value, v.to(u.pc/u.Myr).value
    return r2_prime, T, dTdr, v
    


    
def get_bubble_ODE(r_arr, initial_ODEs, dMdt_params_au):
    """
    Here is the main function that deals with ODE calculation.
    
    The main bulk of the entire simulation. The goal is to estimate the right dMdt.

    Parameters
    ----------
    r_arr : [pc]
        radius at which the ODE is solved.
    initial_ODEs : v_r2Prime, T_r2Prime, dTdr_r2Prime
        These are initial guesses for the ODE, obtained via get_bubble_ODE_initial_conditions().
    dMdt_params_au : 
        Paramerers required to run the ODE. See the main function, get_bubbleproperties() for more.

    Returns
    -------
    dvdr : [pc/Myr/pc]
        distance derivative of velocity.
    dTdr : [K/pc]
        distance derivative of temperature.
    dTdrr : [K/pc**2]
        second distance derivative of temperature.
        
    old code: calc_cons() and get_bubble_ODEs() aka bubble_struct()
    """
    
    # unravel
    v, T, dTdr = initial_ODEs
    
    # semi-correct cooling at low T
    if np.abs(T - 0) < 1e-5:
        print('T is zero')
        sys.exit()
        
    # if T < 10**3.61:
    #     T = 10**3.61
        
    # get density and ionising flux
    # 1/pc3
    ndens = dMdt_params_au['Pb'].value / (2 * dMdt_params_au['k_B_au'].value * T)
    # 1/Myr/pc2
    phi = dMdt_params_au['Qi'].value / (4 * np.pi * r_arr**2)
    
    if np.isnan(T):
        print('Getting np.nan Temeprature')
        print(dMdt_params_au['t_now'].value, ndens, T, phi, v, dTdr)
    
    # dudt is [M_sun/pc/yr3] (erg/cm3/s), because cooling is in units of (erg cm3/s) [M_sun*pc5/s3] 
    dudt = net_coolingcurve.get_dudt(dMdt_params_au['t_now'].value, ndens, T, phi, dMdt_params_au)
    
    # v - a*r but try with right units
    v_term = dMdt_params_au['alpha'].value * r_arr / (dMdt_params_au['t_now'].value)
    
    # change in temeprature gradient
    # old code: dTdrd
    dTdrr = dMdt_params_au['Pb'].value/(dMdt_params_au['c_therm_au'].value * T**(5/2)) * (
        (dMdt_params_au['beta'].value + 2.5 * dMdt_params_au['delta'].value) / dMdt_params_au['t_now'].value   +\
            2.5 * (v - v_term) * dTdr / T - dudt/dMdt_params_au['Pb'].value
        ) - 2.5 * dTdr**2 / T - 2 * dTdr / r_arr

    # velocity gradient
    # old code: vd
    dvdr = (dMdt_params_au['beta'].value + dMdt_params_au['delta'].value) / dMdt_params_au['t_now'].value + (v - v_term) * dTdr / T - 2 *  v / r_arr
    
    return [dvdr, dTdr, dTdrr]
    





