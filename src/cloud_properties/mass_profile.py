#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:58 2022

@author: Jia Wei Teh

This script contains function which computes the mass profile of cloud.
"""

import numpy as np
from src._functions import operations
import scipy.integrate
from src.cloud_properties import bonnorEbertSphere

def get_mass_profile( r_arr, params,
                          return_mdot,
                         **kwargs
                         ):
    """
    Given radius r, and assuming spherical symmetry, calculate the swept-up mass
    at r. I.e., if r is an array, each point in the returned array is the mass contained
    within radius r. 
    

    Parameters
    ----------
    r_arr [pc]: list/array of radius of interest
    params: dictionary
    
    return_mdot: boolean { True | False }
        Whether or not to also compute the time-derivative of mass.
        If True, then further specify an array of velocity.
        - **kwargs -> rdot_arr: None or an array of float
            Time-derivative of radius (i.e., shell velocity). (dr/dt)

    Returns
    -------
    mGas [Msol]: array of float
        The mass profile. 
    mGasdot [Msol/yr]: array of float. Only returned if return_mdot == True.
        The time-derivative mass profile dM/dt. 

    """
    
    # get values
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    mu_neu = params['mu_neu'].value
    mu_ion = params['mu_ion'].value
    mCloud = params['mCloud'].value
    rCloud = params['rCloud'].value
    rCore = params['rCore'].value
    
    if type(r_arr) is not np.ndarray:
        r_arr = np.array([r_arr])
        
    # Setting up values for mass density (from number density) 
    rhoCore = nCore * mu_ion
    rhoISM = nISM * mu_neu
    
    # initialise arrays
    mGas = np.zeros_like(r_arr)  
    mGasdot = np.zeros_like(r_arr) 
    
    # =============================================================================
    # Power-law profile
    # =============================================================================
    if params['dens_profile'].value == 'densPL':
        alpha = params['densPL_alpha'].value
        # ----
        # Case 1: The density profile is homogeneous, i.e., alpha = 0
        if alpha == 0:
            # sphere
            mGas =  4 / 3 * np.pi * r_arr**3 * rhoCore
            # outer region
            mGas[r_arr > rCloud] =  mCloud + 4. / 3. * np.pi * rhoISM * (r_arr[r_arr > rCloud]**3 - rCloud**3)
            
            # if computing mdot is desired
            if return_mdot: 
                try:
                    # try to retrieve velocity array
                    rdot_arr = kwargs.pop('rdot_arr')
                    if type(rdot_arr) is not np.ndarray:
                        rdot_arr = np.array([rdot_arr])
                    # check unit
                    inside_cloud = r_arr <= rCloud
                    mGasdot[inside_cloud] = 4 * np.pi * rhoCore * r_arr[inside_cloud]**2 * rdot_arr[inside_cloud]
                    mGasdot[~inside_cloud] = 4 * np.pi * rhoISM * r_arr[~inside_cloud]**2 * rdot_arr[~inside_cloud]
                    # return value
                    return mGas, mGasdot
                except: 
                    raise Exception('Velocity array expected.')
            else:
                return mGas
            
        # ----
        # Case 2: The density profile has power-law profile (alpha)
        else:
            # input values into mass array
            # inner sphere
                             
            mGas[r_arr <= rCore] = 4 / 3 * np.pi * r_arr[r_arr <= rCore]**3 * rhoCore
            # composite region, see Eq25 in WARPFIELD 2.0 (Rahner et al 2018)
            # assume rho_cl \propto rho (r/rCore)**alpha
            mGas[r_arr > rCore] = 4. * np.pi * rhoCore * (
                           rCore**3/3. +\
                          (r_arr[r_arr > rCore]**(3.+alpha) - rCore**(3.+alpha))/((3.+alpha)*rCore**alpha)
                          )
            # outer sphere
            mGas[r_arr > rCloud] = mCloud + 4. / 3. * np.pi * rhoISM * (r_arr[r_arr > rCloud]**3 - rCloud**3)
            
            # return dM/dt.
            if return_mdot:
                try:
                    rdot_arr = kwargs.pop('rdot_arr')
                    if type(rdot_arr) is not np.ndarray:
                        rdot_arr = np.array([rdot_arr])
                except: 
                    raise Exception('Velocity array expected.')
                rdot_arr = np.array(rdot_arr)
                # input values into mass array
                # dm/dt, see above for expressions of m.
                mGasdot[r_arr <= rCore] = 4 * np.pi * rhoCore * r_arr[r_arr <= rCore]**2 * rdot_arr[r_arr <= rCore]
                mGasdot[r_arr > rCore] = 4 * np.pi * rhoCore * (r_arr[r_arr > rCore]**(2+alpha) / rCore**alpha) * rdot_arr[r_arr > rCore]
                mGasdot[r_arr > rCloud] = 4 * np.pi * rhoISM * r_arr[r_arr > rCloud]**2 * rdot_arr[r_arr > rCloud]
                return mGas, mGasdot
            else:
                return mGas
            
        
    # =============================================================================
    # For Bonnor-Ebert spheres
    # =============================================================================
    elif params['dens_profile'].value == 'densBE':


        # OLD VERSION for mass ----
        # i think this will break if r_arr is given such that it is very large and break interpolation?
        c_s = operations.get_soundspeed(params['densBE_Teff'], params)
        G = params['G'].value
        f_rho_rhoc = params['densBE_f_rho_rhoc'].value
        
        xi_arr = bonnorEbertSphere.r2xi(r_arr, params)
        
        f_mass = lambda xi : 4 * np.pi * rhoCore * (c_s**2 / (4 * np.pi * G * rhoCore))**(3/2) * xi**2 * f_rho_rhoc(xi)
        
        m_arr = np.ones_like(r_arr)
        
        # if r is bigger than cloud and if its smaller than cloud. 
        
        
        for ii, xi in enumerate(xi_arr[r_arr <= rCloud]):
            mass, _ = scipy.integrate.quad(f_mass, 0, xi)
            m_arr[ii] = mass
        # ----
            
            
            
        # # new version for mass -----
        
        # m_arr = np.ones_like(r_arr)
        
        # f_rho_rhoc = params['densBE_f_rho_rhoc'].value
        
        # xi_arr = bonnorEbertSphere.r2xi(r_arr[r_arr <= rCloud], params)
        
        
        
        # #         f_rho_rhoc = params['densBE_f_rho_rhoc'].value
        
        # # xi_arr = bonnorEbertSphere.r2xi(r_arr, params)
        
        # # # print(xi_arr)

        # # rho_rhoc = f_rho_rhoc(xi_arr)
        
        # # n_arr = rho_rhoc * params['nCore'] 
        
        # # n_arr[r_arr > rCloud] = nISM
        
        
        # rho_arr = f_rho_rhoc(xi_arr) * params['nCore'] * params['mu_ion']
        
        # m_arr[r_arr <= rCloud] =  4 / 3 * np.pi * r_arr[r_arr <= rCloud]**3 * rho_arr
        
        
        
        m_arr[r_arr > rCloud] = mCloud + 4 / 3 * np.pi * rhoISM * (r_arr[r_arr > rCloud]**3 - rCloud**3)
            
        
        # -----
        
        
        
        # # mGasDot
        if return_mdot:
            
            
            # this part is a little bit trickier, because there is no analytical solution to the
            # BE spheres. What we can do is to have two seperate parts: for xi <~1 we know that
            # rho ~ rhoCore, so this part can be analytically similar to the 
            # homogeneous sphere. 
            # Once we have enough in the R2_aray and the t_array, we can then use them 
            # to extrapolate to obtain mShell.
            
            # Perhaps what we could do too, is that once collapse happens 
            # we say that mDot is now very small and does not really matter(?)
            # will see. 
            
            try:
                rdot_arr = kwargs.pop('rdot_arr')
                if type(rdot_arr) is not np.ndarray:
                    rdot_arr = np.array([rdot_arr])
            except: 
                raise Exception('Velocity array expected.')

            rdot_arr = np.array(rdot_arr)
            mdot_arr = np.ones_like(rdot_arr)
            
            # the initial cloud arrays
            cloud_n_arr = params['initial_cloud_n_arr'].value
            cloud_r_arr = params['initial_cloud_r_arr'].value
            # try to find threshold
            cloud_getr_interp = scipy.interpolate.interp1d(cloud_n_arr[cloud_r_arr < rCloud], cloud_r_arr[cloud_r_arr < rCloud], kind='cubic', fill_value="extrapolate")
            cloud_getn_interp = scipy.interpolate.interp1d(cloud_r_arr[cloud_r_arr < rCloud], cloud_n_arr[cloud_r_arr < rCloud], kind='cubic', fill_value="extrapolate")
            # calculate threshold
            n_threshold = 0.9 * params['nCore'] 
            # get radius
            r_threshold = cloud_getr_interp(n_threshold)
            
            # print(r_threshold)
            
            rhoGas = cloud_getn_interp(r_arr) * params['mu_ion'].value
            
            # DEBUG remove the time condition
            if params['R2'].value < r_threshold:
            # if params['t_now'].value < 0.01:
                # treat as a homogeneous cloud
                mdot_arr = 4 * np.pi * r_arr**2 * rhoGas * rdot_arr
            else:
                            # # input values into mass array
                            # # dm/dt, see above for expressions of m.
                            # mGasdot[r_arr <= rCore] = 4 * np.pi * rhoCore * r_arr[r_arr <= rCore]**2 * rdot_arr[r_arr <= rCore]
                            # mdot_arr[r_arr > rCore] = 4 * np.pi * rhoCore * (1 / rCore**-2) * rdot_arr[r_arr > rCore]
                            # mdot_arr[r_arr > rCloudud] = 4 * np.pi * rhoISM * r_arr[r_arr > rCloud]**2 * rdot_arr[r_arr > rCloud]
                
                t_arr_previous = params['array_t_now'].value
                r_arr_previous = params['array_R2'].value
                m_arr_previous = params['array_mShell'].value
                
                from scipy.interpolate import CubicSpline
                
                # Cubic spline with extrapolation
                interps = CubicSpline(t_arr_previous, r_arr_previous, extrapolate=True)

                t_next = params['t_next'].value
                # what is the next R2?
                R2_next = interps(t_next)
                
                t_arr_previous = np.concatenate([t_arr_previous, [t_next]])
                r_arr_previous = np.concatenate([r_arr_previous, [R2_next]])
                m_arr_previous = np.concatenate([m_arr_previous, m_arr])
            
                # print('mass profile problems', m_arr, m_arr_previous, t_arr_previous)
                # print(len(m_arr_previous), len(t_arr_previous))
                
                mdot_interp = scipy.interpolate.interp1d(r_arr_previous, np.gradient(m_arr_previous, t_arr_previous), kind='cubic', fill_value="extrapolate")
                
                mdot_arr = mdot_interp(r_arr)

                
            
            # rhoGas = rhoCore * f_rho_rhoc(xi_arr)
            
            # mdot_arr[r_arr <= rCloud] = 4 * np.pi * r_arr[r_arr <= rCloud]**2 * rhoGas[r_arr <= rCloud] * rdot_arr[r_arr <= rCloud]
            # mdot_arr[r_arr > rCloud] = 4 * np.pi * r_arr[r_arr > rCloud]**2 * rhoISM * rdot_arr[r_arr > rCloud]
            
            # print(m_arr, mdot_arr)
            
            return m_arr, mdot_arr
            
        
        return m_arr







