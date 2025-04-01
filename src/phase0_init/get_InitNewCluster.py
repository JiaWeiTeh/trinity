#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:20:29 2023

@author: Jia Wei Teh

This script contains function which makes a new cluster and get new ]
cloud and cluster properties.
"""
# libraries
import astropy.units as u
import astropy.constants as c
import sys
import numpy as np
# get parameter
import src.cloud_properties.bonnorEbert as bE

import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()


def make_new_cluster(Mcloud_au_INPUT, SFE, tcoll, ii_coll):
    """
    make a new cluster and get new cloud and cluster properties
    :param Mcloud_au_INPUT:
    :param SFE:
    :param tcoll:
    :param ii_coll:
    :return:
    """
    # distribute ISM and restart expansion
    if warpfield_params.mult_SF >=1:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("STARBURST No. (", ii_coll+1, ")")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # option 1: create star cluster with the same SFE as before
    if warpfield_params.mult_SF == 1:
        Mcluster_au = SFE * Mcloud_au_INPUT # 2nd cluster
    # option 2 : create star cluster such that a certain star formation efficiency per free-fall time is matched
    elif warpfield_params.mult_SF == 2:
        # time since last collapse
        dtcoll = tcoll[-1] - tcoll[-2]
        # new SFE for this starburst
        # if recollapse happens after more than 10 free-fall times, SFE can be higher than 100%. To prevent this, set maximum of 99%.
        SFE = min([dtcoll* warpfield_params.sfe_tff/warpfield_params.tff, 0.99])
        Mcluster_au = SFE * Mcloud_au_INPUT # 2nd cluster
    # option 0: add a zero mass star cluster
    elif warpfield_params.mult_SF == 0:
        Mcluster_au = 0.0

    print("Mcluster2: ", Mcluster_au)
    Mcloud_au = Mcloud_au_INPUT - Mcluster_au # new cloud mass (reduced by newly formed cluster mass)

    # calculate core radius, cloud radius, and density at edge of cloud
    # remember here the old code mixed up rcore and tbe. 
    rcore_au, rcloud_au, nedge = get_cloud_radius_dens(Mcloud_au)

    rhoa_au =  warpfield_params.nCore * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
    
    # This will become ODEpar
    CloudProp = {'gamma': warpfield_params.gamma_adia, 'Mcloud_au': Mcloud_au, 'rhocore_au': rhoa_au, 'Rcore_au': rcore_au,
                 'nalpha': warpfield_params.dens_a_pL, 'Mcluster_au': Mcluster_au, 'Rcloud_au': rcloud_au, 'SFE': SFE, 'nedge': nedge,
                 'Rsh_max': 0., 't_dissolve':1e30}
    # set the maximum shell radius achieved during this expansion to 0. (i.e. the shell has not started to expand yet)
    # set dissolution time to arbitrary high number (i.e. the shell has not yet dissolved)

    return CloudProp


def get_cloud_radius_dens(mCloud):
 
    rCore = warpfield_params.rCore
    nCore = warpfield_params.nCore
    mu_n = warpfield_params.mu_n
    gamma = warpfield_params.gamma_adia
    # Initialise value if not selected.
    rCore = np.nan
    bE_T = np.nan
    
    # Here is get_cloud_Rn().
    # =============================================================================
    # For power-law density profile    
    # =============================================================================
    if warpfield_params.dens_profile == "pL_prof":
        # initialise value if not selected.
        bE_T = np.nan
        alpha = warpfield_params.dens_a_pL
        # converting to cgs
        mCloud = mCloud * u.M_sun.to(u.g)
        rCore = rCore * u.pc.to(u.cm)
        # compute cloud radius
        rCloud = (
                    (
                        mCloud/(4 * np.pi * nCore * mu_n) - rCore**3/3
                    ) * rCore ** alpha * (alpha + 3)\
                        + rCore**(alpha + 3)
                 )**(1/(alpha + 3))
        # compute the density at edge
        nEdge = nCore * mu_n * (rCloud/rCore)**alpha
        # sanity check
        if nEdge < warpfield_params.nISM:
            sys.exit("Exiting: Density at cloud edge lower than ISM density; please increase nCore.")

        # converting back
        rCore = rCore * u.cm.to(u.pc)
        rCloud = rCloud * u.cm.to(u.pc)

    # =============================================================================
    # For Bonnor-Ebert density profile
    # =============================================================================
    elif warpfield_params.dens_profile == "bE_prof":
        # initialise value if not selected.
        rCore = np.nan
        g = warpfield_params.dens_g_bE
        # Remember that rCore is a property of power-law. it is bE_T for bE spheres.
        # print(mCloud, nCore, g, mu_n, gamma)
        bE_T = bE.get_bE_T(mCloud, nCore, g, mu_n, gamma)
        # print(mCloud, nCore, g, mu_n, gamma)
        # these are the values
        # 1000000.0 1000.0 14.1 2.1287915392418182e-24 1.6666666666666667
        # 4649.954642315685
        rCloud, nEdge = bE.get_bE_rCloud_nEdge(nCore, bE_T, mCloud, mu_n, gamma)
        
    # return
    return rCore, bE_T, rCloud, nEdge,

































