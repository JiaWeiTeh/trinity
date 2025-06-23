# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Sun Jul 24 23:42:14 2022

@author: Jia Wei Teh

This script contains a function that returns initial radius and edge density of the cloud.
"""
import numpy as np
import sys
#--

def get_CloudRadiusEdge(params):
    """
    This function computes the initial properties of the cloud, including (but not all):
        - cloud radius (Units: pc)
        - cloud edge density (Units: 1/cm3)
        And either of these two (depending on density profile):
        - cloud core radius (Units: pc)  (for pL)
        
    Watch out units!

    Returns
    -------
    rCloud : float
        cloud core radius (Units: pc).
    nEdge : float
        cloud edge density (Units: 1/pc3).
    """

    alpha = params['densPL_alpha'].value
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    rCore = params['rCore'].value
    mu_neu = params['mu_neu'].value
    nISM = params['nISM'].value

    # compute cloud radius
    # use core radius/density if there is a power law. If not, use average density.
    if alpha != 0:
        rCloud = (
                    (
                        mCloud/(4 * np.pi * nCore * mu_neu) - rCore**3/3
                    ) * rCore ** alpha * (alpha + 3) + rCore**(alpha + 3)
                 )**(1/(alpha + 3))
        # density at edge
        nEdge = nCore * (rCloud/rCore)**alpha
        
    elif alpha == 0:
        rCloud = (3 * mCloud / 4 / np.pi / (nCore * mu_neu))**(1/3)
        # density at edge should just be the average density
        nEdge = nCore
    
    # sanity check
    if nEdge < nISM:
        print(f'nCore: {nCore}, nISM: {nISM}')
        sys.exit(f"The density at the edge of the cloud ({nEdge}) is lower than the ISM ({nISM}); please consider increasing nCore.")
    # return
    return rCloud, nEdge
    

