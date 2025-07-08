#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:53:59 2025

@author: Jia Wei Teh
"""

import sys
import numpy as np

def create_PLSphere(params):
    """
    """

    alpha = params['densPL_alpha'].value
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    rCore = params['rCore'].value
    mu_neu = params['mu_neu'].value
    nISM = params['nISM'].value
    rhoCore = nCore * mu_neu
    
    print(rCore)
    
    # compute cloud radius
    # use core radius/density if there is a power law. If not, use average density.
    if alpha != 0:
        # rCloud = (
        #             (
        #                 mCloud/(4 * np.pi * nCore * mu_neu) - rCore**3/3
        #             ) * rCore ** alpha * (alpha + 3) + rCore**(alpha + 3)
        #           )**(1/(alpha + 3))
        
        
        rCloud = (mCloud * rCore**alpha * (3+alpha) / (4 * np.pi * rhoCore) -\
                    rCore**3 * rCore**alpha * (3 + alpha) / 3 +\
                        rCore**(3+alpha)) ** (1 / (3 + alpha))
                        
        
        print(mCloud, rCore, alpha)
            
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