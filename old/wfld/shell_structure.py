import numpy as np
import constants as c
import parameters as p
import matplotlib.pyplot as plt
from ODEs import *
import init as i
import scipy.optimize
import scipy.integrate
import auxiliary_functions as aux
from astropy.table import Table
import os


def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


"""calculate structure of a shell whose inner boundary is in pressure equilibrium with stellar winds
(ram-pressure (late) or thermal pressure (early))"""
# input:
#       Rs - inner shell radius in pc
#       Ln - luminosity of non-ionizing radiation in erg/s
#       Li - luminosity of ionizing radiation in erg/s
#       Qi - rate of ionizing photons in 1/s
#       Lw - mechanical luminosity (0.5*mass_loss_rate*terminal_velocity**2) in erg/s (this could actually be calculated from vw and Mwdot)
#       pwind - wind momentum (pwind=vw*Mwdot, where vw - (mechanical luminosity averaged) terminal velocity of winds and supernova ejecta,
#               Mwdot - (mechanical luminosity averaged) mass loss rate of winds and supernova ejecta)
#       ploton - make a plot of the shell structure? (boolean)
#       phase - string: if "Weaver" is specified, inner pressure is set by mechanical luminosity, otherwise by wind/SN momentum (see Martinez-Gonzalez 2014)
# output:
#       fabs_i - trapping fraction of ionizing radiation
#       fabs_n - trapping fraction of non-ionizing radiation
#       fabs   - total trapping fraction
#       shell_ionized  -  boolean: if true, there is no neutral/molecular shell
#       nmax   - maximum reached number density in the shell

def n_from_press(press, Ti, B_cloudy=False):
    n0 = i.mua/i.mui*press/(c.kboltz*Ti)
    if B_cloudy == True:
        # calculate n0_cloudy (density at inner shell edge passed to cloudy (usually includes B-field))
        n0_cloudy = start_dens_B(press, Ti)[0]
    else:
        n0_cloudy = n0
    return n0, n0_cloudy

def shell_structure(Rs, Ln, Li, Qi, Lw, pwind, Msh_fix_au, ploton = False, plotpath = "/home/daniel/Documents/work/loki/data/expansion/", phase="not_Weaver", surpress_warning=False):
    # old version (used for Rahner+2017 "Unison" and Rahner+2018 "30 Dor")

    if (surpress_warning is False):
        print('##########################################################')
        print('OLD VERSION OF SHELL STRUCTURE! SWITCH TO NEW VERSION (2)!')
        print('##########################################################')

    Msh_allthere = False
    phi_zero = False
    full_shell_ionized = False

    # IMPORTANT: this routine uses cgs units
    rStart = Rs*c.pc # get this number from expansion solver
    rInc = min([rStart/100., i.rInc_ion*c.pc])

    Msh_fix = Msh_fix_au * c.Msun
    #Msh_fix = min([Mcluster*(1./SFE-1.), 4.*pi/3*i.rhoa*rStart**3])


    if phase == "Weaver":
        # Weaver radius and pressure, density is set by hot X-ray thermal pressure
        press = 7.*i.rhoa**(1./3.) * (3.*(c.gamma-1.)*Lw/(28.*(9.*c.gamma-4.)*np.pi*rStart**2))**(2./3.)
    ##
    else: # phase is not Weaver, density is set by ram pressure
        press = pwind/(4.*np.pi*rStart**2) # ram pressure at inner boundary

    n0 = i.mua/i.mui*press/(c.kboltz*i.Ti) # ion number density (no B-Field)
    ### cloudy #####################################################################################
    if i.B_cloudy == True:
        # calculate n0_cloudy (density at inner shell edge passed to cloudy (usually includes B-field))
        n0_cloudy = start_dens_B(press, i.Ti)[0]
    else:
        n0_cloudy = n0
    ################################################################################################

    ninner = n0 # density at inner boundary of shell
    phi0 = 1.0
    tau0 = 0.0

    r_ion = []; n_ion = []; phi_ion = []; tau_ion = []; Msh_ion=[]
    # do integration until no ionizing photons left or all the shell mass is accounted for
    while phi_zero is False and Msh_allthere is False:

        # integrate over 1 pc
        rStop = rStart + 1.0*c.pc
        r = np.arange(rStart, rStop, rInc)

        # n0, phi0, tau0
        y0 = [n0, phi0, tau0]
        params = [Ln, Li, Qi]
        #with stdout_redirected():
        psoln = scipy.integrate.odeint(f_drain, y0, r, args=(params,))
        

        n = psoln[:,0]
        phi = psoln[:,1]
        tau = psoln[:,2]


        Msh = r*0.0
        # if this is not the first time to enter this loop get last ionized mass of shell
        if len(Msh_ion) > 0:
            Msh[0] = Msh_ion[-1]

        ii = 0
        while Msh[ii] < Msh_fix and phi[ii]>0.0 and ii<len(r)-1:
            Msh[ii+1]=Msh[ii]+n[ii+1]*i.mui*4.*np.pi*(r[ii+1])**2*rInc # since n is the ion density, multiply with mui and not with mua
            ii = ii+1
        Iend = ii # at this index phi drops below 0 or whole shell mass is accounted for or the integration ends
        if Msh[Iend] >= Msh_fix:
            Msh_allthere = True
            full_shell_ionized = True
            if i.output_verbosity == 2:
                print("Shell fully ionized")
        if phi[Iend] <= 0.0:     phi_zero = True

        r_ion = np.concatenate([r_ion,r[0:Iend]])
        n_ion = np.concatenate([n_ion,n[0:Iend]])
        phi_ion = np.concatenate([phi_ion, phi[0:Iend]])
        tau_ion = np.concatenate([tau_ion, tau[0:Iend]])
        Msh_ion = np.concatenate([Msh_ion, Msh[0:Iend]])

        # set new initial conditions
        #n0 = n[Iend]
        #phi0 = phi[Iend]
        #tau0 = tau[Iend]
        #rStart = r[Iend]
        n0 = n_ion[-1]
        phi0 = phi_ion[-1]
        tau0 = tau_ion[-1]
        rStart = r_ion[-1]

    #################################################################

    #estimate how much of ionizing radiation has been absorbed by dust and how much by hydrogen
    # integrate both dphi/dr terms over dr
    dr_ion = (r_ion[1:len(r_ion)] - r_ion[0:len(r_ion)-1])
    dphi_dust = np.sum(-n_ion[0:len(n_ion)-1] * i.sigmaD * phi_ion[0:len(phi_ion)-1] * dr_ion)
    dphi_rec  = np.sum(-4.*np.pi * c.alphaB * (n_ion[0:len(n_ion)-1])**2 * (r_ion[0:len(r_ion)-1])**2 / Qi * dr_ion)

    if dphi_dust+dphi_rec == 0.0:
        fion_dust = 0.0
        fion_rec = 0.0
    else:
        fion_dust = dphi_dust / (dphi_dust+dphi_rec)
        fion_rec  = dphi_rec / (dphi_dust+dphi_rec)
    #print fion_dust, fion_rec

    #r_noion = r[Iend]
    #n_noion = n[Iend]
    #tau_noion = tau[Iend]
    # maybe there is no ionized shell? This can happen when ionizing radiation is very very weak
    #print "r_ion", r_ion
    #if not r_ion: # check wheter list is empty
    #    print "Problem"
#        fabs_i = 1.0
#        fabs_n = 1.0
#        fabs = 1.0
#        shell_ionized = False
#        iwarn = 1
#        return [fabs_i, fabs_n, fabs, shell_ionized, iwarn]
    #else: # list is not empty, i.e. there is an ionized shell
    r_noion = [r_ion[-1]]; n_noion = [n_ion[-1]]; tau_noion = [tau_ion[-1]]; Msh_noion=[Msh_ion[-1]]

    # if it is not true that the full shell is ionized, calculate structure of non-ionized part
    if full_shell_ionized is False:
        rInc = min([rStart / 100., i.rInc_neutral * c.pc])
        # at the boundary between the ionized and the neutral shell is a temperaure discontinuity: change density
        n0 = n0 * i.mui / i.mua * i.Ti / i.Tn

        while phi_zero is True and Msh_allthere is False:

            rStop = rStart + 1.0*c.pc
            r = np.arange(rStart, rStop, rInc)

            # n0, tau0
            y0 = [n0, tau0]
            
            #with stdout_redirected():
            psoln = scipy.integrate.odeint(f_drain_noion, y0, r, args=(params,))

            n = psoln[:,0]
            tau = psoln[:,1]

            Msh = r*0.0
            Msh[0] = Msh_noion[-1]

            ii = 0
            while Msh[ii] < Msh_fix and ii < len(r) - 1:
                Msh[ii + 1] = Msh[ii] + n[ii + 1] * i.mui * 4. * np.pi * (r[ii + 1]) ** 2 * rInc
                ii = ii + 1
            Iend = ii  # at this index phi drops below 0 or whole shell mass is accounted for or the integration ends
            if Msh[Iend] >= Msh_fix: Msh_allthere = True #; print "all shell mass accounted for in neutral shell"

            r_noion = np.concatenate([r_noion,r[0:Iend]])
            n_noion = np.concatenate([n_noion,n[0:Iend]])
            tau_noion = np.concatenate([tau_noion, tau[0:Iend]])
            Msh_noion = np.concatenate([Msh_noion, Msh[0:Iend]])

            # update initial conditions
            n0 = n_noion[-1]
            tau0 = tau_noion[-1]
            rStart = r_noion[-1]


    #################################################################
    #thickness of shell
    dRs = r_noion[-1]/c.pc - Rs

    if full_shell_ionized is False: # shell contains non-ionized part
        tau_Rend = tau_noion[-1]
        phi_Rend = 0.0
    else:
        tau_Rend = tau_ion[-1]
        phi_Rend = phi_ion[-1]

    fabs_i = 1.0-phi_Rend
    fabs_n = 1-np.exp(-tau_Rend)

    fabs = (fabs_i*Li + fabs_n*Ln) / (Li+Ln)


    if full_shell_ionized is False:
        nmax = max(n_noion)
    else:
        nmax = max(n_ion)

    #print "phi, tau, fabs_i, fabs_n , fabs=", phi_Rend, tau_Rend, fabs_i, fabs_n, fabs

    # calculate rho*dr (this is tau_IR over kIR if kIR is constant)
    if full_shell_ionized is False:
        dr_noion = (r_noion[1:len(r_noion)] - r_noion[0:len(r_noion) - 1])
        rhodr = i.mui * (np.sum(n_ion[0:len(n_ion) - 1] * dr_ion) + np.sum(n_noion[0:len(n_noion) - 1] * dr_noion) )
    else:
        rhodr = i.mui * (np.sum(n_ion[0:len(n_ion) - 1] * dr_ion))

    """
    if ploton == True:

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        ax1.set_xlabel('r (pc)')
        ax1.set_ylabel('log10(n (1/ccm))')
        ax1.set_ylim([-1.2,np.ceil(np.log10(nmax))])
        #ax1.set_ylim([-0.1+r0,1.1+r0])
        ax1.plot((r_ion/c.pc), np.log10(n_ion), 'r--')
        #ax1.plot(r,r**3/6.0,'r:')

        ax3.set_xlabel('r (pc)')
        ax3.set_ylabel('$\\tau$')
        ax3.plot(r_ion/c.pc, tau_ion, 'r--')

        ax2.set_xlabel('r (pc)')
        ax2.set_ylabel('$\phi$ ')
        ax2.plot(r_ion/c.pc, phi_ion, 'r--')

        if full_shell_ionized is False:
            ax1.plot((r_noion / c.pc), np.log10(n_noion), 'b-')
            ax3.plot(r_noion / c.pc, tau_noion, 'b-')
            ax1.plot(([r_noion[-1]/c.pc, r_noion[-1]/c.pc]), np.log10([n_noion[-1], n_intercl]), 'b-')
            ax1.plot(([r_noion[-1]/c.pc, r_noion[-1]/c.pc+(r_ion[-1]-r_ion[0])/c.pc*0.2]), np.log10([n_intercl,n_intercl]), 'k:')
        else:
            ax1.plot(([r_ion[-1]/c.pc, r_ion[-1]/c.pc]), np.log10([n_ion[-1], n_intercl]), 'r--')
            ax1.plot(([r_ion[-1]/c.pc, r_ion[-1]/c.pc+(r_ion[-1]-r_ion[0])/c.pc*0.2]), np.log10([n_intercl,n_intercl]), 'k:')

        #plt.savefig(plotpath)
        #plt.close(fig)
        #fig.clear()
        #plt.show()
    """

    return [fabs_i, fabs_n, fabs, fion_dust, full_shell_ionized, dRs, ninner, nmax, rhodr, n0_cloudy]


def shell_structure2(Rs, press_au, Ln, Li, Qi, Msh_fix_au, cf, ploton = False, plotpath = "/home/mo/Desktop/Hiwi/shell.txt", Minterior=0.):
    # new version (used for Rahner+2019 "WARPFIELD 2.0")
    # IMPORTANT: this routine uses cgs units
    # IMPORTANT: plotpath is a misnomer. Actually, this is the full file name of the density profile of the shell which is stored

    Msh_allthere = False
    phi_zero = False
    full_shell_ionized = False
    shell_dissolved = False

    # convert astro units (Myr, Msun, pc) to cgs units
    if Rs < 1e5: # this means shell radius units are probably parsecs
        rStart0 = Rs*c.pc # convert to centimetres
    rStart = rStart0
    press = press_au*c.press_cgs

    Msh_fix = Msh_fix_au * c.Msun
    #Msh_fix = min([Mcluster*(1./SFE-1.), 4.*pi/3*rhoa*rStart**3])

    [n0,n0_cloudy] = n_from_press(press, i.Ti, B_cloudy=i.B_cloudy)
    
    n0=n0/cf

    # max radius of ionization front if density did not change in shell
    Rionmax = (3.*Qi/(4.*np.pi*c.alphaB*n0**2.) + rStart**3.)**(1./3.)
    mydr = np.min([1.0*c.pc,np.abs(Rionmax-rStart)]) # abs is for safety

    # step size
    rInc = max([mydr / 1e6, min([mydr / 1e3, i.rInc_ion * c.pc])])

    ninner = n0# density at inner boundary of shell
    phi0 = 1.0
    tau_ion0 = 0.0
    Msh_ion0 = 0.0

    r_ion = []; n_ion = []; phi_ion = []
    r_ions = []; n_ions = []; phi_ions = []
    # do integration until no ionizing photons left or all the shell mass is accounted for
    while phi_zero is False and Msh_allthere is False:

        # integrate over 1 pc (if less if above estimate for max thickness is smaller)
        rStop = rStart + mydr

        r = np.arange(rStart, rStop, rInc)

        # n0, phi0, tau0
        y0 = [n0, phi0, tau_ion0]
        
        
        if i.frag_cover == True:
            params = [Ln, Li, Qi,cf]
            #with stdout_redirected():
            psoln = scipy.integrate.odeint(f_drain_cf, y0, r, args=(params,),rtol=1e-3,hmin=1e-7)
            #print('--------------fdrain_cf')
        
        else:
            params = [Ln, Li, Qi]
            #with stdout_redirected():
            psoln = scipy.integrate.odeint(f_drain, y0, r, args=(params,),rtol=1e-3,hmin=1e-7)
        

        n = psoln[:,0]
        phi = psoln[:,1]
        tau = psoln[:,2]

        Msh = np.zeros_like(r)
        Msh[0] = Msh_ion0
        Msh[1:] = n[1:] * i.mui * 4. * np.pi * (r[1:])**2*rInc
        Msh = np.cumsum(Msh)

        ii = np.nonzero((Msh >= Msh_fix) | (phi <= 0))[0]

        if len(ii) > 0:
            ii = ii[0]
        else:
            ii = len(r) - 1
        Msh[ii+1:] = 0.0
        Iend = ii # at this index phi drops below 0 or whole shell mass is accounted for or the integration ends


        if Msh[Iend] >= Msh_fix:
            Msh_allthere = True
            full_shell_ionized = True
            if i.output_verbosity == 2:
                print("Shell fully ionized")
        if phi[Iend] <= 0.0:     phi_zero = True

        r_ions.append(r[0:Iend-1])
        n_ions.append(n[0:Iend-1])
        phi_ions.append(phi[0:Iend-1])

        # set new initial conditions
        n0 = n[Iend - 1]
        phi0 = phi[Iend - 1]
        tau_ion0 = tau[Iend - 1]
        rStart = r[Iend - 1]
        Msh_ion0 = Msh[Iend - 1]
        
        # sanity check: if the shell expands too far or inner density is too low, regard is as dissolved and don't waste computation time
        if ((ninner < 0.001*i.n_intercl) or (rStop > 1.2*(i.rstop*c.pc)) or (rStop-rStart0 > 10.*rStart0)):
            if ninner < 0.001*i.n_intercl:
                print('shell dissolved because ninner < 0.001*i.n_intercl')
            elif rStop > 1.2*(i.rstop*c.pc):
                print('shell dissolved because rStop > 1.2*(i.rstop*c.pc)')
            else:
                print('shell dissolved because rStop-rStart0 > 10.*rStart0')
                
            shell_dissolved = True
            break

    r_ions[-1] = r[0:Iend]
    n_ions[-1] = n[0:Iend]
    phi_ions[-1] = phi[0:Iend]
    r_ion = np.concatenate(r_ions)
    n_ion = np.concatenate(n_ions)
    phi_ion = np.concatenate(phi_ions)


    #################################################################
    if shell_dissolved is False:
        # get graviational potential
        r_Phi_tmp = r_ion
        rho_tmp = n_ion * i.mui
        m_r_tmp = rho_tmp * 4. * np.pi * r_Phi_tmp ** 2 * rInc  # mass per bin
        Mcum_tmp = np.cumsum(m_r_tmp) + Minterior  # cumulative mass
        #Phi_grav_tmp = c.Grav * Mcum_tmp / r_Phi_tmp  # gravitational potential
        Phi_grav_r0s = -4.*np.pi*c.Grav * scipy.integrate.simps(r_Phi_tmp*rho_tmp,x=r_Phi_tmp)
        f_grav_tmp = c.Grav*Mcum_tmp / r_Phi_tmp**2.  # gravitational force per unit mass

        len_r = len(r_Phi_tmp)
        skip = max(int(float(len_r) / float(i.pot_len_intern)),1)
        r_Phi = r_Phi_tmp[0:-1:skip]
        # Phi_grav = Phi_grav_tmp[0:-1:skip]
        f_grav = f_grav_tmp[0:-1:skip]

        #estimate how much of ionizing radiation has been absorbed by dust and how much by hydrogen
        # integrate both dphi/dr terms over dr
        dr_ion = (r_ion[1:len(r_ion)] - r_ion[0:len(r_ion)-1])
        dphi_dust = np.sum(-n_ion[0:len(n_ion)-1] * i.sigmaD * phi_ion[0:len(phi_ion)-1] * dr_ion)
        dphi_rec  = np.sum(-4.*np.pi * c.alphaB * (n_ion[0:len(n_ion)-1])**2 * (r_ion[0:len(r_ion)-1])**2 / Qi * dr_ion)

        if dphi_dust+dphi_rec == 0.0:
            fion_dust = 0.0
            fion_rec = 0.0
        else:
            fion_dust = dphi_dust / (dphi_dust+dphi_rec)
            fion_rec  = dphi_rec / (dphi_dust+dphi_rec)
        #print fion_dust, fion_rec

        #r_noion = r[Iend]
        #n_noion = n[Iend]
        #tau_noion = tau[Iend]
        # maybe there is no ionized shell? This can happen when ionizing radiation is very very weak
        #print "r_ion", r_ion
        #if not r_ion: # check wheter list is empty
        #    print "Problem"
    #        fabs_i = 1.0
    #        fabs_n = 1.0
    #        fabs = 1.0
    #        shell_ionized = False
    #        iwarn = 1
    #        return [fabs_i, fabs_n, fabs, shell_ionized, iwarn]
        #else: # list is not empty, i.e. there is an ionized shell
        r_noion = [r_ion[-1]]; n_noion = [n_ion[-1]]
        r_noions = [[r_ion[-1]]]; n_noions = [[n_ion[-1]]]

        # if it is not true that the full shell is ionized, calculate structure of non-ionized part
        if full_shell_ionized is False:

            # at the boundary between the ionized and the neutral shell is a temperaure discontinuity: change density
            n0 = n0 * i.mui / i.mua * i.Ti / i.Tn

            taumax = 100. # makes no sense to integrate more if tau is already 100.
            mydr = np.min([1.0 * c.pc, np.abs((taumax - tau_ion0)/(n0*i.sigmaD))]) # right side is maximum width of the non-ionzed shell part assuming density remains constant, abs is for safety

            # step size
            rInc = max([mydr/1e6, min([mydr / 1e3, i.rInc_neutral * c.pc])])

            tau_noion0 = tau_ion0
            Msh0 = Msh_ion0
            while phi_zero is True and Msh_allthere is False:

                rStop = rStart + mydr
                r = np.arange(rStart, rStop, rInc)

                # n0, tau0
                y0 = [n0, tau_noion0]
                
                
                
                if i.frag_cover == True:
                    #with stdout_redirected():
                    #print('--------------------noioncf')
                    psoln = scipy.integrate.odeint(f_drain_noion_cf, y0, r, args=(params,),rtol=1e-3,hmin=1e-7)
                else:
                    #with stdout_redirected():
                    psoln = scipy.integrate.odeint(f_drain_noion, y0, r, args=(params,),rtol=1e-3,hmin=1e-7)
                
            

                

                n = psoln[:,0]
                tau = psoln[:,1]

                Msh = np.zeros_like(r)
                Msh[0] = Msh0
                Msh[1:] = n[1:] * i.mui * 4. * np.pi * (r[1:])**2*rInc
                Msh = np.cumsum(Msh)

                ii = np.nonzero(Msh >= Msh_fix)[0]
                if len(ii) > 0:
                    ii = ii[0]
                else:
                    ii = len(r) - 1
                Msh[ii+1:] = 0.0
                Iend = ii  # at this index phi drops below 0 or whole shell mass is accounted for or the integration ends
                if Msh[Iend] >= Msh_fix: Msh_allthere = True #; print "all shell mass accounted for in neutral shell"

                r_noions.append(r[0:Iend-1])
                n_noions.append(n[0:Iend-1])

                # set new initial conditions
                n0 = n[Iend - 1]
                tau_noion0 = tau[Iend - 1]
                rStart = r[Iend - 1]
                Msh0 = Msh[Iend - 1]

            r_noions[-1] = r[0:Iend]
            n_noions[-1] = n[0:Iend]
            r_noion = np.concatenate(r_noions)
            n_noion = np.concatenate(n_noions)

            # get graviational potential
            r_Phi_noion_tmp = r_noion
            rho_noion_tmp = n_noion * i.mui
            m_r_noion_tmp = rho_noion_tmp * 4. * np.pi * r_Phi_noion_tmp ** 2 * rInc  # mass per bin
            Mcum_noion_tmp = np.cumsum(m_r_noion_tmp) + Mcum_tmp[-1]  # cumulative mass
            #Phi_grav_noion_tmp = c.Grav * Mcum_noion_tmp / r_Phi_noion_tmp  # gravitational potential
            Phi_grav_r0s += -4. * np.pi * c.Grav * scipy.integrate.simps(r_Phi_noion_tmp * rho_noion_tmp, x=r_Phi_noion_tmp) # add up 1st and 2nd part of integral (ionized region and non-ionized region)
            f_grav_noion_tmp = c.Grav*Mcum_noion_tmp / r_Phi_noion_tmp**2.  # gravitational force per unit mass

            len_r = len(r_Phi_noion_tmp)
            skip = max(int(float(len_r)/float(i.pot_len_intern)),1)
            r_Phi_noion = r_Phi_noion_tmp[0:-1:skip]
            #Phi_grav_noion = Phi_grav_noion_tmp[0:-1:skip]
            f_grav_noion = f_grav_noion_tmp[0:-1:skip]

            # concatenate with part of ionized shell
            r_Phi = np.concatenate([r_Phi, r_Phi_noion])
            #Phi_grav = np.concatenate([Phi_grav, Phi_grav_noion])
            f_grav = np.concatenate([f_grav, f_grav_noion])

        #################################################################





        #thickness of shell
        dRs = r_noion[-1]/c.pc - Rs
        os.environ["ShTh"] = str(dRs)

        if full_shell_ionized is False: # shell contains non-ionized part
            tau_Rend = tau_noion0
            phi_Rend = 0.0
        else:
            tau_Rend = tau_ion0
            phi_Rend = phi_ion[-1]

        if (tau_Rend > 100.):
            exp_tau_Rend = 0.
        else:
            exp_tau_Rend = np.exp(-tau_Rend)

        fabs_n = 1.0 - exp_tau_Rend
        fabs_i = 1.0 - phi_Rend

        fabs = (fabs_i*Li + fabs_n*Ln) / (Li+Ln)


        if full_shell_ionized is False:
            nmax = np.max(n_noion)
        else:
            nmax = np.max(n_ion)

        #print "phi, tau, fabs_i, fabs_n , fabs=", phi_Rend, tau_Rend, fabs_i, fabs_n, fabs

        # calculate rho*dr (this is tau_IR over kIR if kIR is constant)
        if full_shell_ionized is False:
            dr_noion = (r_noion[1:len(r_noion)] - r_noion[0:len(r_noion) - 1])
            rhodr = i.mui * (np.sum(n_ion[0:len(n_ion) - 1] * dr_ion) + np.sum(n_noion[0:len(n_noion) - 1] * dr_noion) )
        else:
            rhodr = i.mui * (np.sum(n_ion[0:len(n_ion) - 1] * dr_ion))

        # save shell data
        if i.saveshell is True:

            # save shell structure as .txt file (radius, density, temperature)
            # only save Ndat entries (equally spaced in index, skip others)
            Ndat = 500
            Nskip_ion = int(max(1, len(r_ion) / Ndat))
            Nskip_noion = int(max(1, len(r_ion) / Ndat))
            T_ion = i.Ti * np.ones(len(r_ion))
            T_noion = i.Tn * np.ones(len(r_noion))

            if full_shell_ionized is True:
                r_save = np.append(r_ion[0:-1:Nskip_ion], r_ion[-1])
                n_save = np.append(n_ion[0:-1:Nskip_ion], n_ion[-1])
                T_save = np.append(T_ion[0:-1:Nskip_ion], T_ion[-1])

            else:
                r_save = np.append(np.append(r_ion[0:-1:Nskip_ion], r_ion[-1]), np.append(r_noion[0:-1:Nskip_noion], r_noion[-1]))
                n_save = np.append(np.append(n_ion[0:-1:Nskip_ion], n_ion[-1]), np.append(n_noion[0:-1:Nskip_noion], n_noion[-1]))
                T_save = np.append(np.append(T_ion[0:-1:Nskip_ion], T_ion[-1]), np.append(T_noion[0:-1:Nskip_noion], T_noion[-1]))

            sh_savedata = {"r_cm": r_save, "n_cm-3": n_save,
                            "T_K": T_save}
            name_list = ["r_cm", "n_cm-3", "T_K"]
            tab = Table(sh_savedata, names=name_list)
            #mypath = data_struc["mypath"]
            #age1e7_str = ('{:0=5.7f}e+07'.format(
            #    t_now / 10.))  # age in years (factor 1e7 hardcoded), naming convention matches naming convention for cloudy files
            outname = plotpath
            formats = {'r_cm': '%1.6e', 'n_cm-3': '%1.4e', 'T_K': '%1.4e'}
            # ascii.write(bubble_data,outname,names=names,overwrite=True)
            tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)

        """
        if ploton == True:

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            ax1.set_xlabel('r (pc)')
            ax1.set_ylabel('log10(n (1/ccm))')
            ax1.set_ylim([-1.2,np.ceil(np.log10(nmax))])
            #ax1.set_ylim([-0.1+r0,1.1+r0])
            ax1.plot((r_ion/c.pc), np.log10(n_ion), 'r--')
            #ax1.plot(r,r**3/6.0,'r:')

            ax3.set_xlabel('r (pc)')
            ax3.set_ylabel('$\\tau$')
            ax3.plot(r_ion/c.pc, tau_ion, 'r--')

            ax2.set_xlabel('r (pc)')
            ax2.set_ylabel('$\phi$ ')
            ax2.plot(r_ion/c.pc, phi_ion, 'r--')

            if full_shell_ionized is False:
                ax1.plot((r_noion / c.pc), np.log10(n_noion), 'b-')
                ax3.plot(r_noion / c.pc, tau_noion, 'b-')
                ax1.plot(([r_noion[-1]/c.pc, r_noion[-1]/c.pc]), np.log10([n_noion[-1], n_intercl]), 'b-')
                ax1.plot(([r_noion[-1]/c.pc, r_noion[-1]/c.pc+(r_ion[-1]-r_ion[0])/c.pc*0.2]), np.log10([n_intercl,n_intercl]), 'k:')
            else:
                ax1.plot(([r_ion[-1]/c.pc, r_ion[-1]/c.pc]), np.log10([n_ion[-1], n_intercl]), 'r--')
                ax1.plot(([r_ion[-1]/c.pc, r_ion[-1]/c.pc+(r_ion[-1]-r_ion[0])/c.pc*0.2]), np.log10([n_intercl,n_intercl]), 'k:')

            plt.savefig(plotpath)
            plt.close(fig)
            fig.clear()
            plt.show()
        """
        
    elif shell_dissolved is True:
        aux.printl("inside shell_structure2.py: I am assuming the shell has dissolved...")
        fabs_i = 1.0
        fabs_n = 0.0
        fabs = (fabs_i * Li + fabs_n * Ln) / (Li + Ln)
        fion_dust = np.nan
        full_shell_ionized = True
        dRs = np.nan
        nmax = i.n_intercl
        rhodr = 0.0
        r_Phi = np.nan
        Phi_grav_r0s = np.nan
        f_grav = np.nan


    return [fabs_i, fabs_n, fabs, fion_dust, full_shell_ionized, dRs, ninner, nmax, rhodr, n0_cloudy, r_Phi, Phi_grav_r0s, f_grav]



def f_drain(y,r, params):
    n, phi, tau = y
    Ln, Li, Qi = params

    # to circumvent underflow in exp function for very large tau values
    if tau>700.: exp_minustau = 0.0
    else: exp_minustau = np.exp(-tau)

    # sptatial (r) derivatives
    n_sigmaD = n * i.sigmaD
    n_squared_alphaB = c.alphaB * n**2
    four_pi_r_squared = 4. * np.pi * r**2


    nd = i.mua/(i.mui*c.kboltz*i.Ti) * (n_sigmaD / (four_pi_r_squared*c.clight)*(Ln*exp_minustau + Li*phi) + n_squared_alphaB*Li/(c.clight*Qi) )
    phid = -n_sigmaD*phi - four_pi_r_squared * n_squared_alphaB / Qi
    taud = n_sigmaD

    return [nd, phid, taud]

def f_drain_noion(y,r, params):
    n, tau = y
    Ln, Li, Qi = params

    # to circumvent underflow in exp function for very large tau values
    if tau>100.: exp_minustau = 0.0
    else: exp_minustau = np.exp(-tau)

    # sptatial (r) derivatives
    nd = n*i.sigmaD*Ln*exp_minustau/(c.kboltz*i.Tn*4.*np.pi*r**2*c.clight)
    taud = n*i.sigmaD

    return [nd, taud]


def f_drain_cf(y,r, params):
    n, phi, tau = y
    Ln, Li, Qi, cf = params

    # to circumvent underflow in exp function for very large tau values
    if tau>700.: 
        exp_minustau = 0.0
        #print('tau>>>>>>700',tau)
    else: 
        exp_minustau = np.exp(-tau)

    # sptatial (r) derivatives
    n_sigmaD = (n/cf) * i.sigmaD
    n_squared_alphaB = c.alphaB * (n/cf)**2
    four_pi_r_squared = 4. * np.pi * r**2


    nd = i.mua/(i.mui*c.kboltz*i.Ti) * (n_sigmaD / (four_pi_r_squared*c.clight)*(Ln*exp_minustau + Li*phi) + n_squared_alphaB*Li/(c.clight*Qi) )
    phid = -n_sigmaD*phi - four_pi_r_squared * n_squared_alphaB / Qi
    taud = n_sigmaD

    return [nd, phid, taud]

def f_drain_noion_cf(y,r, params):
    n, tau = y
    Ln, Li, Qi, cf = params

    # to circumvent underflow in exp function for very large tau values
    if tau>100.: 
        exp_minustau = 0.0
        #print('tau>>>>>>100',tau)
    else: 
        exp_minustau = np.exp(-tau)

    # sptatial (r) derivatives
    nd = (n/cf)*i.sigmaD*Ln*exp_minustau/(c.kboltz*i.Tn*4.*np.pi*r**2*c.clight)
    taud = (n/cf)*i.sigmaD

    return [nd, taud]


def start_dens_B(Pbubble, T):
    """
    calculates density at the inner edge of the shell assuming pressure equlibrium between the bubble and the shell
    Pwind = Pshell = Ptherm + 2*Pmagnetic (in equipartition: Pturb + Pmagnetic = 2*Pmagnetic)
    for details, see Henney+2005
    for now, this is only used to pass on to CLOUDY and does not affect WARPFIELD
    :input:
    Pbubble: pressure of the bubble (usually ram pressure from winds/SNe or thermal pressure from X-ray emitting gas)
    T: temperature at inner edge of shell (usually 1e4 K if ionized, 1e2 K if neutral)
    :return:
    n: number density at inner edge of shell
    """

    a = (i.mui/i.mua)*c.kboltz*T # thermal pressure term
    b = c.BMW0**2./(4.*np.pi*c.nMW0**c.gmag) # magnetic field and turbulent term (equipartition assumed)
    data = (a,b,Pbubble)

    n_firstguess = 10.

    n = scipy.optimize.fsolve(Fdens, n_firstguess, args=data)

    return n

def Fdens(n,*data):
    a, b, Pbubble = data
    return n*(a+b*n**(1./3.)) - Pbubble
