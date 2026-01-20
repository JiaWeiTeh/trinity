#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:46:31 2022

@author: Jia Wei Teh

This script contains a wrapper that initialises the expansion of
shell.
"""

# libraries
import logging
import numpy as np
import datetime
import sys
import scipy
import src._functions.unit_conversions as cvt

#--
from src.sb99 import read_SB99



from src.phase0_init import (get_InitCloudProp, get_InitPhaseParam)
from src.phase0_init import get_InitPhaseParam
from src._output.simulation_end import write_simulation_end



from src.phase1_energy import run_energy_phase
from src.phase1_energy import run_energy_phase_modified
from src.phase1b_energy_implicit import run_energy_implicit_phase
from src.phase1b_energy_implicit import run_energy_implicit_phase_modified
from src.phase1c_transition import run_transition_phase
from src.phase1c_transition import run_transition_phase_modified
from src.phase2_momentum import run_momentum_phase
from src.phase2_momentum import run_momentum_phase_modified
import src._output.terminal_prints as terminal_prints
from src._input.dictionary import DescribedItem, DescribedDict, COOLING_PHASE_KEYS

# Initialize logger for this module
logger = logging.getLogger(__name__)



def start_expansion(params):
    """
    This wrapper takes in the parameters and feed them into smaller
    functions.

    Parameters
    ----------
    params : Object
        An object describing TRINITY parameters.

    Returns
    -------
    None.

    """
    
    # =============================================================================
    # Step 0: Preliminary
    # =============================================================================

    # TODO: put this in read_param, and make it depend on param file.
    # Note: Logging is configured in run.py before this function is called.
    # This fallback ensures logging works if main.py is called directly.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    # Record timestamp
    startdatetime = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("TRINITY Simulation Starting")
    logger.info("=" * 60)
    logger.info(f"Start time: {startdatetime}")
    terminal_prints.phase0(startdatetime)
    
    # =============================================================================
    # A: Initialising cloud properties.
    # =============================================================================

    # Step 1: Obtain initial cloud properties
    logger.info("Step 1: Initializing cloud properties...")
    get_InitCloudProp.get_InitCloudProp(params)
    logger.debug(f"Cloud radius: {params['rCloud'].value:.4f} pc")
    logger.debug(f"Core density: {params['nCore'].value*cvt.ndens_au2cgs:.4e} cm-3")

    # Step 2: Obtain parameters from Starburst99
    logger.info("Step 2: Loading Starburst99 stellar feedback data...")
    # Scaling factor for cluster masses. Though this might only be accurate for
    # high mass clusters (~>1e5) in which the IMF is fully sampled.
    f_mass = params['mCluster'] / params['SB99_mass']
    logger.debug(f"SB99 mass scaling factor: {f_mass:.4f}")
    # Get SB99 data and interpolation functions.
    SB99_data = read_SB99.read_SB99(f_mass, params)
    SB99f = read_SB99.get_interpolation(SB99_data)
    # TODO:
    # if tSF != 0.: we would actually need to shift the feedback parameters by tSF
    # update
    params['SB99_data'].value = SB99_data
    params['SB99f'].value = SB99f
    logger.info("SB99 data loaded and interpolation functions created")
    
    # Step 3: get cooling structure for CIE (since it is non time dependent).
    logger.info("Step 3: Loading CIE cooling curve...")
    # Values for non-CIE cooling curve will be calculated along the simulation, since it depends on time evolution.

    # for metallicity, here we need to take care of both CIE and nonCIE part.

    # maybe move this to read_params
    # if params['metallicity'].value != 1:
    #     sys.exit('Need to implement non-solar metallicity.')

    # get path to library
    # See example_pl.param for more information.
    cooling_path = params['path_cooling_CIE'].value
    logger.debug(f"Loading cooling curve from: {cooling_path}")
    # unpack from file
    logT, logLambda = np.loadtxt(cooling_path, unpack=True)
    # create interpolation
    cooling_CIE_interpolation = scipy.interpolate.interp1d(logT, logLambda, kind='linear')
    # update
    params['cStruc_cooling_CIE_logT'].value = logT
    params['cStruc_cooling_CIE_logLambda'].value = logLambda
    params['cStruc_cooling_CIE_interpolation'].value = cooling_CIE_interpolation

    logger.info(f"Loaded CIE cooling curve (T range: 10^{logT.min():.1f} - 10^{logT.max():.1f} K)")
    
    # =============================================================================
    # These two are currently not being needed. 
    # =============================================================================
    # create density law for cloudy
    # TODO: add CLOUDY support in the future.
    
    # =============================================================================
    # Begin simulation.
    # =============================================================================
    logger.info("=" * 60)
    logger.info("Initialization complete. Starting bubble expansion simulation...")
    logger.info("=" * 60)

    run_expansion(params)

    # Log completion
    enddatetime = datetime.datetime.now()
    elapsed = enddatetime - startdatetime
    logger.info("=" * 60)
    logger.info("TRINITY Simulation Complete")
    logger.info(f"End time: {enddatetime}")
    logger.info(f"Total elapsed time: {elapsed}")
    logger.info("=" * 60)

    # Write simulation end report to file
    try:
        exit_code = write_simulation_end(params)
        logger.info(f"Simulation end report written (exit code: {exit_code})")
    except Exception as e:
        logger.warning(f"Could not write simulation end report: {e}")
        exit_code = 99


    # ########### STEP 2: In case of recollapse, prepare next expansion ##########################
    # TODO: add loop so that this simulation starts over with old generation of parameter to simulate new starburst environment
    
    return 0
    
    
#%%
    
def run_expansion(params):
    """
    Model evolution of the cloud (both energy- and momentum-phase) until next recollapse or (if no re-collapse) until end of simulation
    """

    # =============================================================================
    # Prep for phases
    # =============================================================================
    # t0 = start time for Weaver phase
    # y0 = [r0, v0, E0, T0]
    # R2 = initial outer bubble/shell radius (pc)
    # v2 = initial velocity (pc/Myr)
    # Eb = initial energy
    # T0 = initial temperature (K)
    logger.info("Computing initial phase parameters (free-streaming -> Weaver transition)...")

    t0, r0, v0, E0, T0 = get_InitPhaseParam.get_y0(params)

    params['t_now'].value = t0
    params['R2'].value = r0
    params['v2'].value = v0
    params['Eb'].value = E0
    params['T0'].value = T0

    # =============================================================================
    # Phase 1a: Energy driven phase.
    # =============================================================================
    
    params['current_phase'].value = 'energy'

    logger.info("-" * 60)
    logger.info("PHASE 1a: Energy-driven phase (constant cooling)")
    logger.info("-" * 60)
    terminal_prints.phase('Entering energy driven phase (constant cooling)')

    phase1a_starttime = datetime.datetime.now()

    # Switch between original (Euler) and modified (solve_ivp) implementations
    # Set use_adaptive_solver=True to use the modified implementation with:
    #   - scipy.integrate.solve_ivp with adaptive RK45/RK23 stepping
    #   - Pure ODE functions that don't mutate params during integration
    #   - Segment-based integration with params updates only after success
    use_adaptive_solver = params.get('use_adaptive_solver', False)
    if hasattr(use_adaptive_solver, 'value'):
        use_adaptive_solver = use_adaptive_solver.value
    

    if use_adaptive_solver:
        logger.info("Using modified energy phase (adaptive solve_ivp)")
        run_energy_phase_modified.run_energy(params)
    else:
        logger.info("Using original energy phase (Euler integration)")
        run_energy_phase.run_energy(params)

    phase1a_endtime = datetime.datetime.now()
    phase1a_elapsed = phase1a_endtime - phase1a_starttime

    logger.info(f"Phase 1a complete. Duration: {phase1a_elapsed}")
    logger.debug(f"  Final R2 = {params['R2'].value:.6e} pc")
    logger.debug(f"  Final v2 = {params['v2'].value:.6e} pc/Myr")
    
    
    # record
    # try:
    #     params.flush()
    # except:
    #     pass

    # sys.exit('done with phase 1a')

    # ------------
    # checkout load_dict_halfway.py to understand how to load dictionaries
    # halfway through the simulation.
    # ------------

    # =============================================================================
    # Phase 1b: implicit energy phase
    # =============================================================================

    params['current_phase'].value = 'implicit'

    logger.info("-" * 60)
    logger.info("PHASE 1b: Energy-driven phase (adaptive cooling)")
    logger.info("-" * 60)
    terminal_prints.phase('Entering energy driven phase (adaptive cooling)')

    phase1b_starttime = datetime.datetime.now()


    if use_adaptive_solver:
        logger.info("Using modified energy phase (adaptive solve_ivp)")
        run_energy_implicit_phase_modified.run_phase_energy(params)
    else:
        logger.info("Using original energy phase (Euler integration)")
        run_energy_implicit_phase.run_phase_energy(params)
        

    phase1b_endtime = datetime.datetime.now()
    phase1b_elapsed = phase1b_endtime - phase1b_starttime
    logger.info(f"Phase 1b complete. Duration: {phase1b_elapsed}")

    # record
    # try:
    #     params.flush()
    # except:
    #     pass

    # make a function that interpolates density so that it goes from top to end of cloud.


    # Since cooling is not needed anymore after this phase, we reset values.
    # COOLING_PHASE_KEYS contains all cooling-related parameters that can be cleared.
    logger.debug("Resetting cooling-related parameters (no longer needed)...")
    params.reset_keys(COOLING_PHASE_KEYS)
    
    # =============================================================================
    # Phase 1c: transition phase
    # =============================================================================

    logger.info("-" * 60)
    logger.info("PHASE 1c: Transition phase (energy -> momentum)")
    logger.info("-" * 60)
    terminal_prints.phase('Entering transition phase (decreasing energy before momentum)')

    params['current_phase'].value = 'transition'

    if params['EndSimulationDirectly'].value == False:
        phase1c_starttime = datetime.datetime.now()

        if use_adaptive_solver:
            logger.info("Using modified transition phase (adaptive solve_ivp)")
            run_transition_phase_modified.run_phase_transition(params)
        else:
            logger.info("Using original transition phase (Euler integration)")
            run_transition_phase.run_phase_transition(params)

        phase1c_endtime = datetime.datetime.now()
        phase1c_elapsed = phase1c_endtime - phase1c_starttime
        logger.info(f"Phase 1c complete. Duration: {phase1c_elapsed}")
    else:
        logger.warning("EndSimulationDirectly=True, skipping transition phase")

    # try:
    #     params.flush()
    # except:
    #     pass

    # =============================================================================
    # Phase 2: momentum phase
    # =============================================================================

    logger.info("-" * 60)
    logger.info("PHASE 2: Momentum-driven phase")
    logger.info("-" * 60)
    terminal_prints.phase('Entering momentum phase')

    params['current_phase'].value = 'momentum'

    params['Eb'].value = 1

    if params['EndSimulationDirectly'].value == False:
        phase2_starttime = datetime.datetime.now()

        if use_adaptive_solver:
            logger.info("Using modified momentum phase (adaptive solve_ivp)")
            run_momentum_phase_modified.run_phase_momentum(params)
        else:
            logger.info("Using original momentum phase (Euler integration)")
            run_momentum_phase.run_phase_momentum(params)

        phase2_endtime = datetime.datetime.now()
        phase2_elapsed = phase2_endtime - phase2_starttime
        logger.info(f"Phase 2 (momentum) complete. Duration: {phase2_elapsed}")
    else:
        logger.warning("EndSimulationDirectly=True, skipping momentum phase")

    # Flush parameters to disk
    try:
        params.flush()
        logger.debug("Parameters flushed to disk")
    except Exception as e:
        logger.warning(f"Could not flush parameters: {e}")

    logger.info("All expansion phases complete")

    return 




def expansion_next(tStart, ODEpar, SB99_data_old, SB99f_old, mypath, cloudypath, ii_coll):

    return



#%%


# def expansion_next(tStart, ODEpar, SB99_data_old, SB99f_old, mypath, cloudypath, ii_coll):

#     aux.printl("Preparing new expansion...")
    
#     print('next_expansion!!')

#     ODEpar['tSF_list'] = np.append(ODEpar['tSF_list'], tStart) # append time of this SF event
#     dtSF = ODEpar['tSF_list'][-1] - ODEpar['tSF_list'][-2] # time difference between this star burst and the previous
#     ODEpar['tStop'] = i.tStop - tStart

#     aux.printl(('list of collapses:', ODEpar['tSF_list']), verbose=0)

#     # get new cloud/cluster properties and overwrite those keys in old dictionary
#     CloudProp2 = get_startvalues.make_new_cluster(ODEpar['Mcloud_au'], ODEpar['SFE'], ODEpar['tSF_list'], ii_coll)
#     for key in CloudProp2:
#         ODEpar[key] = CloudProp2[key]

#     # create density law for cloudy
#     __cloudy__.create_dlaw(mypath, i.namb, i.nalpha, ODEpar['Rcore_au'], ODEpar['Rcloud_au'], ODEpar['Mcluster_au'], ODEpar['SFE'], coll_counter=ii_coll)

#     ODEpar['Mcluster_list'] = np.append(ODEpar['Mcluster_list'], CloudProp2['Mcluster_au']) # contains list of masses of indivdual clusters

#     # new method to get feedback
#     SB99_data, SB99f = getSB99_data.full_sum(ODEpar['tSF_list'], ODEpar['Mcluster_list'], i.Zism, rotation=i.rotation, BHcutoff=i.BHcutoff, return_format='array')

#     # get new feedback parameters
#     #factor_feedback = ODEpar['Mcluster_au'] / i.SB99_mass
#     #SB99_data2 = getSB99_data.load_stellar_tracks(i.Zism, rotation=i.rotation, f_mass=factor_feedback,BHcutoff=i.BHcutoff, return_format="dict") # feedback of only 2nd cluster
#     #SB99_data = getSB99_data.sum_SB99(SB99f_old, SB99_data2, dtSF, return_format = 'array') # feedback of summed cluster --> use this
#     #SB99f = getSB99_data.make_interpfunc(SB99_data) # make interpolation functions for summed cluster (allowed range of t: 0 to (99.9-tStart))

#     if i.write_SB99 is True:
#         SB99_data_write = getSB99_data.SB99_conc(SB99_data_old, getSB99_data.time_shift(SB99_data, tStart))
#         warp_writedata.write_warpSB99(SB99_data_write, mypath)  # create file containing SB99 feedback info

#     # for the purpose of gravity it is important that we pass on the summed mass of all clusters
#     # we can do this as soon as all all other things (like getting the correct feedback) have been finished
#     ODEpar['Mcluster_au'] = np.sum(ODEpar['Mcluster_list'])

#     aux.printl(("ODEpar:", ODEpar), verbose=1)

#     t, r, v, E, T = run_expansion(ODEpar, SB99_data, SB99f, mypath, cloudypath)

#     return t, r, v, E, T, ODEpar, SB99_data, SB99f





# def warp_reconstruct(t, y, ODEpar, SB99f, ii_coll, cloudypath, outdata_file, cloudy_write=True, data_write=False, append=True):
#     """
#     reconstruct output parameters in warpversion 2.1;3
#     :param time: list or array of times SINCE THE LAST STAR FORMATION EVENT
#     :param y: [r,v,E,T]
#     :param ODEpar: cloud/cluster properties dictionary
#     :param SB99f: interpolation dictionary for summed cluster
#     :param ii_coll: number of re-collapses (0 if only 1 cluster present)
#     :param cloudypath: path to directory where cloudy input files will be stored
#     :return:
#     """

#     r, v, E, T = y

#     Ncluster = ii_coll + 1
#     ODEpar['Rsh_max'] = 0.0 # set max previously achieved shell radius to 0 again; necessary because we will the max radius in the following loop
#     SB99f_all = {}

#     for jj in range(0, Ncluster):
#         trash, SB99f_all[str(jj)] = getSB99_data.getSB99_main(init.Zism, rotation=init.rotation,
#                                                               f_mass=ODEpar['Mcluster_list'][jj] / init.SB99_mass,
#                                                               BHcutoff=init.BHcutoff)

#     # minimum time (in Myr) to wait after a star formation event before cloudy files are created
#     tmin = init.cloudy_tmin # default 1e-2
#     # if cloudy_dt or small_cloudy_dt are set to lower values this will prevent output which the user requested
#     # however, I fell it makes no sense to set cloudy_dt to a value smaller than this
#     # TO DO: one should just prevent the user from setting a smaller cloudy_dt than this

#     len_dat = len(t)

#     # add time of last SF event to time vector
#     Data = {'t':t+ODEpar['tSF_list'][-1], 'r':r, 'v':v, 'Eb':E, 'Tb':T}
    
#     if i.frag_cover == True:
#         try:
#             tcf,cfv=np.loadtxt(ODEpar['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODEpar['Mcluster_list']))+".txt", skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
#         except:
#             pass

#     for key in ("Mshell", "fabs", "fabs_i","fabs_n", "Pb", "R1", "n0", "nmax"):
#         Data[key] = np.ones(len_dat) * np.nan

#     for ii in range(0, len_dat):

#         t_real = t[ii] + ODEpar['tSF_list'][-1] # current time (measured from the very first SF event)

#         # reconstruct values like pressure
        
#         if i.frag_cover == True and int(os.environ["Coverfrac?"])==1:
#             if t[ii] <= tcf[1]:
#                 cfr=1
#             else:
#                 ide=aux.find_nearest_id(tcf, t[ii])
#                 cfr=cfv[ide]
#             aux_data = ODE_tot_aux.fE_tot_part1(t[ii], [r[ii], v[ii], E[ii], T[ii]], ODEpar, SB99f,cfs='recon',cf_reconstruct=cfr)
#         else: 
#             aux_data = ODE_tot_aux.fE_tot_part1(t[ii], [r[ii], v[ii], E[ii], T[ii]], ODEpar, SB99f)
       

#         if (cloudy_write is True and t[ii] >= tmin):  # do not write cloudy output very close after SF event; instead wait at least 1e-2 Myr

#             Lbol_list = np.ones(Ncluster) * np.nan  # current bolometric luminosities of all clusters
#             Age_list = t_real - ODEpar['tSF_list']  # current ages of all clusters
#             for jj in range(0, Ncluster):
#                 Lbol_list[jj] = SB99f_all[str(jj)]['fLbol_cgs'](Age_list[jj])

#             # write cloudy data to file
#             __cloudy__.create_model(cloudypath, ODEpar['SFE'], ODEpar['Mcloud_au'], init.namb, init.Zism, aux_data['n0_cloudy'], r[ii], v[ii],
#                          aux_data['Msh'], np.log10(SB99f['fLbol_cgs'](t[ii])), t_real,
#                          ODEpar['Rcloud_au'], ODEpar['nedge'],
#                          SB99model=init.SB99cloudy_file, shell=init.cloudy_stopmass_shell, turb=init.cloudy_turb,
#                          coll_counter=ii_coll, Tarr=Age_list, Larr=Lbol_list,
#                          Li_tot=SB99f['fLi_cgs'](t[ii]), Qi_tot=SB99f['fQi_cgs'](t[ii]),
#                          pdot_tot=SB99f['fpdot_cgs'](t[ii]), Lw_tot=SB99f['fLw_cgs'](t[ii]),
#                          Mcluster=np.sum(ODEpar['Mcluster_list']), phase=np.nan)

#         Data['Mshell'][ii] = aux_data['Msh']
#         for key in ("fabs", "fabs_i", "fabs_n", "Pb", "R1", "n0", "nmax"):
#             Data[key][ii] = aux_data[key]

#         ##########################################

#         # create bubble structure
#         # TO DO: check ii > 1
#         """
#         my_params = {"R2":r[ii], "v2":v[ii], "Eb":E[ii], "t_now":t[ii], 'pwdot':pwdot, 'pwdot_dot':pwdot_dot}

#         dt = t[ii]-t[ii-1]
#         Edot = (E[ii]-E[ii-1])/dt
#         Tdot = (T[ii]-T[ii-1])/dt
#         alpha = t[ii]/r[ii] * v[ii]
#         beta = state_eq.Edot_to_beta(Data["Pb"][ii], Data["R1"][ii], Edot, my_params) # TO DO: need to pass my_params
#         delta = state_eq.Tdot_to_delta(t[ii],T[ii],Tdot)

#         # TO DO: need to write function which saves bubble structure
#         Lb = bubble_structure2.calc_Lb(data_struc, Cool_Struc, 1, rgoal_f=init.r_Tb, verbose=0, plot=0, no_calc=False, error_exit=True,
#                 xtol=1e-6)
#         """


#         ########################

#     Data['Mcloud'] = np.ones(len_dat) * ODEpar['Mcloud_au']
#     Data['Mcluster'] = np.ones(len_dat) * ODEpar['Mcluster_au']

#     Data['Lbol'] = SB99f['fLbol_cgs'](t)
#     Data['Li'] = SB99f['fLi_cgs'](t)
#     Data['Ln'] = SB99f['fLn_cgs'](t)
#     Data['Qi'] = SB99f['fQi_cgs'](t)
#     Data['Fram'] = SB99f['fpdot_cgs'](t)
#     Data['Lw'] = SB99f['fLw_cgs'](t)

#     # write data to file
#     if data_write is True:
#         write_outdata_table(Data, outdata_file, append=append)

#     #print "DEBUG cloudy_write = ", cloudy_write

#     lum_bubble_placeholder = "WARP2_LOG10_LUM_BUBBLE_CL" # placeholder name for total luminosity of bubble

#     #cloudypath = "/home/daniel/Documents/work/loki/code/warpfield/output_test/warpfield2/new3/Z1.00/M6.00/n500.0_nalpha0.00_nc2.70/SFE5.00/cloudy/"
#     #ODEpar = {}; ODEpar['mypath'] = "/home/daniel/Documents/work/loki/code/warpfield/output_test/warpfield2/new3/Z1.00/M6.00/n500.0_nalpha0.00_nc2.70/SFE5.00/"
#     #A = ascii.read(ODEpar['mypath']+'M6.00_SFE5.00_n500.0_Z1.00_data.txt'); t = A['t'], T = A['Tb']

#     # compare list of cloudy files and bubble files
#     # if there is a cloudy file in the energy phase without a corresponding bubble file, we need to make that missing bubble file
#     if (cloudy_write is True and init.savebubble is True):
#         cloudyfiles_all = [f for f in os.listdir(cloudypath) if os.path.isfile(os.path.join(cloudypath, f))]
#         cloudyfiles = sorted([ii for ii in cloudyfiles_all if ('shell' in ii and '.in' in ii)]) # take only shell input files
#         cf = ["" for x in range(len(cloudyfiles))]
#         # create array containing ages of cloudy files
#         for ii in range(0,len(cloudyfiles)):
#             mystring = cloudyfiles[ii]
#             idx = [pos for pos, char in enumerate(mystring) if char == '.']
#             cf[ii] = mystring[idx[-2]-1:idx[-1]]
#         age_c = np.array([float(i) for i in cf]) * 1e-6 # convert to Myr
#         # we now have a list with all ages for which there are cloudy (shell) files


#         bubblepath = os.path.join(ODEpar['mypath'], 'bubble/')
#         bubblefiles_all = [f for f in os.listdir(bubblepath) if os.path.isfile(os.path.join(bubblepath, f))]
#         bubblefiles = sorted([ii for ii in bubblefiles_all if ('bubble' in ii and '.dat' in ii)])

#         #print "DEBUG, bubblepath, bubblefiles", bubblepath, bubblefiles

#         # find times where bubble burst
#         tburst = np.array([0.])
#         for ii in range(1,len(t)):
#             if T[ii] <= 2e4:
#                 if T[ii-1] > 2e4: # if the bubble bursts, the temperature drops to 1e4 K
#                     tburst = np.append(tburst, t[ii])
#         if T[-1] > 2e4: # case: simulation ended in energy phsae
#             tburst = np.append(tburst,1.01*t[-1]) # case where the bubble never burst (take some arbitrary high value)

#         bf = ["" for x in range(len(bubblefiles))]
#         # create array containing ages of bubble files
#         for ii in range(0,len(bubblefiles)):
#             mystring = bubblefiles[ii]
#             idx = [pos for pos, char in enumerate(mystring) if char == '.']
#             bf[ii] = mystring[idx[-2]-1:idx[-1]]
#             #if bf[ii] not in cf:
#             #    rmfile = os.path.join(bubblepath, bubblefiles[ii])
#             #    os.remove(rmfile) # remove bubble files which do not have cloudy counterparts with same age
#         age_b = np.array([float(i) for i in bf]) * 1e-6 # convert to Myr
#         # we now have a list with all ages for which there are bubble files

#         #print "DEBUG age_c, age_b", age_c, age_b, tburst

#         # in case cluster winds are to be modelled: copy all files from cloudy folder to bubble folder
#         # in case cluster winds are not to be modelled: copy only those shell.in and static.ini file from the cloudy folder where the expansion is in the energy limit (will be done later)
#         if init.cloudy_CWcavitiy is True:
#             for file in glob.glob(os.path.join(cloudypath, '*.in*')):
#                 shutil.copy(file, bubblepath)


#         # create interpolated bubble profiles where there is a cloudy file without a bubble file in the energy phase
#         for jj in range(1,len(tburst)): # go through each energy-driven phase seperately (each recollapse leads to another energy phase)
#             msk = (age_b > tburst[jj-1]) * (age_b <= tburst[jj]+1e-6)
#             age_b0 = age_b[msk] # list of bubble file ages in respective energy phase
#             msk = (age_c >= age_b0[0]) * (age_c <= age_b0[-1])
#             age_c0 = age_c[msk] # list of shell file ages in respective energy phase

#             for ii in range(0,len(age_c0)):
#                 age1e7_str = ('{:0=5.7f}e+07'.format(age_c0[ii] / 10.))  # age in years (factor 1e7 hardcoded), naming convention matches naming convention for cloudy files
#                 bubble_base = os.path.join(bubblepath, "bubble_SB99age_" + age1e7_str) # base name of bubble data file
#                 bubble_file = bubble_base + ".in"

#                 # check whether there is a bubble file for this shell file
#                 if age_c0[ii] not in age_b0: # there is a cloudy file but not a corresponding bubble file... we need to make one
#                     bubble_interp(age_c0[ii], age_b0, bubblepath, bubble_base, bubblefiles) # make new bubble file by interpolation

#                 # we are now sure a bubble file exists for the current t
#                 # now need to copy the lines with "table star" and "luminosity total" from the shell file to the bubble file (when the bubble file was created this information was not available)

#                 # copy shell and static file from cloudy directory to bubble directory
#                 # 1) copy shell file to bubble directory, modify extension, and modify prefix inside
#                 if init.cloudy_CWcavitiy is False:  # if colliding winds are to be modelled, copying already happened. If no colliding winds, we only want to copy files in energy phase. Do this now!
#                     for file in glob.glob(os.path.join(cloudypath, 'shell*' + age1e7_str + '.in')): # this is exactly 1 file
#                         #print "copy...", file, bubblepath
#                         shutil.copy(file, bubblepath)
#                 for file in glob.glob(os.path.join(bubblepath, 'shell*' + age1e7_str + '.in')): # this is exactly 1 file
#                     luminosity_line, table_star_line, warp_comments = repair_shell(file) # repair (and rename as .ini) the shell.in file in the bubble folder

#                 # 2) bubble file gets table star and luminosity total from old shell file
#                 repair_bubble(bubble_file, table_star_line, luminosity_line, warp_comments)

#                 # 3) copy static file to bubble directory and modify prefix inside
#                 if init.cloudy_CWcavitiy is False: # if colliding winds are to be modelled, copying already happened. If no colliding winds, we only want to copy files in energy phase. Do this now!
#                     for file in glob.glob(os.path.join(cloudypath, 'static*'+age1e7_str+'.ini')): # this is exactly 1 file
#                         shutil.copy(file, bubblepath)

#         # repair all static files
#         for file in glob.glob(os.path.join(bubblepath, 'static*'+'.ini')):
#             repair_static(file)

#         # now remove superfluous bubble files
#         for ii in range(0,len(bubblefiles)):
#             mystring = bubblefiles[ii]
#             idx = [pos for pos, char in enumerate(mystring) if char == '.']
#             bf[ii] = mystring[idx[-2]-1:idx[-1]]
#             if bf[ii] not in cf:
#                 rmfile1 = os.path.join(bubblepath, bubblefiles[ii]) # this is the .dat file
#                 os.remove(rmfile1) # remove bubble .dat files
#                 rmfile2 = rmfile1[:-3] + "in"
#                 os.remove(rmfile2)  # remove bubble files which do not have cloudy counterparts with same age


#         if init.cloudy_CWcavitiy is True:
#             # add Chevalier and Clegg profile to existing bubble.in files
#             Bfile_list = glob.glob(os.path.join(bubblepath, 'bubble_SB99age_*.in'))
#             for Bfile in Bfile_list:
#                 Lmech, pdot, [rB_start, rB_stop], lines = get_Lmechpdotrstart(Bfile)
#                 rend = (1.0 - 1.e-5)*rB_start # stop radius for colliding winds (make it just slightly smaller than the start radius of the bubble (Weaver) component)
#                 Mdot, vterm = getSB99_data.getMdotv(pdot, Lmech)

#                 cluster_rad_cm = get_cluster_rad(init.fixed_cluster_radius, rB_stop) * c.pc

#                 R,V,rho,T,P = ClusterWind_profile.CW_profile(vterm, Mdot, cluster_rad_cm, rend*c.pc, Rstart=3.0e-3*c.pc)
#                 ndens = rho / init.mui # number density of H atoms
#                 CW_dlaw = "continue -35.0 {:.9f}\n".format(np.log10(ndens[0]))
#                 CW_tlaw = "continue -35.0 {:.9f}\n".format(np.log10(T[0]))
#                 for ll in range(0,len(R)):
#                     CW_dlaw += "continue {:.9f} {:.9f}\n".format(np.log10(R[ll]), np.log10(ndens[ll]))
#                     CW_tlaw += "continue {:.9f} {:.9f}\n".format(np.log10(R[ll]), np.log10(T[ll]))
#                 with open(Bfile, "w") as f:
#                     insert_dlaw = False
#                     insert_tlaw = False
#                     for line in lines:
#                         if "radius" in line and "linear parsec" in line and "stop" not in line: # need to modify start radius (we now start from the cluster center, i.e. r = 0)
#                             f.write("radius 2.0e-02 linear parsec\n") # pick a small number here (but not too small or cloudy gets problems)
#                         elif "continue " not in line: # lines with "continue" (which are part of the density or temperature profile) must not just be copied
#                             f.write(line) # lines which do not include "continue" can be copied
#                             if "dlaw table radius" in line:
#                                 insert_dlaw = True
#                             elif "tlaw table radius" in line:
#                                 insert_tlaw = True
#                         elif "continue " in line:
#                             if insert_dlaw is True:
#                                 # now insert new profile
#                                 f.write(CW_dlaw)
#                                 insert_dlaw = False
#                             elif insert_tlaw is True:
#                                 # now insert new profile
#                                 f.write(CW_tlaw)
#                                 insert_tlaw = False
#                             else:
#                                 f.write(line)

#             # create new bubble files for shell.in files in the momentum phase. For these new bubble files just use the Chevalier and Clegg profile
#             SHfile_list = glob.glob(os.path.join(bubblepath, 'shell_SB99age_*.in'))
#             for ii in range(0,len(SHfile_list)):
#                 SHfile = SHfile_list[ii]
#                 Bfile = SHfile.replace("shell_SB99age", "bubble_SB99age")
#                 bubble_base = Bfile[:-3]
#                 if Bfile not in Bfile_list: # all bubble files in Bfile_list already exist. Now go through the ones which don't exist

#                     Lmech, pdot, [rSH_start, _], lines = get_Lmechpdotrstart(SHfile)

#                     cluster_rad_cm = get_cluster_rad(init.fixed_cluster_radius, rSH_start) * c.pc

#                     rend = (1.0 - 1e-10)*rSH_start  # stop radius for colliding winds (make it just slightly smaller than the start radius of the shell component)
#                     Mdot, vterm = getSB99_data.getMdotv(pdot, Lmech)
#                     R, V, rho, T, P = ClusterWind_profile.CW_profile(vterm, Mdot, cluster_rad_cm, rend * c.pc, Rstart=1.8e-2*c.pc)
#                     ndens = rho / init.mui  # number density of H atoms

#                     bub_savedata = {"r_cm": R, "n_cm-3": ndens, "T_K": T}
#                     name_list = ["r_cm", "n_cm-3", "T_K"]
#                     tab = Table(bub_savedata, names=name_list)
#                     formats = {'r_cm': '%1.9e', 'n_cm-3': '%1.4e', 'T_K': '%1.4e'}
#                     outname = bubble_base + ".dat"
#                     tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)

#                     # write bubble.in file for newly created (interpolated) bubble.dat file
#                     __cloudy_bubble__.write_bubble(outname, Z=init.Zism)

#                     luminosity_line, table_star_line, warp_comments = repair_shell(SHfile) # also renames shell file to .ini
#                     repair_bubble(Bfile, table_star_line, luminosity_line, warp_comments)

#                     #f.write('table read file = "' + prefix.replace("static_SB99age_", "shell_SB99age_") + '.con"\n')

#     return 0

#%%


