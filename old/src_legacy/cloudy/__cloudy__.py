#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:41:38 2023

@author: Jia Wei Teh

This script converts Expansion-Solver models into 1D Cloudy inputs.
"""

import numpy as np
import os.path
import sys
import astropy.constants as c
#--
from src.phase_general import set_phase
# TODO: fix this.
# from src.warpfield.functions import nameparser

# TODO
# Add warpfield_params!


# TODO: give this an option as param
cloudy_use_relative_path = True
# write table of emission lines to shell.in and static.in files
write_line_list = True

warpversion = 2.0

# cloudy B-field (not used for shell structure in WARPFIELD)
# sets the inner shell density passed to cloudy and sets the command "magnetic field tangled ... 2" in the cloudy .in files
B_cloudy = False
    
def create_model(outdir, SFE, M_cloud, n0_cloud, Z_cloud, n0_shell, \
                r0_shell, v0_shell, M_shell, SB99_log10_L0, t0, 
                rcloud_au, nedge, 
                warpfield_params,
                shell = True,
                SB99model='filler',\
                turb = "4 km/s no pressure", coll_counter=0, Tarr=[], Larr=[], 
                Li_tot=np.nan, Qi_tot=np.nan, pdot_tot=np.nan, Lw_tot=np.nan,
                Mcluster=np.nan, phase = 1.0, cloudy_verbosity = 1):
    """
    creates a cloudy .in file
    ---
    The old function looks like this:
        create_model(outdir, SFE, M_cloud, n0_cloud, Z_cloud, n0_shell, \
                r0_shell, v0_shell, M_shell, SB99_log10_L0, t0, 
                rcloud_au, nedge, 
                shell = True,
                SB99model='1e6cluster_rot_Z0014_BH' + str(int(warpfield_params.SB99_BHCUT)),\
                turb = "4 km/s no pressure", coll_counter=0, Tarr=[], Larr=[], 
                Li_tot=np.nan, Qi_tot=np.nan, pdot_tot=np.nan, Lw_tot=np.nan,
                Mcluster=np.nan, phase = 1.0, cloudy_verbosity = 1)
    where these are the paramters:
        :param outdir: directory in which cloudy .in file is written
        :param SFE: star formation efficiency (currently not needed)
        :param M_cloud: cloud mass in Msol
        :param n0_cloud: cloud number density in 1/ccm (only constant density profiles work!)
        :param Z_cloud: metallicity of cloud (and shell) in solar metallicities
        :param n0_shell: number density on the inside of shell in 1/ccm
        :param r0_shell: inner radius of shell in pc
        :param v0_shell: velocity of shell in km/s
        :param M_shell: mass of the shell in Msol
        :param SB99_log10_L0: log10 of bolometric luminosity of 1st cluster (this would not be necesary if we had Larr, but Larr is optional)
        :param t0: time (since beginning of warpfield simulation) in Myr
        :param nedge: density at edge of cloud
        :param shell: Is there a shell or not? (boolean)
        :param SB99model: name of SB99 model (string without ".mod" suffix; necessary to get stellar spectra)
        :param turb: turbulence (not implemented)
        :param coll_counter: counter of star formation events (int)
        :param Tarr: array containing the age of the various clusters
        :param Larr: array containing the log10 of bolometric luminosities of the various clusters
        :param Li_tot: sum of ionizing luminosities of all clusters
        :param Qi_tot: sum of emission rate of ionizing photons of all clusters
        :param pdot_tot: sum of momentum injection rates of all clusters
        :param Lw_tot: sum of mechanical luminosities of all clusters
        :param Mcluster: sum of stellar mass of all clusters in Msol
        :param phase: current evolution phase (see phase_lookup.py)
        :return: 0, if everything went well
    """
    # this is because warpfield_params cannot be called in the function yet,
    # and I don't want to mess up the whole function.
    SB99model = '1e6cluster_rot_Z0014_BH' + str(int(warpfield_params.SB99_BHCUT))
    # TODO in the future: clean up and only use warpfield_params.
    
    # check whether directory for output exists. If not, create it!
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # string to denote additional, i.e. not essential information
    key_moreinfo = "warpfield"+str(3.0)+" additional info"

    # retrieve values
    BMW0 = warpfield_params.BMW0
    nMW0 = warpfield_params.nMW0
    gmag = warpfield_params.gamma_mag
    
    # B-field properties
    # B-field at inner edge of shell
    logB0_shell = np.log10(BMW0*(n0_shell/nMW0)**(gmag/2.)) 
    # B-field in uniform part of cloud
    logB_cloud = np.log10(BMW0*(n0_cloud/nMW0)**(gmag/2.)) 
    # B-field in low density ambient medium (outside cloud)
    logB_ambient = np.log10(BMW0*(warpfield_params.nISM/nMW0)**(gmag/2.)) 

    SB99_age = t0 * 1e6 # convert internal Myr to years for SB99 model
    #SB99_age_str = ('{:0=5.4f}e+07'.format(SB99_age / 1e7))
    SB99_age_str = ('{:0=5.7f}e+07'.format(SB99_age / 1e7)) # age in years (factor 1e7 hardcoded), naming convention matches naming convention for bubble structure files
    log10_M_cloud = np.log10(M_cloud) # assume units are Msol
    log10_M_shell = np.log10(M_shell)
    log10_M_max = max(np.log10(M_cloud), np.log10(M_shell))
    log10_Z_cloud = np.log10(Z_cloud) # assume units are Zsol
    log10_n0_cloud = np.log10(n0_cloud) # assume units are 1/ccm (cubic centimetre)
    log10_n0_shell = np.log10(n0_shell) # assume units are 1/ccm
    log10_n0_ambient = np.log10(warpfield_params.nISM)
    v0_static = 0.0

    # shell has dissolved
    
    if phase == set_phase.dissolve: 
        tmp_str = "dig_SB99"
    elif shell == True:
        tmp_str = "shell_SB99"
    else:
        tmp_str = "cloud_SB99"

    # tell cloudy where the dlaw file is stored
    # we only have a dlawfile if there is a density profile (not constant)
    no_powerlaw = (warpfield_params.dens_profile == 'pL_prof' and warpfield_params.dens_a_pL == 0)
    # If there are no density profile, or if it has dissolved:
    if no_powerlaw: 
        outdir_up = nameparser.dir_up(outdir) # subtract 'cloudy/' from relative pathname to get 1 directory up
        outdir_up = get_set_save_prefix(outdir_up, use_relative_path= cloudy_use_relative_path) # dump cwd if using relative path as cloudy output convention
        dlawfile = os.path.join(outdir_up, "dlaw" + str(coll_counter) + ".ini")# relative path to dlaw file
    else:
        dlawfile = np.nan


    # if there is a shell, create a .in file for the shell (and perhaps for a static component around it)
    if shell == True:
        stopmass_shell = (log10_M_shell + np.log10(c.M_sun.cgs.value))
        #check whether a static component exists
        if (log10_M_cloud + np.log10(c.M_sun.cgs.value)) > stopmass_shell:
            static_exist = True
        else:
            static_exist = False
        cloudy_prefix = os.path.join(outdir, tmp_str + "age_" + SB99_age_str)
        write_shell(cloudy_prefix, Z_cloud, log10_n0_shell, coll_counter, logB0_shell, SB99model, SB99_age, SB99_log10_L0,
                Tarr, t0, Larr, r0_shell, stopmass_shell, key_moreinfo, log10_M_cloud, v0_shell, phase, Mcluster, Li_tot, Qi_tot, 
                pdot_tot, Lw_tot, rcloud_au, nedge, 
                warpfield_params,
                static_exist=static_exist,
                cloudy_verbosity = cloudy_verbosity, dlawfile = np.nan)
        # if the cloud is not fully swept, we need a static component
        if (log10_M_cloud + np.log10(c.M_sun.cgs.value)) > stopmass_shell:
            stopmass_static = np.log10(M_cloud - M_shell) + np.log10(c.M_sun.cgs.value)
            static_prefix = os.path.join(outdir, "static_SB99" + "age_" + SB99_age_str)
            write_static(static_prefix, Z_cloud, log10_n0_cloud, coll_counter, logB_cloud, SB99model, SB99_age, SB99_log10_L0,
                Tarr, t0, Larr, r0_shell, stopmass_static, key_moreinfo, log10_M_cloud, v0_static, phase, Mcluster, Li_tot, Qi_tot, pdot_tot, Lw_tot, rcloud_au, nedge,
                warpfield_params,
                direct_illumination=False, continuum_file=(cloudy_prefix+".con"), cloudy_verbosity = cloudy_verbosity, dlawfile = dlawfile)

    # if the shell has dissolved, we need only a static DIG component
    elif phase == set_phase.dissolve:
        stopmass_ambient = log10_M_max + np.log10(c.M_sun.cgs.value)
        static_prefix = os.path.join(outdir, tmp_str + "age_" + SB99_age_str)
        # allow non-zero velocity
        write_static(static_prefix, Z_cloud, log10_n0_ambient, coll_counter, logB_ambient, SB99model, SB99_age, SB99_log10_L0,
                Tarr, t0, Larr, r0_shell, stopmass_ambient, key_moreinfo, log10_M_cloud, v0_shell, phase, Mcluster, Li_tot, Qi_tot, pdot_tot, Lw_tot, rcloud_au, nedge,
                warpfield_params,
                direct_illumination=True, cloudy_verbosity = cloudy_verbosity, dlawfile = np.nan)

    return 0

def get_set_save_prefix(my_prefix, use_relative_path):
    """
    dumps cwd if use_relative path is true, otherwise does nothing
    output does not have a trailing "/"
    :param my_prefix:
    :param use_relative_path:
    :return:
    """

    if use_relative_path == True:
        cwd = os.getcwd()
        # now model has lost the piece of the path which leads to the current working dir -> relative path
        set_save_prefix = os.path.relpath(my_prefix, cwd) 
    else:
        set_save_prefix = my_prefix

    return set_save_prefix

def write_shell(my_prefix, Z_cloud, log10_n0, coll_counter, logB0, SB99model, SB99_age, SB99_log10_L0,
                Tarr, t0, Larr, r0_shell, stopmass, key_moreinfo, log10_M_cloud, v0, phase, Mcluster, Li_tot, Qi_tot, pdot_tot, Lw_tot, rcloud_au, nedge, 
                warpfield_params,
                static_exist=True,
                cloudy_verbosity = 1, dlawfile = np.nan):
    ############################### 1st (expanding) part ######################################

    # if ((i.output_verbosity > 0) and (cloudy_verbosity >= 1)):
    print("saving cloudy input file to ", my_prefix)

    # open up to write
    model_input_file = open(my_prefix + ".in", mode='w')

    model_input_file.writelines("# use this file for the expanding shell\n")

    ############################### create template with some default settings #########################

    set_save_prefix = get_set_save_prefix(my_prefix, cloudy_use_relative_path)
    create_template(model_input_file, set_save_prefix, 
                    warpfield_params,
                    Z_cloud=Z_cloud, logB0=logB0, log10_n0=log10_n0, stopmass=stopmass)

    model_input_file.writelines("constant pressure\n")  # hydrostatic

    model_input_file.writelines("radius %0.16e linear parsec\n" % r0_shell)

    # this component is moving, i.e. a 'sphere expanding'
    model_input_file.writelines("sphere expanding\n")

    if (not np.isnan(Z_cloud)):
        if Z_cloud >= warpfield_params.z_nodust:
            model_input_file.writelines("grains PAH function sublimation\n") # better use sublimation in shell

    # only if there is no static component (i.e. the cloud has been fully swept by the shell) add the interstellar radiation field to the shell
    if static_exist == False:
        model_input_file.writelines("# table ISM\n") # commented table ISM out because when at late times the shell is at a very low density (huge volume) the X-ray from the ISRF is very high and dominates over the X-ray from the cluster (--> not good when we want to make statements about the X-ray of a cluster)

    ############################# add in additional info ###########################
    add_info_data = {"Mcloud": log10_M_cloud + np.log10(c.M_sun.cgs.value), "velocity_lin_kms": v0, "clustermass_log10_g": np.log10(Mcluster) + np.log10(c.M_sun.cgs.value),\
                     "ionizingluminosity_log10_erg_s": np.log10(Li_tot), "ionizingemissionrate_log10_s": np.log10(Qi_tot),\
                     "mechanical_luminosity_log10_erg/s": np.log10(Lw_tot), "momentum_injection_rate_log10_dyne": np.log10(pdot_tot)}
    if not np.isnan(phase): add_info_data["phase"] = phase
    additional_info(model_input_file, key_moreinfo, add_info_data)

    ############################# add line list ####################################
    # write table of emission lines to shell.in and static.in files
    if write_line_list == True:
        create_line_list_file(model_input_file)

    ############################# spectrum (THIS NEEDS TO GO TO THE END) ###########
    # tell cloudy where to get spectra and write table of luminosities
    # could add check here, whether direct_illumination is True
    write_spectrum(model_input_file, coll_counter, SB99model, SB99_age, SB99_log10_L0, Tarr, t0, Larr,
                   warpfield_params)

    # nothing except the file close command can come after this line!

    model_input_file.close()

    return 0


def write_static(my_prefix, Z_cloud, log10_n0, coll_counter, logB0, SB99model, SB99_age, SB99_log10_L0,
                Tarr, t0, Larr, r0_shell, stopmass, key_moreinfo, log10_M_cloud, v0, phase, Mcluster, Li_tot, Qi_tot, pdot_tot, Lw_tot, rcloud_au, nedge,
                warpfield_params,
                 direct_illumination = False, continuum_file = "", cloudy_verbosity = 1, dlawfile = np.nan):
    ################################# 2nd (static) part ######################################
    # only need this if swept up (shell) mass is less than total cloud mass

    # open file to write
    # if this part is not directly illuminated we need to wait for cloudy to run on the inner (directly illuminated) part first
    # that's why we can not create a cloudy .in file but only an .ini file (which will need to get updated with info from the directly illuminated part)
    if direct_illumination == False:
        model_input_file = open(my_prefix + ".ini", mode='w')
        model_input_file.writelines("# use for static (not yet swept up) part of the cloud\n")
    # if this part is directly illuminated we can directly create a cloudy .in file
    elif direct_illumination == True:
        model_input_file = open(my_prefix + ".in", mode='w')
        model_input_file.writelines("# use for static ISM when the shell has dissolved\n")
        # TODO: verbosity here
        # if ((i.output_verbosity > 0) and (cloudy_verbosity >= 1)):
        print("saving cloudy input file to ", my_prefix)

    set_save_prefix = get_set_save_prefix(my_prefix, cloudy_use_relative_path)
    no_powerlaw = (warpfield_params.dens_profile == 'pL_prof' and warpfield_params.dens_a_pL == 0)
    # If there are no density profile, or if it has dissolved:
    if (no_powerlaw or phase == set_phase.dissolve):
        create_template(model_input_file, set_save_prefix, 
                        warpfield_params,
                        Z_cloud=Z_cloud, logB0=logB0, log10_n0=log10_n0, stopmass=stopmass)
        model_input_file.writelines("# table ISM\n") # commented table ISM out because when at late times the shell is at a very low density (huge volume) the X-ray from the ISRF is very high and dominates over the X-ray from the cluster (--> not good when we want to make statements about the X-ray of a cluster)
        model_input_file.writelines("constant density\n")  # this is not true if there is a density profile
    else:
        # if you want to have a stop radius instead of a stop mass, set stopmass in the next line to np.nan and comment out the line after
        create_template(model_input_file, set_save_prefix, 
                        warpfield_params,
                        Z_cloud=Z_cloud, logB0=np.nan, log10_n0=np.nan, stopmass=stopmass)
        #model_input_file.writelines("stop radius linear parsec %0.3f\n" % (rcloud_au)) # potential worry: cloud radius might be small than outer radius of shell, that's why we use stop mass again
        model_input_file.writelines("init \"%s\"\n" % (dlawfile))

    # this component is not moving, i.e. a 'sphere static'
    model_input_file.writelines("sphere static\n")
    if (not np.isnan(Z_cloud)):
        if Z_cloud >= warpfield_params.z_nodust:
            model_input_file.writelines("grains PAH\n") # static component does not need sublimation

    #################### add in additional info ###########################
    add_info_data = {"Mcloud": log10_M_cloud + np.log10(c.M_sun.cgs.value), "velocity_lin_kms": v0, "clustermass_log10_g": np.log10(Mcluster) + np.log10(c.M_sun.cgs.value),\
                     "ionizingluminosity_log10_erg_s": np.log10(Li_tot), "ionizingemissionrate_log10_s": np.log10(Qi_tot),\
                     "mechanical_luminosity_log10_erg/s": np.log10(Lw_tot), "momentum_injection_rate_log10_dyne": np.log10(pdot_tot)}
    if not np.isnan(phase): add_info_data["phase"] = phase
    additional_info(model_input_file, key_moreinfo, add_info_data)

    #################### add line list ####################################
    if write_line_list == True:
        create_line_list_file(model_input_file)

    # if this part is not directly illuminated, get cloudy continuum file from inner illuminated part
    if direct_illumination == False:
        # model_input_file.writelines("radius PLACEHOLDER linear parsec\n")
        model_input_file.writelines("table read file = \"%s\"\n" % (continuum_file))
        model_input_file.writelines("luminosity total %0.3f\n" % (SB99_log10_L0))
    elif direct_illumination == True:
        write_spectrum(model_input_file, coll_counter, SB99model, SB99_age, SB99_log10_L0, Tarr, t0, Larr,
                       warpfield_params)
        # if this part is directly ilumminated by the star cluster, we know its inner radius
        model_input_file.writelines("radius %0.16e linear parsec\n" % r0_shell)

    model_input_file.close()

    return 0

def write_spectrum(model_input_file, coll_counter, SB99model, SB99_age, SB99_log10_L0, Tarr, t0, Larr,
                   warpfield_params):
    """
    write spectrum
    CAUTION: There is one version of this routine for warpversion 1.0 and another version for warpversion >= 2.0
    TO DO: Merge
    :param model_input_file:
    :param coll_counter:
    :param SB99model:
    :param SB99_age:
    :param SB99_log10_L0:
    :param Tarr:
    :param t0:
    :param Larr:
    :return:
    """

    ##################### WARPVERSION 1.0 ############################
    if warpversion == 1.0:

        # case: 1 cluster (single star formation event)
        if coll_counter == 0:
            model_input_file.writelines(
                "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age, warpfield_params.SB99_age_min])))
            model_input_file.writelines("luminosity total %0.5f\n" % (SB99_log10_L0))
        # case: several clusters (more than 1 star formation event)
        elif coll_counter >= 1:
            for ii in range(0, coll_counter + 1):  # loop over lines in Tarr (each line is a star cluster)
                # find correct entry in SB99 data
                jj = 0
                while Tarr[ii, jj] < t0:
                    jj += 1
                # jj is the correct index
                SB99_age_ii = (Tarr[ii, jj] - Tarr[ii, 0]) * 1e6
                model_input_file.writelines(
                    "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age_ii, warpfield_params.SB99_age_min])))
                model_input_file.writelines("luminosity total %0.5f\n" % (Larr[ii, jj]))

        return 0


    ##################### WARPVERSION >= 2.0 (not tested!) ############################
    elif warpversion >= 2.0:

        for ii in range(0, coll_counter + 1):  # loop over lines in Tarr
            SB99_age_ii = Tarr[ii] * 1e6
            model_input_file.writelines(
                "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age_ii, warpfield_params.SB99_age_min])))
            if Larr[ii] < 200:
               model_input_file.writelines("luminosity total %0.5e\n" % (Larr[ii]))
            else:
               model_input_file.writelines("luminosity total linear %0.5e\n" % (Larr[ii]))

        return 0




def additional_info(model_input_file, key_moreinfo, data_dict):
    # add additional information to cloudy files

    for key in data_dict.keys():
        model_input_file.writelines("# %s %s %0.8f\n" % (key_moreinfo, key, data_dict[key]))
    return 0

def create_template(model_input_file, set_save_prefix, 
                    warpfield_params,
                    Z_cloud=np.nan, logB0=np.nan, log10_n0=np.nan, stopmass=np.nan):
    # some standard commands for cloudy
    model_input_file.writelines("set save prefix \"%s\"\n" % (set_save_prefix))  # model save prefix

    model_input_file.writelines("iterate\n")
    #model_input_file.writelines("stop temperature 5 K linear\n")
    model_input_file.writelines("cosmic rays background\n")
    # not sure the next few lines should be included (currently commented)
    #model_input_file.writelines("# cosmic rays background 1.60\n")
    #model_input_file.writelines("# stop Av extended 6.36\n")
    model_input_file.writelines("stop temperature off\n")

    model_input_file.writelines("database H2\n")
    # model_input_file.writelines("turbulence equipartition no pressure\n")
    model_input_file.writelines("turbulence = 4.0 km/s\n")  # To do: shouldn't be a fixed value
    model_input_file.writelines("CMB\n")
    model_input_file.writelines("save transmitted continuum last \".con\"\n")
    model_input_file.writelines("save continuum last \".full_con\"\n")
    model_input_file.writelines("save molecule \".mol\" last\n")
    model_input_file.writelines("save pressure \".press\" last\n")
    model_input_file.writelines("save pdr \".pdr\" last\n")
    model_input_file.writelines('save hydrogen conditions last ".hyd"\n')
    model_input_file.writelines("save line list absolute \".lines\" \"default_bubble_lines.lst\" last no hash\n")
    # model_input_file.writelines("table ism\n") # interstellar radiation field only for outermost component

    if (not np.isnan(Z_cloud)):
        log10_Z_cloud = np.log10(Z_cloud)
        if Z_cloud >= warpfield_params.z_nodust:
            model_input_file.writelines("abundance orion\n") #changed hii to orion because not sure whether the key word is "hii" or "h ii" and orion should give the same abundances, orion already includes grains!
            model_input_file.writelines("metals and grains log %0.2f\n" % (log10_Z_cloud))
        else:
            model_input_file.writelines("abundance orion no grains\n")
            model_input_file.writelines("metals log %0.2f\n" % (log10_Z_cloud))

    if (not np.isnan(logB0)):
        if B_cloudy == True:
            model_input_file.writelines("magnetic field tangled %0.4f 2\n" % (logB0))

    if (not np.isnan(log10_n0)):
        model_input_file.writelines("hden %0.3f\n" % (log10_n0))  # log of hydrogen density

    if (not np.isnan(stopmass)):
        model_input_file.writelines("stop mass %0.3f\n" % (stopmass))

    return 0


def opiate_line_list():
    my_opiate_line_list = """N  2 121.767m
N  2 205.244m
N  2 6583.45A
Blnd 5755.00A
O  3 88.3323m
O  3 5006.84A
Blnd 4363.00A
O  2 3726.03A
Blnd 3727.00A
O  2 3728.81A
Blnd 7323.00A
Blnd 7332.00A
O  1 63.1679m
O  1 145.495m
Blnd 6300.00A
S  3 18.7078m
S  3 33.4704m
S  3 9530.62A
S  3 9068.62A
S  3 6312.06A
S  2 6716.44A
S  2 6730.82A
Blnd 6720.00A
H  1 6562.81A
H  1 4861.33A
C  2 157.636m
C  1 370.269m
C  1 609.590m
Blnd 2.12100m
CO   2600.05m
CO   1300.05m
CO   866.727m
CO   650.074m
CO   520.089m
CO   433.438m
CO   371.549m
CO   325.137m
CO   289.041m
CO   260.169m
CO   236.549m
CO   216.868m
CO   200.218m
HCO+ 373.490m
HCN  375.844m
Si 2 34.8046m
TIR  1800.00m
TIR  1100.00m
F12  12.0000m
F25  25.0000m
F60  60.0000m
F100 100.000m
MIPS 24.0000m
MIPS 70.0000m
MIPS 160.000m
IRAC 3.60000m
IRAC 4.50000m
IRAC 5.80000m
IRAC 8.00000m
SPR1 250.000m
SPR2 350.000m
SPR3 500.000m
PAC1 70.0000m
PAC2 100.000m
PAC3 160.000m"""
    return my_opiate_line_list

def create_line_list_file(outdir):

    my_opiate_line_list = opiate_line_list()
    outdir.writelines("save lines last emissivity \".ems\"\n")
    outdir.writelines(my_opiate_line_list+"\n")
    outdir.writelines("end of lines\n")

    return 0

