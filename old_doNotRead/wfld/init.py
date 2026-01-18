import os
import constants as c
import parameters as p
import numpy as np
import imp
import sys
import auxiliary_functions as aux
import warnings
import time
import warprandom

# make sure that myconfig is imported from current working directory
cwd = os.getcwd()
try:
    mc = imp.load_source('myconfig', cwd + '/myconfig.py')
except:
    print(('myconfig.py does not exist in current working directory: '+cwd))
    sys.exit('Fatal Error')




#if p.Zism < (1.0-0.15)/2.:
#    SB99file = p.SB99file + "_Z0002"
#else:
#    SB99file = p.SB99file
#SB99file = SB99file + ".ods"


# This loop attemps to load values from the myconfig file. If the value is not set
# in the myconfig file it attemps to load the value from the parameter file.

parameter_list = ["Ti","Tn","Qi","Ln","namb","Lw","pdot",\
                 "kIR", "tInc","tCheck","tStart", "mui", "mua","sigmaD0",\
                 "T_intercl","n_intercl","tInc_small","rotation",\
                  "tCheck_small","SB99","Zism","vstop", "Mcloud_INPUT_array",\
                  "rotation", "rInc_ion", "rInc_neutral", "SFE_array",\
                  "rcore_au", "nalpha", "ndiss", "tStop_unit", "tStop_tmp",\
                  "dt_min_diss", "BHcutoff", "output_verbosity", "basedir",\
                  "write_cloudy", "mult_exp", "mult_SF", "cloudy_dt", "small_cloudy_dt", "SB99_age_min",\
                  "cloudy_stopmass_shell", "B_cloudy", "cloudy_turb",\
                  "cloudyVersion", "norm", "cloudy_dig", "Mcloud_beforeSF",\
                  "SB99_mass", "n_proc", "write_data", "plot_data", "SFE_ff",\
                  "Weav_grav", "rdotdot0", "small_dt_save", "rstop",\
                  "adiabatic_core_only", "r_Tb", "myloss", "delta_error",\
                  "lum_error", "lum_error2", "cs_bubble", "dt_Emax", "dt_Emin",\
                  "cloudy_t_switch", "rcoll", "write_SB99", "dt_Estart", "dt_switchon",\
                  "immediate_leak", "ncluster_check", "warpversion", "r_restart", \
                  "write_potential", "Z_nodust", "pot_len_intern", "pot_len_write",\
                  "dens_profile", "fit_len_max", "fit_len_min", "frag_c",\
                  "cfmin", "Emin", "savebubble", "force_SB99file","Nrandom_models",\
                  "random_input", "random_Mcl_lim", "random_SFE_lim", "random_namb_lim",\
                  "saveshell", "thermcoeff_clwind", "thermcoeff_SN", "cloudy_tmin",\
                  "cloudy_use_relative_path", "f_Mcold_W", "f_Mcold_SN","v_SN","cloudy_CWcavitiy",\
                  "cluster_radius", "gravfrag_burst", "RT_burst", "inhomfrag_burst",\
                  "old_output_names","write_line_list", "fixed_cluster_radius", "scale_cluster_radius",\
                  "g_BE","frag_cover","cf_end","frag_t_iter","rtol","warns"] # unused: #["Mcluster","SB99file"]

# load parameters from myconfig or default to parameters.py
for var in parameter_list:
    try:
        exec("%s = mc.%s"%(var,var))
    except:
        exec("%s = p.%s"%(var,var))

if random_input is True:
    Mcloud_random, SFE_random, namb = warprandom.randominput(random_Mcl_lim, random_SFE_lim, random_namb_lim)
    Mcloud_INPUT_array = np.array([Mcloud_random])
    SFE_array = np.array([SFE_random])

# only use relative paths for cloudy if basedir is relative path as well or a relative path can be reconstructed (which is the case if basedir is a subdirectory of the current working directory)
if os.path.isabs(basedir) and cloudy_use_relative_path is True:
    print("ATTENTION: provided absolute path of basedir conflicts with usage of relative path for cloudy")
    print("ATTENTION (cont'd): I will set cloudy_use_relative_path to False. If you want a relative path for cloudy, provide a relative path for basedir as well!")
    cloudy_use_relative_path = False


# change basedir to an absolute path with a single fixed trailing slash
basedir = os.path.abspath(basedir) + '/' # trailing slash for legacy issues
print("basedir = " + basedir)

if (nalpha == 0.):
    navg = namb
else:
    try: navg = mc.navg
    except: sys.exit("No parameter navg (average cloud number density) provided in myconfig!")

# calculate free fall time
# ONLY FOR CONSTANT DENSITY (for density profile, average density will be used)
tff = np.sqrt(3. * np.pi / (32. * c.Grav * navg * mui)) / c.Myr
if tStop_unit == "tff":
    tStop = tStop_tmp * tff
    print(("free fall time = ", tff, ", end time = ", tStop))
elif tStop_unit == "Myr":
    tStop = tStop_tmp  # if the unit is Myr then that's ok because that is also the unit used in expansion_solver.py

# we assume the dust cross section scales linearly with metallicity (but below a certain metallicity, there is no dust)
if Zism >= Z_nodust:
    sigmaD = sigmaD0 * Zism
else:
    sigmaD = 0.

# ambient mass density
rhoa = namb * mui

# intercluster mass density
rho_intercl = n_intercl * mui

# mui SI:
muiSI=mui*10**(-3)

# if there is a density gradient
if nalpha != 0.: density_gradient = True
else: density_gradient = False

# convert parameters to Msun, pc, Myr
kIR_au = kIR * c.Msun / c.pc**2.  # IR opacity (of order unity?)
rhoa_au = rhoa * c.pc**3. / c.Msun  # density of ambient medium
rho_intercl_au = rho_intercl * c.pc**3. / c.Msun
n_intercl_au = n_intercl * c.pc**3.

if ((Zism <= 0.15) and (BHcutoff >= 120.)):
    warnings.warn("You are running models with low metallicity, but have set the direct collapse BH threshold: BHcutoff >= 120. ")
    warnings.warn("I will set BHcutoff = 40. Be aware!")
    BHcutoff = 40.

# figure out the SB99 file for use in cloudy (cloudy will not interpolate between files, just pick the one that comes closest)
if force_SB99file == 0: # no specific cloudy file is forced, determine which file to use from BHcutoff, metallicity, ...
    if rotation is True: rot_string = "_rot_"
    else: rot_string = "_norot_"
    BH_string = "_BH" + str(int(BHcutoff))
    if abs(Zism-1.0) < abs(Zism-0.15): Z_string = "Z0014"
    else: Z_string = "Z0002"
    SB99cloudy_file = '1e6cluster'+rot_string+Z_string+BH_string
else:
    # if a specific file is forced, remove extension for cloudy
    print(("forcing specific starburst99 file: " + force_SB99file))
    idx = [pos for pos, char in enumerate(force_SB99file) if char == '.'] # find position of last '.' (beginning of file extension)
    SB99cloudy_file = force_SB99file[:idx[-1]] # remove extension




if ((n_proc > 1) and (output_verbosity > -1)):
    output_verbosity = -1
    print("Setting output_verbosity to -1 because more than 1 processor is in use.")

if ((nalpha != 0.) and (mult_SF == 2)):
    # since the free-fall time as calculated in this routine is only correct for constant density, do not use a density profile and a specified star formation efficiency per free-fall time
    #print("Since the free-fall time as calculated in .init is only correct for constant density, do not use a density profile and a specified star formation efficiency per free-fall time at the same time!")
    #print("Change nalpha or mult_SF!")
    #sys.exit("Exiting: SFE per free fall time not implemented for density profiles! Change mult_SF or nalpha!")
    warnings.warn("SFE per free fall time is calculated from AVERAGE density!")

if namb > 1e4:
    warnings.warn("Very high density can lead to crashes. No guarantee for density > 1e4.")
    warnings.warn("Maybe you will need to reduce the minimum time step.")

if ((B_cloudy is True) and (nalpha != 0.)):
    sys.exit('B-Field not yet implemented for dlaw (nalpha != 0). Static cloudy components will be wrong. Change cloud profile or set B_cloudy to False!')

if n_intercl > ndiss:
    sys.exit('ndiss must be larger or equal to n_intercl! How can you keep calling something a shell if its density is lower than the ambient low-density medium?')

if ((write_potential is True) and (pot_len_intern < 1000)):
    warnings.warn("Writing potential files with very coarse spatial resolution. The result might be not precise.")

if not (dens_profile == "powerlaw" or dens_profile == "BonnorEbert" ):
    sys.exit("Chosen dens_profile not implemented (only 'powerlaw' or 'BonnorEbert' work)!")


if small_cloudy_dt > cloudy_dt:
    small_cloudy_dt = cloudy_dt






start_time = time.time()


