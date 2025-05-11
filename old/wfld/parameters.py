# specify parameters to use for solution of ODEs
# if the same parameter is specified in myconfig, it will overwrite the value specified here
import numpy as np
import constants as c

################## MODEL TYPE AND OUTPUT ###################
# after recollapse, start expansion again
mult_exp = True

# after recollapse, form stars again (with same SFE as before)... if mult_exp == False, this switch will not do anything
# 0: no starburst after collapse
# 1: starburst with same SFE as before
# 2: starburst with SFE calculated such that a specified SFE per free-fall time is achieved
mult_SF = 1

# star formation efficiency per free-fall time (only used if mult_SF == 2)
SFE_ff = 0.01 # default: 0.01 (i.e. 1%)

# collapse radius (if the shell has a smaller radius while also having a negative velocity, collapse and possibly renewed star formation occurs)
rcoll = 1.0 # (pc)

# radius at which to restart expansion after 2nd (or later) star formation event occured
# (default is 0., larger values have not been properly tested for warpversion 3)
r_restart = 0.

# number of processors to use
n_proc = 1

# write output data (radius, velocity, absorbed fraction of radiation etc.) to folders which are created according to (Z, Mcloud, SFE_density)
# should be set to True
write_data = True

# create some plots (radius and velocity vs time)
plot_data = False

# how many print statements do you want?
# -2: minimum, -1: few, 0: some, 1: a lot
output_verbosity = -2

# do you want to create ..._expansion.txt files with very high time output?
small_dt_save = False

############### CLOUD PROPERTIES #########################

# density profile
# namb = 1000.0 # cloud number density (if switch on density gradient, this will be the core density)
# navg = namb # average cloud density (if no density gradient: same as core density)
nalpha = 0.0 # density gradient power law index: n(r) = namb*(r/rcore)**nalpha (usually 0. [for constant density] or negative: -1.5 or -2.0)
rcore_au = 0.099 # core radius in pc (must be > 0.0) # DO NOT CHANGE! HAS NOT BEEN PROPERLY TESTED! DO NOT CHANGE IN MYCONFIG!

# metallicity
# this sets the metallicity of both the ISM and the stars
# (limited to metallicites in starburst99 models, only Z = 1.0 and Z = 0.15)
Zism = 1.0

# Is the input cloud mass measured before stars have formed?
# (if this is set to TRUE, when you specify 1e6 Msol cloud with SFE 10%, the cloud becomes a 9e5 Msol cloud after SF and forms a 1e5 Msol cluster
# if this is set to FALSE, 1e6 Msol cloud with SFE 10% means the cloud mass after SF is 1e6 Msol and the cluster is 1.1e5 Msol)
Mcloud_beforeSF = True
# temperaure of HII and neutral region
Ti = 1e4
Tn = 100.0
# ambient medium (behind the edges of the cloud --> intercloud material)
T_intercl = 1e4
n_intercl = 0.1 #density

# kappa infrared
kIR = 4.0

#dust cross section for solar metallicity (Draine2011)
sigmaD0 = 1.5e-21

# metallicity below which there is no dust (in solar metallicity) Zwicky
Z_nodust = 0.05

# mean molecular weight
mua = c.fH_p*c.mH + c.fHe_p*c.mHe + c.fe_p*c.me # mean mass per particle
mui = c.fH_i*c.mH + c.fHe_i*c.mHe # mean mass per ion

# if we always assume 1 helium atom per 10 hydrogen atoms:
#mua = (10.+4.)/(20.+3.) * c.mp
#mui = (10.+4.)/(10.+1.) * c.mp

# if Helium abundance scaled with metallicity:
#mua = (10.+Zism*4.)/(20.+Zism*3.) * c.mp
#mui = (10.+Zism*4.)/(10.+Zism) * c.mp

# thermalisation efficiency for colliding winds and colliding Supernova ejecta (see Stevens and Hartwell 2003 or Kavanagh 2020 for a review)
# if no coling losses due to colliding winds are assumed, the efficiency = 1, otherwise < 1
# the new mechanical energy will be: Lw_new = thermcoeff * Lw_old (respectively for winds and supernovae)
thermcoeff_clwind = 1.0 # for colliding stellar winds # default: 1.0
thermcoeff_SN = 1.0 # for supernovae; if unsure, use same number as for colliding winds # default: 1.0

# additional fraction of mass injected into the cloud from sweeping of cold material, e.g. from protostars and disks inside the star cluster
# the total mass loss rate of the cluster will be Mdot_new = Mdot_old(SB99) * (1 + f_Mcold)
f_Mcold_W = 0.0 # default: 0.0, i.e. no extra mass loss
f_Mcold_SN = 0.0 # default: 0.0, i.e. no extra mass loss

# velocity of supernova ejecta
v_SN = 1e4*1e5 # in cm/s (default: 1e9 cm/s i.e. 1e4 km/s)

############### CLUSTER PROPERTIES #############################
# star formation efficiciency (if you did not specify anything in myconfig)
SFE = 0.1

# starburst 99 stuff

# Do you want to use SB99?
SB99 = True

# rotating or non-rotating stars
rotation = True

# starburst99 stellar mass (in Msol) above which no SNe occur
BHcutoff = 120.

# the cluster mass the starburst99 file uses (in Msol)
SB99_mass = 1e6 # default in SB99 is 1e6

# Minimum age to take when reading in SB99 cluster spectrum for CLOUDY (in years). This prevents a model age from requesting a cluster spectrum out of the bounds of the SB99 calculation
SB99_age_min = 5.0e5
# TO DO: read in SB99_age_min from SB99 file

# force warpfield to use a specific starburst99 file (independent of the set BH cutoff mass, rotation of stars, metallicity, ...)
force_SB99file = 0 # use 0 if you do not want to force a file, otherwise use full filename (with path to file), e.g. /local/data/SB99_39_7b.txt if the file SB99_39_7b.txt lies in /local/data/
# make sure SB99_mass is set correctly!

############### CLOUDY ####################################

# do you want to create cloudy input files?
write_cloudy = False

# write table of emission lines to shell.in and static.in files
write_line_list = True

# cloudy time steps (only relevant for warpversion 1.0)
small_cloudy_dt = 0.1 # (Myr), before a time of cloudy_t_switch has passed since the last SF event, use this small dt for writing cloudy output
cloudy_dt = 0.5 # (Myr) after more than cloudy_t_switch has passed since last SF event, use this bigger dt
cloudy_t_switch = 1.0 # (Myr) at this time after the last SF event, cloudy_dt is udes instead of small_cloudy_dt

cloudy_tmin = 1e-2 # minimum time (in Myr) to wait after a star formation event before cloudy files are created

cloudy_use_relative_path = True # if True, cloudy input files (and bubble files) will use relative path in the set save prefix command

old_output_names = False # use old file name convention (OLD: file name of main data output in input files repeats most important model parameters, NEW: evo.dat and input.dat)

# cloudy B-field (not used for shell structure in WARPFIELD)
# sets the inner shell density passed to cloudy and sets the command "magnetic field tangled ... 2" in the cloudy .in files
B_cloudy = False

# for cloudy models, use shell mass as stop mass (if set to False, use cloud mass)
cloudy_stopmass_shell = True

# turbulence in cloudy models
cloudy_turb = "4 km/s no pressure" # this parameter is passed to __cloudy__ but IS NOT USED

# Default version of Cloudy. Like all parameters, this option is overwritten in a local "myconfig.py" file
cloudyVersion = "C17"

# Should we create cloudy input files for after the shell has dissolved to account for the cluster still existing in the diffuse ionized gas (DIG)?
cloudy_dig = True # NOT IMPLEMENTED. TO DO!

# Right now this only specifies the line name that cloudy is normalizing line fluxes to.
# This should be in my config as well, and when invoked should change the cloudy input files.
class norm(object):
    label = "H  1 4861.33A"
    species, ion, wave = label.split()

cloudy_CWcavitiy = True # include zones A and B (Chevalier and Clegg 1985) in cloudy calculation
fixed_cluster_radius = True # use a fixed cluster radius which does not change with time?
cluster_radius = 1.0 # cluster radius in pc (important for Chevalier and Clegg 1985 profile), only used if fixed_cluster_radius is True
scale_cluster_radius = 0.3 # scale factor for evolving cluster radius (cluster radius = scale_cluster_radius * shell radius), only used if fixed_cluster_radius is False

################### CLOUD DISSOLUTION ################

# threshold for shell dissolution (1/ccm), if density lower for extended period of time --> stop simulation
ndiss = 1.0
# minimum time span during which the max density of shell has to fall below ndiss, in order to warrant stoppage of simulation (Myr)
dt_min_diss = 2.0

################## TIME STEPPING and STEP SIZE #######################
# timesteps for integration (in Myr)

# tStart should be a value slightly above 0.0 (depracated)
tStart = 0.0 #1e-6 (unit: Myr)

# end time of simulation
tStop_tmp = 15.05 # can be Myr or as multiples of the free-fall time, specify in next line
# HINT: Don't go after the last SN (44 Myr for single cluster), code becomes slow

# unit of end time
tStop_unit = "Myr" # unit: "tff" (free fall times) or "Myr"

# output time step for ODEs
tInc = 5e-4 # default: 5e4, unit Myr

# time step for calculation of shell structure
tCheck = 0.04166 #0.2 # this is the max main time step (used to calculated shell structure etc.) in Myr

dt_Emax = 0.04166
dt_Emin = 1e-5
dt_Estart = 1e-4
dt_switchon = 1e-3 # gradually switch on things during this time period

# time steps for first 0.1 Myr
#tCheck_small = 0.0187 #0.02
tCheck_small = 1e-2
tInc_small = 5e-5 #5e-5



# if collapse becomes very fast, calculation becomes unstable --> stop
vstop = -10000.0 # in km/s

# stop radius (if shell expands further, stop simulation)
rstop = 1050. # in pc (very high number = never stop)

# initial condition for acceleration (at beginning of adiabatic phase I)
rdotdot0 = -30000.0 # in km/s/Myr

rtol=1e-3 #relative tolerance of stiffest ODE solver (energy phase),decrease to 1e-4 if you have crashes --> slower but more stable

warns=False #show warnings?
################ shell structure #########################

#step size for shell integration
# ionized part of shell
rInc_ion = 5e-4                 # 5e-4, unit pc
# netural part of shell
rInc_neutral = 5e-5             # 5e-5, unit pc

# save the density of the shell as .txt file
saveshell = False

################ bubble structure ########################

# relative radius xi = r/R2 at which to measure the bubble temperature (needed to calculate delta), must be < 1.0    default: 0.99
r_Tb = 0.99

# save the density and temperature structure of the bubble as .txt file
savebubble = False

######################## MISC #########################

# loss function for alpha, beta, delta fitting
myloss = 'soft_l1' # high robustness: 'cauchy', 'arctan', medium robustness: 'soft_l1', 'huber', low robustness: 'linear'
# (for more info, see http://scipy-cookbook.readthedocs.io/items/robust_regression.html)

delta_error = 0.03 # max allowed relative error in calculation of delta (dln(T)/dln(t))    default: 0.05
lum_error = 0.005 # set time step such that mechanical luminosity does not change more than this (relative, i.e. 0.01 is 1%)    default: 0.01
lum_error2 = 0.005 # allowed error for Lw-Lb (percentage of mechanical luminosity)

# include gravity in phase I? 0.0 for No, 1.0 for Yes
Weav_grav = 1.0

# # adiabatic phase only in core?
adiabatic_core_only = False

# properties of star cluster if SB99 not used

# mass of the cluster
Mcluster = 1e4*c.Msun

# radiation (if SB99 == True: these values are meaningless)
Qi = 10 ** 52.55
Li = 1e42
Ln = 10 ** 42.4

# winds (if SB99 == True: these values are meaningless)
Lw = 1e40
vw = 2700. * 1e5
Mwdot = 2. * Lw / vw ** 2
pdot = vw * Mwdot

# write table with used feedback parameters (SB99.txt)
write_SB99 = True

# write table with graviational potential
write_potential = False

# length of potential file used internally (to pass between routines)
pot_len_intern = 10000 # used seperately in bubble, shell, and outer region (i.e. x3)

# length of potential files as it is written
pot_len_write = 500



# if BonnorEbert is used as a density profile, the critical value (>14.1 for critical BE sphere) and core density of the isothermal cloud is needed
g_BE = 15
#core density is defined as namb 


# density profile (default is "powerlaw")
dens_profile = "powerlaw" # 'powerlaw' or 'BonnorEbert'


cs_bubble = 1000. # sound speed in bubble (temporary!)

# go to momentum driving immediately after bubble bursts?
# (it is advised this parameter is set to True. If it is False, the code might crash)
immediate_leak = True # OLD

# number of cluster to check (if for each SF-event the time between SF events becomes shorter, stop)
ncluster_check = 5

# warpversions for printing
warpversion = 3.0

# for fitting of alpha, beta, delta
fit_len_max = 13
fit_len_min = 7

# fragmentation constant, might be 0.67 or 0.73 (compare McCray & Kafatos 1987), or might be 0.37 (compare Elmegreen 1994)
# if you want to switch fragmentation off, set this number to 0.0 (or better: set gravfrag_burst = False)
frag_c = 0.67

# additional events which switch to momentum driving
gravfrag_burst = False
RT_burst = False
inhomfrag_burst = False

frag_t_iter=True #should a fragmentation timescale be considered?

# should a geometrical cover fraction be considered?

frag_cover=True
cf_end=0.1   #coverfraction at the end of the fragmentation process

# minimum covering fraction
cfmin = 0.4

# minimum energy at end of transition phase, afterwards: momentum-driving
Emin = 1e-4

########################### random input ##############################
# use random input (cloud mass, SFE, density) for warpfield instead of predefined values?
random_input = False # boolean (default: False)

# How many random models do you want to run?
Nrandom_models = 1 # (integer >= 1)

# if using random input, specify boundary values for cloud mass (random numbers will be drawn at constant probability density between the logarithms of these limits
random_Mcl_lim = [1e5,3e7]

# if using random input, specify boundary values for star formation efficiency (random numbers will be drawn at constant probability density between the logarithms of these limits
random_SFE_lim = [0.01, 0.10]

# if using random input, specify boundary values for cloud (core) density (random numbers will be drawn at constant probability density between the logarithms of these limits
random_namb_lim = [100.,1000.]
