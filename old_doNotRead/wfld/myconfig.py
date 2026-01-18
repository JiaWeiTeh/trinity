import numpy as np
import sys
import subprocess

# Define cloud mass arrays
def linList(min,max,step):
    return np.linspace(min, max, endpoint=True, num=int(np.round(1+(max-min)/step)))


# Define Cloud masses
log10_Mcloud_INPUT_dict = {}
log10_Mcloud_INPUT_dict["a"] = np.linspace(7,9.0,7,endpoint=True)  #linList(5.0,8.0,0.25)
log10_Mcloud_INPUT_dict["b"] = np.array([7.00, 7.25, 7.5])
log10_Mcloud_INPUT_dict["c"] = np.array([8.00])

# Define SFE arrays
SFE_dict = {}
SFE_dict["a"] = np.linspace(1,25,17,endpoint=True)/100
SFE_dict["b"] = [0.05,0.10,0.15]
SFE_dict["c"] = [0.15]

############ BASIC MODEL PROPERTIES ###################

# pick one of the lists specified above

Mcloud_INPUT_array =10**log10_Mcloud_INPUT_dict["c"]
SFE_array = SFE_dict["c"]



#density profile of clump:

dens_profile = "powerlaw" #"powerlaw" #"BonnorEbert"

# central density of cloud
namb = 1000

#powerlaw parameters:
nalpha = -2
navg = 170

#Bonnor-Ebert parameters:
g_BE= 15

#intercloud density; dissolving density of shell
n_intercl = 5
ndiss = 6

#include coverfraction? And with which end value?
frag_cover=True
cf_end=1


Zism = 1 #metallicity (0.15 or 1)
r_Tb=0.98 #temperature measure radius (<1!)

# end time; end radius
tStop_tmp = 50  #Myr 
rstop = 5050   #pc --> high number, never stop at a specific radius

#relative tolerance of stiffest ODE solver (energy phase)
rtol=1e-3 #decrease to 1e-4 if you have crashes --> slower but more stable

frag_t_iter=True #should a fragmentation timescale be considered?

# restart expansion after recollapse?
mult_exp = False   

# form more stars after recollapse?
# 0: no starburst after collapse
# 1: starburst with same SFE as before
# 2: starburst with SFE calculated such that a specified SFE per free-fall time is achieved
mult_SF = 1 

###################################################
# folder to save output in
basedir =    './model_test_output/' 

# how much print statements would you like?
output_verbosity = 1# -2: minimum, -1: few, 0: some, 1: a lot  #def:1

#number of processors to use
n_proc = 1
# full path to warpfield code files
path_to_code = "./"
