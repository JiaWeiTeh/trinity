#from pyexcel_ods import get_data
import numpy as np
import sys
import os
import auxiliary_functions as aux
import warp_nameparser
import scipy.interpolate
import init as i
import warnings
import collections
import pathlib

# assume this is the structure of the ods-file:
# time(yr), log10(Qi), log10(Li/Lbol), log10(Lbol), log10(Lw w/ SNe), log10(pw_dot), log10(Lw w/o SNe)

# input:
#   file: path to file that contains all necessary feedback paramters as a function of time
#   f_mass: normalize to different cluster mass
#         e.g. if you are reading in a file for a 1e6 Msol cluster but want to simulate a 1e5 cluster, set f_mass = 1e5/1e6 = 0.1
#   f_Zism: normalize to reference metallicity. Only wind output is affected by this scaling
#         e.g. If using 0.7 solar Z, the code will use stellar tracks for closest reference Z and then scale winds accordingly, i.e. with f_Zism = Z/Z_reference

# output:
#   t: time in Myr
#   Qi: rate of ionizing photons (1/s)
#   Li: luminosity of ionizing radiation (erg/s)
#   Ln: luminosity of non-ionizing radiation (erg/s)
#   Lbol: Li+Ln
#   Lw: mechanical wind luminosity
#   vw: wind velocity (cm/s)
#   Mw_dot: mass loss rate due to winds (g/s)
#   pdot_SNe: momentum flux of SN ejecta (SN mass loss rate times 1e4 cm/s)

def getMdotv(pdot,Lmech):
    """
    calculate mass loss rate Mdot and terminal velocity v from momentum injection rate pdot and mechanical luminosity Lmech
    :param pdot: momentum injection rate
    :param Lmech: mechanical luminosity
    :return: mass loss rate, terminal velocity
    """

    Mdot = pdot**2/(2.*Lmech)
    v = 2.*Lmech/pdot

    return Mdot, v

def getpdotLmech(Mdot,v):
    """
    calculate momentum injection rate and mechanical luminosity from mass loss rate and terminal velocity
    :param Mdot: mass loss rate
    :param v: terminal velocity
    :return: momentum injection rate, mechanical luminosity
    """

    pdot = Mdot * v
    Lmech = 0.5 * Mdot * v**2.0

    return pdot, Lmech

def getSB99_data(file, f_mass=1.0, f_met=1.0, test_plot = False, log_t = False, tmax = 30., verbose=0, ylim=[37.,43.]):
    """
    subroutine for load_stellar_tracks.py
    :param file: file to read
    :param f_mass: mass scaling (as default: in units of 1e6 Msol)
    :param f_met: metallicity scale factor (NB: only use a number different from 1.0 when you are scaling down or up an SB99 file with a different metallicity to the one you set in myconfig. E.g. If you want Zism = 0.15 and a SB99 file with that metallicity exists, f_met must be 1.0. If you want to run Zism = 0.15 and want to scale down a SB99 file with Z = 0.3, set f_met = 0.5)
    :param test_plot: do you want to see a plot of the feedback parameters? (boolean, use only for debugging)
    :param log_t: in that test plot, do you want the time axis to be log?
    :param tmax: in that test plot, what is the maximum time you want to plot?
    :return:
    """
    aux.printl("getSB99_data: mass scaling f_mass = %.3f" %(f_mass), verbose=verbose)

    #data_dict = get_data(file)
    #data_list = data_dict['Sheet1']
    #data = np.array(data_list)

    if os.path.isfile(file):
        data = np.loadtxt(file)
    elif "WARPFIELD_CORE" in os.environ:
        full_path = os.environ["WARPFIELD_CORE"] + "/star_clusters/sb99/" + file
        if os.path.isfile(full_path):
            data = np.loadtxt(full_path)
        else:
            sys.exit("Specified SB99 file does not exist in WARPFIELD_CORE:", file)
    elif os.path.isfile(pathlib.Path(__file__).parent / file):
        data = np.loadtxt(pathlib.Path(__file__).parent / file)
    else:
        #print(("Specified SB99 file does not exist:", file))
        sys.exit("Specified SB99 file does not exist:", file)

    t = data[:,0]/1e6 # in Myr
    # all other quantities are in cgs
    Qi = 10.0**data[:,1] *f_mass # emission rate of ionizing photons (number per second)
    fi = 10**data[:,2] # fraction of ionizing radiation
    Lbol = 10**data[:,3] *f_mass # bolometric luminosity (erg/s)
    Li = fi*Lbol # luminosity in the ionizing part of the spectrum (>13.6 eV)
    Ln = (1.0-fi)*Lbol # luminosity in the non-ionizing part of the spectrum (<13.6 eV)

    #get mechanical luminosity of SNe before scaling wind luminosity according to metallicity and other factors:
    pdot_W_tmp = 10**data[:, 5] * f_mass  # momentum rate for winds before scale factors considered (other than mass scaling)
    Lmech_tmp = 10**data[:,4] * f_mass # mechanical luminosity of winds and SNe
    Lmech_W_tmp = 10 ** data[:, 6] * f_mass # only wind
    Lmech_SN_tmp = Lmech_tmp - Lmech_W_tmp # only SNe

    # winds
    Mdot_W, v_W = getMdotv(pdot_W_tmp, Lmech_W_tmp) # convert pdot and Lmech to mass loss rate and terminal velocity
    Mdot_W *= f_met * (1. + i.f_Mcold_W) # modify mass injection rate according to 1) metallicity and 2) cold mass content in cluster (NB: metallicity affects mainly the mass loss rate, not the terminal velocity)
    v_W *= np.sqrt(i.thermcoeff_clwind / (1. + i.f_Mcold_W)) # modifiy terminal velocity according to 1) thermal efficiency and 2) cold mass content in cluster
    pdot_W, Lmech_W = getpdotLmech(Mdot_W, v_W)

    # supernovae
    v_SN = i.v_SN # assuming a constant ejecta velocity (which is not quite right, TO DO: get time-dependent velocity, e.g. when mass of ejecta are known)
    Mdot_SN = 2.* Lmech_SN_tmp/v_SN**2
    Mdot_SN *= (1. + i.f_Mcold_SN) # # modify mass injection rate according to 1) cold mass content in cluster during SN explosions, do not modify according to metallicity
    v_SN *= np.sqrt(i.thermcoeff_SN / (1. + i.f_Mcold_SN)) # modifiy terminal velocity according to 1) thermal efficiency and 2) cold mass content in cluster during SN explosions
    pdot_SN, Lmech_SN = getpdotLmech(Mdot_SN, v_SN)

    # add mechanical energy and momentum injection rate, respectively, from winds and supernovae
    Lmech= Lmech_W + Lmech_SN
    pdot = pdot_W + pdot_SN

    # insert 1 element at t=0
    t = np.insert(t, 0, 0.0)
    Qi = np.insert(Qi, 0, Qi[0])
    Li = np.insert(Li, 0, Li[0])
    Ln = np.insert(Ln, 0, Ln[0])
    Lbol = np.insert(Lbol, 0, Lbol[0])
    Lmech= np.insert(Lmech, 0, Lmech[0])
    pdot = np.insert(pdot, 0, pdot[0])
    pdot_SN = np.insert(pdot_SN, 0, pdot_SN[0])

    # test plot (only use for debugging)
    if test_plot: testplot(t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN, log_t=log_t, t_max=tmax, ylim=ylim)

    return[t,Qi,Li,Ln,Lbol,Lmech,pdot,pdot_SN]



def combineSFB_data(file1, file2, f_mass1=1.0, f_mass2=1.0, interfile1 = ' ', interfile2 = ' ', Zfile1 = 0.15, Zfile2 = 1.0, Zism = 0.5):
    """
    adds up feedback from two cluster populations
    """
    # if file1 is "inter" or "interpolate" then an interpolation between 2 metallicities is performed

    if (file1 == 'interpolate' or file1 == 'inter'):
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data_interp(Zism, interfile1, Zfile1, interfile2, Zfile2, f_mass = f_mass1)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file2, f_mass = f_mass2)
    else:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data(file1, f_mass = f_mass1)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file2, f_mass = f_mass2)

    tend1 = t1[-1]
    tend2 = t2[-2]

    tend = np.min([tend1, tend2])

    # cut to same length

    Qi = add_vector(Qi1, Qi2)
    Li = add_vector(Li1, Li2)
    Ln = add_vector(Ln1, Ln2)
    Lbol = add_vector(Lbol1, Lbol2)
    Lw = add_vector(Lw1, Lw2)
    pdot = add_vector(pdot1, pdot2)
    pdot_SNe = add_vector(pdot_SNe1, pdot_SNe2)

    # check that the times are the same
    t1_tmp = t1[t1 <= tend]
    t2_tmp = t2[t2 <= tend]
    if not all(t1_tmp == t2_tmp):
        print("FATAL: files do not have the same time vectors")
        sys.exit("Exiting: SB99 files time arrays do not match!")

    if tend1 > tend2: t = t1
    else: t = t2

    return[t,Qi,Li,Ln,Lbol,Lw,pdot,pdot_SNe]

def getSB99_data_interp(Zism, file1, Zfile1, file2, Zfile2, f_mass = 1.0):
    """
    interpolate metallicities from SB99 data
    :param Zism: metallicity you want (between metallicity 1 and metallicity 2)
    :param file1: path to file for tracks with metallicity 1
    :param Zfile1: metallicity 1
    :param file2: path to file for tracks with metallicity 2
    :param Zfile: metallicity 2
    :return:
    """

    # let's ensure that index 1 belongs to the lower metallicity
    if Zfile1 < Zfile2:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data(file1, f_mass = f_mass)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file2, f_mass = f_mass)
        Z1 = Zfile1
        Z2 = Zfile2
    elif Zfile1 > Zfile2:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data(file2, f_mass = f_mass)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file1, f_mass = f_mass)
        Z1 = Zfile2
        Z2 = Zfile1

    tend1 = t1[-1]
    tend2 = t2[-2]

    tend = np.min([tend1, tend2])

    # cut to same length

    Qi1 = Qi1[t1 <= tend]
    Li1 = Li1[t1 <= tend]
    Ln1 = Ln1[t1 <= tend]
    Lbol1 = Lbol1[t1 <= tend]
    Lw1 = Lw1[t1 <= tend]
    pdot1 = pdot1[t1 <= tend]
    pdot_SNe1 = pdot_SNe1[t1 <= tend]

    Qi2 = Qi2[t2 <= tend]
    Li2 = Li2[t2 <= tend]
    Ln2 = Ln2[t2 <= tend]
    Lbol2 = Lbol2[t2 <= tend]
    Lw2 = Lw2[t2 <= tend]
    pdot2 = pdot2[t2 <= tend]
    pdot_SNe2 = pdot_SNe2[t2 <= tend]

    t1 = t1[t1 <= tend]
    t2 = t2[t2 <= tend]

    if not all(t1 == t2):
        print("FATAL: files do not have the same time vectors")
        sys.exit("Exiting: SB99 files time arrays do not match!")

    t = t1
    Qi = (Qi1 * (Z2 - Zism) + Qi2 * (Zism - Z1)) / (Z2 - Z1)
    Li = (Li1 * (Z2 - Zism) + Li2 * (Zism - Z1)) / (Z2 - Z1)
    Ln = (Ln1 * (Z2 - Zism) + Ln2 * (Zism - Z1)) / (Z2 - Z1)
    Lbol = (Lbol1 * (Z2 - Zism) + Lbol2 * (Zism - Z1)) / (Z2 - Z1)
    Lw = (Lw1 * (Z2 - Zism) + Lw2 * (Zism - Z1)) / (Z2 - Z1)
    pdot = (pdot1 * (Z2 - Zism) + pdot2 * (Zism - Z1)) / (Z2 - Z1)
    pdot_SNe = (pdot_SNe1 * (Z2 - Zism) + pdot_SNe2 * (Zism - Z1)) / (Z2 - Z1)

    return[t,Qi,Li,Ln,Lbol,Lw,pdot,pdot_SNe]

def testplot(t,Qi,Li,Ln,Lbol,Lw,pdot,pdot_SNe, log_t = False, t_max = 30., ylim=[39.,43.]):

    import matplotlib.pyplot as plt

    if log_t:
        plt.semilogx(t, np.log10(Li), 'b', label="$L_i$")
        plt.semilogx(t, np.log10(Ln), 'r', label="$L_n$")
        plt.semilogx(t, np.log10(Lbol), 'g--', label="$L_{bol}$")
        plt.semilogx(t, np.log10(Lw), 'k', label="$L_{wind}$")
        plt.semilogx(t, np.log10(Qi) - 10.0, 'm', label="$Q_{i}-10$")
        plt.semilogx(t, np.log10(pdot) + 10.0, 'c', label="$\dot{p_{w}}+10$")
    else:
        plt.plot(t, np.log10(Li), 'b', label="$L_i$")
        plt.plot(t, np.log10(Ln), 'r', label="$L_n$")
        plt.plot(t, np.log10(Lbol), 'g--', label="$L_{bol}$")
        plt.plot(t, np.log10(Lw), 'k', label="$L_{wind}$")
        plt.plot(t, np.log10(Qi) - 10.0, 'm', label="$Q_{i}-10$")
        plt.plot(t, np.log10(pdot) + 10.0, 'c', label="$\dot{p_{w}}+10$")
    plt.xlabel("t in Myr")
    plt.xlim([0.9,t_max])
    plt.ylim(ylim)
    plt.ylabel("log10(L) in erg/s")
    plt.legend()

    plt.show()
    return 0


"""
file1 = '/home/daniel/Documents/work/loki/code/warpfield/1e6cluster_rot_Z0002.txt'
file2 = '/home/daniel/Documents/work/loki/code/warpfield/1e6cluster_rot.txt'
Zfile1 = 0.002
Zfile2 = 0.014
[t,Qi,Li,Ln,Lbol,Lw,pdot,pdot_SNe] = getSB99_data_interp(0.43*0.014, file1, Zfile1, file2, Zfile2, f_mass = 1.0)
[t1,Qi1,Li1,Ln1,Lbol1,Lw1,pdot1,pdot_SNe1] =  getSB99_data(file1, f_mass = 1.0)
[t2,Qi2,Li2,Ln2,Lbol2,Lw2,pdot2,pdot_SNe2] =  getSB99_data(file2, f_mass = 1.0)

import matplotlib.pyplot as plt

plt.plot(t,np.log10(Li),'b',label="$L_i$")
plt.plot(t,np.log10(Ln),'r',label="$L_n$")
plt.plot(t,np.log10(Lbol),'g--',label="$L_{bol}$")
plt.plot(t,np.log10(Lw),'k',label="$L_{wind}$")
plt.plot(t,np.log10(Qi)-10.0,'m',label="$Q_{i}-10$")
plt.plot(t,np.log10(pdot)+10.0,'c',label="$\dot{p_{w}}+10$")
plt.xlabel("t in Myr")
plt.ylabel("log10(L) in erg/s")
plt.legend()

print t1[Lw1 == np.max(Lw1)]

plt.show()
"""

def load_stellar_tracks(Zism, rotation= True, f_mass = 1.0, BHcutoff = 120., force_file = i.force_SB99file, test_plot = False, log_t = False, tmax=30., return_format="array"):
    """
    wrapper for loading stellar evolution tracks
    :param Zism: metallicity
    :param rotation: rotating stars (boolean, optional)
    :param f_mass: cluster mass in units of 1e6 Msol (optional)
    :param force_file: you can force the code to use a SB99 file (the other optional parameters "rotation" and "BHcutoff" will be ignored in that case)
                        if this is set to 0, it will not force a file
                        otherwise write the name of the file, e.g. force_file == "1e6cluster.txt"
    :param test_plot: do you want to see a plot of the feedback parameters? (boolean, use only for debugging)
    :param log_t: in that test plot, do you want the time axis to be log?
    :param tmax: in that test plot, what is the maximum time you want to plot?
    :param return_format: "array" or "dict"
    """
    # output:
    #   t: time in Myr
    #   Qi: rate of ionizing photons (1/s)
    #   Li: luminosity of ionizing radiation (erg/s)
    #   Ln: luminosity of non-ionizing radiation (erg/s)
    #   Lbol: Li+Ln
    #   Lw: mechanical wind luminosity
    #   vw: wind velocity (cm/s)
    #   Mw_dot: mass loss rate due to winds (g/s)
    #   pdot_SNe: momentum flux of SN ejecta (SN mass loss rate times 1e4 cm/s)

    # get SB99 filenames
   
    SB99_file_Z0002 = warp_nameparser.get_SB99_filename(0.15, rotation, BHcutoff, SB99_mass=i.SB99_mass) # low metallicity
    SB99_file_Z0014 = warp_nameparser.get_SB99_filename(1.0, rotation, BHcutoff, SB99_mass=i.SB99_mass) # high metallicity

    if force_file != 0:  # case: specific file is forced
        SB99file = force_file
        warnings.warn("Forcing WARPFIELD to use the following SB99 file: %s" % (force_file))
        warnings.warn("WARNING: Make sure you still provided the correct metallicity and mass scaling")
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data(SB99file, f_mass=f_mass,
                                                                                             test_plot=test_plot,
                                                                                             log_t=log_t, tmax=tmax)
        if SB99file != i.SB99cloudy_file + '.txt':
            warnings.warn("SB99 file in getSB99_data and SB99cloudy_file from init do not agree!")
            print(("SB99file: " + SB99file))
            print(("SB99cloudy_file +.txt: " + i.SB99cloudy_file))

    # if metallicity is in between: interpolate!
    elif (Zism < 1.0 and Zism > 0.15):
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data_interp(Zism,
                                                                                                       SB99_file_Z0002,
                                                                                                       0.15,
                                                                                                       SB99_file_Z0014, 1.0,
                                                                                                       f_mass=f_mass)
    elif (Zism == 1.0 or (Zism <= 0.15 and Zism >= 0.14)):
        if Zism == 1.0:
            SB99file = SB99_file_Z0014
        elif (Zism <= 0.15 and Zism >= 0.14):
            SB99file = SB99_file_Z0002
        if SB99file != i.SB99cloudy_file+'.txt':
            warnings.warn("SB99 file in getSB99_data and SB99cloudy_file from init do not agree!")
            print(("SB99file: " + SB99file))
            print(("SB99cloudy_file +.txt: " + i.SB99cloudy_file))
       
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data(SB99file, f_mass=f_mass, test_plot=test_plot, log_t=log_t, tmax=tmax)
    else:
        #print("No Stellar evolutionary tracks for chosen metalicity (Z=" + str(Zism) + ") provided. Run Starburst99 first!")
        #sys.exit("Stop Code: Feedback parameters not provided!")
        warnings.warn("Your metallicity is either too high or too low. There are no stellar tracks! I will choose the closest track and scale the winds linearly but be careful!")
        if Zism < 0.15: SB99file = SB99_file_Z0002; Zism_rel = 0.15
        elif Zism > 1.0: SB99file = SB99_file_Z0014; Zism_rel = 1.0
        print(("Using the following SB99 file: "+SB99file))

        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data(SB99file, f_mass=f_mass, f_met=Zism/Zism_rel)

    Data = [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo]

    if return_format is 'dict': Data_out = return_as_dict(Data)
    elif return_format is 'array': Data_out = return_as_array(Data)

    return Data_out

def add_vector(a,b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c

def return_as_dict(SB99_data):
    """
    return input as dict indep. of whether it is a dictionary or an array
    :param SB99_data:
    :return:
    """

    if isinstance(SB99_data, collections.Mapping):
        return SB99_data
    else: # an array
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = SB99_data
        SB99_data_out = {'t_Myr': t_evo, 'Qi_cgs': Qi_evo, 'Li_cgs': Li_evo, 'Ln_cgs': Ln_evo, 'Lbol_cgs': Lbol_evo, 'Lw_cgs': Lw_evo, 'pdot_cgs': pdot_evo, 'pdot_SNe_cgs': pdot_SNe_evo}
        return SB99_data_out

def return_as_array(SB99_data):
    """
    return input as array indep. of whether it is a dictionary or an array
    :param SB99_data:
    :return:
    """

    # check whether SB99_data is a dictionary
    if isinstance(SB99_data, collections.Mapping):
        t_Myr = SB99_data['t_Myr']
        Qi_cgs = SB99_data['Qi_cgs']
        Li_cgs = SB99_data['Li_cgs']
        Ln_cgs = SB99_data['Ln_cgs']
        Lbol_cgs = SB99_data['Lbol_cgs']
        Lw_cgs = SB99_data['Lw_cgs']
        pdot_cgs = SB99_data['pdot_cgs']
        pdot_SNe_cgs = SB99_data['pdot_SNe_cgs']
        SB99_data_out = [t_Myr, Qi_cgs, Li_cgs, Ln_cgs, Lbol_cgs, Lw_cgs, pdot_cgs, pdot_SNe_cgs]
        return SB99_data_out
    else:  # already an array
        return SB99_data

def make_interpfunc(SB99_data_IN):
    """
    get starburst99 interpolation functions
    :param SB99_data: array (of SB99 data) or dictionary
    :return: dictionary (containing interpolation functions)
    """

    # convert to an array
    SB99_data = return_as_array(SB99_data_IN)
    [t_Myr, Qi_cgs, Li_cgs, Ln_cgs, Lbol_cgs, Lw_cgs, pdot_cgs, pdot_SNe_cgs] = SB99_data

    fQi_cgs = scipy.interpolate.interp1d(t_Myr, Qi_cgs, kind='cubic')
    fLi_cgs = scipy.interpolate.interp1d(t_Myr, Li_cgs, kind='cubic')
    fLn_cgs = scipy.interpolate.interp1d(t_Myr, Ln_cgs, kind='cubic')
    fLbol_cgs = scipy.interpolate.interp1d(t_Myr, Lbol_cgs, kind='cubic')
    fLw_cgs = scipy.interpolate.interp1d(t_Myr, Lw_cgs, kind='cubic')
    fpdot_cgs = scipy.interpolate.interp1d(t_Myr, pdot_cgs, kind='cubic')
    fpdot_SNe_cgs = scipy.interpolate.interp1d(t_Myr, pdot_SNe_cgs, kind='cubic')

    SB99f = {'fQi_cgs': fQi_cgs, 'fLi_cgs': fLi_cgs, 'fLn_cgs': fLn_cgs, 'fLbol_cgs': fLbol_cgs, 'fLw_cgs': fLw_cgs,
             'fpdot_cgs': fpdot_cgs, 'fpdot_SNe_cgs': fpdot_SNe_cgs}

    return SB99f


def getSB99_main(Zism, rotation=True, f_mass=1e6, BHcutoff=120., return_format="array"):
    """
    get starburst99 data and corresponding interpolation functions
    :param Zism: metallicity (in solar units)
    :param rotation: boolean
    :param f_mass: mass of cluster (in solar masses)
    :param BHcutoff: cut off mass for direct collapse black holes in solar masses (stars above this mass, will not inject energy via supernova explosions)
    :return: array (of SB99 data), dictionary (containing interpolation functions)
    """

    SB99_data = load_stellar_tracks(Zism, rotation=rotation, f_mass=f_mass, BHcutoff=BHcutoff)
    SB99f = make_interpfunc(SB99_data)

    if return_format == "dict": SB99_data = return_as_dict(SB99_data)
    elif return_format == "array": SB99_data = return_as_array(SB99_data)


    return SB99_data, SB99f

def sum_SB99(SB99f, SB99_data2_IN, dtSF, return_format='array'):
    """
    sum 2 SB99 files (dictionaries).
    :param SB99f: Interpolation dictionary for 1st SB99 file
    :param SB99_data2: Data dictionary for 2nd SB99 file (cluster forms after cluster corresponding to SB99f)
    :param dtSF: time difference between 1st and 2nd file
    :return: Data dictionary
    """

    SB99_data2 = return_as_dict(SB99_data2_IN)
    ttemp = SB99_data2['t_Myr']

    # mask out late times for which no interpolation exists
    mask = ttemp+dtSF <= max(SB99f['fQi_cgs'].x) # np.max(ttemp)
    t = ttemp[mask]

    #print "max t, dtSF", max(t), dtSF

    # initialize Dsum
    Dsum = {'t_Myr': t}

    # loop through keys and summ up feedback
    for key in SB99_data2:
        if key != 't_Myr': # do not sum time
            Dsum[key] = SB99f['f'+key](t + dtSF) + SB99_data2[key][mask]

    # what format should the result be in?
    if return_format == 'array': # array
        Dsum_array = [Dsum['t_Myr'], Dsum['Qi_cgs'], Dsum['Li_cgs'], Dsum['Ln_cgs'], Dsum['Lbol_cgs'], Dsum['Lw_cgs'], Dsum['pdot_cgs'], Dsum['pdot_SNe_cgs']]
        return Dsum_array
    else: # dictionary
        return Dsum

def time_shift(SB99_data_IN, t):
    """
    adds a time time to time vector of SB99 data dictionary or array
    :param SB99_data: SB99 data dictionary
    :param t: time offset (float)
    :return:
    """

    # check whether input is a dictionary
    if isinstance(SB99_data_IN, collections.Mapping):
        SB99_data = SB99_data_IN.copy()
        SB99_data['t_Myr'] += t
    else: # not a dictionary
        SB99_data = np.copy(SB99_data_IN)
        SB99_data[0] += t

    return SB99_data

def SB99_conc(SB1, SB2):
    """
    concatenate 2 files (assuming file 2 has a later start time)
    :param SB1: SB99 data array or dictionary
    :param SB2: SB99 data array or dictionary
    :return: array
    """

    [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = return_as_array(SB1)
    [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = return_as_array(SB2)

    ii_time = aux.find_nearest_lower(t1, t2[0])

    t = np.append(t1[:ii_time+1], t2)
    Qi = np.append(Qi1[:ii_time + 1], Qi2)
    Li = np.append(Li1[:ii_time + 1], Li2)
    Ln = np.append(Ln1[:ii_time + 1], Ln2)
    Lbol = np.append(Lbol1[:ii_time + 1], Lbol2)
    Lw = np.append(Lw1[:ii_time + 1], Lw2)
    pdot = np.append(pdot1[:ii_time + 1], pdot2)
    pdot_SNe = np.append(pdot_SNe1[:ii_time + 1], pdot_SNe2)

    return [t, Qi, Li, Ln, Lbol, Lw, pdot, pdot_SNe]

def full_sum(t_list, Mcluster_list, Zism, rotation=True, BHcutoff=120., return_format='array'):

    Data = {}
    Data_interp = {}

    t_now = t_list[-1]

    N = len(t_list)
    for ii in range(0,N):
        f_mass = Mcluster_list[ii]/i.SB99_mass
        key = str(ii)
        Data[key] = load_stellar_tracks(Zism, rotation=rotation, f_mass=f_mass, BHcutoff=BHcutoff, return_format='dict')
        Data_interp[key] = make_interpfunc(Data[key])

    Data_tot = Data[str(N-1)]
    if N > 1:
        for ii in range(0,N-1):
            dtSF = t_now - t_list[ii]
            Data_tot = sum_SB99(Data_interp[str(ii)], Data_tot, dtSF)

    Data_tot_interp = make_interpfunc(Data_tot)

    if return_format is 'dict': Data_tot = return_as_dict(Data_tot)
    elif return_format is 'array': Data_tot = return_as_array(Data_tot)

    return Data_tot, Data_tot_interp

def sum_SB99_old(SB99_data1, SB99_data2, dtSF):
    """
    # depricated
    sum two SB99 data tables (where the clusters formed at different times)
    :param SB99_data1:
    :param SB99_data2:
    :param dtSF: time of SF for second cluster minus time of SF for first cluster
    :return:
    """

    [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = SB99_data1
    [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = SB99_data2


    # add birth time of 2nd cluster to evolution time
    #t2 += dtSF
    # find index in first time vector where 2nd cluster formed
    ii_time = aux.find_nearest_lower(t1, dtSF)

    t_evo_old_p2 = t1[ii_time:]
    Qi_evo_old_p2 = Qi1[ii_time:]
    Lw_evo_old_p2 = Lw1[ii_time:]
    Lbol_evo_old_p2 = Lbol1[ii_time:]
    Ln_evo_old_p2 = Ln1[ii_time:]
    Li_evo_old_p2 = Li1[ii_time:]
    pdot_evo_old_p2 = pdot1[ii_time:]
    pdot_SNe_evo_old_p2 = pdot_SNe1[ii_time:]

    t_evo_old_p1 = t1[:ii_time]
    Qi_evo_old_p1 = Qi1[:ii_time]
    Lw_evo_old_p1 = Lw1[:ii_time]
    Lbol_evo_old_p1 = Lbol1[:ii_time]
    Ln_evo_old_p1 = Ln1[:ii_time]
    Li_evo_old_p1 = Li1[:ii_time]
    pdot_evo_old_p1 = pdot1[:ii_time]
    pdot_SNe_evo_old_p1 = pdot_SNe1[:ii_time]

    len_gen1_p2 = len(Qi_evo_old_p2)
    # add up feedback from both clusters
    t_evo = np.concatenate([t_evo_old_p1, t_evo_old_p2])
    Qi_evo = np.concatenate([Qi_evo_old_p1, Qi_evo_old_p2 + Qi2[0:len_gen1_p2]])
    Lw_evo = np.concatenate([Lw_evo_old_p1, Lw_evo_old_p2 + Lw2[0:len_gen1_p2]])
    Lbol_evo = np.concatenate([Lbol_evo_old_p1, Lbol_evo_old_p2 + Lbol2[0:len_gen1_p2]])
    Li_evo = np.concatenate([Li_evo_old_p1, Li_evo_old_p2 + Li2[0:len_gen1_p2]])
    Ln_evo = np.concatenate([Ln_evo_old_p1, Ln_evo_old_p2 + Ln2[0:len_gen1_p2]])
    pdot_evo = np.concatenate([pdot_evo_old_p1, pdot_evo_old_p2 + pdot2[0:len_gen1_p2]])
    pdot_SNe_evo = np.concatenate([pdot_SNe_evo_old_p1, pdot_SNe_evo_old_p2 + pdot_SNe2[0:len_gen1_p2]])

    SB99_data = [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo]

    return SB99_data

def time_shift_old(SB99_data, t):
    """
    add t to time vector in SB99_data
    :param SB99_data:
    :param t: float
    :return:
    """

    [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = SB99_data
    t1 += t
    SB99_data = [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1]

    return SB99_data

