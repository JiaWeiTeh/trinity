import numpy as np
import sys
import myconfig as mc
import os
from astropy.io import ascii
from scipy.interpolate import LinearNDInterpolator
import warnings
#import matplotlib.pyplot as plt
import interpolation as interp
import cool
import constants as myc
import interpolation
import auxiliary_functions as aux
import init as i
import time



# Search ""giving a wrong residual here"" to comment off sys.exit()


def get_Cool_dat_timedep(Zism, age, basename="opiate_cooling", extension=".dat", cool_folder="cooling_tables", indiv_CH=False):
    """
    gets time dependent cooling by interpolating cooling structures at different times
    :param t: in years
    :param Zism:
    :param basename:
    :param extension:
    :param cool_folder:
    :param indiv_CH:
    :return:
    """

    # print(age)
    # sys.exit()
    
    
    
    #Cool_dat1 = get_Cool_dat(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)
    age_lo, age_hi = get_NN_ages(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder)

    # print(age_lo, age_hi)


    # if there exists a file for the requested age, we can just load that file and be done
    if age in [age_lo, age_hi]:
        Cool_dat = get_Cool_dat(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)
        return Cool_dat
    # if there is no file with a lower (or higher) age, do not interpolate (because: not possible) but use only that file instead
    elif age_lo == age_hi:
        Cool_dat = get_Cool_dat(Zism, age_lo, basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)
    # if the indices are different, we need to interpolate
    else:
        # this is the data at the closest lower age: (use as template for final data structure)
        Cool_dat = get_Cool_dat(Zism, age_lo, basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)
        # this is the data at the closest higher age:
        Cool_dat1 = get_Cool_dat(Zism, age_hi, basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)

        t0 = age_lo
        t1 = age_hi
        dt = t1 - t0 # time difference between the closest lower and higher ages
        # now: interpolate (linearly)
        Cool_dat["Netcool"] = Interp_lin(age, [t0, t1], [Cool_dat["Netcool"], Cool_dat1["Netcool"]])
        if indiv_CH:
            Cool_dat["Cool"] = Interp_lin(age, [t0, t1], [Cool_dat["Cool"], Cool_dat1["Cool"]])
            Cool_dat["Heat"] = Interp_lin(age, [t0, t1], [Cool_dat["Heat"], Cool_dat1["Heat"]])

    return Cool_dat





def get_NN_ages(Zism, age, basename="opiate_cooling", extension=".dat", cool_folder="cooling_tables", rotation=i.rotation):
    """
    get closest ages (lower and higher) for which cooling files exist
    :param Zism:
    :param age:
    :param basename:
    :param extension:
    :param cool_folder:
    :param indiv_CH:
    :return:
    """

    # string in filenames which stores information about stellar rotation
    if rotation is True: rot_str = "_rot"
    else: rot_str = "_norot"

    file_list = []
    age_list = []

    # check all files in cooling table folder with the correct filenames (rotating/non-rotating stars, metallicity, extension)
    cooltable_dir = get_cooltable_dir(cool_folder)
    for file in os.listdir(cooltable_dir):
        if (basename in file and rot_str in file and get_Zstring(Zism) in file and file.endswith(extension)): # check whether files with the correct metallicity and data extension exist
            # files will be read in any order, need to sort them later
            file_list.append(file)
            age_list.append(extract_keyvalue(file, key="age"))

    if len(file_list) == 0:
        sys.exist("No cooling tables! E.g., I am expecting: "+make_cooling_filename(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder))

    # sort according to age
    pairs = list(zip(age_list, file_list))
    pairs.sort()
    age_list = [ x[0] for x in pairs ]
    # find files with the closest lower and higher age
    idx0 = aux.find_nearest_lower(age_list,age)
    idx1 = aux.find_nearest_higher(age_list, age)
    age_lower = age_list[idx0]
    age_higher = age_list[idx1]
    
    return age_lower, age_higher


def get_cooltable_dir(cool_folder):

    cooltable_dir = mc.path_to_code + "/" + cool_folder + "/"

    return cooltable_dir

def get_Zstring(Zism):

    Zstr = "Z" + format(Zism, '.2f')

    return Zstr

def get_agestring(age):

    agestr = "age" + format(age, '.2e')

    return agestr

def extract_keyvalue(string, key="age"):

    len_key = len(key)
    if key == "age":
        number_len = len(get_agestring(1e5)) - len_key # get the length of the number in the age-string by checking an example age
    elif key == "Z":
        number_len = len(get_Zstring(1e5)) - len_key
    idx = string.find(key)
    idx0 = idx + len_key
    idx1 = idx0 + number_len
    value = float(string[idx0:idx1])

    return value

def make_cooling_filename(Zism, age, basename="opiate_cooling", extension = ".dat", cool_folder = "cooling_tables", rotation=i.rotation):

    if rotation is True: rot_str = "rot"
    else: rot_str = "norot"

    cooltable_dir = get_cooltable_dir(cool_folder)
    agestr = get_agestring(age)
    Zstr = get_Zstring(Zism)  # use 2 digits after point
    cooltable_file = cooltable_dir + basename + "_" + rot_str + "_" + Zstr + "_" + agestr + extension

    return cooltable_file


def get_Cool_dat(Zism, age, basename="opiate_cooling", extension=".dat", cool_folder="cooling_tables", indiv_CH=False):
    """
    get cooling data
    this is the main routine that provides the data which can be interpolated
    :param Zism: metallicity in solar units (1.0 corresponds to solar metallicity)
    :param age: age of cluster in years
    :param basename: (optional) name of cooling table without metallicity identifier and extension
    :param extension: (optional) extension of cooling table file
    :param cool_folder: folder name where cooling table is stored
    :return: Data structure containing:
                log_n: number density, data structure containing:
                        dat: list from min log_n to max log_n
                        min: minimum
                        max: maximum
                        d: distance between 2 elements in log10 (can only deal with equal distances)
                log_T: temperature, data structure like log_n
                log_Phi: Ionizing Photons number flux, data structure like log_n and log_T
                Netcool: net cooling values (ne*np*Lambda) 3D-array (NOT LOG) on grid which is spanned by
                            1D: n, 2D: T, 3rd D: Phi
    """

    # if a numpy-readable version of the cooling table exists, use it
    # if not, create such a file

    np_coolfilename = make_cooling_filename(Zism, age, basename=basename, extension = ".npy", cool_folder = cool_folder)

    if not indiv_CH:
        if os.path.isfile(np_coolfilename):
            ln_dat, lT_dat, lP_dat = get_opiate_gridstruc(Zism, age, basename="opiate_cooling", extension=".dat", cool_folder="cooling_tables")
            NetCool = np.load(np_coolfilename)
            Cool_dat = {"Netcool": NetCool, "log_n": ln_dat, "log_T": lT_dat, "log_Phi": lP_dat}
        else:
            Cool_dat = prep_coolingtable(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)

    elif indiv_CH:
        np_onlycoolfilename = make_cooling_filename(Zism, age, basename=basename + "C", extension = ".npy", cool_folder = cool_folder)
        np_onlyheatfilename = make_cooling_filename(Zism, age, basename=basename + "H", extension=".npy", cool_folder=cool_folder)
        # do the numpy readable-files already exist?
        if (os.path.isfile(np_coolfilename) and os.path.isfile(np_onlycoolfilename)) and os.path.isfile(np_onlyheatfilename):
            ln_dat, lT_dat, lP_dat = get_opiate_gridstruc(Zism, age, basename="opiate_cooling", extension=".dat", cool_folder="cooling_tables")
            NetCool = np.load(np_coolfilename)
            Cool_dat = {"Netcool": NetCool, "log_n": ln_dat, "log_T": lT_dat, "log_Phi": lP_dat}
            Cool = np.load(make_cooling_filename(Zism, age, basename=basename + "C", extension=".npy", cool_folder=cool_folder))
            Cool_dat["Cool"] = Cool
            Heat = np.load(make_cooling_filename(Zism, age, basename=basename + "H", extension=".npy", cool_folder=cool_folder))
            Cool_dat["Heat"] = Heat
        else:
            Cool_dat = prep_coolingtable(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)

    return Cool_dat




def read_opiatetable(Zism, age, basename="opiate_cooling", extension = ".dat", cool_folder = "cooling_tables"):
    """
    load the external cooling table
    (IMPORTANT: assumes file name contains metallicity with 2 digits after point)
    :param Zism: metallicity in solar units (1.0 corresponds to solar metallicity)
    :param age: age of cluster in years
    :param basename: (optional) name of cooling table without metallicity identifier and extension
    :param extension: (optional) extension of cooling table file
    :param cool_folder: folder name where cooling table is stored
    :return: number density, temperature, photon number flux, (only) cooling, (only) heating, net cooling
    """

    # construct name of cooling table
    cooltable_file = make_cooling_filename(Zism, age, basename=basename, extension = extension, cool_folder = cool_folder)

    # print('filename', cooltable_file)

    # check whether cooling table exists, exit if it does not
    if not os.path.isfile(cooltable_file):
        sys.exit("cooling table does not exist: {}".format(cooltable_file))


    # read in cooling table
    datatable = ascii.read(cooltable_file)

    ndens = datatable["ndens"]
    temp = datatable["temp"]
    Phi = datatable["phi"]

    cool = datatable["cool"]
    heat = datatable["heat"]

    # IMPORTANT: CHECK WHETHER SIGNS IN HEATING IN COOLING COLUMN ARE DIFFERENT
    # I WANT POSITIVE SIGNS FOR BOTH
    if np.sign(heat[0]) == -1.:
        warnings.warn("Heating column has negative signs in {}. I will change the signs to positive.".format(cooltable_file))
        heat = -1.0 * heat  # now they have the same signs
    if np.sign(cool[0]) == -1.:
        warnings.warn("Cooling column has negative signs in {}. I will change the signs to positive.".format(cooltable_file))
        cool = -1.0 * cool  # now they have the same signs

    netcool = cool - heat

    return ndens, temp, Phi, cool, heat, netcool


def get_opiate_gridstruc(Zism, age, basename="opiate_cooling", extension = ".dat", cool_folder = "cooling_tables"):
    """
    analyze the structure of the opiate cooling grid
    :param Zism: metallicity in solar units (1.0 corresponds to solar metallicity)
    :param age: age of cluster in years
    :param basename: (optional) name of cooling table without metallicity identifier and extension
    :param extension: (optional) extension of cooling table file
    :param cool_folder: folder name where cooling table is stored
    :return: log_n: number density, data structure containing:
                        dat: list from min log_n to max log_n
                        min: minimum
                        max: maximum
                        d: distance between 2 elements in log10 (can only deal with equal distances)
            log_T: temperature, data structure like log_n
            log_Phi: Ionizing Photons number flux, data structure like log_n and log_T
    """

    ndens, temp, Phi, cool, heat, netcool = read_opiatetable(Zism, age, basename=basename, extension=extension,
                                                             cool_folder=cool_folder)

    ldens = np.log10(ndens)
    ltemp = np.log10(temp)
    lPhi = np.log10(Phi)

    min_ln = np.min(ldens)
    min_lT = np.min(ltemp)
    min_lP = np.min(lPhi)

    max_ln = np.max(ldens)
    max_lT = np.max(ltemp)
    max_lP = np.max(lPhi)

    list_ldens = np.round(np.sort(np.array(list(set(ldens)))), decimals=3)  # actually an array
    list_ltemp = np.round(np.sort(np.array(list(set(ltemp)))), decimals=3)
    list_lPhi = np.round(np.sort(np.array(list(set(np.round(lPhi))))), decimals=3)


    d_ln = np.round(np.diff(list_ldens), decimals=3)
    d_lT = np.around(np.diff(list_ltemp), decimals=3)
    d_lP = np.round(np.diff(list_lPhi), decimals=3)

    # check whether distance between grid points in log is constant
    if (len(set(d_ln)) != 1) or (len(set(d_lT)) != 1) or (len(set(d_lP)) != 1):
        sys.exit("Structure of cooling table not recognized! Distance between grid points in log is not constant.")

    d_ln = d_ln[0]
    d_lT = d_lT[0]
    d_lP = d_lP[0]

    ln_dat = {"dat": list_ldens, "min": min_ln, "max": max_ln, "d": d_ln}
    lT_dat = {"dat": list_ltemp, "min": min_lT, "max": max_lT, "d": d_lT}
    lP_dat = {"dat": list_lPhi, "min": min_lP, "max": max_lP, "d": d_lP}

    return ln_dat, lT_dat, lP_dat


def read_opiatetable_NEDENS(Zism, age, basename="opiate_cooling", extension = ".dat", cool_folder = "cooling_tables"):
    """
    copy of read_opiatetable with the addition that also the electron density is read out
    :param Zism: metallicity in solar units (1.0 corresponds to solar metallicity)
    :param age: age of cluster in years
    :param basename: (optional) name of cooling table without metallicity identifier and extension
    :param extension: (optional) extension of cooling table file
    :param cool_folder: folder name where cooling table is stored
    :return: number density, electron density, temperature, photon number flux, (only) cooling, (only) heating, net cooling
    """

    # construct name of cooling table
    cooltable_file = make_cooling_filename(Zism, age, basename=basename, extension = extension, cool_folder = cool_folder)

    # check whether cooling table exists, exit if it does not
    if not os.path.isfile(cooltable_file):
        sys.exit("cooling table does not exist: {}".format(cooltable_file))


    # read in cooling table
    datatable = ascii.read(cooltable_file)

    ndens = datatable["ndens"]
    nedens = datatable["nedens"]
    temp = datatable["temp"]
    Phi = datatable["phi"]

    cool = datatable["cool"]
    heat = datatable["heat"]

    # IMPORTANT: CHECK WHETHER SIGNS IN HEATING IN COOLING COLUMN ARE DIFFERENT
    # I WANT POSITIVE SIGNS FOR BOTH
    if np.sign(heat[0]) == -1.:
        warnings.warn("Heating column has negative signs in {}. I will change the signs to positive.".format(cooltable_file))
        heat = -1.0 * heat  # now they have the same signs
    if np.sign(cool[0]) == -1.:
        warnings.warn("Cooling column has negative signs in {}. I will change the signs to positive.".format(cooltable_file))
        cool = -1.0 * cool  # now they have the same signs

    netcool = cool - heat

    return ndens, nedens, temp, Phi, cool, heat, netcool


def prep_coolingtable(Zism, age, basename="opiate_cooling", extension = ".dat", cool_folder = "cooling_tables", indiv_CH=False):
    """
    prepare an ordered, fast-to-access run-time version of the cooling table
    :param Zism: metallicity in solar units (1.0 corresponds to solar metallicity)
    :param age: age of cluster in years
    :param basename: (optional) name of cooling table without metallicity identifier and extension
    :param extension: (optional) extension of cooling table file
    :param cool_folder: folder name where cooling table is stored
    :return: Data structure containing:
                log_n: number density, data structure containing:
                        dat: list from min log_n to max log_n
                        min: minimum
                        max: maximum
                        d: distance between 2 elements in log10 (can only deal with equal distances)
                log_T: temperature, data structure like log_n
                log_Phi: Ionizing Photons number flux, data structure like log_n and log_T
                Netcool: net cooling values (ne*np*Lambda) 3D-array (NOT LOG) on grid which is spanned by
                            1D: n, 2D: T, 3rd D: Phi
    """

    ndens, temp, Phi, cool, heat, netcool = read_opiatetable(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder)

    # save data in a numpy readable way
    np_file_full = make_cooling_filename(Zism, age, basename=basename+"_full", extension="", cool_folder=cool_folder)  # no extension because the extension .npy is added automatically
    if not os.path.isfile(np_file_full):
        np.save(np_file_full, [ndens, temp, Phi, cool, heat])

    ldens = np.log10(ndens)
    ltemp = np.log10(temp)
    lPhi = np.log10(Phi)

    # find minimum and maximum extent of grid
    min_ln = np.min(ldens)
    min_lT = np.min(ltemp)
    min_lP = np.min(lPhi)

    max_ln = np.max(ldens)
    max_lT = np.max(ltemp)
    max_lP = np.max(lPhi)

    # get ordered lists in log_dens, log_temp, and log_Phi which span grid
    list_ldens = np.round(np.sort(np.array(list(set(ldens)))),decimals=3) # actually an array
    list_ltemp = np.round(np.sort(np.array(list(set(ltemp)))),decimals=3)
    list_lPhi = np.round(np.sort(np.array(list(set(np.round(lPhi))))),decimals=3)

    len_dens = len(list_ldens)
    len_temp = len(list_ltemp)
    len_Phi = len(list_lPhi)

    # calculated distance between points in each of the 3 dimensions
    d_ln = np.round(np.diff(list_ldens), decimals=3)
    d_lT = np.around(np.diff(list_ltemp), decimals=3)
    d_lP = np.round(np.diff(list_lPhi), decimals=3)

    # check whether distance between grid points in log is constant
    if (len(set(d_ln))!=1) or (len(set(d_lT))!=1) or (len(set(d_lP))!=1):
        sys.exit("Structure of cooling table not recognized! Distance between grid points in log is not constant.")

    # now we know the distances in log are constant, can as well take the first entry
    d_ln = d_ln[0]
    d_lT = d_lT[0]
    d_lP = d_lP[0]

    # store important info in data structure
    ln_dat = {"dat":list_ldens, "min": min_ln, "max": max_ln, "d":d_ln}
    lT_dat = {"dat":list_ltemp, "min": min_lT, "max": max_lT, "d": d_lT}
    lP_dat = {"dat":list_lPhi, "min": min_lP, "max": max_lP, "d": d_lP}


    # now access original cooling table and rewrite it in a more efficiently numpy-accessible way

    NetCool = np.ones([len_dens, len_temp, len_Phi])*np.nan  # create array of size len_dens x len_temp x len_Phi and fill with NaN
    Cool = np.ones([len_dens, len_temp, len_Phi])*np.nan
    Heat = np.ones([len_dens, len_temp, len_Phi])*np.nan

    ii_Phi = 0
    for lP in list_lPhi:
        jj_temp = 0
        mask1 = np.array(np.abs(lP-lPhi)<0.1*d_lP)
        for lT in list_ltemp:
            kk_dens = 0
            mask2 = np.array(np.abs(lT-ltemp)<0.1*d_lT)
            for ln in list_ldens:
                mask3 = np.array(np.abs(ln-ldens)<0.1*d_ln)

                mask = mask1*mask2*mask3

                # only overwrite NaN if entry exists
                if np.size(netcool[mask]) != 0:
                    NetCool[kk_dens, jj_temp, ii_Phi] = netcool[mask][0]
                    Cool[kk_dens, jj_temp, ii_Phi] = cool[mask][0]
                    Heat[kk_dens, jj_temp, ii_Phi] = heat[mask][0]

                kk_dens += 1
            jj_temp += 1
        ii_Phi += 1

    # save array cube to file
    print("saving numpy-readable version of net cooling table for future uses...")
    file1 = make_cooling_filename(Zism, age, basename=basename, extension = "", cool_folder = cool_folder) # no extension because the extension .npy is added automatically
    np.save(file1, NetCool)

    # save individual Cooling and Heating tables
    if indiv_CH:
        print("saving numpy-readable version of cooling and heating tables for future uses...")
        fileC = make_cooling_filename(Zism, age, basename=basename+"C", extension="",
                                     cool_folder=cool_folder)  # no extension because the extension .npy is added automatically
        np.save(fileC, Cool)
        fileH = make_cooling_filename(Zism, age, basename=basename+"H", extension="",
                                     cool_folder=cool_folder)  # no extension because the extension .npy is added automatically
        np.save(fileH, Heat)

    Cool_dat = {"Netcool": NetCool, "Cool": Cool, "Heat": Heat, "log_n": ln_dat, "log_T": lT_dat, "log_Phi": lP_dat}

    return Cool_dat



def Interp_lin(x, xarr, yarr):
    """
    linear interpolation
    :param x: point at which to interpolate
    :param xarr: list or array with 2 elements
    :param yarr: list or array with 2 elements
    :return:
    """

    y = yarr[0] + (yarr[1] - yarr[0]) * (x-xarr[0])/(xarr[1]-xarr[0])

    return y


def Interp3_dudt(point, Cool_Struc, element = "Netcool"):
    """
    Interpolates cooling function which depends on density, temperature, and photon number flux (ionizing)
    This is the main routine to call every time you request a cooling value for some parameter tuple
    :param point: structure (not log), containing number density "n", temperature "T", and photon number flux (ionizing) "Phi"
    :param Cool_Struc: see output of get_Cool_dat()
    :return: energy (net cooling/heating) rate du/dt, i.e. ne*np*(Lambda - Gamma)
    """
    
    # print('Interp3 is used')

    # point at which to interpolate (not log)
    x = point["n"]
    y = point["T"]
    z = point["Phi"]

    # got to log (necessary to find corners of surrounding cuboid as distance between points in constant in log only)
    log_x = np.log10(x)
    log_y = np.log10(y)
    log_z = np.log10(z)

    # unpack tabulated data
    my_element = Cool_Struc[element]
    
    # np.save('cool_interp3dudtmyelement', my_element)
    # sys.exit()
    
    
    ln_dat = Cool_Struc["log_n"]
    lT_dat = Cool_Struc["log_T"]
    lP_dat = Cool_Struc["log_Phi"]

    # find indices of cuboid in which "point" lies
    ii_n_0 = int((log_x-ln_dat["min"])/ln_dat["d"])
    jj_T_0 = int((log_y-lT_dat["min"])/lT_dat["d"])
    kk_P_0 = int((log_z-lP_dat["min"])/lP_dat["d"])

    ii_n_1 = ii_n_0 + 1
    jj_T_1 = jj_T_0 + 1
    kk_P_1 = kk_P_0 + 1

    # to have true linear interpolation go to linear space instead of log-space
    x0 = 10. ** ln_dat["dat"][ii_n_0]
    x1 = 10. ** ln_dat["dat"][ii_n_1]

    y0 = 10. ** lT_dat["dat"][jj_T_0]
    y1 = 10. ** lT_dat["dat"][jj_T_1]

    z0 = 10. ** lP_dat["dat"][kk_P_0]
    z1 = 10. ** lP_dat["dat"][kk_P_1]


    # call interpolator
    dudt = interp.trilinear([x, y, z], [x0, y0, z0], [x1, y1, z1],
                         my_element[ii_n_0:ii_n_0 + 2, jj_T_0:jj_T_0 + 2, kk_P_0:kk_P_0 + 2])
    
    # print('\n\nto interpolate', 
    #       't_now', point['t_now'],
    #       '\nx:', x, 
    #       '\ny:', y, 
    #       '\nz', z, 
    #       '\nsecond three',
    #       'x0:', x0, 
    #       '\ny0:', y0, 
    #       '\nz0', z0, 
    #       '\nthird three',
    #       '\nx1:', x1, 
    #       '\ny1:', y1, 
    #       '\nz1', z1,     
    #       '\ngridpoints',
    #       '\narray', my_element[ii_n_0:ii_n_0 + 2, jj_T_0:jj_T_0 + 2, kk_P_0:kk_P_0 + 2],
    #       )
    
    # sys.exit()

    # print('value', dudt)

    return dudt


def cool_interp_master(point, Cool_Struc, log_T_intermin = 3.9, log_T_noeqmin = 4.0, log_T_noeqmax = 5.4, log_T_intermax=5.499):

    #log_T_intermin = log_T_noeqmin - log_T_interd
    #log_T_intermax = log_T_noeqmax + log_T_interd

    if (np.log10(point["T"]) > log_T_intermax) or (np.log10(point["T"]) < log_T_intermin):
        #start = time.time()
        Lambda = cool.coolfunc(point["T"])
        dudt = -1. * (point["n"]) ** 2 * Lambda / myc.dudt_cgs
        #end = time.time()
        #print(end - start, '######')

    elif (np.log10(point["T"]) >= log_T_noeqmax):
        #start = time.time()
        dudt1 = -1. * (point["n"]) ** 2 * cool.coolfunc(point["T"]) / myc.dudt_cgs
        dudt0 = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"], 't_now': point['t_now']}, Cool_Struc) / myc.dudt_cgs
        dudt = interpolation.linear(np.log10(point["T"]), [log_T_noeqmax, log_T_intermax], [dudt0, dudt1])
        #end = time.time()
        #print(end - start, '#####')

    elif (np.log10(point["T"]) <= log_T_noeqmin):
        #start = time.time()
        dudt0 = -1. * (point["n"]) ** 2 * cool.coolfunc(point["T"]) / myc.dudt_cgs
        dudt1 = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"], 't_now': point['t_now']}, Cool_Struc) / myc.dudt_cgs
        dudt = interpolation.linear(np.log10(point["T"]), [log_T_intermin, log_T_noeqmin], [dudt0, dudt1])
        #end = time.time()
        #print(end - start, '##')

    else:
        #start = time.time()
        dudt = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"], 't_now': point['t_now']}, Cool_Struc) / myc.dudt_cgs
        #end = time.time()
        #print(end - start, '####')


    #if np.isnan(dudt):
    #    dudt1 = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"]}, Cool_Struc)
    #    sys.exit("Cooling Function is NaN")
    
    # print(point["n"], point["T"], point["Phi"], dudt)
    # sys.exit()

    # print(point, dudt)
    
    return dudt

#### Functions below are depricated.

def create_coolingfunc(Zism, age):
    # depricated
    """
    creates a cooling function in the case of no collisional ionization equilibrium by interpolating OPIATE cooling tables
    :param Zism: metallicity in Zsol (1.0 is solar metallicity)
    :param age: age of cluster in years
    :return: coolfunc: function object which takes 1 argument of dim 3 (or dimension x times 3, if array):
            [lg_ndens, lg_temp, lg_Phi]
            [log10 number density (1/ccm), log10 temperature (K), log10 photon flux of ionizing photons (1/cm^2/s)]
            coolfunc returns net cooling (erg cm^3/s), not log10
    remark:
    call coolfunc via: coolnoeq.coolfunc([lg_dens, lg_temp, lg_Phi]), e.g. coolnoeq.coolfunc([2.1, 5.1, 10.9])
    works also for arrays: e.g. coolnoeq.coolfunc(np.array([[2.1, 5.1, 10.9],[2.2, 5.2, 11.0]])
    """

    ndens, temp, Phi, cool, heat, netcool = read_opiatetable(Zism, age)

    # stack density, temperature and photon flux together
    phase_points = np.transpose(np.vstack([ndens,temp,Phi])) # seems I need to transpose

    # interpolate cooling rate
    coolfunc = LinearNDInterpolator(np.log10(phase_points),np.log10(netcool)) # TAKE LOG, OTHERWISE INTERPOLATION CRASHES

    # return the function object
    return coolfunc

def create_onlycoolheat(Zism, age, basename="opiate_cooling", extension=".dat", cool_folder="cooling_tables"):
    """
    :param Zism: metallicity in solar units (1.0 = solar)
    :param age: age of cluster in years
    :return: interpolation functions which return the log10 of heating and cooling [erg/ccm/s] (i.e. n^2 is already incoorporated)
    """

    age_lo, age_hi = get_NN_ages(Zism, age, basename=basename, extension=extension, cool_folder=cool_folder)



    # determine which file to read in and whether to interpolate (time)
    if age in [age_lo, age_hi]:
        #ndens, temp, Phi, cool, heat, netcool = read_opiatetable(Zism, age)

        np_file_full = make_cooling_filename(Zism, age, basename=basename + "_full", extension="", cool_folder=cool_folder)
        Bmatrix = np.load(np_file_full+'.npy')
        cool = Bmatrix[3,:]
        heat = Bmatrix[4,:]



    elif age_lo == age_hi:
        #ndens, temp, Phi, cool, heat, netcool = read_opiatetable(Zism, age_lo)

        np_file_full = make_cooling_filename(Zism, age_lo, basename=basename + "_full", extension="", cool_folder=cool_folder)
        Bmatrix = np.load(np_file_full+'.npy')
        cool = Bmatrix[3,:]
        heat = Bmatrix[4,:]


    else:
        #ndens, temp, Phi, cool0, heat0, netcool = read_opiatetable(Zism, age_lo)
        #ndens, temp, Phi, cool1, heat1, netcool = read_opiatetable(Zism, age_hi)

        np_file_full0 = make_cooling_filename(Zism, age_lo, basename=basename + "_full", extension="", cool_folder=cool_folder)
        np_file_full1 = make_cooling_filename(Zism, age_hi, basename=basename + "_full", extension="", cool_folder=cool_folder)
        Bmatrix = np.load(np_file_full0+'.npy')
        cool0 = Bmatrix[3,:]
        heat0 = Bmatrix[4,:]
        Bmatrix = np.load(np_file_full1+'.npy')
        cool1 = Bmatrix[3,:]
        heat1 = Bmatrix[4,:]

        cool = Interp_lin(age, [age_lo, age_hi], [cool0, cool1])
        heat = Interp_lin(age, [age_lo, age_hi], [heat0, heat1])

    ndens = Bmatrix[0, :]
    temp = Bmatrix[1, :]
    Phi = Bmatrix[2, :]

    # stack density, temperature and photon flux together
    phase_points = np.transpose(np.vstack([ndens, temp, Phi]))  # seems I need to transpose

    # interpolate cooling rate
    onlycoolfunc = LinearNDInterpolator(np.log10(phase_points), np.log10(cool)) # TAKE LOG, OTHERWISE INTERPOLATION CRASHES
    onlyheatfunc = LinearNDInterpolator(np.log10(phase_points), np.log10(heat)) # TAKE LOG, OTHERWISE INTERPOLATION CRASHES

    # return the function object
    return onlycoolfunc, onlyheatfunc