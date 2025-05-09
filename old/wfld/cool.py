import constants as c
import numpy as np
import sys
from scipy import spatial
import auxiliary_functions as aux
import datetime
from scipy.interpolate import interp1d
import init as i
import time

# data from see Joung & Mac Low 2006 Figure 1, uses Dalgarmo & McCray 1972 for T < 2e4K and Sutherland & Dopita 1993 for T > 2e4K)
# the first element in the following 3 lists has been added by hand (DR), i.e. cooling is not correct below 10**0.75K but the code doesn't crash if some low temperature appears
#logT_list = np.array([-6.,0.75,0.971374,1.34494,1.55248,2.24427,2.81155,3.47567,3.69704,3.95992,4.19513,4.55487,4.87309,5.37118,5.71708,6.22901,6.42271,7.48807,8.16603])
#logLambda_list = np.array([-28.0288,-28.0288,-28.0288,-26.5666,-26.2503,-25.5429,-25.3027,-25.0628,-25.0331,-24.9431,-21.7462,-21.8224,-21.2046,-21.1906,-21.8851,-21.8862,-22.4295,-22.7786,-22.4332])
#slope_exp_list = np.array([0.,0.,3.9141677776,1.5240435579,1.0225646511,0.4234240587,0.3612298982,0.1341645209,0.3423615338,13.591684027,-0.2118196475,1.941424172,0.0281073702,-2.0078057242,-0.0021487313,-2.8048528653,-0.3276826613,0.5094695852,0.5])


# Cloudy (HII region, metals 1.0, no grains), (first two elements added by hand)
logT_cloudy_nograins =      np.array([1.0 , 2.391,  2.4    , 2.5    , 2.6    , 2.7    , 2.8    , 2.9    , 3.     , 3.1    , 3.2   , 3.3    , 3.4    , 3.5    , 3.6    , 3.7   , 3.8     , 3.9    , 4.     , 4.1    , 4.2    ,  4.3   ,  4.4   ,  4.5   ,  4.6   ,  4.7   ,  4.8   ,  4.9   ,  5.    ,   5.1  ,  5.2   ,  5.3   ,  5.4   , 5.5    ,  5.6   ,  5.7   ,  5.8   ,  5.9   ,  6.    ,   6.1  ,  6.2   ,  6.3   ,  6.4   ,  6.5   ,  6.6   ,  6.7   ,  6.8   ,  6.9   , 7.    ,  7.1   ,  7.2   ,  7.3   ,  7.4   ,  7.5   ,  7.6   ,  7.7   ,  7.8   ,  7.9  ,  8.     ,   8.1  ,  8.2  ,  8.3    ,  8.4   , 8.5    ,  8.6  ,  8.7    ,  8.8   ,  8.9   ,  9.    ,   9.1  ,  9.2   ,  9.3   ,  9.4   ,  9.5   ,  9.6   ,  9.7  ,  9.8    ,  9.9])
logLambda_cloudy_nograins = np.array([-30., -30., -27.309, -26.759, -26.487, -26.301, -26.148, -26.014, -25.882, -25.771, -25.65, -25.525, -25.408, -25.285, -25.156, -25.06,  -25.058, -25.115, -24.809, -23.566, -22.061, -21.861, -21.987, -22.002, -21.926, -21.777, -21.584, -21.397, -21.326, -21.387, -21.408, -21.399, -21.409, -21.626, -21.956, -22.109, -22.281, -22.486, -22.547, -22.547, -22.545, -22.583, -22.631, -22.715, -22.799, -22.852, -22.875, -22.874, -22.862, -22.85, -22.836, -22.813, -22.782, -22.745, -22.705, -22.662, -22.616, -22.57,  -22.522, -22.472, -22.42,  -22.367, -22.311, -22.252, -22.19,  -22.124, -22.052, -21.974, -21.888, -21.792, -21.686, -21.569, -21.443, -21.307, -21.166, -21.02,  -20.872, -20.725])

# Cloudy (HII region, metals 1.0, with grains and sublimation), (first two elements added by hand)
logT_cloudy_grains =      np.array([1.0 , 2.291,    2.3,    2.4,      2.5,     2.6,    2.7,     2.8,      2.9,     3.,      3.1,     3.2,     3.3,     3.4,     3.5,     3.6,     3.7,     3.8,     3.9,      4.,     4.1,     4.2,     4.3,     4.4,     4.5,     4.6,    4.7,      4.8,     4.9,     5.,      5.1,     5.2,     5.3,     5.4,    5.5,      5.6,     5.7,     5.8,     5.9,      6.,     6.1,     6.2,     6.3,     6.4,     6.5,     6.6,     6.7,     6.8,     6.9,      7.,     7.1,     7.2,     7.3,     7.4,     7.5,     7.6,     7.7,     7.8,     7.9,      8.,     8.1,     8.2,     8.3,     8.4,     8.5,     8.6,     8.7,     8.8,     8.9,      9.,     9.1,     9.2,     9.3,     9.4,     9.5,     9.6,    9.7])
logLambda_cloudy_grains = np.array([-30., -30., -27.046, -26.52,  -26.243, -26.044, -25.89,  -25.76,  -25.646, -25.54,  -25.454, -25.379, -25.318, -25.258, -25.177, -25.075, -24.999, -25.036, -25.162, -24.948, -23.755, -22.062, -21.846, -21.969, -21.987, -21.915, -21.77,  -21.579, -21.393, -21.32,  -21.379, -21.398, -21.388, -21.394, -21.59,  -21.861, -21.936, -21.967, -21.942, -21.829, -21.688, -21.541, -21.392, -21.239, -21.085, -20.929, -20.773, -20.617, -20.462, -20.308, -20.154, -20.002, -19.849, -19.698, -19.546, -19.395, -19.244, -19.094, -18.943, -18.793, -18.642, -18.492, -18.342, -18.192, -18.042, -17.892, -17.742, -17.592, -17.443, -17.293, -17.143, -16.994, -16.845, -16.696, -16.547, -16.398, -16.25])


#logT_mod_list = logT_list
#logLambda_mod_list = logLambda_list
#slope_exp_mod_list = slope_exp_list

# Gnat & Ferland 2012 (last element added by hand)
logT_GnatFerland =      np.array([3.99999,     4.05817,     4.14765,     4.21924,      4.29978,      4.41611,      4.50112,     4.62192,    4.86801,     4.94855,     5.02013,      5.11857,     5.33333,     5.3915,          5.45861,         5.5481,        5.61074,        5.6868,        5.74944,        5.84787,        6.03579,        6.17897,        6.37136,        6.54586,        6.76063,        7.01119,        7.20358,        7.40045,        7.65548,        8.00001,    10.00001])
logLambda_GnatFerland = np.array([-22.8853,    -22.5492,    -21.9223,    -21.7133,     -21.8159,     -21.9758,     -22.0046,    -21.9187,   -21.4599,    -21.3821,    -21.3658,     -21.3987,    -21.3415,    -21.3375,        -21.4195,       -21.5877,       -21.6205,       -21.6370,      -21.6821,       -21.8134,       -21.8832,       -21.9612,       -22.2811,       -22.5108,       -22.5889,       -22.5728,       -22.6631,       -22.7412,       -22.7169,       -22.6475,   -21.6475])

# modified Gnat & Ferland (between 10**5.4 and 10**5.5)
logT_GF_mod_list =      np.array([3.99999,     4.05817,     4.14765,     4.21924,      4.29978,      4.41611,      4.50112,     4.62192, 4.8,             5.0,            5.3,            5.35,           5.4,            5.45,        5.5,           5.5481,        5.61074,        5.6868,         5.74944,        5.84787,        6.03579,        6.17897,        6.37136,        6.54586,        6.76063,        7.01119,        7.20358,        7.40045,        7.65548,        8.00001,    10.00001])
logLambda_GF_mod_list = np.array([-22.8853,    -22.5492,    -21.9223,    -21.7133,     -21.8159,     -21.9758,     -22.0046,    -21.9187,-21.58367835,    -21.32552679,   -21.39893145,   -21.40349377,   -21.40866418,   -21.451105,  -21.49729414,  -21.5877,      -21.6205,       -21.6370,       -21.6821,       -21.8134,       -21.8832,       -21.9612,       -22.2811,       -22.5108,       -22.5889,       -22.5728,       -22.6631,       -22.7412,       -22.7169,       -22.6475,   -21.6475])
slope_GF_exp_mod_list = np.append((logLambda_GF_mod_list[1:] - logLambda_GF_mod_list[0:-1]) / (logT_GF_mod_list[1:] - logT_GF_mod_list[0:-1]), 0.5)

# Gnedin, Hollon 2012 (only compilation)
#logT_Gnedin = np.array([3.90909,4.,4.125,4.20455,4.30682,4.40909,4.52273,4.60227,4.75,4.90909,5.,5.05682,5.20455,5.39773,5.61364,5.70455,5.92045,6.11364,6.22727,6.51136,6.67045,7.04545,7.31818,7.52273,8.19318])
#logL_Gnedin = np.array([-25.4065,-25.1367,-23.5,-22.3489,-22.1691,-22.277,-22.295,-22.223,-21.9892,-21.6835,-21.6295,-21.6475,-21.6475,-21.5935,-21.8813,-21.8813,-22.0971,-22.1691,-22.277,-22.7626,-22.8705,-22.8165,-22.9604,-22.9784,-22.7086])

# Gnat, Sternberg 2007
#logT_GnatSternberg = np.array([4.0,4.02363,4.04136,4.17725,4.22452,4.26588,4.41359,4.4904,4.54948,4.61448,4.88626,5.01625,5.13442,5.37075,5.43575,5.54801,5.613,5.70753,5.83752,5.8966,5.97341,6.19202,6.27474,6.57016,6.70606,6.83604,7.05465,7.30871,7.46824,7.89365,7.99409])
#logL_GnatSternberg = np.array([-22.2935,-22.5163,-22.538,-21.7772,-21.712,-21.7446,-21.9783,-22.0109,-21.9946,-21.9293,-21.4348,-21.3587,-21.3967,-21.3315,-21.3696,-21.587,-21.625,-21.6413,-21.7989,-21.837,-21.8424,-21.9348,-22.0272,-22.413,-22.4891,-22.5217,-22.5489,-22.7011,-22.7228,-22.5924,-22.5815])

# Schurre, Kosenko, et al. 2009
#logT_Schurre = np.array([3.8,4.0125,4.15,4.2125,4.275,4.3625,4.4,4.4375,4.7625,4.9625,5.1,5.225,5.3875,5.6,5.725,5.85,6.2125,6.5125,6.675,6.8875,7.3375,7.55,8.1375])
#logL_Schurre = np.array([-25.6818,-22.6136,-21.4545,-21.3182,-21.4091,-21.6136,-21.6591,-21.6364,-20.8864,-20.7273,-20.7955,-20.7273,-20.75,-21.4545,-21.5227,-21.7273,-21.8409,-22.3864,-22.4318,-22.4091,-22.7045,-22.6818,-22.4773])

# Sutherland and Dopita1993 for [Fe/H] = -1
logT_SD_Z002 = np.array([3.9999, 4.2, 4.4, 4.52,4.66,4.82,4.94,4.99,5.09,5.29,5.38,5.45,5.51,5.57,5.64,5.70,5.77,5.85,5.92,6.03,6.12,6.24,6.35,6.45,6.63,6.78,7.00,7.16,7.45,7.65,7.93,8.34,8.49, 10.01])
logLambda_SD_Z002 = np.array([-23.31,-21.88,-22.16,-22.20,-22.09,-21.83,-21.66,-21.63,-21.62,-21.50,-21.47,-21.55,-21.76,-21.98,-22.16,-22.20,-22.22,-22.33,-22.51,-22.63,-22.67,-22.68,-22.78,-22.90,-22.99,-23.02,-22.99,-22.97,-22.90,-22.83,-22.71,-22.53,-22.47,-21.64])

# determine which CIE cooling curves to use
if i.Zism == 1.0:
    logT_list = logT_GnatFerland
    logLambda_list = logLambda_GnatFerland
    logT_mod_list = logT_GF_mod_list
    logLambda_mod_list = logLambda_GF_mod_list
elif (i.Zism == 0.15 or i.Zism == 0.14):
    logT_list = logT_SD_Z002
    logLambda_list = logLambda_SD_Z002
    logT_mod_list = logT_list           # check this!
    logLambda_mod_list = logLambda_list # check this!
else:
    print("The chosen metallicity has not been implemented. Cooling tables are missing.")

slope_exp_list = np.append((logLambda_list[1:] - logLambda_list[0:-1]) / (logT_list[1:] - logT_list[0:-1]), 0.5)
slope_exp_mod_list = np.append((logLambda_mod_list[1:] - logLambda_mod_list[0:-1]) / (logT_mod_list[1:] - logT_mod_list[0:-1]), 0.5)


def create_coolCIE(Zism):
    """
    creates interpolation function for CIE cooling curve
    :param Zism: metallicity in solar units
    :return:
    """
    if not ((Zism == 1.0) or (Zism == 0.15) or (Zism == 0.14)):
        sys.exit("cooling curve not implemented for chosen metallicity (cool.py)")
    f_logLambdaCIE = interp1d(logT_mod_list, logLambda_mod_list, kind = 'linear')
    return f_logLambdaCIE

####################################### only created in header of bubble_structure2.py (maybe only need to create it once?)
#f_logLambdaCIE = create_coolCIE(i.Zism)
#######################################

def coolfunc(T):
    """
    calculates cooling function, piecewise power law approximation 
    (see Joung & Mac Low 2006 Figure 1, 
         uses Dalgarmo & McCray 1972 for T < 2e4K and 
         Sutherland & Dopita 1993 for T > 2e4K)
    :param T: temperature in K
    :return: Lambda in erg cm**3 / s
    """

    logT = np.log10(T)

    # old version
    if logT < logT_list[0]:
        if T < 0.0:
            sys.exit("Negative temperature in cooling function!")
        #else:
        #    print("log10(T)=%.4f, log10(Tmin)=%.4f" %(logT, logT_list[0]))
        #    sys.exit("Temperature too low for cooling function!")

    idx = aux.find_nearest_lower(logT_list,logT)
    logLambda = logLambda_list[idx] + slope_exp_list[idx]*(logT - logT_list[idx])



    Lambda = 10.**logLambda

    # for debugging:
    #Lambda = 0.

    return Lambda

def coolfunc_kb(T, my_kboltz = c.kboltz):
    """
    return cooling function over boltzmann constant squared: Lambda/kB^2
    :param T: temperature in K
    :return: Lambda/kB^2 (Lambda is in erg cm**3 / s)
    """
    return coolfunc(T)/my_kboltz**2.

def coolfunc_arr(T):
    """
    same as coolfunc but for list of temperatures
    :param T: list of temperatures in K
    :return: list of Lambdas in erg cm**3 / s
    """

    #startdatetime = datetime.datetime.now()

    logT = np.log10(T)
    # find nearest neighbors in logT_list (see top of this module)
    idx = aux.find_NN(logT, logT_list) # closest temperature indeces but not the closest one for which T_list[idx] is smaller than T

    # OLD VERSION: reshape lists for cKTree
    #logT_rsh = np.reshape(logT,(len(logT),1))
    #logT_list_rsh = np.reshape(logT_list,(len(logT_list),1))
    #tree = spatial.cKDTree(zip(logT_list_rsh.ravel())) # find closest temperature neighbors
    #idx = tree.query(logT_rsh)[1] # closest temperature index but not the closest one for which T_list[idx] is smaller than T

    idx[logT<logT_list[idx]] += -1 # these are the closest indeces for which T_list[idx] is smaller than T
    logLambda = logLambda_list[idx] + slope_exp_list[idx]*(logT - logT_list[idx])
    Lambda = 10.**logLambda

    #enddatetime = datetime.datetime.now()
    #print enddatetime - startdatetime

    return Lambda

""" OLD
def coolfunc_arr(T):
    # T is an array
    Lambda = np.zeros(len(T))
    for ii in range(0,len(T)):
        Lambda[ii] = coolfunc(T[ii])
    return Lambda
"""

