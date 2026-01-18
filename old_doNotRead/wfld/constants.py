import numpy as np

# constants in cgs
Lsun = 4e33  # solar luminosity in erg/s
Msun = 1.989e33 # solar mass
pc = 3.085677581491367e18 # parsec
mp = 1.67e-24 # proton mass
alphaB = 2.59e-13  # case B recombination coefficient
kboltz = 1.380658e-16 # erg/K
hplanck = 6.6260755e-27 # Planck constant in erg s

mH = 1.673e-24 # hydrogen mass
mHe = 4.003*1.661e-24 # helium mass
me = 9.11e-28 #electron mass

gamma = 5. / 3. # adiabatic index (5/3 for ideal gas)

clight = 3e10 # speed of light
Myr  = 3.15e13 # 1 Megayear
yr = 3.15e7 # 1 year
kms = 1e5
E_cgs = Msun * (pc/Myr) ** 2
L_cgs =  E_cgs/Myr
Lambda_cgs = (Msun*pc**5)/Myr**3
dudt_cgs = Msun/(pc*Myr**3) # ne*nb*(Gamma-Lambda)
press_cgs = Msun/(Myr**2 * pc)

Grav = 6.67e-8

LTM = 1500. # light-to-mass ratio in erg/s/g for a Kroupa IMF up to 120 Msol at zero-age main sequence (only fully sampled clusters)
#########################
#constants in SI (used for BE-profile)
MsunSI = 1.989e30 
RgasSI=8.31446261815324 #universal gas constant in SI
GravSI=6.67e-11  # gravitational constant in SI
pcSI=3.085677581e16 # pc in m
auSI= (pc**3. / Msun)**(-1)
Mdotconvert= (1/pcSI)**2 * 10**(-3)*auSI # from kg/m in g/(cm^3)_(in au) * pc^2



############################
# constants in Msun, Myr, pc
# au stands for "astro-units"

Grav_au=Grav*Msun/(pc*(pc/Myr)**2)
clight_au = clight/(pc/Myr)
kboltz_au = kboltz / (Msun * (pc/Myr)**2)

############################
# magnetic field

gmag = 4./3. # for a sphere, see Henney 2005
BMW0 = 10.**(-4.3125) # source: Eric (see also Han, Manchester, Line, Qiao, 2002, ApJL 570)
nMW0 = 10.**2.065 # source: Eric


#Cspitzer = 6e-7 # cgs (this is the value used in MacLow 1988, in agreement with Spitzer 1956)
Cspitzer = 1.2e-6 # this is the value used in Weaver 1977 and Harper-Clark & Murray 2009 (presumably from Spitzer 1962, table 5.1)
Cspitzer_au = Cspitzer * (Myr/Msun)*(Myr/pc)*Myr

# helium abundance
He_ion = 2. # degree of He ionization
aH = 10.0
aHe = 1.0 # number of He atoms per aH hydrogen atoms
ae = aH+aHe*He_ion
ap_sum = aH + aHe + ae
ai_sum = aH + aHe
fH_p = aH/ap_sum # fraction of H atom cores per particle in fully ionized gas
fHe_p = aHe/ap_sum # fraction of He atom cores per particle in fully ionized gas
fe_p = ae/ap_sum # fraction of electrons per particle in fully ionized gas
fH_i = aH/ai_sum # fraction of H atom cores per ion
fHe_i = aHe/ai_sum # fraction of He atom cores per ion
