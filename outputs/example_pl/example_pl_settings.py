# =============================================================================
# Summary of parameters in the 'example_pl' run.
# Created at 01/04/2025 15:43:24.
# =============================================================================

#-- import library for units
import astropy.units as u


model_name="example_pl"
out_dir="/Users/jwt/unsync/Code/Trinity/outputs/example_pl/"
verbose = 2.0
output_format="ASCII"
rand_input = 0.0
log_mCloud = 7.0 * u.M_sun
is_mCloud_beforeSF = 1.0
sfe = 0.01
nCore = 10000.0 / u.cm**3
rCore = 0.099 * u.pc
metallicity = 1.0
stochastic_sampling = 0.0
mult_exp = 0.0
r_coll = 1.0 * u.pc
mult_SF = 1.0
sfe_tff = 0.01
imf="kroupa.imf"
stellar_tracks="geneva"
SB99_mass = 1000000.0 * u.M_sun
SB99_rotation = 1.0
SB99_BHCUT = 120.0 * u.M_sun
SB99_forcefile = 0.0
SB99_age_min = 500000.0 * u.yr
dens_profile="pL_prof"
dens_a_pL = 0.0
dens_navg_pL = 10000.0 / u.cm**3
frag_enabled = 0.0
frag_cf_end = 0.1
stop_n_diss = 1.0 / u.cm**3  
stop_t_diss = 2.0 * u.Myr
stop_r = 5000.0 * u.pc
stop_v = -10000.0 * u.km/u.s
stop_t = 15.05 * u.Myr
stop_t_unit="Myr"
adiabaticOnlyInCore = 0.0
immediate_leak = 1.0
phase_Emin = 0.0001 * u.erg
write_main = 1.0
write_stellar_prop = 0.0
write_bubble = 0.0
write_bubble_CLOUDY = 0.0
write_shell = 0.0
write_figures = 0.0
write_potential = 0.0
cooling_alpha = 0.6
cooling_beta = 0.8
cooling_delta = -0.17142857142857143
path_cooling_nonCIE="/Users/jwt/unsync/Code/Trinity/lib/cooling/opiate/"
path_cooling_CIE="/Users/jwt/unsync/Code/Trinity/lib/cooling/CIE/coolingCIE_3_Gnat-Ferland2012.dat"
path_sps="/Users/jwt/unsync/Code/Trinity/lib/sps/starburst99/"
xi_Tb = 0.9
inc_grav = 1.0
f_Mcold_wind = 0.0
f_Mcold_SN = 0.0
v_SN = 10000.0 * u.km/u.s
sigma0 = 1.5e-21 * u.cm**2
z_nodust = 0.05
mu_n = 2.1287915392418182e-24 * u.g
mu_p = 1.0181176926808696e-24 * u.g
t_ion = 10000.0 * u.K
t_neu = 100.0 * u.K
nISM = 0.1 / u.cm**3
kappa_IR = 4.0 * u.cm**2 / u.g
gamma_adia = 1.6666666666666667
thermcoeff_wind = 1.0
thermcoeff_SN = 1.0
alpha_B = 2.59e-13 * u.cm**3 / u.s
gamma_mag = 1.3333333333333333
log_BMW = -4.3125
log_nMW = 2.065
c_therm = 6e-07 * u.erg / u.cm / u.s * u.K**(-7/2)
dMdt_factor = 1.646
T_r2Prime = 30000.0 * u.K
sigma_d = 1.5e-21 * u.cm**2
tff = 0.45623350704921234 * u.Myr
mCloud = 9900000.0 * u.M_sun
mCloud_beforeSF = 10000000.0
mCluster = 100000.0 * u.M_sun
BMW = 4.869675251658631e-05
nMW = 116.14486138403426
density_gradient = 0.0
