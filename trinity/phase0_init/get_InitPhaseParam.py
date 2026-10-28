#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:44:00 2022

@author: Jia Wei Teh

This script contains a function that computes the initial values for the
energy-driven phase (from a short free-streaming phase).
"""

import logging
import numpy as np
import scipy.integrate
import scipy.optimize

import trinity._functions.unit_conversions as cvt


# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS (with literature references)
# =============================================================================

# Energy fraction in bubble interior: E0 = (5/11) * Lw * dt
# From Weaver+77, Eq. 20 - assumes adiabatic index gamma = 5/3
WEAVER_ENERGY_FRACTION = 5.0 / 11.0

# Temperature coefficient in Weaver+77, Eq. 37
# T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4
WEAVER_TEMP_COEFFICIENT = 1.51e6  # Kelvin

# Reference luminosity for temperature scaling [erg/s]
WEAVER_L_REF = 1e36

# Minimum valid values to prevent division by zero
MIN_LUMINOSITY = 1e-100  # Prevent div by zero in Mdot calculation
MIN_MOMENTUM = 1e-100    # Prevent div by zero in velocity calculation
MIN_VELOCITY = 1e-100    # Prevent div by zero in dt_phase0 calculation

# Top of the non-CIE cooling cube's photon-flux axis (lib/default/opiate, log10 phi = 0 .. 21, 22 points; a test
# pins this to the shipped cube). The first phase-1a structure solve looks the cube up at phi = Qi/(4 pi r^2) for
# r <= R2 (bubble_luminosity._get_bubble_ODE), so a Weaver start with phi(r0) above the axis cannot run phase 1a
# (docs/dev/phii-identity/weak-wind/PLAN.md, L1, D6): it is refused in get_y0 rather than run for 0 segments.
NONCIE_PHI_MAX_CGS = 1e21   # cm^-2 s^-1

# The Stroemgren start (start_scheme 'stromgren'; docs/dev/phii-identity/weak-wind/PLAN.md section 4). zeta is the wind
# bubble's equilibrium radius against the ionised gas, R_eq = sqrt(pdot_w / (4 pi P_i)), over the Stroemgren radius
# (Lancaster et al. 2025a, arXiv:2505.22730, eq. 25). At zeta >= ZETA_REFUSE the bubble would outgrow the H II region
# and the start does not apply; from ZETA_WARN up the early wind momentum it leaves out is of order zeta^3/2 of the
# shell's (their eq. 43) and the run is warned (D5). Provisional, to be replaced by the zeta ladders' measurement
# (G-WW3b, D14); no automatic choice between the two starts until then (D17).
ZETA_REFUSE = 1.0
ZETA_WARN = 0.3



def get_y0(params):
    """
    Obtain initial values for the energy-driven (Weaver) phase by integrating
    a brief free-streaming phase from the SPS wind feedback at tSF.

    Physics references
    ------------------
    - Free-streaming phase duration: Rahner thesis Eq. 1.15
      https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf pg 17
    - Bubble energy: Weaver+77, Eq. 20
    - Bubble temperature: Weaver+77, Eq. 37

    Parameters
    ----------
    params : DescribedDict
        Must contain: tSF, sps_f, nCore, mu_convert, bubble_xi_Tb.

    Returns
    -------
    t0 : float [Myr]
        Start time for Weaver phase (= tSF + free-streaming duration).
    r0 : float [pc]
        Initial bubble outer radius R2 (= terminal velocity * free-streaming duration).
    v0 : float [pc/Myr]
        Initial expansion velocity (wind terminal velocity).
    E0 : float [au]
        Initial bubble thermal energy.
    T0 : float [K]
        Initial characteristic bubble temperature.
    """

    # Core properties - handle both DescribedItem and raw value access
    mu_convert = params['mu_convert'].value  # mass per H nucleus — for rho = n_H * mu_convert
    nCore = params['nCore'].value
    bubble_xi_Tb = params['bubble_xi_Tb'].value
        
    # =========================================================================
    # EXTRACT PARAMETERS
    # =========================================================================

    # Time of star formation [Myr]
    tSF = params['tSF'].value

    # SPS interpolation functions.
    sps_f = params['sps_f'].value

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    if tSF < 0:
        raise ValueError(f"tSF must be non-negative, got {tSF}")

    if nCore <= 0:
        raise ValueError(f"nCore must be positive, got {nCore}")

    if not (0 <= bubble_xi_Tb <= 1):
        raise ValueError(f"bubble_xi_Tb must be in [0,1], got {bubble_xi_Tb}")

    # =========================================================================
    # GET SPS FEEDBACK VALUES AT tSF
    # =========================================================================

    # CRITICAL: Use WIND-ONLY quantities for wind velocity calculation.
    # The free-streaming phase describes the stellar wind expansion before
    # the Weaver phase. The wind terminal velocity v = 2L/pdot is only
    # physical when using wind-only L and pdot (not total which includes SNe).
    Lmech_W = sps_f['fLmech_W'](tSF)
    pdot_W = sps_f['fpdot_W'](tSF)

    # Validate SPS values
    if Lmech_W < MIN_LUMINOSITY:
        logger.warning(f"Lmech_W={Lmech_W * cvt.L_au2cgs:.3e} erg/s is very small at tSF={tSF} Myr")
        Lmech_W = MIN_LUMINOSITY

    if pdot_W < MIN_MOMENTUM:
        logger.warning(f"pdot_W={pdot_W} is very small at tSF={tSF} Myr")
        pdot_W = MIN_MOMENTUM

    # =========================================================================
    # COMPUTE WIND PROPERTIES (WIND-ONLY - BUG FIX)
    # =========================================================================

    # Mass loss rate from winds [AU units]
    # From: L = 0.5 * Mdot * v^2 and pdot = Mdot * v
    # => Mdot = pdot^2 / (2 * L)
    Mdot0 = pdot_W**2 / (2.0 * Lmech_W)

    # Terminal velocity from winds [pc/Myr in AU units]
    # From: v = 2 * L / pdot  (wind-only quantities)
    v0 = 2.0 * Lmech_W / pdot_W

    if v0 < MIN_VELOCITY:
        logger.warning(f"v0={v0} is very small, may cause numerical issues")
        v0 = MIN_VELOCITY

    # =========================================================================
    # COMPUTE FREE-STREAMING PHASE DURATION
    # =========================================================================

    # Ambient density [AU units: Msun/pc^3]
    # nCore is hydrogen nuclei density n_H; use mu_convert (=1.4) for mass density
    rhoa = nCore * mu_convert

    # Duration of free-streaming phase [Myr]
    # From Rahner thesis Eq. 1.15:
    # dt = sqrt(3 * Mdot / (4 * pi * rho_a * v^3))
    dt_phase0 = np.sqrt(3.0 * Mdot0 / (4.0 * np.pi * rhoa * v0**3))

    logger.debug(f"Free-streaming phase duration: dt_phase0 = {dt_phase0:.6e} Myr")

    # =========================================================================
    # COMPUTE INITIAL VALUES FOR WEAVER PHASE
    # =========================================================================

    # Start time for Weaver phase [Myr]
    t0 = tSF + dt_phase0

    # Initial separation / bubble radius [pc]
    r0 = v0 * dt_phase0

    # Initial bubble energy [AU units]
    # From Weaver+77, Eq. 20: E = (5/11) * L_w * t
    E0 = WEAVER_ENERGY_FRACTION * Lmech_W * dt_phase0

    # Initial temperature [K]
    # From Weaver+77, Eq. 37:
    # T = 1.51e6 * (L/10^36)^(8/35) * (n)^(2/35) * t^(-6/35) * (1-xi)^0.4
    T0 = WEAVER_TEMP_COEFFICIENT * \
         (Lmech_W * cvt.L_au2cgs / WEAVER_L_REF)**(8.0/35.0) * \
         (nCore * cvt.ndens_au2cgs)**(2.0/35.0) * \
         (dt_phase0)**(-6.0/35.0) * \
         (1.0 - bubble_xi_Tb)**0.4
         

    logger.info(
        f"Initial Weaver phase values: "
        f"t0={t0:.6f} Myr, r0={r0:.6e} pc, v0={v0:.6e} pc/Myr, "
        f"E0={E0:.6e}, T0={T0:.2e} K"
    )

    # One-time feedback summary at SF onset, in conventional (paper) units:
    #   log Q [1/s], L [erg/s], Mdot [Msun/yr], E0 [erg].
    Qi_tSF = float(sps_f['fQi'](tSF))      # internal [1/Myr]
    Lbol_tSF = float(sps_f['fLbol'](tSF))  # internal [Msun*pc^2/Myr^3]
    logger.info(
        "Feedback at SF onset (tSF=%.4f Myr): log Q=%.3f [1/s], "
        "Lbol=%.3e erg/s, Lmech_wind=%.3e erg/s, "
        "Mdot_wind=%.3e Msun/yr, E0=%.3e erg"
        % (tSF, np.log10(Qi_tSF * cvt.s2Myr),
           Lbol_tSF * cvt.L_au2cgs, Lmech_W * cvt.L_au2cgs,
           Mdot0 * cvt.Mdot_au2Msunyr, E0 * cvt.E_au2cgs)
    )

    # The start-up limit (docs/dev/phii-identity/weak-wind/PLAN.md, D6): refuse a Weaver start whose first structure
    # solve would fall off the cooling cube, instead of running phase 1a for 0 segments and continuing on a dead state.
    phi_r0_cgs = _phi_cgs(r0, Qi_tSF)
    if phi_r0_cgs > NONCIE_PHI_MAX_CGS:
        raise ValueError(
            f"Weaver start below the start-up limit: phi(r0) = Qi/(4 pi r0^2) = {phi_r0_cgs:.3e} cm^-2 s^-1 exceeds the "
            f"non-CIE cooling cube's top {NONCIE_PHI_MAX_CGS:.0e} (r0 = {r0:.3e} pc; phase 1a needs "
            f"3 pdot_w 1e21 >= Qi rho_core v_inf^2). This is a weak wind in a dense core: use start_scheme stromgren "
            f"under phii_scheme swept, or a lower nCore."
        )

    return t0, r0, v0, E0, T0


def _phi_cgs(r_pc, Qi_au):
    """Ionising photon flux Qi / (4 pi r^2) in cm^-2 s^-1, from Qi in 1/Myr and r in pc."""
    return (Qi_au * cvt.s2Myr) / (4.0 * np.pi * (r_pc * cvt.pc2cm) ** 2)


def stromgren_radius(params, Qi):
    """The Stroemgren radius on the run's own density profile (docs/dev/phii-identity/weak-wind/PLAN.md, D15), pc.

    The root of 4 pi chi_e alpha_B int_0^R n(r)^2 r^2 dr = Qi (case B, on the spot; the gas at rest). Equal to the
    uniform-core value [3 Qi / (4 pi chi_e alpha_B nCore^2)]^(1/3) while the sphere lies inside a flat core, and larger
    where the density falls off (n <= nCore everywhere, so the uniform value brackets the root from below). np.inf if
    the photons are not used up within 1e3 rCloud.
    """
    from trinity.cloud_properties import density_profile
    g = lambda k: params[k].value  # noqa: E731
    chi_aB = g('chi_e_shell') * g('caseB_alpha')
    R_uni = (3.0 * Qi / (4.0 * np.pi * chi_aB * g('nCore') ** 2)) ** (1.0 / 3.0)
    n = lambda r: float(np.atleast_1d(density_profile.get_density_profile(np.array([r]), params))[0])  # noqa: E731

    def deficit(R):
        pts = [x for x in (g('rCore'), g('rCloud')) if 0.0 < x < R]
        return 4.0 * np.pi * chi_aB * scipy.integrate.quad(lambda r: n(r) ** 2 * r ** 2, 0.0, R, points=pts or None,
                                                           limit=200)[0] - Qi

    d_uni = deficit(R_uni)
    if abs(d_uni) <= 1e-9 * Qi:            # inside a flat core: the closed form, not a root-find's rounding
        return R_uni
    hi = 2.0 * R_uni
    while deficit(hi) < 0.0:
        if hi > 1e3 * g('rCloud'):
            return np.inf
        hi *= 2.0
    return scipy.optimize.brentq(deficit, R_uni, hi, rtol=1e-10)


def stromgren_zeta(params):
    """(zeta_MD [-], zeta_ED [-], R_St [pc], R_eq [pc], c_i [pc/Myr]) at tSF (docs/dev/phii-identity/weak-wind/PLAN.md 4.2).

    R_St from stromgren_radius (the run's profile, D15) and P_i = (mu_c/mu_i) k_B T_ion nCore, as swept.root_R2 and the
    benchmarks (harness/option3_bench.Synth) define them. zeta_MD: the momentum-driven bubble, R_eq from
    pdot_w = 4 pi R_eq^2 P_i. zeta_ED: the energy-conserving Weaver bubble, E = (5/11) L_w t and
    R = (250/308 pi)^(1/5) (L_w t^3 / rho)^(1/5), at the time its pressure (2/3) E / V equals P_i (identical to Lancaster
    et al. 2025a eq. 23 at theta = 0).
    """
    g = lambda k: params[k].value  # noqa: E731
    tSF, sps_f = g('tSF'), g('sps_f')
    Qi = float(sps_f['fQi'](tSF))
    pdot_W = max(float(sps_f['fpdot_W'](tSF)), MIN_MOMENTUM)
    Lmech_W = max(float(sps_f['fLmech_W'](tSF)), MIN_LUMINOSITY)
    nCore = g('nCore')
    p_ion = g('mu_convert') / g('mu_ion_shell') * g('k_B') * g('TShell_ion')     # P_i per unit n_H
    c_i = np.sqrt(p_ion / g('mu_convert'))                                     # pc/Myr
    R_St = stromgren_radius(params, Qi)
    P_i = p_ion * nCore
    R_eq = np.sqrt(pdot_W / (4.0 * np.pi * P_i))
    rho = g('mu_convert') * nCore
    A = (250.0 / (308.0 * np.pi)) ** 0.2
    # P_b(t) = (5 / 22 pi) L_w t / R^3 = [5 / (22 pi A^3)] L_w^(2/5) rho^(3/5) t^(-4/5) = P_i  ->  t_eq
    t_eq = ((5.0 / (22.0 * np.pi * A ** 3)) * Lmech_W ** 0.4 * rho ** 0.6 / P_i) ** 1.25
    R_eq_ED = A * (Lmech_W * t_eq ** 3 / rho) ** 0.2
    return R_eq / R_St, R_eq_ED / R_St, R_St, R_eq, c_i


def resolve_start_scheme(params):
    """'weaver' or 'stromgren', as the .param sets it; zeta and R_St are logged for every start (D5).

    No automatic choice between the two starts yet (D17): the switch value is set after the zeta ladders.
    """
    scheme = str(params['start_scheme'].value).strip().lower()
    try:
        zeta_MD, zeta_ED, R_St, R_eq, _ = stromgren_zeta(params)
        logger.info(f"start_scheme {scheme}: zeta_MD = R_eq/R_St = {zeta_MD:.4g} (zeta_ED = {zeta_ED:.4g}; "
                    f"R_eq = {R_eq:.4e} pc, R_St = {R_St:.4e} pc on the run's density profile)")
    except Exception as e:  # noqa: BLE001 -- a diagnostic only; the Weaver start must not depend on it
        logger.warning(f"start_scheme {scheme}: zeta not computed ({type(e).__name__}: {e})")
    return scheme


def rtype_duration(R_St, c_i, params):
    """The R-type phase before the D-type start (docs/dev/phii-identity/weak-wind/PLAN.md 4.1a).

    With the gas at rest the front follows R_St (1 - e^(-t/t_rec))^(1/3), t_rec = 1/(alpha_B n_e) (Spitzer 1978;
    Pittard, Kupilas & Wareing 2022 eq. 1, with n_e = chi_e n_H as the code's R_St has it), and the D-type phase begins
    when its speed falls to 2 c_i (Williams et al. 2018 sec. 1): x / (1 - x)^(2/3) = beta with x = e^(-t/t_rec) and
    beta = 6 c_i t_rec / R_St (one root for any beta > 0). The uniform-core formation law: exact inside a flat core.
    Returns (t_R [Myr], R_R [pc], t_rec [Myr], beta [-]): the duration, the front's radius then, the recombination
    time, beta.
    """
    g = lambda k: params[k].value  # noqa: E731
    t_rec = 1.0 / (g('caseB_alpha') * g('chi_e_shell') * g('nCore'))
    beta = 6.0 * c_i * t_rec / R_St
    x = scipy.optimize.brentq(lambda x: x / (1.0 - x) ** (2.0 / 3.0) - beta, 1e-300, 1.0 - 1e-12)
    t_R = -t_rec * np.log(x)
    R_R = R_St * (1.0 - x) ** (1.0 / 3.0)
    return t_R, R_R, t_rec, beta


def get_y0_stromgren(params):
    """The Stroemgren (D-type) start for phii_scheme 'swept' (docs/dev/phii-identity/weak-wind/PLAN.md section 4).

    Phases 1a-1c are skipped: the swept momentum runner starts at the end of the R-type phase from Hosokawa &
    Inutsuka 2006's initial state (their eq. 36 at t = 0; Williams et al. 2018 eq. 16): the outer edge at the
    Stroemgren radius, moving at 2 c_i / sqrt(3), carrying all the gas inside, with the wind bubble inside it at
    its ram-pressure equilibrium radius (Lancaster et al. 2025a section 3.5; the runner root-finds R2 from this
    guess). Refused for phii_scheme other than 'swept', at zeta_MD >= ZETA_REFUSE, and where R_St > rCloud (D15).
    Returns (t0 [Myr], rs [pc], u0 [pc/Myr], Ms [Msun], R2_guess [pc]).
    """
    import trinity.cloud_properties.mass_profile as mass_profile
    if str(params['phii_scheme'].value).strip().lower() != 'swept':
        raise ValueError("start_scheme 'stromgren' needs phii_scheme 'swept' "
                         "(the c3c closure has no thick-layer state to start from)")
    zeta_MD, zeta_ED, R_St, R_eq, c_i = stromgren_zeta(params)
    rCloud = params['rCloud'].value
    if not R_St <= rCloud:
        raise ValueError(
            f"start_scheme 'stromgren' does not apply: the Stroemgren radius on the cloud's profile, {R_St:.3e} pc, "
            f"exceeds rCloud = {rCloud:.3e} pc (the ionising photons leave the cloud). Use start_scheme weaver."
        )
    if zeta_MD >= ZETA_REFUSE:
        raise ValueError(
            f"start_scheme 'stromgren' does not apply: zeta = R_eq/R_St = {zeta_MD:.3g} >= {ZETA_REFUSE} "
            f"(R_eq = {R_eq:.3e} pc, R_St = {R_St:.3e} pc): the wind bubble would outgrow the H II region. "
            f"Use start_scheme weaver."
        )
    if zeta_MD >= ZETA_WARN:
        logger.warning(f"start_scheme 'stromgren' at zeta_MD = {zeta_MD:.3g} (zeta_ED = {zeta_ED:.3g}): the wind's "
                       f"early momentum, of order zeta^3/2 = {zeta_MD ** 3 / 2:.2g} of the shell's, is not carried")
    t_R, R_R, t_rec, beta = rtype_duration(R_St, c_i, params)
    t0 = params['tSF'].value + t_R
    rs = R_St
    u0 = 2.0 * c_i / np.sqrt(3.0)
    Ms = float(np.atleast_1d(mass_profile.get_mass_profile(rs, params))[0])
    logger.info(
        f"R-type phase (start_scheme stromgren): t_rec={t_rec * 1e6:.4g} yr, beta=6 c_i t_rec/R_St={beta:.4g}, "
        f"t_R={t_R * 1e6:.4g} yr (the front's speed falls to 2 c_i), R_R={R_R:.6e} pc = {R_R / R_St:.4f} R_St"
    )
    logger.info(
        f"Initial Stroemgren-start values (start_scheme stromgren): t0=tSF+t_R={t0:.6e} Myr, r_s=R_St={rs:.6e} pc, "
        f"u0=2c_i/sqrt(3)={u0:.6e} pc/Myr ({u0 * cvt.v_au2kms:.3f} km/s), M_s={Ms:.6e} Msun, "
        f"R2 guess=R_eq={R_eq:.6e} pc, zeta_MD={zeta_MD:.4g}, zeta_ED={zeta_ED:.4g}"
    )
    return t0, rs, u0, Ms, R_eq

