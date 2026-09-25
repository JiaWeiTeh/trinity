#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:36:10 2022

@author: Jia Wei Teh

This script contains useful functions that help compute properties and parameters
of the bubble. grep "Section" so jump between different sections.
"""
# libraries
import numpy as np
import scipy.optimize
import logging
import astropy.units as u
import trinity._functions.unit_conversions as cvt

logger = logging.getLogger(__name__)

#--

# =============================================================================
# This section contains function which computes the ODEs that dictate the 
# structure (e.g., temperature, velocity) of the bubble. 
# =============================================================================

def delta2dTdt(t, T, delta):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        Temperature at xi = r/R2.

    Returns
    -------
    dTdt : float
    """
    dTdt = (T/t) * delta

    return dTdt


def dTdt2delta(t, T, dTdt):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        DESCRIPTION.

    Returns
    -------
    delta : float
    """
    
    delta = (t/T) * dTdt
    
    return delta



def cool_beta_to_Ebdot(params):
    # old code: beta_to_Edot(), previously beta2Edot()
    """
    Convert Weaver cooling parameter beta to dE_b/dt.

    See pg 80, Eq A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf

    Equation implemented (bubble energy rate):

        E_b_dot = [ 2*pi * Pb_dot * d^2
                  + 3 * E_b * R_b_dot * R_b^2 * (1 - c/(E_b+c))
                  - a * R_ts^3 * E_b^2 / (E_b + c) ]
                 / [ d * (1 - c/(E_b+c)) ]

        a ≡ (3/2) * F_ram_dot / F_ram           [1/time]
        c ≡ (3/4) * F_ram     * R_ts            [energy]
        d ≡ R_b^3 - R_ts^3                      [length^3]

    Code ↔ equation mapping
    -----------------------
    Pb_dot        <- d(P_b)/dt (from beta definition: beta = -(t/Pb)(dPb/dt))
    Eb            <- E_b (bubble energy)
    R2, v2        <- R_b (outer bubble radius) and R_b_dot
    R1            <- R_ts (termination shock radius, inner)
    pdot_total    <- F_ram (total mechanical momentum injection rate)
    pdotdot_total <- F_ram_dot
    a_coeff       <- equation symbol `a`  = (3/2) * pdotdot_total / pdot_total
    c_coeff       <- equation symbol `c`  = (3/4) * pdot_total * R1
    d_coeff       <- equation symbol `d`  = R2^3 - R1^3
    c_frac        <- c/(E_b + c)

    Parameters
    ----------
    params : dict-like
        Must provide .value for: Pb, cool_beta, t_now, R1, R2, v2, Eb,
        pdot_total, pdotdot_total.

    Returns
    -------
    Eb_dot : float
        d(E_b)/dt.
    """
    # dPb/dt from the Weaver cooling parameter: beta = -(t/Pb)(dPb/dt)
    Pb_dot = -params['Pb'].value * params['cool_beta'].value / params['t_now'].value

    # Pull state
    R1 = params['R1'].value                        # R_ts
    R2 = params['R2'].value                        # R_b
    v2 = params['v2'].value                        # R_b_dot
    Eb = params['Eb'].value
    pdot_total = params['pdot_total'].value        # F_ram
    pdotdot_total = params['pdotdot_total'].value  # F_ram_dot

    # Equation coefficients (see docstring)
    a_coeff = 1.5 * pdotdot_total / pdot_total
    c_coeff = 0.75 * pdot_total * R1
    d_coeff = R2**3 - R1**3
    c_frac = c_coeff / (Eb + c_coeff)              # c/(E_b + c)

    # Main equation (Rahner thesis A12)
    numerator = (
        2 * np.pi * Pb_dot * d_coeff**2
        + 3 * Eb * v2 * R2**2 * (1 - c_frac)
        - a_coeff * R1**3 * Eb**2 / (Eb + c_coeff)
    )
    denominator = d_coeff * (1 - c_frac)

    Eb_dot = numerator / denominator
    return Eb_dot


def Ebdot_to_cool_beta(bubble_P, r1, bubble_Edot, my_params):
    # old code: Edot_to_beta(), previously Edot2beta()
    """
    Inverse of cool_beta_to_Ebdot: convert dE_b/dt to Weaver cooling parameter beta.

    See pg 80, Eq A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf

    Solves the A12 equation for Pb_dot and then returns
        cool_beta = - Pb_dot * t_now / P_b.

    See cool_beta_to_Ebdot for the equation↔code variable map.

    Parameters
    ----------
    bubble_P : float
        Bubble pressure P_b.
    r1 : float
        Termination shock radius R_ts (inner).
    bubble_Edot : float
        d(E_b)/dt.
    my_params : dict-like
        Must provide t_now, pdot_total, pdotdot_total, R2, v2, Eb
        (plain float values, not .value-wrapped).

    Returns
    -------
    cool_beta : float
        Weaver cooling parameter beta = -(t/P_b) * dP_b/dt.
    """
    t_now = my_params["t_now"]
    pdot_total = my_params["pdot_total"]           # F_ram
    pdotdot_total = my_params["pdotdot_total"]     # F_ram_dot
    R2 = my_params["R2"]                           # R_b
    v2 = my_params["v2"]                           # R_b_dot
    Eb = my_params["Eb"]

    # Equation coefficients
    a_coeff = 1.5 * pdotdot_total / pdot_total
    c_coeff = 0.75 * pdot_total * r1
    d_coeff = R2**3 - r1**3
    c_frac = c_coeff / (Eb + c_coeff)

    # Invert A12 for Pb_dot
    Pb_dot = (
        d_coeff * (1 - c_frac) * bubble_Edot
        - 3 * Eb * v2 * R2**2 * (1 - c_frac)
        + a_coeff * r1**3 * Eb**2 / (Eb + c_coeff)
    ) / (2 * np.pi * d_coeff**2)

    cool_beta = -Pb_dot * t_now / bubble_P
    return cool_beta



# =============================================================================
# Section: conversion between bubble energy and pressure. Calculation of ram pressure.
# =============================================================================

def bubble_E2P(Eb, r2, r1, gamma):
    """
    Convert bubble thermal energy to bubble pressure.

    Parameters
    ----------
    Eb : float
        Bubble thermal energy [au].
    r2 : float
        Outer bubble radius (= inner shell edge) [pc].
    r1 : float
        Inner bubble radius (wind termination shock) [pc].
    gamma : float
        Adiabatic index.

    Returns
    -------
    bubble_P : float
        Bubble pressure [au].
    """
    
    # Make sure units are in cgs
    r1 *= cvt.pc2cm
    r2 *= cvt.pc2cm
    Eb *= cvt.E_au2cgs
    # avoid division by zero
    r2 += 1e-10 
    
    # pressure, see https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf
    # pg71 Eq 6.
    shell_volume = r2**3 - r1**3
    if shell_volume <= 0:
        # Catastrophic-cooling degeneracy: Eb collapses, the wind shock R1 -> R2,
        # so (r2**3 - r1**3) underflows to 0 in float64 and the divide blows up
        # (-> inf/ZeroDivisionError -> Eb=nan). Floor it so the divide stays finite;
        # the energy phases detect the collapse (Eb<=0) and hand off (phase 1b routes
        # to momentum; phase 1a stops -- see docs/dev/transition/pdv-trigger/
        # HIMASS_HANDOFF_PLAN.md). Bit-identical on every physical bubble (shell_volume > 0).
        shell_volume = 1e-13 * r2**3
    Pb = (gamma - 1) * Eb / shell_volume / (4 * np.pi / 3)
    # return back in au
    return Pb * cvt.Pb_cgs2au


def get_leak_luminosity(coverFraction, R2, Pb, c_sound, gamma):
    """
    Geometry-set covering-fraction energy leak (enthalpy flux through the
    open fraction of the bubble wall):

        Lleak = gamma/(gamma-1) * (1 - Cf) * 4*pi*R2**2 * Pb * c_sound

    Cf = coverFraction is the *closed* fraction of the wall; Cf = 1 is a
    sealed (Weaver) bubble and returns exactly 0. Hot gas escapes through the
    open area (1-Cf)*4*pi*R2**2 at the interior sound speed, carrying its
    enthalpy. See the leakage spec, Eq. (leak).

    All quantities are in code units [Msun, pc, Myr]; the product
    Pb*c_sound*R2**2 already lands in the luminosity unit Msun*pc**2/Myr**3,
    so no conversion is applied (asserted in test/test_cf_leak.py).

    Parameters
    ----------
    coverFraction : float
        Closed fraction of the bubble wall, Cf in (0, 1].
    R2 : float
        Outer bubble radius (contact discontinuity) [pc].
    Pb : float
        Bubble (interior) pressure [Msun/pc/Myr**2].
    c_sound : float
        Hot-bubble sound speed [pc/Myr], evaluated at bubble_Tavg (NOT the
        cold-shell value).
    gamma : float
        Adiabatic index.

    Returns
    -------
    float
        Leak luminosity [Msun*pc**2/Myr**3], >= 0. Returns 0 when Cf >= 1
        (sealed), Pb <= 0 (depressurised / numerical undershoot), or
        c_sound <= 0 (no hot-gas temperature yet), so the term self-limits
        and never injects energy.
    """
    # Cf = 1 must reproduce the sealed bubble exactly; the other guards keep
    # the leak from injecting energy when the bubble state is degenerate.
    if coverFraction >= 1.0 or Pb <= 0.0 or c_sound <= 0.0:
        return 0.0
    return gamma / (gamma - 1.0) * (1.0 - coverFraction) * 4.0 * np.pi * R2**2 * Pb * c_sound

def pRam(r, Lmech, v_mech):
    """
    Ram pressure from a freely streaming wind: P_ram = L_mech / (2 pi r^2 v_mech).

    In current usage this is called with the *total* mechanical luminosity
    (winds + SNe) and the corresponding total mechanical velocity, e.g. in
    the momentum and transition phases.

    Parameters
    ----------
    r : float
        Outer bubble radius R2 [pc].
    Lmech : float
        Mechanical luminosity (typically Lmech_total) [au].
    v_mech : float
        Mechanical terminal velocity (typically v_mech_total) [pc/Myr].

    Returns
    -------
    P_ram : float
        Ram pressure [au].
    """
    return Lmech / (2 * np.pi * r**2 * v_mech)


def mass_freeWind(r, Lmech, v_mech):
    """Mass of freely streaming wind in transit inside radius r [Msun].

    Companion to pRam and deliberately shares its convention: Lmech = 0.5 Mdot v**2,
    so Mdot = 2 Lmech / v**2, rho(r) = Mdot / (4 pi r**2 v), and the enclosed mass is

        M = integral_0^r rho 4 pi r'**2 dr' = Mdot r / v = 2 Lmech r / v**3

    This applies the INSTANTANEOUS Mdot over the whole flight time r/v, i.e. it assumes
    the wind has been steady for that long. At B3M scales the flight time is ~7e-3 Myr,
    so the approximation is benign; it would need revisiting if Mdot ever varied on that
    timescale.

    WHY THIS EXISTS. In the MOMENTUM phase the shocked-wind bubble has vanished --
    R1 == R2 and Eb == 0 exactly on every row -- so there is no bubble whose mass to
    carry, and the only mass enclosed within R2 is the free wind in flight. That is
    what shell_structure.py:268 wants when it adds params['bubble_mass'] to the shell's
    cumulative gravity mass. Before this, bubble_mass was a stale carry-over from the
    last implicit step (99.64 Msun on B3M, ~860x too large); see
    docs/dev/phii-identity/PLAN.md, 2026-08-29.
    """
    if not (r > 0 and Lmech > 0 and v_mech > 0):
        return 0.0
    return 2.0 * Lmech * r / v_mech**3


def get_phii_c3c(params, shell_props):
    """Photoionised pressure as a regime switch (the C3c scheme).

    The ionised gas either is, or is not, confined by the surrounding pressure, and
    those two cases are physically different:

        P_C3a = (mu_c/mu_i) * k_B * T * sqrt(3 Qi_abs / (4 pi chi_e alpha_B R2**3))

        P_C3a <= P_conf :  confinement holds it as a thin skin. The skin TRANSMITS the
                           confining pressure and contributes nothing of its own, so
                           this returns 0.0.
        P_C3a >  P_conf :  confinement cannot hold it. It fills its own volume and
                           drives at P_C3a.

    Returning exactly 0.0 on the confined branch is load-bearing: it is what makes
    every existing P_drive expression correct without editing any of them --

        energy/implicit   max(Pb_eff, 0)       = Pb_eff
        transition        max(Pb, 0 + P_ram)   = max(Pb, P_ram)
        momentum          0 + P_ram            = P_ram alone

    -- so a change that returns a small non-zero value there silently alters all four
    phases. test/test_phii_c3c.py pins this.

    This replaces computing P_HII from the CAPPED Stromgren density, which made it an
    exact algebraic relabelling of the confining pressure (the cap's shell_n0 is
    Pb/(k_B T) * mu), carrying no information about Qi or the ionised volume.

    P_conf is read as params['Pb'], which IS the wind ram pressure in the momentum
    phase (run_momentum_phase.py assigns it so) and the bubble pressure elsewhere.
    Note this is the un-ramped Pb. The choice cannot flip the branch at nominal wind,
    where P_C3a/Pb <= 0.04 in the energy and implicit phases, but it can at weak wind,
    where the ratio crosses 1 inside the energy phase (B3MW001, t = 1.2 kyr).

    KNOWN OPEN BEHAVIOUR: the momentum phase comes out photoionisation-dominated in
    every configuration measured so far, and the cause is the R2**-1.5 cavity geometry
    rather than an O(1) normalisation. The Lw**-0.33 scaling and its inversion near
    Lw ~ 260, quoted here until 2026-09-10, came from an offline screen on PRE-C3c
    trajectories and are retired: refitting that ladder gives -0.36, and Batch 10 --
    real C3c runs -- gives -0.40 (cavity form) and -0.11 (profile form).
    See docs/dev/phii-identity/PLAN.md.
    """
    R2 = params['R2'].value
    Qi = params['Qi'].value
    if not (R2 > 0 and Qi > 0):
        return 0.0
    # W69: the GAS-absorbed budget. n_c3a below is a recombination-balance density, so
    # the dust share of the absorbed LyC must not enter it. shell_fAbsorbedIon is kept
    # as the total for F_rad and the P_ext gate; this is the gas term alone.
    f_abs = getattr(shell_props, 'shell_fAbsorbedIonGas', None)
    if f_abs is None:
        f_abs = getattr(shell_props, 'shell_fAbsorbedIon', 1.0)
    if not (isinstance(f_abs, float) and 0.0 <= f_abs <= 1.0):
        f_abs = 1.0
    Qi_abs = Qi * f_abs
    denom = 4.0 * np.pi * params['chi_e_shell'].value * params['caseB_alpha'].value * R2**3
    if not (denom > 0.0 and Qi_abs > 0.0):
        return 0.0
    n_c3a = np.sqrt(3.0 * Qi_abs / denom)
    P_c3a = ((params['mu_convert'].value / params['mu_ion_shell'].value)
             * n_c3a * params['k_B'].value * params['TShell_ion'].value)
    return float(P_c3a) if P_c3a > params['Pb'].value else 0.0


def _k11_skin_density(R_IF, pdot_w, Q_eff, pref, chi_e, alpha_B):
    """Geen 2019's coupled skin density: the unique positive root of

        f(n) = n**2 R_IF**3 - C_W sqrt(n) - C_Q = 0
        C_W  = (pdot_w / (4 pi pref))**1.5      C_Q = 3 Q_eff / (4 pi chi_e alpha_B)

    obtained by eliminating the wind radius r_w between the paper's two conditions,
    recombination equilibrium over the cavity-excluded layer (wind:photoequilibrium) and
    wind/photoionised pressure balance at r_w (wind:windpressurebalance). The elimination
    is exact -- re-derived symbolically, not transcribed. Never use the appendix form:
    Geen 2019's appendix carries a (r_i - r_w)**3 typo for (r_i**3 - r_w**3).

    UNIQUENESS AND BRACKET, both proved rather than tuned. f(0) = -C_Q <= 0 and f has a
    single stationary point on n > 0, so there is exactly one positive root. With
    n_St = sqrt(C_Q/R_IF**3) (the photon-only root) and n_w = (C_W/R_IF**3)**(2/3) (the
    wind-only root):
        f(n_St) = -C_W sqrt(n_St) <= 0  and  f(n_w) = -C_Q <= 0   => max(n_St, n_w) <= n*
        f(n_St + n_w) >= 0, since with sqrt(a+b) <= sqrt(a) + sqrt(b) the residual reduces
        to n_w + 2 n_St >= sqrt(n_w) sqrt(n_St + n_w), whose squares differ by
        3 n_St n_w + 4 n_St**2 >= 0.
    So the bracket always contains the root; the pad keeps the sign change strict in the
    degenerate limits (one term exactly zero), where the bracket collapses onto the
    analytic answer.

    LIMITS, which are the point of the scheme:
        wind -> 0     n -> sqrt(3 Q_eff / (4 pi chi_e alpha_B R_IF**3)), the Stromgren
                      density over R_IF -- the photoionisation-only floor, delivered
                      additively with no branch and no max().
        photons -> 0  pref n -> pdot_w / (4 pi R_IF**2), i.e. r_w -> R_IF, a thin skin at
                      the front, and the drive at R2 becomes pdot_w/(4 pi R2**2).

    Returns 0.0 rather than raising if the root-find cannot be set up or converged.
    ponytail: silent zero on failure matches this module's other guards; the bracket is
    provable so the only reachable failures are float overflow at absurd inputs. If a run
    ever needs to distinguish "confined" from "solver gave up", this is the line to
    instrument -- the arm's own P_HII > 0 diagnostic is what currently covers it.
    """
    try:
        C_W = (pdot_w / (4.0 * np.pi * pref)) ** 1.5 if pdot_w > 0.0 else 0.0
        C_Q = 3.0 * Q_eff / (4.0 * np.pi * chi_e * alpha_B) if Q_eff > 0.0 else 0.0
        R3 = R_IF ** 3
        n_st = np.sqrt(C_Q / R3) if C_Q > 0.0 else 0.0
        n_w = (C_W / R3) ** (2.0 / 3.0) if C_W > 0.0 else 0.0
        if not (n_st > 0.0 or n_w > 0.0):
            return 0.0

        def f(n):
            return n * n * R3 - C_W * np.sqrt(n) - C_Q

        lo = max(n_st, n_w) * (1.0 - 1e-9)
        hi = (n_st + n_w) * (1.0 + 1e-9)
        return float(scipy.optimize.brentq(f, lo, hi, xtol=1e-300, rtol=8.9e-16,
                                           maxiter=300))
    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        return 0.0


def get_phii_k11(params, shell_props):
    """Photoionised pressure from Geen 2019's ADDITIVE closure on trinity's own geometry.

    ARM CODE (docs/dev/phii-identity/PLAN.md Batch 22, stage 2). NOT the shipped scheme;
    adopting it is a maintainer decision (D5, still open).

        n     = _k11_skin_density(R_IF, pdot_w, Qi, ...)      <- one scalar Brent
        drive = pref * n * (R_IF/R2)**2 ,   pref = (mu_c/mu_i) k_B T

    RELATION TO O1, which is the whole comparison. O1 uses the shell solver's own inner
    boundary density n0 = P_conf/pref and amplifies it by the same (R_IF/R2)**2, so

        drive_K11 / drive_O1  ==  n / n0        exactly, on every row.

    Both pin r_i = R_IF, so the area amplification is SHARED and the entire difference
    between the two candidates is the skin density. What K11 adds is that n solves the
    wind and photon conditions together, so the photoionisation-only limit exists instead
    of the drive following P_conf to zero as the wind weakens.

    R_IF is read verbatim from the shell solve (ShellProperties.R_IF,
    shell_structure.py:227 = rShell_arr_ion[-1]) -- the front trinity already computes
    with the real density profile, the real dust term and the real termination conditions.
    Nothing is re-derived, rescaled or clipped, and no second front solve is introduced.

    Qi is used WHOLE, not Qi*f_abs: the shell solve starts at phi = 1 and the
    recombination/dust/escape split is an OUTPUT of that solve, not an input to this one
    (the Batch 17 convention). The dusty variant is Q_eff = Qi*(1 - shell_fIonisedDust),
    a one-line change measured in stage 1 at roughly half the driving-branch excess.

    P_conf is the RAMPED effective pressure, not params['Pb'] -- inside dt_switchon the
    two differ by 0.330-0.995x (G16.3), and the un-ramped value would re-admit the D-ramp
    defect class C3c removed.

    RETURN CONVENTION (Batch 16, verified to 2.22e-16 through the real P_drive
    expressions): the closure's drive MINUS whatever the phase's own composition adds --

        energy/implicit   max(P_conf, P_HII)            <- return P_conf*rho
        transition        max(P_conf, P_HII + P_ram)    <- return P_conf*rho - P_ram
        momentum          P_HII + P_ram                 <- return P_ram*(rho - 1)

    with rho := drive/P_conf. In the momentum phase get_effective_bubble_pressure returns
    pRam by construction, so P_conf == P_ram there and the momentum form composes exactly.

    KNOWN, NOT FIXED BY THIS VARIANT (Batch 22 stage 1, measured):
      * G22.4 FAILS -- the implied ionised layer exceeds the shell's own mass on 6 of 17
        B3M momentum rows (max 1.91) where O1 is inside the shell on 140/140, and the
        closure's implied r_w sits at 0.65 R2 (B3M) / 0.49 R2 (B3MW01).
      * the per-segment freeze ratchet is UNTOUCHED: drive >= P_conf always (provably,
        since n >= n_w = n0 (R2/R_IF)**2), so this wins max(press_bubble, P_HII) on 100%
        of confined rows exactly as O1 does.
      * on trinity's driving rows the closure is photon-dominated by 10-92x, so the
        promised floor sits ABOVE O1 (2.25-4.20x), not underneath it.
    """
    R2 = params['R2'].value
    Qi = params['Qi'].value
    if not (R2 > 0 and Qi > 0):
        return 0.0

    R_IF = getattr(shell_props, 'R_IF', None)
    if not (isinstance(R_IF, float) and R_IF >= R2):
        return 0.0

    phase = params['current_phase'].value
    Lmech_total = params['Lmech_total'].value
    v_mech_total = params['v_mech_total'].value
    Eb = params['Eb'].value
    # R1 via solve_R1 rather than params['R1'], matching the ODE's own canonical call
    # (energy_phase_ODEs.py:221) so this cannot read a stale radius.
    R1 = solve_R1(R2, Eb, Lmech_total, v_mech_total)
    P_conf = get_effective_bubble_pressure(
        current_phase=phase,
        Eb=Eb, R2=R2, R1=R1, gamma=params['gamma_adia'].value,
        Lmech_total=Lmech_total, v_mech_total=v_mech_total,
        t=params['t_now'].value, tSF=params['tSF'].value,
    )
    # NOTE the bar is >= 0, not > 0, and this is the whole point of the scheme. O1 must
    # guard P_conf > 0 because its drive IS P_conf*(R_IF/R2)**2, so P_conf -> 0 forces the
    # drive to 0 -- that is O1's missing photoionisation-only limit, structural and
    # unfixable by any implementation choice. K11's drive does not contain P_conf as a
    # factor: at pdot_w -> 0 the closure returns the Stromgren density over R_IF and the
    # drive stays finite. Copying O1's guard (and its P_conf*rho form, which divides by
    # P_conf) threw that limit away -- caught by G22.7 and fixed here, 2026-08-31.
    if not (P_conf >= 0.0):
        return 0.0

    pref = (params['mu_convert'].value / params['mu_ion_shell'].value
            * params['k_B'].value * params['TShell_ion'].value)
    # pdot_w: the wind momentum flux the closure balances against. Momentum phase uses
    # pRam directly (the registered mapping); elsewhere the ramped confining pressure.
    P_w = pRam(R2, Lmech_total, v_mech_total) if phase == 'momentum' else P_conf
    pdot_w = 4.0 * np.pi * R2 * R2 * P_w

    n = _k11_skin_density(R_IF, pdot_w, Qi, pref,
                          params['chi_e_shell'].value, params['caseB_alpha'].value)
    if not (n > 0.0):
        return 0.0

    drive = pref * n * (R_IF / R2) ** 2

    # Batch 16's mapping written on `drive` DIRECTLY rather than on P_conf*rho. The two
    # are algebraically identical wherever P_conf > 0 (in momentum P_conf == pRam by
    # get_effective_bubble_pressure's own construction, so pRam*(rho-1) == drive - pRam),
    # but this form has no division by P_conf and so survives P_conf -> 0.
    # P_ram computed directly, never read from params['P_ram'], so this helper carries
    # no dependence on call ordering within the phase runners.
    if phase in ('momentum', 'transition'):
        return float(drive - pRam(R2, Lmech_total, v_mech_total))
    return float(drive)



def get_phii_k10(params, shell_props):
    """Photoionised pressure from the coupled (CEM) closure on trinity's OWN geometry.

    ARM CODE (docs/dev/phii-identity/PLAN.md Batch 21, variant O1). NOT the shipped
    scheme; adopting it is a maintainer decision (D5, still open).

        drive = P_conf * (R_IF/R2)**2

    R_IF is read verbatim from the shell solve (ShellProperties.R_IF,
    shell_structure.py:227 = rShell_arr_ion[-1]) -- the ionisation front trinity already
    computes with the REAL density profile, the REAL dust term (get_shellODE.py:120) and
    the real termination conditions (photon depletion OR shell mass exhaustion). Nothing
    is re-derived, rescaled or clipped.

    WHY READ IT RATHER THAN SOLVE IT (Batch 20). The earlier variant solved its own front
    from a uniform pressure-equilibrium density; that front was measured OUTSIDE the
    neutral shell on 18/18 B3M momentum rows and outside the cloud on 100% of driving
    rows (R_i up to 72.7 pc in a 5.0 pc cloud). Reading R_IF makes the front physical by
    construction, and deletes the closed-form/brentq machinery with its guard and bracket
    defects along with it.

    P_conf is the RAMPED effective pressure, not params['Pb']: inside dt_switchon the two
    differ by 0.330-0.995x (Batch 16 G16.3), and the un-ramped value would re-admit the
    D-ramp defect class C3c removed.

    RETURN CONVENTION (Batch 16, verified to 2.22e-16 through the real P_drive
    expressions): the CEM drive MINUS whatever the phase's own composition contributes --

        energy/implicit   max(P_conf, P_HII)            <- return P_conf*rho
        transition        max(P_conf, P_HII + P_ram)    <- return P_conf*rho - P_ram
        momentum          P_HII + P_ram                 <- return P_ram*(rho - 1)

    KNOWN, NOT FIXED BY THIS VARIANT (Batch 21 G21.5): the drive is still proportional to
    P_conf, so the photo-only limit is still absent and test_phii_c3c_spitzer.py still
    fails; and P_HII is still frozen per ODE segment by the phase runners while
    press_bubble is recomputed live, which is the freeze ratchet Batch 20 slice 4
    measured.
    """
    R2 = params['R2'].value
    Qi = params['Qi'].value
    if not (R2 > 0 and Qi > 0):
        return 0.0

    R_IF = getattr(shell_props, 'R_IF', None)
    if not (isinstance(R_IF, float) and R_IF >= R2):
        return 0.0

    phase = params['current_phase'].value
    Lmech_total = params['Lmech_total'].value
    v_mech_total = params['v_mech_total'].value
    Eb = params['Eb'].value
    # R1 via solve_R1 rather than params['R1'], matching the ODE's own canonical call
    # (energy_phase_ODEs.py:221) so this cannot read a stale radius.
    R1 = solve_R1(R2, Eb, Lmech_total, v_mech_total)
    P_conf = get_effective_bubble_pressure(
        current_phase=phase,
        Eb=Eb, R2=R2, R1=R1, gamma=params['gamma_adia'].value,
        Lmech_total=Lmech_total, v_mech_total=v_mech_total,
        t=params['t_now'].value, tSF=params['tSF'].value,
    )
    if not (P_conf > 0.0):
        return 0.0

    rho = (R_IF / R2) ** 2

    # P_ram computed directly, never read from params['P_ram'], so this helper carries
    # no dependence on call ordering within the phase runners.
    if phase == 'momentum':
        return float(pRam(R2, Lmech_total, v_mech_total) * (rho - 1.0))
    if phase == 'transition':
        return float(P_conf * rho - pRam(R2, Lmech_total, v_mech_total))
    return float(P_conf * rho)


def get_effective_bubble_pressure(current_phase, Eb, R2, R1, gamma,
                                   Lmech_total=None, v_mech_total=None,
                                   t=None, tSF=None):
    """
    Effective interior pressure felt by the shell.

    Energy phase: thermal pressure from hot bubble via bubble_E2P.
    Momentum phase: ram pressure from freely streaming wind via pRam.

    This function MUST be called in both the ODE and in compute_derived_quantities
    to guarantee consistency between the integrator and diagnostics.

    Parameters
    ----------
    current_phase : str
        Current simulation phase ('energy', 'momentum', etc.)
    Eb : float
        Bubble energy [au]
    R2 : float
        Outer bubble radius [pc]
    R1 : float
        Inner bubble radius [pc]
    gamma : float
        Adiabatic index
    Lmech_total : float, optional
        Mechanical wind luminosity (required for momentum phase)
    v_mech_total : float, optional
        Terminal wind velocity (required for momentum phase)
    t : float, optional
        Current time [Myr] (for early-phase R1 ramp-up)
    tSF : float, optional
        Star formation time [Myr] (for early-phase R1 ramp-up)

    Returns
    -------
    press_bubble : float
        Effective bubble pressure [au]
    """
    if current_phase == 'momentum':
        # Momentum phase: ram pressure from freely streaming wind
        return pRam(R2, Lmech_total, v_mech_total)
    elif current_phase == 'transition':
        # Transition phase: use max(P_thermal, P_ram) to ensure smooth
        # handoff to momentum phase.  As Eb decays on the sound-crossing
        # timescale, P_thermal drops while P_ram stays roughly constant.
        # By the time Eb hits the energy floor, P_ram already dominates,
        # so switching to momentum phase (P_ram only) is continuous.
        P_thermal = bubble_E2P(Eb, R2, R1, gamma)
        P_ram = pRam(R2, Lmech_total, v_mech_total)
        P_eff = max(P_thermal, P_ram)
        logger.debug(f"Transition pressure (P/k_B): P_thermal={P_thermal*cvt.Pb_au2_KcmInv:.4e}, "
                     f"P_ram={P_ram*cvt.Pb_au2_KcmInv:.4e} K cm⁻³, "
                     f"using={'P_ram' if P_ram >= P_thermal else 'P_thermal'}, Eb={Eb:.4e}")
        return P_eff
    else:
        # Energy/implicit phases: thermal pressure from hot bubble.
        # Include the early-phase R1 ramp-up if timing info provided.
        #
        # POST-MERGE NOTE (2026-08-14): the C3c photoionised regime switch
        # (`get_phii_c3c`, merged in c43a50e) changed what this ramp controls.
        # Phase 1a drives the shell with max(press_bubble, P_HII). Before C3c,
        # P_HII equalled the UNRAMPED Pb exactly, so the drive was the unramped
        # pressure and this ramp acted only on the energy equation's PdV drain.
        # After C3c, P_HII = 0 in the energy phase (measured 0.0000 on
        # simple_cluster), so the drive is the ramped pressure too -- the ramp
        # is now strictly MORE load-bearing than the numbers below were measured
        # under. The algebra is unaffected (re-verified on merged main: R1/R2 =
        # 0.869167, PdV/Lmech = 2.647425 on all five configs); the trajectory and
        # fate percentages below predate C3c and should be re-measured before
        # being quoted as current.
        #
        # LOAD-BEARING — do not delete as "inert" (magic-number audit #2), and
        # do not "improve" it without reading why four replacements failed.
        #
        # WHAT IT DOES. For the first 1e-3 Myr after star formation, R1 is
        # ramped linearly into bubble_E2P, enlarging the shocked-wind volume and
        # so holding the early driving pressure down.
        #
        # PROVENANCE — the measurements below predate C3c (2026-08-14). They were
        # taken when P_drive = max(this ramped pressure, P_HII) and P_HII was
        # params['Pb'] relabelled, i.e. the UN-ramped pressure, frozen per
        # segment. The max therefore selected the un-ramped floor and this ramp
        # never reached the shell momentum equation at all -- it acted only
        # through Edot and L_leak. get_phii_c3c now returns exactly 0.0 on the
        # confined branch, so the ramp throttles vd for the first time. The
        # ALGEBRA below (the PdV/Lmech identity, the seed universality) is in the
        # energy equation and is unaffected; the ABLATION FIGURES (fate flips,
        # the Weaver Eq.20 distances, the dR2 cost bound) were measured with the
        # ramp half-connected and are pending a re-run. Do not quote them as
        # current. See docs/dev/switchon-successor/PLAN.md Status block and
        # docs/dev/phii-identity/PLAN.md section 3 item 3 ("D-ramp").
        #
        # WHY IT IS NEEDED — the handover is inconsistent, and provably so.
        # solve_R1 puts R1 where the free wind's ram pressure balances the
        # bubble pressure, i.e. Pb = Lmech/(2 pi v_wind R1**2). Substituting
        # that into phase 1a's energy equation collapses the work term to
        #
        #     PdV / Lmech = 2 (v2/v_wind) / (R1/R2)**2
        #
        # in which Eb does not appear (verified to 1e-12 along a whole run:
        # docs/dev/switchon-successor/data/s4_identity_check.csv). Since
        # R1/R2 <= 1, PdV/Lmech >= 2 (v2/v_wind) for ANY seed energy — and
        # phase 0 hands over v2 = v_wind by construction, because the
        # free-expansion phase ends with the shell at the wind terminal speed.
        # So the energy-driven phase starts doing work ~2.6x faster than the
        # wind supplies it, on every config: the seed state is identical to six
        # digits across four decades of density and mass (R1/R2 = 0.869167,
        # PdV/Lmech = 2.647425; data/s4_seed_anatomy.csv). Unramped, Eb drains,
        # which drives R1 -> R2, which raises Pb further; the runaway ends the
        # bubble on 3 of 5 screen configs including the default published one
        # (docs/dev/phase1a-stiffness/data/dt_switchon_removability.csv).
        #
        # WHY THIS SHAPE, GIVEN IT IS UNCALIBRATED. The 1e-3 Myr window is
        # absolute, not scale-relative, and runs 500-87,000x longer than
        # dt_phase0, the establishment time the code itself computes — that is
        # a real wart. Four successors were pre-registered and measured
        # (docs/dev/switchon-successor/PLAN.md), and all four failed:
        #   - a physical clock (tmin = k*dt_phase0) flips fates on 3 of 5, and
        #     not in order of window length, so no k rescues it (D2);
        #   - a sustainability cap on Pb clears every fate but pins dEb/dt ~ 0,
        #     so Eb plateaus and the solution lands ~2x further from the
        #     Weaver Eq.20 reference than this ramp does (D3);
        #   - reseeding E0 cannot work at all — see the identity above (D4);
        #   - reseeding v0 rescues 2 of the 3 fates but still fails on
        #     f1edge_hidens and is 3.6-6.0x worse on the physics bar, because
        #     starting marginal only delays the runaway (D4).
        # With the ramp, Eb/t tracks Weaver Eq.20 within ~12%; without it, it
        # falls 154x below. The constant is kept because it measurably beats
        # every derived alternative tried, not because nobody looked.
        #
        # COST. Bounded at |dR2| <= 0.006-0.017% beyond the early window on the
        # two configs that survive ablation; on the three that do not, the ramp
        # is the difference between a bubble and no bubble, so no cost figure
        # is meaningful there.
        #
        # THE REAL FIX, not attempted here: a decelerating phase between free
        # expansion and the energy-driven solution, so the handover does not
        # happen while v2 is still v_wind. TRINITY has no such phase.
        #
        # Pinned by test/test_dt_switchon_ramp.py.
        dt_switchon = 1e-3
        tmin = dt_switchon

        if t is not None and tSF is not None:
            if t <= (tmin + tSF):
                R1_tmp = (t - tSF) / tmin * R1
                return bubble_E2P(Eb, R2, R1_tmp, gamma)

        return bubble_E2P(Eb, R2, R1, gamma)


# =============================================================================
# Find inner discontinuity
# R1 = interface separating inner bubble radius and outer solar wind
# =============================================================================

def get_r1(r1, params):
    """
    Root of this equation sets r1 (see Rahners thesis, eq 1.25).
    This is derived by balancing pressure.
    
    units of au
    
    Parameters
    ----------
    r1 : variable for solving the equation 
        The inner radius of the bubble.

    Returns
    -------
    equation : equation to be solved for r1.

    """
    # Note
    # old code: R1_zero()
    Lmech_total, Ebubble, v_mech_total, r2 = params
    
    # set minimum energy to avoid zero
    if Ebubble < 1e-30:
        Ebubble = 1e-30
    # the equation to solve
    equation = np.sqrt( Lmech_total / v_mech_total / Ebubble * (r2**3 - r1**3) ) - r1
    # return
    return equation


def solve_R1(R2, Eb, Lmech_total, v_mech_total):
    """
    Solve get_r1 for the inner bubble radius R1 (wind termination shock) [pc].

    Uses the full bracket [0, R2]: for Lmech_total > 0 the equation is
    sqrt(Lmech/v/Eb * R2**3) > 0 at r1 = 0 and -R2 < 0 at r1 = R2, so the
    bracket always contains the root (the former [1e-3*R2, R2] bracket
    missed roots below 1e-3*R2 and raised). Lmech_total <= 0 means no wind
    ram pressure, hence no termination shock: returns 0.0. A non-physical
    R2 <= 0 (a transient ODE-integrator excursion during the energy-driven
    Eb -> 0 collapse) likewise has no shock: returns 0.0, so the energy-phase
    RHS stays finite and the integrator's error control rejects the bad step
    instead of get_r1 hitting sqrt(<0) -> NaN -> brentq raising and crashing
    the run (see docs/dev/failed-large-clouds).

    Raises on root-finding failure for a physical bubble instead of
    fabricating a value. Non-finite Eb/Lmech/v_mech with a physical R2 raise
    explicitly: scipy < 1.11 brentq silently converges on a NaN-poisoned
    function (returns ~1e-12 instead of raising), so the no-fabrication
    guarantee must not depend on the scipy version.
    """
    if Lmech_total <= 0:
        return 0.0
    if not (R2 > 0):  # R2 <= 0 or NaN: non-physical radius, no wind shock
        return 0.0
    if not (np.isfinite(Eb) and np.isfinite(Lmech_total) and np.isfinite(v_mech_total)):
        raise ValueError(
            f"solve_R1 got non-finite input for a physical R2={R2:.6e}: Eb={Eb}, "
            f"Lmech_total={Lmech_total}, v_mech_total={v_mech_total}"
        )
    try:
        return scipy.optimize.brentq(
            get_r1, 0.0, R2,
            args=([Lmech_total, Eb, v_mech_total, R2]),
        )
    except (ValueError, RuntimeError):
        logger.error(
            f"R1 root finding failed on [0, R2]: R2={R2:.6e}, Eb={Eb:.6e}, "
            f"Lmech_total={Lmech_total:.6e}, v_mech_total={v_mech_total:.6e}"
        )
        raise


# =============================================================================
# P_HII closures and their dispatcher.
#
# ⛔ KEEP THESE AT THE END OF THE FILE. The held candidate arms
# docs/dev/phii-identity/hpc/b14/{k10_o1,k11}_arm.patch insert immediately after
# get_phii_c3c, and test/test_phii_limits.py builds both by `git apply`-ing them to
# a temp copy of this file. Anything added between get_phii_c3c and the next def
# shifts their context and takes 14 tracked limit gates with it (found by doing it,
# 2026-09-12). Appending here leaves their hunks untouched.
# =============================================================================

def get_phii_front(params, shell_props):
    """Photoionised pressure as the FRONT pressure (option C, D16 ruled 2026-09-11).

    Under the maintainer's geometry ruling G1-G3 the photoionised gas is the shell's
    inner layer [R2, R_IF]. It pushes the neutral gas beyond R_IF, and what it pushes
    with is the pressure AT the front:

    ⛔ NO END-STATE BRANCH, by ruling 2026-09-14. This paragraph used to read "a fully
    ionised shell (end state 1) has no neutral gas to push, so P_HII = 0". That was C v1
    and it is RETIRED; the block comment in the body explains why (a fully ionised shell
    means photons LEAK, i.e. the front has run PAST rShell into cloud the model no longer
    tracks -- there IS neutral gas, the code has merely lost the front's radius). The
    docstring outlived the ruling by eight days and misled a reader. If the branch
    behaviour ever changes again, change BOTH.

        P_front = (mu_c/mu_i) * n_IF * k_B * T_ion

    The caller applies it over 4*pi*R_IF**2 -- the front's area, not R2's -- and does
    NOT add Pb or P_ram, because the wind acts on the ionised layer, not on the neutral
    shell. So this helper is only half of C; the other half is the area and the
    composition in the two ODE right-hand sides.

    WHY THE FRONT AND NOT THE LAYER'S OWN PRESSURE. Decided on the limits, not the
    magnitudes (PLAN.md 0.0.1 item 3):
      * weak ionisation: Qi and Li -> 0 gives R_IF/R2 -> 1.0000000 and n_IF/n0 ->
        1.000000, continuously and including at exactly zero, with `has_neutral` still
        true -- so the drive tends to 4*pi*R2**2*Pb, which in the momentum phase is
        4*pi*R2**2*P_ram: the pure wind solution, recovered with no branch. Measured
        2026-09-11. (Scaling Qi ALONE gives 1.907 and looks like a Qi=0+ discontinuity;
        that is an artifact of leaving Li at full strength, which keeps compressing the
        shell radiatively. Qi and Li are one physical quantity. Do not repeat it.)
      * weak wind: a layer pressure composed additively gives P_HII + P_ram -> 2*P_ram,
        discontinuous at Qi = 0+. C adds nothing, so it has no such term.

    ⛔ SUPERSEDED (2026-09-14) -- the paragraph below describes C v1's state-1 zero,
    which no longer exists. It is kept because the RETURN CONVENTION it documents is
    still how the phase runners read this helper. Under the current closure P_HII > 0 on
    every row (measured: 1581/1581 rows across 8 configs), so the "branch signal" it
    describes never fires.
    RETURNING EXACTLY 0.0 IN STATE 1 WAS LOAD-BEARING in v1, exactly as it is for
    get_phii_c3c: it is what lets the state-1 force assembly stay byte-identical to the
    shipped one, and it is the branch signal the ODE reads (P_HII > 0 <=> state 2 under
    this scheme), so no `has_neutral` has to be plumbed into the snapshots.

    NOTE what this does NOT depend on: any photon budget. C3c inverts a recombination
    balance and so had to be corrected for the gas-vs-total absorbed LyC (W69) and would
    have to be corrected again for the sky partition (the 2026-09-12 coverFraction
    ruling, `shell_fAbsorbedIonGas_total`). n_IF is the shell ODE's own value at the
    front on a covered ray, so C is insulated from both.
    """
    # ⛔ NO end-state branch here, by ruling 2026-09-14. It is tempting to return 0.0 on
    # a fully ionised shell ("nothing neutral left to push"), and C v1 did. It is wrong:
    # a fully ionised shell means photons LEAK, which means the front has run PAST rShell
    # into cloud the code no longer tracks -- there IS neutral gas, the model has just
    # lost the front's radius. R_IF == rShell there (231/231 archived rows) is the front
    # AT the shell's outer edge on its way out. n_IF and R_IF are both continuous through
    # that moment (measured: a -1.2% step in p_ref*n_IF/Pb across B3M's momentum flip,
    # inside its existing trend), so a branch there injects a factor-7.5 discontinuity
    # into an ODE that the physics does not have. See data/b32_alwayson.csv and
    # shell_frontEscaped, which flags the regime instead of switching on it.
    n_IF = getattr(shell_props, 'n_IF', 0.0)
    try:
        n_IF = float(n_IF)
    except (TypeError, ValueError):
        return 0.0
    if not (np.isfinite(n_IF) and n_IF > 0.0):
        return 0.0
    P_front = ((params['mu_convert'].value / params['mu_ion_shell'].value)
               * n_IF * params['k_B'].value * params['TShell_ion'].value)
    return float(P_front) if (np.isfinite(P_front) and P_front > 0.0) else 0.0


# name -> closure.  The shipped scheme is first and is the default.
PHII_SCHEMES = {
    'c3c': get_phii_c3c,
    'front': get_phii_front,
    'o1': get_phii_k10,
    'k11': get_phii_k11,
}


def phii_is_active(params, shell_props):
    """Scheme-aware pre-gate for the P_HII call sites.

    The phase runners guard the closure call so that a scheme's own degenerate cases do
    not have to be re-derived at six sites. The gate is NOT the same question for the
    two schemes, and conflating them was a real defect (found by /xcheck 2026-09-12):

      c3c    `n_IF_Str > 0`. The shipped gate, kept verbatim so c3c stays bit-identical.
             ⚠ It is a LEGACY gate and it does not belong to get_phii_c3c at all.
             `n_IF_Str` is the SHELL-LAYER Stroemgren density, built on the layer volume
             (R_IF**3 - R2**3); get_phii_c3c builds its own CAVITY density `n_c3a` on
             R2**3 and never reads n_IF_Str. The layer density is precisely the quantity
             C3c replaced on 2026-08-14 (see get_phii_c3c's docstring), so the gate
             outlived the closure it was written for. It is kept only for bit-identity,
             and it has never been observed to fire: n_IF_Str > 0 on 867/867 rows of the
             C trajectory pair (data/local/c_20260914T061301Z/) and on 895/895 archived
             states. ⇒ do not reason about C3c's activation from it. What actually holds
             C3c at 0.0 through the whole energy era is its CONFINEMENT branch
             (P_c3a <= Pb), measured on 303/303 energy+implicit rows of that pair -- NOT
             a vanishing density, which was my error and is corrected here.
      front  True. get_phii_front self-gates on `has_neutral` and `n_IF`, which are the
             front's own quantities. Gating it on `n_IF_Str` instead made option C's
             ACTIVATION depend on a legacy layer-volume term -- and that term vanishes
             precisely in the weak-ionisation limit (R_IF -> R2) that C was chosen for.
             On the 895 archived states the two gates happen to agree on every row (0
             disagreements), so this changes no measured number; it removes a coupling
             that would have bitten the next time n_IF_Str's definition moved, as it
             already did at W60.
    """
    item = params.get('phii_scheme', None)
    name = str(item.value if hasattr(item, 'value') else 'c3c').strip().lower()
    if name == 'front':
        return True
    return getattr(shell_props, 'n_IF_Str', 0.0) > 0


def get_phii(params, shell_props):
    """Dispatch to the P_HII closure named by params['phii_scheme'].

    One dispatcher so the six phase-runner call sites do not each grow a branch, and so
    `get_phii_c3c` stays byte-identical as the control arm.

    An unknown name falls back to the shipped scheme rather than raising. That is belt
    and braces, not the primary defence: `registry._validate_phii_scheme` rejects an
    unrecognised value at startup, so a typo in a .param never reaches here. (Until
    2026-09-12 this docstring asserted that validation while none existed -- found by
    /xcheck. The validator is now real; if you remove it, remove this sentence too.)
    """
    item = params.get('phii_scheme', None)
    name = item.value if hasattr(item, 'value') else 'c3c'
    return PHII_SCHEMES.get(str(name).strip().lower(), get_phii_c3c)(params, shell_props)


