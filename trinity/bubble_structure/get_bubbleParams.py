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
    Note this is the un-ramped Pb; in the energy phase P_C3a/Pb is far below 1 by
    either measure, so the branch outcome is insensitive to that choice.

    KNOWN OPEN BEHAVIOUR: the momentum phase comes out photoionisation-dominated in
    every configuration measured so far. P_C3a/P_ram falls only as Lw**-0.33 with wind
    strength, so an inversion would need an unphysical Lw ~ 260. This is NOT an O(1)
    normalisation error -- the same normalisation predicts the transition-phase
    crossover to within 7% -- it is the R2**-1.5 cavity geometry. See
    docs/dev/phii-identity/PLAN.md 3c stage 3.
    """
    R2 = params['R2'].value
    Qi = params['Qi'].value
    if not (R2 > 0 and Qi > 0):
        return 0.0
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
