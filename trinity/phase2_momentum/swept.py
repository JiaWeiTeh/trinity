#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum phase under phii_scheme 'swept' (docs/dev/phii-identity PLAN.md 0.0.11.3, Step 4; sweep speed 0.0.11.10).

The shell is one body between R2 and its outer radius r_s. Evolved: r_s and u, the speed of the shell gas. The
mass is everything the outer edge has passed, M_s = M(<r_s): it never decreases, and it is frozen while u <= 0
(ruling D-T1, 2026-10-21). R2 is algebraic: each segment it is root-found so that the shell solve started at R2
(inner density from P_ram(R2)) with mass M_s ends exactly at r_s. The drive is the shell profile's own balance,

    F = 4 pi R2^2 P_ram(R2) + K_s - 4 pi r_s^2 P_ext(r_s) + F_rad,

with K_s = INT 8 pi r P dr over the profile (the curvature term the c3c runner cannot have), equivalently
4 pi r_s^2 P(r_s) + (F_rad - F_rad,sh) by the identity 4 pi R2^2 P_ram + K_s + F_rad,sh = 4 pi r_s^2 P(r_s) (L8,
exported as K_s, F_rad_sh, P_outer). There is no P_HII: the photoionised gas's pressure is inside P(r_s).

The outer edge is the leading shock (ruling D-T2 (b), 2026-10-21): it runs at U = (u + sqrt(u^2 + 4 c0^2))/2,
the isothermal jump at the gas ahead, c0^2 = P_ext / rho_amb at r_s, where 0 < c0 <= c_i and the gas ahead is cloud
gas (r_s < rCloud; Step 7a, A1, 2026-10-26); elsewhere U = u. In a cold cloud (P_ext = 0) that is one speed exactly,
and past the cloud the shell moves with its gas. Mass is swept at 4 pi r_s^2 rho_amb U and carried at u:

    dr_s/dt = U,      du/dt = (F - F_grav - Mdot_s u) / M_s,      F_grav = G M_s (M_* + M_s/2) / r_s^2.

R2/r_s, K_s and F_rad are held over a segment, as the c3c runner holds its shell solve (PLAN 0.0.11.3: one
root-find per segment); P_ram, P_ext, the mass and the jump are evaluated at the live r_s.
"""

import logging
import math

import numpy as np
import scipy.integrate
import scipy.optimize

import trinity._output.terminal_prints as terminal_prints
import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
import trinity.cloud_properties.mass_profile as mass_profile
from trinity._input.dictionary import updateDict
from trinity._output.simulation_end import SimulationEndCode
from trinity.cloud_properties import density_profile
from trinity.phase1_energy.energy_phase_ODEs import get_press_ion
from trinity.phase2_momentum.run_momentum_phase import (
    ADAPTIVE_FACTOR, ADAPTIVE_MONITOR_KEYS, ADAPTIVE_THRESHOLD_DEX, DT_SEGMENT_COLLAPSE, DT_SEGMENT_INIT,
    DT_SEGMENT_MAX, DT_SEGMENT_MIN, MAX_SEGMENTS, ODE_ATOL, ODE_MAX_STEP, ODE_METHOD, ODE_MIN_STEP, ODE_RTOL,
    VELOCITY_THRESHOLD_COLLAPSE, VELOCITY_THRESHOLD_EXTREME, MomentumPhaseResults, compute_max_dex_change,
    get_monitor_values,
)
from trinity.phase_general.phase_events import (
    apply_event_result, build_momentum_phase_events, build_transition_phase_events, check_event_termination,
    collapse_reason, update_collapse_radius,
)
import trinity._functions.operations as operations
from trinity.shell_structure.shell_structure import shell_structure_pure
from trinity.sps.update_feedback import get_current_sps_feedback

logger = logging.getLogger(__name__)

# D12 (docs/dev/phii-identity/weak-wind/PLAN.md, ruled 2026-10-28): segment control after a Stroemgren start, in units
# of the shell's dynamical time r_s/|u|. Measured on the G-WW1 test cloud (R_St/c_i = 3.2 kyr; the runner's default
# segments, 2e-3 / 1e-3 Myr there, reverse the shell in its first segment): the outer edge's largest deviation from
# Hosokawa & Inutsuka 2006 over 2-4.3 R_St is 1.5e-3, 4.2e-4 and 8.5e-5 at a fraction 0.03, 0.01 and 0.003 (93, 246,
# 792 rows to 10 R_St); 0.01 meets the 1e-3 bar with a factor 2.4 to spare (weak-wind/data/g_ww1_cap_20261028/).
STROMGREN_DT_FIRST = 1e-3   # fraction of r_s/|u| for the first segment (not Myr)
STROMGREN_DT_FRAC = 0.01    # fraction of r_s/|u| for every later segment (not Myr)

FOUR_PI = 4.0 * math.pi
MONITOR_KEYS = ADAPTIVE_MONITOR_KEYS + ['rs_dyn']     # R2 is algebraic here; the evolved radius is rs_dyn


def _monitor(params):
    # get_monitor_values reads ADAPTIVE_MONITOR_KEYS only; the shared list stays as c3c has it
    return {**get_monitor_values(params), 'rs_dyn': params['rs_dyn'].value}


def is_swept(params):
    item = params.get('phii_scheme', None)
    return str(item.value if hasattr(item, 'value') else 'c3c').strip().lower() == 'swept'


def _scalar(x):
    return float(np.atleast_1d(x)[0])


def p_ext(r, fabs_ion, params, turbulent=True):
    """External pressure at r, as the c3c runner has it: photoionised gas if photons escape, PISM past rCloud, and the
    cloud gas's own pressure inside the cloud (density_profile.get_press_cloud; opt-in, 0 by default)."""
    P = _scalar(get_press_ion(r, params)) if fabs_ion < 1.0 else 0.0
    if r >= params['rCloud'].value:
        P += params['PISM'].value * params['k_B'].value
    P += density_profile.get_press_cloud(r, fabs_ion, params, turbulent)
    return P


def edge_speed(u, r, fabs_ion, params):
    """The outer edge's speed U for shell-gas speed u: the isothermal jump at the gas ahead (D-T2 (b)).

    Applies where u > 0, the gas ahead is cloud gas (r < rCloud; PLAN Step 7a, A1) and 0 < c0 <= c_i; elsewhere
    U = u. c0 = c_i exactly when photons escape inside the cloud (P_ext/rho = k T_ion / mu_ion_shell), so the cut
    carries a 1e-9 relative allowance for rounding. Past the cloud the dense shell moves with its gas, as in every
    single-speed shell model; a separately tracked weak front there (A2) is the future upgrade.
    """
    if u <= 0.0 or r >= params['rCloud'].value:
        return u
    rho = params['mu_convert'].value * _scalar(density_profile.get_density_profile(np.array([r]), params))
    if rho <= 0.0:
        return u
    c0_sq = p_ext(r, fabs_ion, params, turbulent=False) / rho   # thermal only: the jump is a sound-speed condition
    c_i_sq = params['k_B'].value * params['TShell_ion'].value / params['mu_ion_shell'].value
    if not (0.0 < c0_sq <= c_i_sq * (1.0 + 1e-9)):
        return u
    return 0.5 * (u + math.sqrt(u * u + 4.0 * c0_sq))


def swept_mass(r, U, M_floor, frozen, params):
    """(M_s, Mdot_s): the cloud mass inside r swept at speed U, never below M_floor; frozen gives (M_floor, 0)."""
    if frozen:
        return M_floor, 0.0
    m, md = mass_profile.get_mass_profile(r, params, return_mdot=True, rdot=U)
    m, md = _scalar(m), _scalar(md)
    return (M_floor, 0.0) if m < M_floor else (m, md)


_TRIAL_KEYS = ('R2', 'R1', 'Pb', 'shell_mass', 'bubble_mass', 'rShell')     # what _shell_at writes


def _p_conf(R2, Eb, fb, params, phase='transition', t=None):
    """(P_conf, R1) at the inner face (PLAN 0.0.11.3): P_ram in momentum (Eb None); max(P_th, P_ram) in transition;
    in the implicit phase the bubble's own pressure at time t, the switch-on ramp included, exactly as its drive."""
    if Eb is None:
        return get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total), R2
    R1 = get_bubbleParams.solve_R1(R2, Eb, fb.Lmech_total, fb.v_mech_total)
    if phase == 'implicit':
        dt_on = params['dt_switchon'].value if 'dt_switchon' in params else None
        return get_bubbleParams.get_effective_bubble_pressure(
            'implicit', Eb, R2, R1, params['gamma_adia'].value, fb.Lmech_total, fb.v_mech_total,
            t=t, tSF=params['tSF'].value, dt_switchon=dt_on), R1
    return get_bubbleParams.get_effective_bubble_pressure('transition', Eb, R2, R1, params['gamma_adia'].value,
                                                          fb.Lmech_total, fb.v_mech_total), R1


def _shell_at(R2, rs, Ms, fb, params, Eb=None, phase='transition', t=None):
    """The shell solve started at R2 (inner density from P_conf(R2)) with mass Ms; r_s is only the solver's guess.

    With Eb given (implicit, transition) bubble_mass is left as the c3c runners leave it.
    """
    Pc, R1 = _p_conf(R2, Eb, fb, params, phase, t)
    params['R2'].value, params['R1'].value, params['Pb'].value = R2, R1, Pc
    params['shell_mass'].value = Ms
    if Eb is None:
        params['bubble_mass'].value = get_bubbleParams.mass_freeWind(R2, fb.Lmech_total, fb.v_mech_total)
    params['rShell'].value = rs
    return shell_structure_pure(params), Pc


TAN_STEP = 1e-4        # relative step of the tangent's one-sided differences (Step 7b-0: partials flat over 1e-5..1e-3)
TAN_MISS_EPS = 1e-3    # the closure's miss at the next root, in Step 6's energy units, above which the next segment is cut


def tangent_partials(R2, rs, Ms, Eb, fb, params, t, sp, h=TAN_STEP):
    """Partials of the outer edge r_out(R2, Eb, M, t) at an implicit-phase root (PLAN.md Step 7b-1, 2026-10-26).

    One-sided differences from the root's own solve sp (r_out = sp.rShell = r_s): N = dr_out/dR2 with R2 stepped DOWN
    (a thin shell sits next to r_s), D_E = dr_out/dEb with Eb stepped up, D_t = dr_out/dt with the feedback at t (1 + h);
    D_M = 1 / (4 pi r_out^2 rho_out) in closed form (equal to the finite difference to 4e-4, row 71). params' trial keys
    and feedback are put back. None when a solve fails or N <= 0: the caller then keeps the secant.
    """
    saved = {k: params[k].value for k in _TRIAL_KEYS}
    t2 = t * (1.0 + h)
    try:
        base = float(sp.rShell)
        r_dn = float(_shell_at(R2 * (1.0 - h), rs, Ms, fb, params, Eb=Eb, phase='implicit', t=t)[0].rShell)
        r_eb = float(_shell_at(R2, rs, Ms, fb, params, Eb=Eb * (1.0 + h), phase='implicit', t=t)[0].rShell)
        fb2 = get_current_sps_feedback(t2, params)
        updateDict(params, fb2)
        r_t = float(_shell_at(R2, rs, Ms, fb2, params, Eb=Eb, phase='implicit', t=t2)[0].rShell)
    except Exception as e:  # a failed trial solve: fall back, never stop the run
        logger.warning(f"[swept] tangent partials failed at t={t:.6e}: {e}")
        return None
    finally:
        updateDict(params, fb)
        for k, v in saved.items():
            params[k].value = v
    rho_out = params['mu_convert'].value * float(np.asarray(sp.shell_n_arr, float)[-1])
    out = dict(N=(base - r_dn) / (h * R2), DE=(r_eb - base) / (h * Eb), DT=(r_t - base) / (t2 - t),
               DM=1.0 / (FOUR_PI * base**2 * rho_out) if rho_out > 0 else float('nan'))
    if not (out['N'] > 0 and all(np.isfinite(v) for v in out.values())):
        logger.warning(f"[swept] tangent partials unusable at t={t:.6e} (N={out['N']:.3e}): the secant stands in")
        return None
    return out


def escape_indicator(sp):
    """Signed distance to the photon-escape switch (PLAN.md Step 7b-2): 1 - R_IF/r_out while the shell traps every ionising
    photon, -(1 - f_abs,ion) while some escape. Continuous, zero at the switch, where r_out has a kink."""
    f = float(sp.shell_fAbsorbedIon)
    return -(1.0 - f) if f < 1.0 - 1e-12 else 1.0 - float(sp.R_IF) / float(sp.rShell)


def switch_dt(s_prev, t_prev, s, t, dt):
    """The segment length that ends just past a predicted escape switch, or dt (Step 7b-2).

    The crossing t_x of the indicator's linear trend through (t_prev, s_prev), (t, s); if s moves toward zero and t < t_x < t + dt
    the segment ends at t + 1.05 (t_x - t), so the kink lies just inside its end and the next segment starts on the new branch.
    Offline on the 7a and 7b-1 runs this flagged every switch at its own segment start and nothing else (data/b75_*).
    s_prev, s dimensionless; t_prev, t, dt [Myr]; returns [Myr].
    """
    if s_prev is None or t <= t_prev:
        return dt
    rate = (s - s_prev) / (t - t_prev)
    if rate == 0 or s * rate >= 0:
        return dt
    t_x = t - s / rate
    return min(dt, 1.05 * (t_x - t)) if t < t_x < t + dt else dt


def live_rates(Ed, W0, k_t, U_t, Mdot_t, tan, zeta):
    """dR2/dt and dEb/dt inside a 1b segment with the wall's work live (Step 7b-2).

    dEb/dt = Ed - (k_t dR2/dt - W0), k_t = 4 pi R2^2 Pb, W0 the work rate the beta/delta solve charged at the segment start.
    With the tangent, dR2/dt is the closure's own derivative, N dR2/dt + D_E dEb/dt + D_M dM/dt + D_t = U, solved with dEb/dt
    in closed form; with the secant, dR2/dt = zeta U.
    Ed, W0 [Msun pc^2/Myr^3]; k_t [Msun pc/Myr^2]; U_t [pc/Myr]; Mdot_t [Msun/Myr]; tan: N [-], DE [Myr^2/(Msun pc)],
    DM [pc/Msun], DT [pc/Myr]. Returns (dR2/dt [pc/Myr], dEb/dt [Msun pc^2/Myr^3]).
    """
    if tan is None:
        rd = zeta * U_t
    else:
        rd = (U_t - tan['DE'] * (Ed + W0) - tan['DM'] * Mdot_t - tan['DT']) / (tan['N'] - tan['DE'] * k_t)
    return rd, Ed + W0 - k_t * rd


def layer_terms(sp, R2, Pc, params):
    """K_s, F_rad,sh, P(r_s) and the L8 residual of 4 pi R2^2 P_ram + K_s + F_rad,sh = 4 pi r_s^2 P(r_s).

    Trapezoids over the solve's own grid (thousands of points): L8 <= 3.2e-7 on Step 1's 427 momentum segments.
    The two points at R_IF (ionised and neutral sides) carry the same pressure, so that zero-width interval adds
    nothing.
    """
    g = lambda k: params[k].value  # noqa: E731
    r = np.asarray(sp.shell_r_arr, float)
    n = np.asarray(sp.shell_n_arr, float)
    i = int(sp.shell_ion_idx)
    p_ion = g('mu_convert') / g('mu_ion_shell') * g('k_B') * g('TShell_ion')
    p_neu = g('mu_convert') / g('mu_atom') * g('k_B') * g('TShell_neu')
    pref = np.where(np.arange(len(r)) <= i, p_ion, p_neu) if i >= 0 else np.full(len(r), p_neu)
    P = pref * n
    K = float(np.sum(0.5 * (8 * math.pi * r[1:] * P[1:] + 8 * math.pi * r[:-1] * P[:-1]) * np.diff(r)))
    light = g('Li') + g('Ln')
    F_rad_sh = sp.shell_fAbsorbedWeightedTotal * light / g('c_light') if light else 0.0
    P_out = float(P[-1])
    rs = float(sp.rShell)
    outer = FOUR_PI * rs**2 * P_out
    L8 = (FOUR_PI * R2**2 * Pc + K + F_rad_sh - outer) / outer if outer else float('nan')
    return K, F_rad_sh, P_out, L8


def f_rad(sp, params):
    """Radiation force on the shell, exactly as the c3c momentum runner computes it."""
    if sp.isDissolved:
        return 0.0
    return (sp.shell_fAbsorbedWeightedTotal * params['Lbol'].value / params['c_light'].value
            * (1.0 + sp.shell_tauKappaRatio * params['dust_KappaIR'].value))


def root_R2(rs, Ms, fb, params, guess=None, Eb=None, phase='transition', t=None):
    """R2 in (0, r_s) whose shell solve ends at r_s (PLAN 0.0.11.3). Returns a dict; ok=False with a reason.

    The residual rShell(R2) - r_s rises with R2: near R2 -> 0 the mass fits in a thin skin, at R2 -> r_s it cannot be
    held. Lower end: given `guess` (the previous segment's root, scaled to this r_s), guess/1.5; otherwise
    lo = min(1e-3 r_s, the R2 whose inner density is 100 n_St(r_s)), stepping up through 10 lo, 100 lo, 0.1 r_s,
    0.5 r_s while the shell solve rejects a very thin trial. Upper end (Step 4c, 2026-10-24): 1.5 guess, or 4x the
    lower end, stepping up by 4x until the residual is positive -- each trial still below the root becomes the lower
    end. R2 = r_s (1 - 1e-9) is only the last resort: there the whole mass sits in a sliver at r_s and the profile
    runs to millions of points (6.3M at B2's late states, each solve leaving ~9 MB of C-level memory behind). The
    residual is cached, so brentq re-solves neither end. Eb given: the implicit or transition inner pressure (_p_conf).
    Thin shells (Step 6, 2026-10-25): with a guess the bracket also works in the gap g = r_s - R2 (lower end no lower
    than r_s - 2g, upper end no higher than r_s - g/2, stepping the gap down by 4), so a shell with R2/r_s near 1 does
    not send the upper end to the last resort; for R2/r_s < 1/2 the ends are the R2 ones above.
    """
    g = lambda k: params[k].value  # noqa: E731
    count = {'n': 0}
    saved = {k: params[k].value for k in _TRIAL_KEYS}

    def failed(why):
        for k, v in saved.items():          # the trial solves leave their last R2, Pb, ... in params
            params[k].value = v
        return dict(ok=False, why=why, nsolve=count['n'])

    cache = {}

    def resid(R2):
        if R2 not in cache:
            count['n'] += 1
            sp, _ = _shell_at(R2, rs, Ms, fb, params, Eb, phase, t)
            cache[R2] = float(sp.rShell) - rs
        return cache[R2]

    lo = 1e-3 * rs
    pdot = 2.0 * fb.Lmech_total / fb.v_mech_total if fb.v_mech_total > 0 else 0.0
    n_st = math.sqrt(3 * fb.Qi / (FOUR_PI * g('chi_e_shell') * g('caseB_alpha') * rs**3)) if fb.Qi > 0 else 0.0
    if pdot > 0 and n_st > 0:
        p_ion = g('mu_convert') / g('mu_ion_shell') * g('k_B') * g('TShell_ion')
        lo = min(lo, math.sqrt(pdot / (FOUR_PI * p_ion * 100.0 * n_st)))
    hi = rs * (1 - 1e-9)
    flo, why = None, ''
    lo_g = max(guess / 1.5, rs - 2.0 * (rs - guess)) if guess is not None else None
    if lo_g is not None and lo < lo_g < hi:
        try:
            f = resid(lo_g)
            if f < 0:
                lo, flo = lo_g, f
        except Exception:  # noqa: BLE001 -- fall back to the ladder
            pass
    if flo is None:
        for lo_try in (lo, 10 * lo, 100 * lo, 0.1 * rs, 0.5 * rs):
            try:
                flo = resid(lo_try)
                lo = lo_try
                break
            except Exception as exc:  # noqa: BLE001 -- a trial the shell solve rejects; the next one is thicker
                why = f'{type(exc).__name__}: {exc}'
    if flo is None:
        return failed(f'lower bracket solve failed: {why}')
    if flo >= 0:
        return failed(f'no bracket: f(lo)={flo:.3e} >= 0 at lo={lo:.4g}')
    up = min(1.5 * guess, rs - 0.5 * (rs - guess), hi) if guess is not None else lo
    if up <= lo:
        up = min(4 * lo, hi)
    while True:
        try:
            fhi = resid(up)
        except Exception as exc:  # noqa: BLE001
            if up >= hi:
                return failed(f'upper bracket solve failed: {type(exc).__name__}: {exc}')
            up = hi                          # a trial the solve rejects: the last resort
            continue
        if fhi > 0:
            break
        if up >= hi:
            return failed(f'no bracket: f(lo)={flo:.3e} f(hi)={fhi:.3e}')
        lo, flo, up = up, fhi, min(4 * up, rs - 0.25 * (rs - up), hi)
    n0 = count['n']
    try:
        R2 = scipy.optimize.brentq(resid, lo, up, xtol=1e-10 * rs, maxiter=100)
    except Exception as exc:  # noqa: BLE001
        return failed(f'brentq: {exc}')
    iters = count['n'] - n0
    sp, Pc = _shell_at(R2, rs, Ms, fb, params, Eb, phase, t)
    count['n'] += 1
    K, F_rad_sh, P_out, L8 = layer_terms(sp, R2, Pc, params)
    return dict(ok=True, R2=R2, sp=sp, Pc=Pc, K=K, F_rad_sh=F_rad_sh, P_out=P_out, L8=L8,
                resid=(float(sp.rShell) - rs) / rs, iters=iters, nsolve=count['n'])


def initial_state(params, fb, Eb=None, phase='transition', t=None):
    """Hand-off (PLAN 0.0.11.3): r_s from the mass-matched shell solve at the entry R2, u conserving momentum.

    r_s is the fixed point r_s = rShell(R2, M(<r_s)) (row 60), and u = M_sh v2 / M(<r_s): the shell's momentum
    shared with the swept gas, taken at rest (a merge, as Lancaster et al.'s eq. 37 divides the momentum injected
    so far by the mass carried). Eb given: the implicit or transition inner pressure.
    """
    R2, v2, m0 = params['R2'].value, params['v2'].value, params['shell_mass'].value
    m = m0
    for _ in range(60):
        sp, _ = _shell_at(R2, params['rShell'].value, m, fb, params, Eb, phase, t)
        m_new = max(_scalar(mass_profile.get_mass_profile(float(sp.rShell), params)), m0)
        if abs(m_new - m) <= 1e-10 * m:
            break
        m_used, m = m, m_new
    else:
        m = m_used          # the mass the last solve used, so r_s and M_s stay a pair
        logger.warning(f"[swept] hand-off fixed point not converged in 60 solves (M {m:.6g} -> {m_new:.6g} Msun)")
    return float(sp.rShell), m, m0 * v2 / m


def _forces_to_params(params, root, rs, u, Ms, Mdot, U, fabs_ion, fb, Eb=None, phase='transition'):
    """The snapshot's force and pressure columns under 'swept' (no P_HII; F_ion_in over the OUTER area).

    F_ram and P_drive are the inner face's P_conf; with Eb given Pb and P_ram keep the c3c runners' meanings: Pb the
    bubble's (unramped) thermal pressure, P_ram the free wind's ram pressure in transition and 0 in the implicit phase.
    """
    sp, R2, Pc = root['sp'], root['R2'], root['Pc']
    Pext = p_ext(rs, fabs_ion, params)
    params['shell_mass'].value, params['shell_massDot'].value = Ms, Mdot
    params['F_grav'].value = params['G'].value * Ms / rs**2 * (params['mCluster'].value + 0.5 * Ms)
    params['F_ram'].value = FOUR_PI * R2**2 * Pc
    params['F_rad'].value = f_rad(sp, params)
    params['F_ion_in'].value = FOUR_PI * rs**2 * Pext
    params['P_HII'].value, params['F_HII'].value = 0.0, 0.0
    params['n_IF'].value, params['R_IF'].value = sp.n_IF, sp.R_IF
    params['P_drive'].value = params['P_ram'].value = params['Pb'].value = Pc
    if Eb is not None:
        params['Pb'].value = get_bubbleParams.bubble_E2P(Eb, R2, params['R1'].value, params['gamma_adia'].value)
        params['P_ram'].value = (0.0 if phase == 'implicit'
                                 else get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total))
    params['press_HII_in'].value = Pext
    params['F_ram_wind'].value, params['F_ram_SN'].value = fb.pdot_W, fb.pdot_SN
    params['rs_dyn'].value, params['v2'].value, params['rs_dot'].value = rs, u, U
    params['K_s'].value, params['F_rad_sh'].value, params['P_outer'].value = root['K'], root['F_rad_sh'], root['P_out']
    params['swept_root_iter'].value, params['swept_root_resid'].value = root['iters'], root['resid']
    params['n_shell_solves'].value = root['nsolve']


def _end(params, reason, code, message):
    params['SimulationEndReason'].value = message
    params['SimulationEndCode'].value = code.code
    params['EndSimulationDirectly'].value = True
    return reason


class ImplicitSwept:
    """The swept state through phase 1b (PLAN.md Step 6, 2026-10-25): r_s, u, M_s and the closure's carry.

    The c3c implicit runner keeps its loop -- the beta/delta solve and everything around it -- and calls these at its
    hooks. Per segment: R2 root-found with the bubble's own pressure (switch-on ramp included); the RHS evolves (r_s, u)
    while Eb and T0 keep their frozen beta/delta rates. Step 7b-1 (2026-10-26): inside a segment R2 follows the tangent of
    the constraint r_out(R2, Eb, M, t) = r_s at the root, R2 = R2_0 + [(r_s - r_s0) - D_E (Eb - Eb0) - D_M (M - M0)
    - D_t (t - t0)] / N, and the beta/delta solve sees the coupled wall speed (wall_speed); where the partials cannot be
    had the secant zeta = Delta R2 / Delta r_s of successive roots stands in (1 on the first segment). Step 7b-2: Eb's rate
    charges the wall's work at R2's own speed inside the segment (live_rates), and a segment with a predicted photon-escape
    switch inside it ends just past the switch (switch_cut).
    """

    def __init__(self, params):
        t = params['t_now'].value
        fb = get_current_sps_feedback(t, params)
        self.rs, self.Ms, self.u = initial_state(params, fb, params['Eb'].value, 'implicit', t)
        logger.info(f"[swept] implicit hand-off: r_s={self.rs:.6g} pc (R2={params['R2'].value:.6g}), "
                    f"M_s={self.Ms:.6g} Msun, u={self.u:.6g} pc/Myr (v2={params['v2'].value:.6g})")
        self.x_prev, self.rs_prev = params['R2'].value / self.rs, self.rs
        self.zeta, self.R2_last, self.rs_last, self.root = 1.0, None, None, None
        self.tan, self.miss_eps = None, 0.0
        self.n_tan = self.n_fallback = self.n_recouple = self.n_capped = 0
        self.s_prev, self.t_prev, self.s_esc, self.t_seg, self.Eb0 = None, None, None, None, None   # Step 7b-2
        self.n_switch = 0

    def segment_start(self, params, fb, t, Eb):
        """Mass, root, zeta and the speeds at the segment start; returns the root dict (ok=False on failure)."""
        self.Ms = max(self.Ms, swept_mass(self.rs, self.u, self.Ms, self.u <= 0.0, params)[0])
        root = root_R2(self.rs, self.Ms, fb, params, guess=self.x_prev * self.rs, Eb=Eb, phase='implicit', t=t)
        if not root['ok']:
            return root
        # Step 7b-1: how far the last segment's closure missed this root, in Step 6's energy units (step control)
        self.miss_eps = 0.0
        if self.tan is not None and Eb > 0:
            miss = root['R2'] - self._closure(self.rs, Eb, self.Ms, t)
            self.miss_eps = FOUR_PI * root['R2']**2 * params['Pb'].value * miss / Eb
        # ponytail: zeta is a backward difference (as in 1c); ill-conditioned when r_s barely moves, kept then
        if self.R2_last is not None and abs(self.rs - self.rs_last) > 1e-9 * self.rs:
            self.zeta = (root['R2'] - self.R2_last) / (self.rs - self.rs_last)
        self.R2_last, self.rs_last, self.root = root['R2'], self.rs, root
        self.fabs_ion = root['sp'].shell_fAbsorbedIon
        self.U = edge_speed(self.u, self.rs, self.fabs_ion, params)
        self.Mdot = swept_mass(self.rs, self.U, self.Ms, self.u <= 0.0, params)[1]
        # Step 7b-2: the escape indicator at this root (the previous one is kept for switch_dt)
        self.s_prev, self.t_prev = self.s_esc, self.t_seg
        self.s_esc, self.t_seg, self.Eb0 = escape_indicator(root['sp']), t, Eb
        p = tangent_partials(root['R2'], self.rs, self.Ms, Eb, fb, params, t, root['sp'])
        if p is None:
            self.tan = None
            self.n_fallback += 1
        else:
            self.tan = dict(p, R2_0=root['R2'], rs_0=self.rs, Eb_0=Eb, M_0=self.Ms, t_0=t)
            self.n_tan += 1
        # the speed the beta/delta solve sees: coupled with the previous segment's loss (recouple() refines it)
        self.Rdot2 = self.wall_speed(params, Eb, params['Lmech_total'].value, params['bubble_Lloss'].value)
        return root

    def _closure(self, r, Eb, M, t):
        """R2 inside the current segment: the tangent of the constraint, or the secant where there is none."""
        p = self.tan
        if p is None:
            return self.root['R2'] + self.zeta * (r - self.rs_last)
        return p['R2_0'] + ((r - p['rs_0']) - p['DE'] * (Eb - p['Eb_0']) - p['DM'] * (M - p['M_0'])
                            - p['DT'] * (t - p['t_0'])) / p['N']

    def wall_speed(self, params, Eb, Lgain, Lloss):
        """dR2/dt for the beta/delta solve, coupled with the energy balance it enforces (PLAN.md Step 7b-1).

        From N dR2/dt + D_E dEb/dt + D_M dM/dt + D_t = U and dEb/dt = L_gain - L_loss - 4 pi R2^2 Pb dR2/dt (L_loss with
        the leak, Pb the unramped thermal pressure, as get_betadelta's balance):
        dR2/dt = (U - D_E (L_gain - L_loss) - D_M dM/dt - D_t) / (N - D_E 4 pi R2^2 Pb). The secant's zeta U without partials.
        """
        p = self.tan
        # no partials, or no luminosities yet (the cooling keys are NaN at the 1b hand-off): the secant's speed
        if p is None or not (Lgain is not None and Lloss is not None and np.isfinite(Lgain) and np.isfinite(Lloss)):
            return self.zeta * self.U
        from trinity.phase1b_energy_implicit.get_betadelta import compute_R1_Pb
        R2 = self.root['R2']
        Pb = compute_R1_Pb(R2, Eb, params['Lmech_total'].value, params['v_mech_total'].value,
                           params['gamma_adia'].value)[1]
        v = (self.U - p['DE'] * (Lgain - Lloss) - p['DM'] * self.Mdot - p['DT']) / (p['N'] - p['DE'] * FOUR_PI * R2**2 * Pb)
        return v if np.isfinite(v) else self.zeta * self.U

    def recouple(self, params, Eb, Lgain, Lloss):
        """After the beta/delta solve: the wall speed with its own luminosities. True (and Rdot2 updated) if it moved by
        more than 1 %; the runner then solves beta/delta once more with it."""
        if self.tan is None or Lgain is None or Lloss is None or not (np.isfinite(Lgain) and np.isfinite(Lloss)):
            return False
        new = self.wall_speed(params, Eb, Lgain, Lloss)
        if abs(new - self.Rdot2) <= 0.01 * abs(self.Rdot2):
            return False
        self.Rdot2 = new
        self.n_recouple += 1
        return True

    def cap_dt(self, dt, floor):
        """Step control (PLAN.md Step 7b-1): a closure miss above TAN_MISS_EPS makes the segment about to run a fraction of the
        one that missed (t_seg - t_prev), never longer than dt nor below floor; counted only when it cuts. dt, floor [Myr].
        (2026-10-27, /xcheck: it used to scale dt, which the runner may already have grown past the missed segment.)"""
        e = abs(self.miss_eps)
        if e <= TAN_MISS_EPS or self.t_prev is None:
            return dt
        run = max(min(dt, (self.t_seg - self.t_prev) * max(0.3, 0.9 * math.sqrt(TAN_MISS_EPS / e))), floor)
        if run < dt:
            self.n_capped += 1
        return run

    def switch_cut(self, dt, floor):
        """This segment's run length (Step 7b-2): dt, or just past a photon-escape switch predicted inside it (switch_dt),
        never below floor. Unlike cap_dt it is not carried into the next segment's dt. dt, floor [Myr]."""
        run = max(switch_dt(self.s_prev, self.t_prev, self.s_esc, self.t_seg, dt), floor)
        if run < dt:
            self.n_switch += 1
            logger.info(f"[swept] segment from t={self.t_seg:.6e} Myr ends just past a predicted escape switch: runs {run:.3e} "
                        f"of {dt:.3e} Myr")
        return run

    def to_params(self, params, fb, Eb):
        """The snapshot's swept columns (v2 back to the gas speed u; cool_alpha keeps t Rdot2 / R2)."""
        _forces_to_params(params, self.root, self.rs, self.u, self.Ms, self.Mdot, self.U, self.fabs_ion, fb, Eb,
                          phase='implicit')
        if abs(self.root['L8']) > 1e-6:
            logger.warning(f"[swept] L8 = {self.root['L8']:.2e} at t={params['t_now'].value:.6e} (bar 1e-6)")

    def y0(self, Eb, T0):
        return np.array([self.rs, self.u, Eb, T0])

    def rhs(self, params, Ed, Td):
        """The segment's RHS: (r_s, u) evolve with the live drive; T0 at its frozen delta rate, Eb at its beta/delta rate with
        the wall's work live (Step 7b-2: the work charged at R2's own speed inside the segment, not the start speed)."""
        from trinity.phase1b_energy_implicit.get_betadelta import compute_R1_Pb
        K, F_rad = self.root['K'], params['F_rad'].value
        M_floor, fabs_ion = self.Ms, self.fabs_ion
        G, mCluster, gamma = params['G'].value, params['mCluster'].value, params['gamma_adia'].value
        closure, tan, zeta = self._closure, self.tan, self.zeta
        # Step 7b-2: the work rate the beta/delta solve charged, W0 = 4 pi R2_0^2 Pb_0 Rdot2_0 (unramped Pb, as its balance)
        R2_0 = self.root['R2']
        Pb_0 = compute_R1_Pb(R2_0, self.Eb0, params['Lmech_total'].value, params['v_mech_total'].value, gamma)[1]
        W0 = FOUR_PI * R2_0**2 * Pb_0 * self.Rdot2

        def f(t, y):
            r, v = max(y[0], 1e-10), y[1]
            fb_t = get_current_sps_feedback(t, params)       # live, as the c3c RHS
            U_t = edge_speed(v, r, fabs_ion, params)
            m, md = swept_mass(r, U_t, M_floor, v <= 0.0, params)
            R2t = closure(r, y[2], m, t)
            Pc, R1t = _p_conf(R2t, y[2], fb_t, params, 'implicit', t)
            F = FOUR_PI * R2t**2 * Pc + K - FOUR_PI * r**2 * p_ext(r, fabs_ion, params) + F_rad
            k_t = FOUR_PI * R2t**2 * get_bubbleParams.bubble_E2P(y[2], R2t, R1t, gamma)   # unramped Pb, as compute_R1_Pb
            _, Edot = live_rates(Ed, W0, k_t, U_t, md, tan, zeta)
            return np.array([U_t, (F - G * m / r**2 * (mCluster + 0.5 * m) - md * v) / m, Edot, Td])
        return f

    def R2_at(self, r, Eb, M, t):
        """The bubble radius the closure gives at the state (r, Eb, M, t) within the current segment."""
        return self._closure(r, Eb, M, t)

    def advance(self, params, y, r_max):
        """Take the segment's end state y = (r_s, u, ...); mass up to the farthest r_s reached. params R2 stays the
        last root (with rShell, the pair the next phase's first zeta is taken from)."""
        self.rs, self.u = float(y[0]), float(y[1])
        self.Ms = max(self.Ms, _scalar(mass_profile.get_mass_profile(float(r_max), params)))
        params['rs_dyn'].value, params['v2'].value, params['shell_mass'].value = self.rs, self.u, self.Ms
        params['R2'].value = self.root['R2']

    def moved(self):
        """The next guess, once a segment has advanced (see the momentum runner)."""
        self.x_prev = self.root['R2'] / self.rs_last

    def collapse_flags(self, params):
        """The c3c rule on (u, r_s), as Steps 4-5."""
        if self.u < 0 and self.rs < self.rs_prev:
            params['isCollapse'].value = True
        elif self.u > 0 and self.rs > self.rs_prev:
            params['isCollapse'].value = False
        self.rs_prev = self.rs

    def reconcile(self, params, t, Eb):
        """The exit row when the run ends in 1b; when 1c follows it continues this state and writes none."""
        if not params['EndSimulationDirectly'].value:
            return
        fb = get_current_sps_feedback(t, params)
        updateDict(params, fb)
        root = root_R2(self.rs, self.Ms, fb, params, guess=self.x_prev * self.rs, Eb=Eb, phase='implicit', t=t)
        if not root['ok']:
            logger.warning(f"[swept] phase-boundary reconciliation skipped: {root['why']}")
            return
        updateDict(params, root['sp'])
        self.root, self.rs_last = root, self.rs
        self.fabs_ion = root['sp'].shell_fAbsorbedIon
        self.U = edge_speed(self.u, self.rs, self.fabs_ion, params)
        self.Mdot = swept_mass(self.rs, self.U, self.Ms, self.u <= 0.0, params)[1]
        self.to_params(params, fb, Eb)
        params.save_snapshot()


def run_phase_momentum_swept(params) -> MomentumPhaseResults:
    """The momentum phase with state (r_s, u); same segment loop, events and stop rules as the c3c runner."""
    tmin, tmax = params['t_now'].value, params['stop_t'].value
    T0 = params['T0'].value
    params['Eb'].value = 0.0
    logger.info(terminal_prints.format_state(params, label="momentum phase entry (swept)"))

    fb = get_current_sps_feedback(tmin, params)
    updateDict(params, fb)
    if params['rs_dyn'].value > 0:
        # the transition ran under 'swept' (Step 5; only swept writes rs_dyn): its state goes on, and this phase's
        # first row is the transition's exit row
        rs, Ms, u = params['rs_dyn'].value, params['shell_mass'].value, params['v2'].value
        logger.info(f"[swept] continuing the transition's state: r_s={rs:.12e} pc, M_s={Ms:.12e} Msun, "
                    f"u={u:.12e} pc/Myr")
    else:
        rs, Ms, u = initial_state(params, fb)
        logger.info(f"[swept] hand-off: r_s={rs:.6g} pc (R2={params['R2'].value:.6g}), M_s={Ms:.6g} Msun, "
                    f"u={u:.6g} pc/Myr (v2={params['v2'].value:.6g})")

    # the hand-off's fixed point solved (r_s, M_s) at the entry R2, so that R2 is the first segment's root
    t_now, rs_prev, segment_count, termination_reason, x_prev = tmin, rs, 0, None, params['R2'].value / rs
    t_results, R2_results, u_results = [tmin], [params['R2'].value], [u]
    dt_segment, t_diss_onset = DT_SEGMENT_INIT, np.inf
    # D12 (docs/dev/phii-identity/weak-wind/PLAN.md): after a Stroemgren start the shell's dynamical time r_s/u is of
    # order R_St/c_i (kyr in a dense core), shorter than the default segments, and the first segment holds the
    # start's photon-escape flag; so under start_scheme 'stromgren' only, the first segment is at most STROMGREN_DT_FIRST
    # of r_s/|u| and every later one at most STROMGREN_DT_FRAC of r_s/|u| (both dimensionless fractions, not Myr), with
    # the floor lowered to match. Weaver starts: unchanged.
    stromgren = str(getattr(params.get('start_scheme'), 'value', 'weaver')).strip().lower() == 'stromgren'
    dt_floor = DT_SEGMENT_MIN
    if stromgren:
        dt_segment = min(dt_segment, STROMGREN_DT_FIRST * rs / max(abs(u), 1e-30))
        dt_floor = min(DT_SEGMENT_MIN, dt_segment)
    ode_events = build_momentum_phase_events(params)
    G, mCluster = params['G'].value, params['mCluster'].value

    while t_now <= tmax and segment_count < MAX_SEGMENTS:
        segment_count += 1
        nSnap_rCloud = params['stop_at_rCloud_nSnap'].value
        if (nSnap_rCloud is not None and rs > params['rCloud'].value
                and params['_snapshots_after_rCloud'].value >= nSnap_rCloud):
            termination_reason = _end(params, "stop_at_rCloud", SimulationEndCode.RCLOUD_BOUNDARY,
                                      f"Reached {nSnap_rCloud} segment(s) past rCloud (stop_at_rCloud_nSnap)")
            break

        params['t_now'].value, params['Eb'].value, params['T0'].value = t_now, 0.0, T0
        update_collapse_radius(params, rs)
        fb = get_current_sps_feedback(t_now, params)
        updateDict(params, fb)

        # mass at the segment start: never decreasing, frozen while u <= 0 (D-T1)
        Ms = max(Ms, swept_mass(rs, u, Ms, u <= 0.0, params)[0])
        root = root_R2(rs, Ms, fb, params, guess=x_prev * rs if x_prev else None)
        if not root['ok']:
            termination_reason = _end(params, f"swept_root_failed: {root['why']}", SimulationEndCode.ERROR_SOLVER,
                                      f"swept R2 root-find failed at t={t_now:.6e} Myr: {root['why']}")
            logger.error(params['SimulationEndReason'].value)
            break
        sp = root['sp']
        updateDict(params, sp)
        fabs_ion = sp.shell_fAbsorbedIon
        U = edge_speed(u, rs, fabs_ion, params)
        _, Mdot = swept_mass(rs, U, Ms, u <= 0.0, params)
        _forces_to_params(params, root, rs, u, Ms, Mdot, U, fabs_ion, fb)
        if abs(root['L8']) > 1e-6:
            logger.warning(f"[swept] L8 = {root['L8']:.2e} at t={t_now:.6e} (shell identity, bar 1e-6)")

        _save_count_before = params.save_count
        params.save_snapshot()
        if (params['stop_at_rCloud_nSnap'].value is not None and params.save_count > _save_count_before
                and rs > params['rCloud'].value):
            params['_snapshots_after_rCloud'].value += 1

        if tmax is not None and t_now >= tmax:
            termination_reason = _end(params, "reached_tmax", SimulationEndCode.STOPPING_TIME, 'Stopping time reached')
            break

        x, K, F_rad, M_floor = root['R2'] / rs, root['K'], params['F_rad'].value, Ms

        def rhs(t, y):
            r, v = max(y[0], 1e-10), y[1]
            fb_t = get_current_sps_feedback(t, params)       # live, as the c3c RHS (SN turn-on mid-segment)
            R2t = x * r
            F = (FOUR_PI * R2t**2 * get_bubbleParams.pRam(R2t, fb_t.Lmech_total, fb_t.v_mech_total) + K
                 - FOUR_PI * r**2 * p_ext(r, fabs_ion, params) + F_rad)
            U_t = edge_speed(v, r, fabs_ion, params)
            m, md = swept_mass(r, U_t, M_floor, v <= 0.0, params)
            return np.array([U_t, (F - G * m / r**2 * (mCluster + 0.5 * m) - md * v) / m])

        values_before = _monitor(params)
        solver_kwargs = dict(fun=rhs, t_span=(t_now, min(t_now + dt_segment, tmax)), y0=np.array([rs, u]),
                             method=ODE_METHOD, rtol=ODE_RTOL, atol=ODE_ATOL, max_step=ODE_MAX_STEP, events=ode_events)
        if ODE_METHOD == 'LSODA':
            solver_kwargs['min_step'] = ODE_MIN_STEP
        try:
            sol = scipy.integrate.solve_ivp(**solver_kwargs)
        except Exception as e:
            logger.error(f"solve_ivp failed at t={t_now:.6e}: {e}")
            termination_reason = f"solver_error: {e}"
            break
        if not sol.success or len(sol.t) == 0:
            termination_reason = f"solver_failed: {sol.message}"
            break

        event_result = check_event_termination(sol, ode_events)
        if event_result.triggered:
            logger.info(f"Event '{event_result.name}' triggered at t={event_result.t:.6e} Myr: "
                        f"r_s={event_result.y[0]:.4e} pc, u={event_result.y[1]:.4e} pc/Myr")
            termination_reason = event_result.reason_code
            rs, u, t_now = float(event_result.y[0]), float(event_result.y[1]), event_result.t
            Ms = max(Ms, _scalar(mass_profile.get_mass_profile(float(np.max(sol.y[0])), params)))
            t_results.append(t_now)
            R2_results.append(params['R2'].value)
            u_results.append(u)
            apply_event_result(params, event_result, t_now, event_result.y, state_keys=['rs_dyn', 'v2'])
            break

        rs, u, t_now = float(sol.y[0, -1]), float(sol.y[1, -1]), float(sol.t[-1])
        # the next guess only once the state has moved: a segment that breaks off (solver error) leaves the state
        # where it was, and the reconciliation must re-find the same root -- bit-identical, so save_snapshot's
        # (t_now, R2) guard drops it rather than writing a second row at the same t
        x_prev = x
        # never decreasing: everything the edge passed this segment, up to its farthest point (a turnaround
        # inside the segment would otherwise drop the gas between the start and the peak)
        Ms = max(Ms, _scalar(mass_profile.get_mass_profile(float(np.max(sol.y[0])), params)))
        params['t_now'].value, params['rs_dyn'].value, params['v2'].value = t_now, rs, u
        params['shell_mass'].value = Ms
        terminal_prints.heartbeat(params, "2 momentum (swept)", segment_count, tmin, tmax)

        max_dex_change = compute_max_dex_change(values_before, _monitor(params), MONITOR_KEYS)
        if max_dex_change > ADAPTIVE_THRESHOLD_DEX:
            dt_segment = max(dt_segment / ADAPTIVE_FACTOR, dt_floor)
        else:
            dt_segment = min(dt_segment * ADAPTIVE_FACTOR, DT_SEGMENT_MAX)
        if u < 0:
            if -u > VELOCITY_THRESHOLD_EXTREME:
                dt_segment = DT_SEGMENT_COLLAPSE
            elif -u > VELOCITY_THRESHOLD_COLLAPSE:
                dt_segment = min(dt_segment, DT_SEGMENT_MIN)
        if stromgren:                                    # D12: at most a fraction of the dynamical time
            dt_segment = min(dt_segment, STROMGREN_DT_FRAC * rs / max(abs(u), 1e-30))

        t_results.append(t_now)
        R2_results.append(params['R2'].value)
        u_results.append(u)

        # the c3c runner's rules, applied to r_s and u (PLAN 0.0.11.3)
        if u < 0 and rs < rs_prev:
            params['isCollapse'].value = True
        elif u > 0 and rs > rs_prev:
            params['isCollapse'].value = False
        rs_prev = rs
        if tmax is not None and t_now > tmax:
            termination_reason = _end(params, "reached_tmax", SimulationEndCode.STOPPING_TIME, 'Stopping time reached')
            break
        if params['isCollapse'].value and rs < update_collapse_radius(params, rs):
            termination_reason = _end(params, "small_radius", SimulationEndCode.SHELL_COLLAPSED, collapse_reason(params))
            break
        stop_r = params['stop_r'].value
        if stop_r is not None and rs > stop_r:
            termination_reason = _end(params, "large_radius", SimulationEndCode.LARGE_RADIUS, 'Large radius reached')
            break
        if params['shell_nMax'].value < params['nISM'].value:
            if t_diss_onset == np.inf:
                t_diss_onset = t_now
            if (t_now - t_diss_onset) >= params['stop_t_diss'].value:
                params['isDissolved'].value = True
                termination_reason = _end(params, "dissolved", SimulationEndCode.SHELL_DISSOLVED, 'Shell dissolved')
                break
        else:
            t_diss_onset = np.inf

    # phase-boundary reconciliation snapshot at the final (r_s, u), as the c3c runner writes one
    fb = get_current_sps_feedback(t_now, params)
    updateDict(params, fb)
    if params['isDissolved'].value:
        # a dissolved shell has no outer radius to root-find to (the solve returns the previous rShell):
        # as the c3c runner, one solve at the last R2, then the final snapshot
        R2 = params['R2'].value
        params['Pb'].value = get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total)
        params['R1'].value = R2
        params['bubble_mass'].value = get_bubbleParams.mass_freeWind(R2, fb.Lmech_total, fb.v_mech_total)
        updateDict(params, shell_structure_pure(params))
        params['rs_dyn'].value, params['v2'].value, params['shell_mass'].value = rs, u, Ms
        params.save_snapshot()
    else:
        root = root_R2(rs, Ms, fb, params, guess=x_prev * rs if x_prev else None)
        if root['ok']:
            updateDict(params, root['sp'])
            U = edge_speed(u, rs, root['sp'].shell_fAbsorbedIon, params)
            _forces_to_params(params, root, rs, u, Ms, swept_mass(rs, U, Ms, u <= 0.0, params)[1], U,
                              root['sp'].shell_fAbsorbedIon, fb)
            params.save_snapshot()
        else:
            logger.warning(f"[swept] phase-boundary reconciliation skipped: {root['why']}")

    if termination_reason is None:
        termination_reason = "max_segments" if segment_count >= MAX_SEGMENTS else "unknown"
    (logger.warning if termination_reason == "unknown" else logger.info)(
        f"Momentum phase (swept) completed: {termination_reason}; t={t_now:.6e} Myr, r_s={rs:.6e} pc, "
        f"segments {segment_count}")
    logger.info(terminal_prints.format_state(params, label="momentum phase exit (swept)"))
    return MomentumPhaseResults(t=np.array(t_results), R2=np.array(R2_results), v2=np.array(u_results),
                                termination_reason=termination_reason, final_time=t_now)


def run_phase_transition_swept(params):
    """The transition phase with state (r_s, u, Eb) (PLAN.md Step 5, registered 2026-10-25).

    The momentum runner's closure and drive with the inner pressure P_conf = max(P_th, P_ram), and the c3c
    transition's energy equation, min(L_mech - L_bubble - 4 pi R2^2 P_conf Rdot2 - L_leak, -Eb c_s / R2), with the
    bubble wall's speed Rdot2 := zeta U, zeta = Delta R2 / Delta r_s over the two latest root-finds (1 on the first
    segment; PLAN 0.0.11.3). Within a segment R2 = R2_root + zeta (r_s - r_s,root), so the volume in P_th moves at
    the Rdot2 the work term charges. After a swept implicit phase (Step 6) its state continues, and the first zeta is the
    secant across the boundary. Exits, events and stops as the c3c transition, on (r_s, u). The exit row is
    written only if the run ends here; otherwise the momentum runner continues this state and its first row is it.
    """
    from trinity.phase1c_transition import run_transition_phase as TP     # TP dispatches here: import late
    tmin, tmax = params['t_now'].value, params['stop_t'].value
    T0, gamma = params['T0'].value, params['gamma_adia'].value
    fb = get_current_sps_feedback(tmin, params)
    updateDict(params, fb)
    Eb = params['Eb'].value
    if params['rs_dyn'].value > 0:
        # 1b ran under 'swept' (Step 6): its state goes on, and the first zeta is the secant from 1b's last root
        # (params R2 and rShell hold that pair) to this phase's first
        rs, Ms, u = params['rs_dyn'].value, params['shell_mass'].value, params['v2'].value
        R2_last, rs_last = params['R2'].value, params['rShell'].value
        logger.info(f"[swept] continuing the implicit phase's state: r_s={rs:.12e} pc, M_s={Ms:.12e} Msun, "
                    f"u={u:.12e} pc/Myr")
    else:
        rs, Ms, u = initial_state(params, fb, Eb)
        R2_last = rs_last = None
        logger.info(f"[swept] transition hand-off: r_s={rs:.6g} pc (R2={params['R2'].value:.6g}), M_s={Ms:.6g} Msun, "
                    f"u={u:.6g} pc/Myr (v2={params['v2'].value:.6g})")

    t_now, rs_prev, segment_count, termination_reason, x_prev = tmin, rs, 0, None, params['R2'].value / rs
    zeta = 1.0
    t_results, R2_results, u_results, Eb_results = [tmin], [params['R2'].value], [u], [Eb]
    dt_segment, t_diss_onset = TP.DT_SEGMENT_INIT, np.inf
    ode_events = build_transition_phase_events(params, energy_floor=TP.ENERGY_FLOOR)
    G, mCluster = params['G'].value, params['mCluster'].value

    while t_now <= tmax and segment_count < TP.MAX_SEGMENTS:
        segment_count += 1
        nSnap_rCloud = params['stop_at_rCloud_nSnap'].value
        if (nSnap_rCloud is not None and rs > params['rCloud'].value
                and params['_snapshots_after_rCloud'].value >= nSnap_rCloud):
            termination_reason = _end(params, "stop_at_rCloud", SimulationEndCode.RCLOUD_BOUNDARY,
                                      f"Reached {nSnap_rCloud} segment(s) past rCloud (stop_at_rCloud_nSnap)")
            break

        params['t_now'].value, params['Eb'].value, params['T0'].value = t_now, Eb, T0
        update_collapse_radius(params, rs)
        fb = get_current_sps_feedback(t_now, params)
        updateDict(params, fb)
        T_b = params['bubble_Tavg'].value if 'bubble_Tavg' in params else None
        c_sound = operations.get_soundspeed(T_b if T_b else 1e6, params)          # as the c3c transition
        params['c_sound'].value = c_sound

        Ms = max(Ms, swept_mass(rs, u, Ms, u <= 0.0, params)[0])
        root = root_R2(rs, Ms, fb, params, guess=x_prev * rs if x_prev else None, Eb=Eb)
        if not root['ok']:
            termination_reason = _end(params, f"swept_root_failed: {root['why']}", SimulationEndCode.ERROR_SOLVER,
                                      f"swept R2 root-find failed at t={t_now:.6e} Myr: {root['why']}")
            logger.error(params['SimulationEndReason'].value)
            break
        # ponytail: zeta is a backward difference, ill-conditioned when r_s barely moves (a turnaround); there
        # the previous zeta is kept. Upgrade if Step 6 needs it: Rdot2 = Delta R2 / Delta t
        if R2_last is not None and abs(rs - rs_last) > 1e-9 * rs:
            zeta = (root['R2'] - R2_last) / (rs - rs_last)
        R2_last, rs_last = root['R2'], rs
        sp = root['sp']
        updateDict(params, sp)
        fabs_ion = sp.shell_fAbsorbedIon
        U = edge_speed(u, rs, fabs_ion, params)
        _, Mdot = swept_mass(rs, U, Ms, u <= 0.0, params)
        _forces_to_params(params, root, rs, u, Ms, Mdot, U, fabs_ion, fb, Eb)
        if abs(root['L8']) > 1e-6:
            logger.warning(f"[swept] L8 = {root['L8']:.2e} at t={t_now:.6e} (shell identity, bar 1e-6)")
        logger.debug(f"[swept] transition t={t_now:.6e}: zeta={zeta:.6g}, R2/r_s={root['R2'] / rs:.6g}")

        _save_count_before = params.save_count
        params.save_snapshot()
        if (params['stop_at_rCloud_nSnap'].value is not None and params.save_count > _save_count_before
                and rs > params['rCloud'].value):
            params['_snapshots_after_rCloud'].value += 1

        if tmax is not None and t_now >= tmax:
            termination_reason = _end(params, "reached_tmax", SimulationEndCode.STOPPING_TIME, 'Stopping time reached')
            break

        # R2 moves with r_s at the slope zeta over the segment, so the bubble volume in P_th changes at the same
        # Rdot2 = zeta U the work term charges (a held R2/r_s would grow it at (R2/r_s) U instead)
        R2a, rsa, K, F_rad, M_floor, z = root['R2'], rs, root['K'], params['F_rad'].value, Ms, zeta
        L_bubble, Cf = params['bubble_LTotal'].value, params['coverFraction'].value

        def rhs(t, y):
            r, v, E = max(y[0], 1e-10), y[1], y[2]
            fb_t = get_current_sps_feedback(t, params)       # live, as the c3c RHS
            R2t = R2a + z * (r - rsa)
            Pc, _ = _p_conf(R2t, E, fb_t, params)
            F = FOUR_PI * R2t**2 * Pc + K - FOUR_PI * r**2 * p_ext(r, fabs_ion, params) + F_rad
            U_t = edge_speed(v, r, fabs_ion, params)
            m, md = swept_mass(r, U_t, M_floor, v <= 0.0, params)
            Ed = (fb_t.Lmech_total - L_bubble - FOUR_PI * R2t**2 * Pc * z * U_t
                  - get_bubbleParams.get_leak_luminosity(Cf, R2t, Pc, c_sound, gamma))
            Ed_sc = -E / (R2t / c_sound) if c_sound > 0 else 0.0
            return np.array([U_t, (F - G * m / r**2 * (mCluster + 0.5 * m) - md * v) / m, min(Ed, Ed_sc)])

        values_before = _monitor(params)
        solver_kwargs = dict(fun=rhs, t_span=(t_now, min(t_now + dt_segment, tmax)), y0=np.array([rs, u, Eb]),
                             method=TP.ODE_METHOD, rtol=TP.ODE_RTOL, atol=TP.ODE_ATOL, max_step=TP.ODE_MAX_STEP,
                             events=ode_events)
        if TP.ODE_METHOD == 'LSODA':
            solver_kwargs['min_step'] = TP.ODE_MIN_STEP
        try:
            sol = scipy.integrate.solve_ivp(**solver_kwargs)
        except Exception as e:
            logger.error(f"solve_ivp failed at t={t_now:.6e}: {e}")
            termination_reason = f"solver_error: {e}"
            break
        if not sol.success or len(sol.t) == 0:
            termination_reason = f"solver_failed: {sol.message}"
            break

        event_result = check_event_termination(sol, ode_events)
        if event_result.triggered:
            logger.info(f"Event '{event_result.name}' triggered at t={event_result.t:.6e} Myr: r_s={event_result.y[0]:.4e} "
                        f"pc, u={event_result.y[1]:.4e} pc/Myr, Eb={event_result.y[2]:.4e}")
            termination_reason = event_result.reason_code
            rs, u, Eb, t_now = (float(event_result.y[0]), float(event_result.y[1]), float(event_result.y[2]),
                                event_result.t)
            Ms = max(Ms, _scalar(mass_profile.get_mass_profile(float(np.max(sol.y[0])), params)))
            params['shell_mass'].value = Ms
            t_results.append(t_now)
            R2_results.append(R2a + z * (rs - rsa))
            u_results.append(u)
            Eb_results.append(Eb)
            apply_event_result(params, event_result, t_now, event_result.y, state_keys=['rs_dyn', 'v2', 'Eb'])
            if event_result.name == 'velocity_sign':
                # turnaround: straight to momentum, as the c3c transition (ruled 2026-10-01)
                if Eb > TP.ENERGY_FLOOR:
                    params['Eb_handoff'].value = Eb
                    logger.warning(f"Shell turned around in 1c at t={t_now:.6e} Myr (r_s={rs:.4f} pc): Eb={Eb:.3e} "
                                   f"dropped to the floor -> momentum [velocity_sign_change].")
                Eb = TP.ENERGY_FLOOR
                params['Eb'].value = Eb
            break

        rs, u, Eb, t_now = float(sol.y[0, -1]), float(sol.y[1, -1]), float(sol.y[2, -1]), float(sol.t[-1])
        x_prev = R2a / rsa  # only once the state has moved (see the momentum runner)
        Ms = max(Ms, _scalar(mass_profile.get_mass_profile(float(np.max(sol.y[0])), params)))
        params['t_now'].value, params['rs_dyn'].value, params['v2'].value, params['Eb'].value = t_now, rs, u, Eb
        params['shell_mass'].value = Ms
        terminal_prints.heartbeat(params, "1c transition (swept)", segment_count, tmin, tmax)

        max_dex_change = compute_max_dex_change(values_before, _monitor(params), MONITOR_KEYS)
        if max_dex_change > TP.ADAPTIVE_THRESHOLD_DEX:
            dt_segment = max(dt_segment / TP.ADAPTIVE_FACTOR, TP.DT_SEGMENT_MIN)
        else:
            dt_segment = min(dt_segment * TP.ADAPTIVE_FACTOR, TP.DT_SEGMENT_MAX)
        if u < 0:
            if -u > TP.VELOCITY_THRESHOLD_EXTREME:
                dt_segment = TP.DT_SEGMENT_COLLAPSE
            elif -u > TP.VELOCITY_THRESHOLD_COLLAPSE:
                dt_segment = min(dt_segment, TP.DT_SEGMENT_MIN)

        t_results.append(t_now)
        R2_results.append(R2a + z * (rs - rsa))
        u_results.append(u)
        Eb_results.append(Eb)

        # the c3c transition's exits, in its order: ram-dominated, energy floor, then the stops
        fb_post = get_current_sps_feedback(t_now, params)
        R2e = R2a + z * (rs - rsa)
        P_th = get_bubbleParams.bubble_E2P(
            Eb, R2e, get_bubbleParams.solve_R1(R2e, Eb, fb_post.Lmech_total, fb_post.v_mech_total), gamma)
        P_ram = get_bubbleParams.pRam(R2e, fb_post.Lmech_total, fb_post.v_mech_total)
        if P_th + P_ram > 0 and P_ram / (P_th + P_ram) > 0.9:
            termination_reason = "ram_dominated"
            logger.info(f"Ram fraction {P_ram / (P_th + P_ram):.3f} > 0.9, transitioning to momentum "
                        f"(t={t_now:.4e}, Eb={Eb:.4e})")
            break
        if Eb < TP.ENERGY_FLOOR:
            termination_reason = "energy_floor"
            break
        if u < 0 and rs < rs_prev:
            params['isCollapse'].value = True
        elif u > 0 and rs > rs_prev:
            params['isCollapse'].value = False
        rs_prev = rs
        if tmax is not None and t_now > tmax:
            termination_reason = _end(params, "reached_tmax", SimulationEndCode.STOPPING_TIME, 'Stopping time reached')
            break
        if params['isCollapse'].value and rs < update_collapse_radius(params, rs):
            termination_reason = _end(params, "small_radius", SimulationEndCode.SHELL_COLLAPSED, collapse_reason(params))
            break
        stop_r = params['stop_r'].value
        if stop_r is not None and rs > stop_r:
            termination_reason = _end(params, "large_radius", SimulationEndCode.LARGE_RADIUS, 'Large radius reached')
            break
        if params['shell_nMax'].value < params['nISM'].value:
            if t_diss_onset == np.inf:
                t_diss_onset = t_now
            if (t_now - t_diss_onset) >= params['stop_t_diss'].value:
                params['isDissolved'].value = True
                termination_reason = _end(params, "dissolved", SimulationEndCode.SHELL_DISSOLVED, 'Shell dissolved')
                break
        else:
            t_diss_onset = np.inf

    if params['EndSimulationDirectly'].value:
        # the run ends in 1c: its exit row, at the final (r_s, u, Eb), as the c3c transition writes one
        fb = get_current_sps_feedback(t_now, params)
        updateDict(params, fb)
        if params['isDissolved'].value:
            # as the momentum runner: one solve at the last R2 (inner pressure P_conf), then the final row
            R2 = params['R2'].value
            params['Pb'].value, params['R1'].value = _p_conf(R2, Eb, fb, params)
            updateDict(params, shell_structure_pure(params))
            params['Pb'].value = get_bubbleParams.bubble_E2P(Eb, R2, params['R1'].value, gamma)
            params['rs_dyn'].value, params['v2'].value, params['shell_mass'].value = rs, u, Ms
            params.save_snapshot()
        else:
            root = root_R2(rs, Ms, fb, params, guess=x_prev * rs if x_prev else None, Eb=Eb)
            if root['ok']:
                updateDict(params, root['sp'])
                U = edge_speed(u, rs, root['sp'].shell_fAbsorbedIon, params)
                _forces_to_params(params, root, rs, u, Ms, swept_mass(rs, U, Ms, u <= 0.0, params)[1], U,
                                  root['sp'].shell_fAbsorbedIon, fb, Eb)
                params.save_snapshot()
            else:
                logger.warning(f"[swept] phase-boundary reconciliation skipped: {root['why']}")

    if termination_reason is None:
        termination_reason = "max_segments" if segment_count >= TP.MAX_SEGMENTS else "unknown"
    (logger.warning if termination_reason == "unknown" else logger.info)(
        f"Transition phase (swept) completed: {termination_reason}; t={t_now:.12e} Myr, r_s={rs:.12e} pc, "
        f"u={u:.12e} pc/Myr, M_s={Ms:.12e} Msun, Eb={Eb:.6e}, segments {segment_count}")
    logger.info(terminal_prints.format_state(params, label="transition phase exit (swept)"))
    return TP.TransitionPhaseResults(t=np.array(t_results), R2=np.array(R2_results), v2=np.array(u_results),
                                     Eb=np.array(Eb_results), termination_reason=termination_reason,
                                     final_time=t_now)
