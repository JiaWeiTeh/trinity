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
the isothermal jump at the gas ahead, c0^2 = P_ext / rho_amb at r_s, where 0 < c0 <= c_i; elsewhere U = u. In a
cold cloud (P_ext = 0) that is one speed exactly. Mass is swept at 4 pi r_s^2 rho_amb U and carried at u:

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


def p_ext(r, fabs_ion, params):
    """External pressure at r, as the c3c runner has it: photoionised gas if photons escape, PISM past rCloud."""
    P = _scalar(get_press_ion(r, params)) if fabs_ion < 1.0 else 0.0
    if r >= params['rCloud'].value:
        P += params['PISM'].value * params['k_B'].value
    return P


def edge_speed(u, r, fabs_ion, params):
    """The outer edge's speed U for shell-gas speed u: the isothermal jump at the gas ahead (D-T2 (b)).

    Applies where u > 0 and 0 < c0 <= c_i; elsewhere U = u. c0 = c_i exactly when photons escape inside the
    cloud (P_ext/rho = k T_ion / mu_ion_shell), so the cut carries a 1e-9 relative allowance for rounding.
    """
    if u <= 0.0:
        return u
    rho = params['mu_convert'].value * _scalar(density_profile.get_density_profile(np.array([r]), params))
    if rho <= 0.0:
        return u
    c0_sq = p_ext(r, fabs_ion, params) / rho
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


def _p_conf(R2, Eb, fb, params):
    """(P_conf, R1) at the inner face: P_ram in momentum (Eb None); max(P_th, P_ram) in transition (PLAN 0.0.11.3)."""
    if Eb is None:
        return get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total), R2
    R1 = get_bubbleParams.solve_R1(R2, Eb, fb.Lmech_total, fb.v_mech_total)
    return get_bubbleParams.get_effective_bubble_pressure('transition', Eb, R2, R1, params['gamma_adia'].value,
                                                          fb.Lmech_total, fb.v_mech_total), R1


def _shell_at(R2, rs, Ms, fb, params, Eb=None):
    """The shell solve started at R2 (inner density from P_conf(R2)) with mass Ms; r_s is only the solver's guess.

    In transition (Eb given) bubble_mass is left as the c3c transition leaves it (the implicit phase's last value).
    """
    Pc, R1 = _p_conf(R2, Eb, fb, params)
    params['R2'].value, params['R1'].value, params['Pb'].value = R2, R1, Pc
    params['shell_mass'].value = Ms
    if Eb is None:
        params['bubble_mass'].value = get_bubbleParams.mass_freeWind(R2, fb.Lmech_total, fb.v_mech_total)
    params['rShell'].value = rs
    return shell_structure_pure(params), Pc


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


def root_R2(rs, Ms, fb, params, guess=None, Eb=None):
    """R2 in (0, r_s) whose shell solve ends at r_s (PLAN 0.0.11.3). Returns a dict; ok=False with a reason.

    The residual rShell(R2) - r_s rises with R2: near R2 -> 0 the mass fits in a thin skin, at R2 -> r_s it cannot be
    held. Lower end: given `guess` (the previous segment's root, scaled to this r_s), guess/1.5; otherwise
    lo = min(1e-3 r_s, the R2 whose inner density is 100 n_St(r_s)), stepping up through 10 lo, 100 lo, 0.1 r_s,
    0.5 r_s while the shell solve rejects a very thin trial. Upper end (Step 4c, 2026-10-24): 1.5 guess, or 4x the
    lower end, stepping up by 4x until the residual is positive -- each trial still below the root becomes the lower
    end. R2 = r_s (1 - 1e-9) is only the last resort: there the whole mass sits in a sliver at r_s and the profile
    runs to millions of points (6.3M at B2's late states, each solve leaving ~9 MB of C-level memory behind). The
    residual is cached, so brentq re-solves neither end. Eb given: the transition's inner pressure (_p_conf).
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
            sp, _ = _shell_at(R2, rs, Ms, fb, params, Eb)
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
    if guess is not None and lo < guess / 1.5 < hi:
        try:
            f = resid(guess / 1.5)
            if f < 0:
                lo, flo = guess / 1.5, f
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
    up = min(1.5 * guess if (guess is not None and 1.5 * guess > lo) else 4 * lo, hi)
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
        lo, flo, up = up, fhi, min(4 * up, hi)
    n0 = count['n']
    try:
        R2 = scipy.optimize.brentq(resid, lo, up, xtol=1e-10 * rs, maxiter=100)
    except Exception as exc:  # noqa: BLE001
        return failed(f'brentq: {exc}')
    iters = count['n'] - n0
    sp, Pc = _shell_at(R2, rs, Ms, fb, params, Eb)
    count['n'] += 1
    K, F_rad_sh, P_out, L8 = layer_terms(sp, R2, Pc, params)
    return dict(ok=True, R2=R2, sp=sp, Pc=Pc, K=K, F_rad_sh=F_rad_sh, P_out=P_out, L8=L8,
                resid=(float(sp.rShell) - rs) / rs, iters=iters, nsolve=count['n'])


def initial_state(params, fb, Eb=None):
    """Hand-off (PLAN 0.0.11.3): r_s from the mass-matched shell solve at the entry R2, u conserving momentum.

    r_s is the fixed point r_s = rShell(R2, M(<r_s)) (row 60), and u = M_sh v2 / M(<r_s): the shell's momentum
    shared with the swept gas, taken at rest (a merge, as Lancaster et al.'s eq. 37 divides the momentum injected
    so far by the mass carried). Eb given: the transition's inner pressure.
    """
    R2, v2, m0 = params['R2'].value, params['v2'].value, params['shell_mass'].value
    m = m0
    for _ in range(60):
        sp, _ = _shell_at(R2, params['rShell'].value, m, fb, params, Eb)
        m_new = max(_scalar(mass_profile.get_mass_profile(float(sp.rShell), params)), m0)
        if abs(m_new - m) <= 1e-10 * m:
            break
        m_used, m = m, m_new
    else:
        m = m_used          # the mass the last solve used, so r_s and M_s stay a pair
        logger.warning(f"[swept] hand-off fixed point not converged in 60 solves (M {m:.6g} -> {m_new:.6g} Msun)")
    return float(sp.rShell), m, m0 * v2 / m


def _forces_to_params(params, root, rs, u, Ms, Mdot, U, fabs_ion, fb, Eb=None):
    """The snapshot's force and pressure columns under 'swept' (no P_HII; F_ion_in over the OUTER area).

    F_ram and P_drive are the inner face's P_conf; in transition (Eb given) Pb and P_ram keep the c3c transition's
    meanings, the bubble's thermal pressure and the free wind's ram pressure.
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
        params['P_ram'].value = get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total)
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
            dt_segment = max(dt_segment / ADAPTIVE_FACTOR, DT_SEGMENT_MIN)
        else:
            dt_segment = min(dt_segment * ADAPTIVE_FACTOR, DT_SEGMENT_MAX)
        if u < 0:
            if -u > VELOCITY_THRESHOLD_EXTREME:
                dt_segment = DT_SEGMENT_COLLAPSE
            elif -u > VELOCITY_THRESHOLD_COLLAPSE:
                dt_segment = min(dt_segment, DT_SEGMENT_MIN)

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
    the Rdot2 the work term charges. Exits, events and stops as the c3c transition, on (r_s, u). The exit row is
    written only if the run ends here; otherwise the momentum runner continues this state and its first row is it.
    """
    from trinity.phase1c_transition import run_transition_phase as TP     # TP dispatches here: import late
    tmin, tmax = params['t_now'].value, params['stop_t'].value
    T0, gamma = params['T0'].value, params['gamma_adia'].value
    fb = get_current_sps_feedback(tmin, params)
    updateDict(params, fb)
    Eb = params['Eb'].value
    rs, Ms, u = initial_state(params, fb, Eb)
    logger.info(f"[swept] transition hand-off: r_s={rs:.6g} pc (R2={params['R2'].value:.6g}), M_s={Ms:.6g} Msun, "
                f"u={u:.6g} pc/Myr (v2={params['v2'].value:.6g})")

    t_now, rs_prev, segment_count, termination_reason, x_prev = tmin, rs, 0, None, params['R2'].value / rs
    zeta, R2_last, rs_last = 1.0, None, None
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
