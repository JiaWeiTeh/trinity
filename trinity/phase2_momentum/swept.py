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
    apply_event_result, build_momentum_phase_events, check_event_termination, collapse_reason,
    update_collapse_radius,
)
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


def _shell_at(R2, rs, Ms, fb, params):
    """The shell solve started at R2 (inner density from P_ram(R2)) with mass Ms; r_s is only the solver's guess."""
    Pc = get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total)
    params['R2'].value, params['R1'].value, params['Pb'].value = R2, R2, Pc
    params['shell_mass'].value = Ms
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


def root_R2(rs, Ms, fb, params):
    """R2 in (0, r_s) whose shell solve ends at r_s (PLAN 0.0.11.3). Returns a dict; ok=False with a reason.

    Bracket [lo, r_s (1 - 1e-9)], lo = min(1e-3 r_s, the R2 whose inner density is 100 n_St(r_s)): at lo the mass
    fits in a thin skin, at the top it cannot be held, so the residual rShell(R2) - r_s changes sign. A very thin
    trial can be rejected by the shell solve, so lo steps up through 10 lo, 100 lo, 0.1 r_s, 0.5 r_s.
    """
    g = lambda k: params[k].value  # noqa: E731
    count = {'n': 0}
    saved = {k: params[k].value for k in _TRIAL_KEYS}

    def failed(why):
        for k, v in saved.items():          # the trial solves leave their last R2, Pb, ... in params
            params[k].value = v
        return dict(ok=False, why=why, nsolve=count['n'])

    def resid(R2):
        count['n'] += 1
        sp, _ = _shell_at(R2, rs, Ms, fb, params)
        return float(sp.rShell) - rs

    lo = 1e-3 * rs
    pdot = 2.0 * fb.Lmech_total / fb.v_mech_total if fb.v_mech_total > 0 else 0.0
    n_st = math.sqrt(3 * fb.Qi / (FOUR_PI * g('chi_e_shell') * g('caseB_alpha') * rs**3)) if fb.Qi > 0 else 0.0
    if pdot > 0 and n_st > 0:
        p_ion = g('mu_convert') / g('mu_ion_shell') * g('k_B') * g('TShell_ion')
        lo = min(lo, math.sqrt(pdot / (FOUR_PI * p_ion * 100.0 * n_st)))
    hi = rs * (1 - 1e-9)
    flo, why = None, ''
    for lo_try in (lo, 10 * lo, 100 * lo, 0.1 * rs, 0.5 * rs):
        try:
            flo = resid(lo_try)
            lo = lo_try
            break
        except Exception as exc:  # noqa: BLE001 -- a trial the shell solve rejects; the next one is thicker
            why = f'{type(exc).__name__}: {exc}'
    if flo is None:
        return failed(f'lower bracket solve failed: {why}')
    try:
        fhi = resid(hi)
    except Exception as exc:  # noqa: BLE001
        return failed(f'upper bracket solve failed: {type(exc).__name__}: {exc}')
    if not (flo < 0 < fhi):
        return failed(f'no bracket: f(lo)={flo:.3e} f(hi)={fhi:.3e}')
    n0 = count['n']
    try:
        R2 = scipy.optimize.brentq(resid, lo, hi, xtol=1e-10 * rs, maxiter=100)
    except Exception as exc:  # noqa: BLE001
        return failed(f'brentq: {exc}')
    iters = count['n'] - n0
    sp, Pc = _shell_at(R2, rs, Ms, fb, params)
    count['n'] += 1
    K, F_rad_sh, P_out, L8 = layer_terms(sp, R2, Pc, params)
    return dict(ok=True, R2=R2, sp=sp, Pc=Pc, K=K, F_rad_sh=F_rad_sh, P_out=P_out, L8=L8,
                resid=(float(sp.rShell) - rs) / rs, iters=iters, nsolve=count['n'])


def initial_state(params, fb):
    """Hand-off (PLAN 0.0.11.3): r_s from the mass-matched shell solve at the entry R2, u conserving momentum.

    r_s is the fixed point r_s = rShell(R2, M(<r_s)) (row 60), and u = M_sh v2 / M(<r_s): the shell's momentum
    shared with the swept gas, taken at rest (a merge, as Lancaster et al.'s eq. 37 divides the momentum injected
    so far by the mass carried).
    """
    R2, v2, m0 = params['R2'].value, params['v2'].value, params['shell_mass'].value
    m = m0
    for _ in range(60):
        sp, _ = _shell_at(R2, params['rShell'].value, m, fb, params)
        m_new = max(_scalar(mass_profile.get_mass_profile(float(sp.rShell), params)), m0)
        if abs(m_new - m) <= 1e-10 * m:
            break
        m_used, m = m, m_new
    else:
        m = m_used          # the mass the last solve used, so r_s and M_s stay a pair
        logger.warning(f"[swept] hand-off fixed point not converged in 60 solves (M {m:.6g} -> {m_new:.6g} Msun)")
    return float(sp.rShell), m, m0 * v2 / m


def _forces_to_params(params, root, rs, u, Ms, Mdot, U, fabs_ion, fb):
    """The snapshot's force and pressure columns under 'swept' (no P_HII; F_ion_in over the OUTER area)."""
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
    rs, Ms, u = initial_state(params, fb)
    logger.info(f"[swept] hand-off: r_s={rs:.6g} pc (R2={params['R2'].value:.6g}), M_s={Ms:.6g} Msun, "
                f"u={u:.6g} pc/Myr (v2={params['v2'].value:.6g})")

    t_now, rs_prev, segment_count, termination_reason = tmin, rs, 0, None
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
        root = root_R2(rs, Ms, fb, params)
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
        root = root_R2(rs, Ms, fb, params)
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
