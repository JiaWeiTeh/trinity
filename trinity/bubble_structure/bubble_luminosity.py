#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble luminosity with dataclass returns.

This module provides bubble property calculations that return a dataclass
instead of mutating the params dictionary. This is essential for use with
adaptive ODE solvers.

Key difference from bubble_luminosity.py:
- get_bubbleproperties_pure() returns a BubbleProperties dataclass
- No dictionary mutations during calculation
- Use updateDict(params, bubble_data) after call returns

TODO: add docstrings for each function

@author: Jia Wei Teh
"""

import numpy as np
import os
import sys
import pickle
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional
import logging

import trinity._functions.operations as operations
import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
from trinity.cooling import net_coolingcurve
import trinity._functions.unit_conversions as cvt

logger = logging.getLogger(__name__)

# NumPy compatibility: trapz was renamed to trapezoid in NumPy 2.0
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

MIN_SPACING = 1e-12

# =============================================================================
# Gated bubble-integration diagnostic (observational only).
#
# Set the environment variable TRINITY_BUBBLE_DIAG=1 to capture every
# problematic bubble temperature profile from the L172 odeint call:
# the arrays + integration context are saved to <path2output>/bubble_diag/
# and a one-line mode classification is logged. This exists to disambiguate
# the two known triggers of the downstream `MonotonicError`
# (see analysis/bubble-integrator-robustness.md):
#   - "dead_integrator": LSODA gives up, T-profile has a zero/non-finite
#     tail at the hot (inner) end.
#   - "boundary_transient": a small, smooth dip at the T_init=3e4 (outer)
#     edge, confined to the first ~0.1% of points; the bulk is monotonic.
#
# IMPORTANT: this is purely observational. When the flag is unset the
# odeint call and all control flow are byte-identical to before; when set
# it only saves files and logs — it never alters T_array or the result.
_BUBBLE_DIAG_MAX = 100          # cap on saved events per process
_bubble_diag_count = 0


def _bubble_diag_enabled():
    """True iff the gated bubble-integration diagnostic is requested."""
    return bool(os.environ.get('TRINITY_BUBBLE_DIAG'))


def _capture_bubble_integration(params, r_array, psoln, infodict,
                                R1, Pb, initial_conditions, bubble_dMdt):
    """Save + classify a problematic bubble T-profile (gated diagnostic).

    Called only when TRINITY_BUBBLE_DIAG is set. Returns immediately unless
    the profile is non-finite, non-monotonic, or has a sub-floor tail, so a
    healthy run produces no output. Never mutates state or raises into the
    caller.
    """
    global _bubble_diag_count
    try:
        T = np.asarray(psoln[:, 1], dtype=float)
        n = T.size
        if n < 2:
            return
        finite = bool(np.isfinite(T).all())
        diffs = np.diff(T)
        negs = np.where(diffs < 0)[0]
        strictly_monotonic = (negs.size == 0
                              or bool(np.all(diffs >= 0))
                              or bool(np.all(diffs <= 0)))
        floor = 1e4
        tail = T[-max(10, n // 100):]
        tail_below_floor = int(np.sum(tail < floor))
        problem = (not finite) or (not strictly_monotonic) or (tail_below_floor > 0)
        if not problem:
            return

        cmax = np.maximum.accumulate(T)
        drawdown = (cmax - T) / np.maximum(np.abs(cmax), 1e-300)
        max_dd = float(drawdown.max())
        dd_loc = int(np.argmax(drawdown))
        last_bad = int(negs.max()) if negs.size else -1

        if (not finite) or tail_below_floor > 0:
            mode = "dead_integrator(zero/nonfinite tail)"
        elif max_dd <= 1e-2 and (last_bad < 0.01 * n):
            mode = "boundary_transient"
        else:
            mode = "bulk_nonmonotonic(possible real inversion)"

        ier = None
        message = ""
        if isinstance(infodict, dict):
            _ier = infodict.get('ier')
            ier = int(_ier) if _ier is not None else None
            message = str(infodict.get('message', ''))

        beta = params['cool_beta'].value
        delta = params['cool_delta'].value
        R2 = params['R2'].value
        Eb = params['Eb'].value
        t_now = params['t_now'].value

        logger.warning(
            f"[bubble-diag] mode={mode} ier={ier} "
            f"max_drawdown={max_dd:.3e}@frac{dd_loc / n:.4f} "
            f"last_bad_idx={last_bad}/{n} T0={T[0]:.3e} Tend={T[-1]:.3e} "
            f"dMdt={bubble_dMdt:.3e} beta={beta:.4f} delta={delta:.4f} "
            f"R2={R2:.4e} Eb={Eb:.4e} t={t_now:.4e} msg={message!r}"
        )

        if _bubble_diag_count >= _BUBBLE_DIAG_MAX:
            return
        # geometry context — lets the inspector localize an event against the
        # cloud/grid (rCloud was a red herring for the spike; kept as reference).
        geom = {}
        for _gk in ('rCloud', 'rCore'):
            try:
                geom[_gk] = float(params[_gk].value)
            except Exception:
                pass
        # full LSODA step diagnostics (present because the gated call used
        # full_output=True): info_hu (step sizes), info_mused (method),
        # info_nst (cumulative steps), info_tcur/info_tsw, scalars. These
        # confirm whether the solver hiccuped at the violation. NB: odeint's
        # infodict has no 'ier' key (success is via 'message').
        info_save = {}
        if isinstance(infodict, dict):
            for _k, _v in infodict.items():
                try:
                    info_save[f"info_{_k}"] = np.asarray(_v)
                except Exception:
                    pass
        outdir = os.path.join(params['path2output'].value, 'bubble_diag')
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f"event_{_bubble_diag_count:04d}_t{t_now:.6e}.npz")
        np.savez(
            fname,
            r=r_array, v=psoln[:, 0], T=T, dTdr=psoln[:, 2],
            ier=(ier if ier is not None else -999), message=message,
            R1=R1, Pb=Pb, R2=R2, Eb=Eb, t_now=t_now,
            beta=beta, delta=delta, bubble_dMdt=bubble_dMdt,
            initial_conditions=np.asarray(initial_conditions, dtype=float),
            mode=mode, max_drawdown=max_dd, last_bad_idx=last_bad,
            **geom, **info_save,
        )
        _bubble_diag_count += 1
        if _bubble_diag_count == _BUBBLE_DIAG_MAX:
            logger.warning(
                f"[bubble-diag] reached cap of {_BUBBLE_DIAG_MAX} saved "
                "events; suppressing further saves (logging continues)"
            )
    except Exception as e:
        logger.warning(f"[bubble-diag] capture failed (ignored): {e}")


# =============================================================================
# Gated bubble-state dump (observational only) — for the offline audit harness.
#
# Set TRINITY_BUBBLE_STATE_DUMP=<N> to pickle the first N bubble-structure call
# states to <path2output>/bubble_state/. Each dump holds every picklable param
# value (the runtime cooling cubes are skipped — they are reconstructed
# deterministically offline via read_param + get_coolingStructure), the solved
# inputs (R1, Pb, dMdt, r2Prime, initial_conditions), and the structure arrays
# so the harness can verify it reproduces this exact call. Byte-identical to
# before when unset; never mutates state or raises into the caller.
_bubble_state_dump_count = 0
_bubble_state_last_t = 0.0


def _dump_bubble_state(params, R1, Pb, bubble_dMdt, bubble_r_Tb, r2Prime,
                       initial_conditions, r_array, v_array, T_array, dTdr_array):
    """Pickle one bubble-call state for the offline correctness audit (gated)."""
    global _bubble_state_dump_count, _bubble_state_last_t
    try:
        cap = int(os.environ.get('TRINITY_BUBBLE_STATE_DUMP') or 0)
        if cap <= 0 or _bubble_state_dump_count >= cap:
            return
        # Optional log-spacing in time so dumped states span the evolution:
        # require t_now to have grown by TRINITY_BUBBLE_STATE_DT between dumps
        # (default 1.0 = no spacing = first-N behavior).
        dt_factor = float(os.environ.get('TRINITY_BUBBLE_STATE_DT') or 1.0)
        t_now = params['t_now'].value
        if _bubble_state_dump_count > 0 and t_now < _bubble_state_last_t * dt_factor:
            return
        _bubble_state_last_t = t_now
        pvals, skipped = {}, []
        for k in params.keys():
            try:
                v = params[k].value
                pickle.dumps(v)
                pvals[k] = v
            except Exception:
                skipped.append(k)
        state = {
            'param_values': pvals,
            'skipped_param_keys': skipped,
            'R1': float(R1), 'Pb': float(Pb),
            'bubble_dMdt': float(bubble_dMdt), 'bubble_r_Tb': float(bubble_r_Tb),
            'r2Prime': float(r2Prime),
            'initial_conditions': np.asarray(initial_conditions, dtype=float),
            'r_array': np.asarray(r_array, dtype=float),
            'v_array': np.asarray(v_array, dtype=float),
            'T_array': np.asarray(T_array, dtype=float),
            'dTdr_array': np.asarray(dTdr_array, dtype=float),
        }
        t_now = params['t_now'].value
        outdir = os.path.join(params['path2output'].value, 'bubble_state')
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f"state_{_bubble_state_dump_count:04d}_t{t_now:.6e}.pkl")
        with open(fname, 'wb') as fh:
            pickle.dump(state, fh)
        _bubble_state_dump_count += 1
        logger.warning(f"[bubble-state] dumped {fname} "
                       f"(skipped {len(skipped)} unpicklable params)")
    except Exception as e:
        logger.warning(f"[bubble-state] dump failed (ignored): {e}")


@dataclass
class BubbleProperties:
    """
    Dataclass containing all bubble properties.

    This can be used with updateDict(params, bubble_properties) to
    update the params dictionary after bubble calculation completes.
    """
    # Core bubble properties
    bubble_LTotal: float  # Total luminosity loss
    bubble_T_r_Tb: float  # Temperature at bubble_r_Tb
    bubble_Tavg: float  # Average temperature
    bubble_mass: float  # Bubble mass

    # Luminosity components
    bubble_L1Bubble: float  # Luminosity in bubble region (CIE)
    bubble_L2Conduction: float  # Luminosity in conduction zone (non-CIE)
    bubble_L3Intermediate: float  # Luminosity in intermediate region

    # Bubble structure arrays
    bubble_v_arr: np.ndarray  # Velocity array
    bubble_T_arr: np.ndarray  # Temperature array
    bubble_dTdr_arr: np.ndarray  # Temperature gradient array
    bubble_r_arr: np.ndarray  # Radius array
    bubble_n_arr: np.ndarray  # Density array

    # Mass flux
    bubble_dMdt: float  # Mass flux from shell into hot region

    # Derived quantities
    R1: float  # Inner bubble radius
    Pb: float  # Bubble pressure
    bubble_r_Tb: float  # Radius at T_goal


def get_bubbleproperties_pure(params) -> BubbleProperties:
    """
    Calculate bubble properties and return as a dataclass.

    This is the pure version that does NOT mutate params.
    All calculated values are returned in a BubbleProperties dataclass.

    Parameters
    ----------
    params : DescribedDict
        Parameter dictionary (read-only access)

    Returns
    -------
    BubbleProperties
        Dataclass containing all bubble properties
    """
    logger.debug('Entering get_bubbleproperties_pure')

    # =============================================================================
    # Step 1: Get necessary parameters
    # =============================================================================

    # Inner bubble radius
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * params['R2'].value, params['R2'].value,
        args=([params['Lmech_total'].value, params['Eb'].value,
               params['v_mech_total'].value, params['R2'].value])
    )

    # Bubble pressure
    Pb = get_bubbleParams.bubble_E2P(
        params['Eb'].value,
        params['R2'].value,
        R1,
        params['gamma_adia'].value
    )

    # =============================================================================
    # Step 2: Calculate dMdt
    # The mass flux from the shell back into the hot region (b, hot stellar wind)
    # if it isn't yet computed, set it via estimation from Equation 33 in Weaver+77.
    # =============================================================================

    # Get initial dMdt guess
    bubble_dMdt = params['bubble_dMdt'].value
    if np.isnan(bubble_dMdt):
        # if no guesses, begin from formula
        # if value already exist, use previous as current guess.
        bubble_dMdt = _get_init_dMdt(params, Pb)
        logger.debug(f"Initial dMdt guess: {bubble_dMdt:.3e} Msun/Myr")

    # Calculate r_Tb, i.e., radius at which we evaluate bubble temperature
    xi_Tb = params['bubble_xi_Tb'].value
    # this is relative to bubble thickness.
    bubble_r_Tb = R1 + xi_Tb * (params['R2'].value - R1)
    # sanity check: r_Tb cannot be smaller than the inner bubble radius R1
    assert bubble_r_Tb > R1, f"r_Tb ({bubble_r_Tb}) < R1 ({R1})"

    # Solve for dMdt using boundary condition
    # Create a wrapper that uses local Pb instead of params['Pb']
    def velocity_residuals_wrapper(dMdt_guess):
        return _get_velocity_residuals(dMdt_guess, params, Pb, R1)

    bubble_dMdt = scipy.optimize.fsolve(
            velocity_residuals_wrapper,
            bubble_dMdt,
            xtol=1e-4,
            factor=50,
            epsfcn=1e-4
        )[0]

    # =============================================================================
    # Step 3: Calculate bubble structure arrays
    # We note here that T_array, the temperature structure, is in increasing order.
    # This means r_array, the radius, is in DECREASING order.
    # =============================================================================

    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = _get_bubble_ODE_initial_conditions(
        bubble_dMdt, params, Pb, R1
    )

    # Create radius array and solve ODE
    # NOTE: The adaptive grid (_create_adaptive_radius_grid) is disabled because it
    # concentrates points around shock regions but under-samples the conduction zone
    # (T = 1e4 to 10^5.5 K), causing incorrect cooling calculations. The cleaned
    # legacy grid fixes the LSODA warnings while maintaining accuracy.
    #
    # TODO: To re-enable adaptive grid, ensure it maintains sufficient resolution
    # in the conduction zone, not just shock regions. The cooling calculations
    # at lines 267+ expect ~100+ points between index_cooling_switch and index_CIE_switch.
    initial_conditions = [v_r2Prime, T_r2Prime, dTdr_r2Prime]

    # Step 1 of the solve_ivp rewrite: the grid-based luminosity is extracted
    # into _bubble_luminosity_legacy (below) and retained as the runtime
    # fallback. The solve_ivp primary path + try/fallback is added next.
    return _bubble_luminosity_legacy(
        params, R1, Pb, r2Prime, initial_conditions, bubble_r_Tb, bubble_dMdt)


def _bubble_luminosity_legacy(params, R1, Pb, r2Prime, initial_conditions,
                              bubble_r_Tb, bubble_dMdt):
    """Legacy grid-based bubble luminosity, retained as a runtime fallback.

    60k-point legacy grid + odeint + find_nearest_higher region split +
    trapezoid integration (the original get_bubbleproperties_pure body). Kept
    so the solve_ivp path can fall back here on failure.
    """
    # Always use cleaned legacy grid for now (adaptive grid causes accuracy issues)
    r_array = _create_legacy_radius_grid(R1, r2Prime)
    # When the diagnostic is off this is byte-identical to the original call
    # (full_output defaults to False); when on we keep the same psoln and also
    # retain LSODA's infodict for the capture below.
    _diag = _bubble_diag_enabled()
    if _diag:
        psoln, _infodict = scipy.integrate.odeint(
            _get_bubble_ODE,
            initial_conditions,
            r_array,
            args=(params, Pb),
            tfirst=True,
            full_output=True,
        )
    else:
        psoln = scipy.integrate.odeint(
            _get_bubble_ODE,
            initial_conditions,
            r_array,
            args=(params, Pb),
            tfirst=True
        )
        _infodict = None

    if _diag:
        _capture_bubble_integration(
            params, r_array, psoln, _infodict, R1, Pb,
            initial_conditions, bubble_dMdt,
        )

    v_array = psoln[:, 0]
    T_array = psoln[:, 1]
    dTdr_array = psoln[:, 2]
    n_array = Pb / ((params['mu_convert'].value / params['mu_ion'].value) * params['k_B'].value * T_array)

    logger.debug(f'Bubble structure: r=[{r_array[0]:.4f}, {r_array[-1]:.4f}], '
                 f'T=[{T_array[0]:.2e}, {T_array[-1]:.2e}]'
                 f'n=[{n_array[0]:.2e}, {n_array[-1]:.2e}]')

    if os.environ.get('TRINITY_BUBBLE_STATE_DUMP'):
        _dump_bubble_state(params, R1, Pb, bubble_dMdt, bubble_r_Tb, r2Prime,
                           initial_conditions, r_array, v_array, T_array, dTdr_array)

    # =============================================================================
    # Step 4: Calculate cooling losses
    # The bubble will have these regions:
    #   1. Low resolution (bubble) region. This is the CIE region, where T > 10**5.5 K.
    #   2. High resolution (conduction zone) region. This is the non-CIE region.
    #   3. Intermediate region. This is between 1e4 and T[index_cooling_switch].
    # -----
    # Goal: calculate power-loss (luminosity) in these regions due to cooling. To do this
    #       for each zone we calculate T, dTdr and n. 
    # Remember r is monotonically decreasing, so temperature increases!
    # 
    # 
    # Two things to make sure in this section:
    #   1. All cooling calculations take in au values, but the inner operations and outputs are cgs.
    #      The exception is get_dudt(), which takes in au and returns in au.
    #   2. Dobule check units when using interpolation functions; some of them also only take log10 or 10^.
    # 
    # =============================================================================

    # Temperature at which any lower will have no cooling
    _coolingswitch = 1e4
    # Temperature of switching between CIE and non-CIE. Temperatures higher than 
    # _CIEswitch results in switching on CIE.
    _CIEswitch = 10**5.5

    index_CIE_switch = operations.find_nearest_higher(T_array, _CIEswitch)
    index_cooling_switch = operations.find_nearest_higher(T_array, _coolingswitch)

    if any(T_array < 0):
        logger.critical('Negative temperature detected')
        sys.exit('Negative temperature in bubble structure')

    # Process CIE/non-CIE transition
    if index_cooling_switch != index_CIE_switch:
        # extra points to add
        _xtra = 20
        # array sliced from beginning until somewhere after _CIEswitch
        r_interp = r_array[:index_CIE_switch + _xtra]
        # interpolation function for T and dTdr.
        fdTdr_interp = interp1d(r_interp, dTdr_array[:index_CIE_switch + _xtra], kind='linear')
        # subtract so that it is zero at _CIEswitch
        fT_interp = interp1d(r_interp, T_array[:index_CIE_switch + _xtra] - _CIEswitch, kind='cubic')
        fv_interp = interp1d(r_interp, v_array[:index_CIE_switch + _xtra], kind='linear')

        # calculate quantities
        r_CIEswitch = scipy.optimize.brentq(fT_interp, np.min(r_interp), np.max(r_interp), xtol=1e-8)
        n_CIEswitch = Pb / ((params['mu_convert'].value / params['mu_ion'].value) * params['k_B'].value * _CIEswitch)
        dTdr_CIEswitch = fdTdr_interp(r_CIEswitch)
        v_CIEswitch = fv_interp(r_CIEswitch)

        # insert into array
        T_array = np.insert(T_array, index_CIE_switch, _CIEswitch)
        r_array = np.insert(r_array, index_CIE_switch, r_CIEswitch)
        n_array = np.insert(n_array, index_CIE_switch, n_CIEswitch)
        dTdr_array = np.insert(dTdr_array, index_CIE_switch, dTdr_CIEswitch)
        v_array = np.insert(v_array, index_CIE_switch, v_CIEswitch)

    #---------------- 1. Bubble. Low resolution region, T > 10**5.5 K. CIE is used. 
    T_bubble = T_array[index_CIE_switch:]
    r_bubble = r_array[index_CIE_switch:]
    n_bubble = n_array[index_CIE_switch:]
    dTdr_bubble = dTdr_array[index_CIE_switch:]
    # import values from two cooling curves
    cooling_CIE = params['cStruc_cooling_CIE_interpolation'].value
    # cooling rate [au]
    Lambda_bubble = 10**(cooling_CIE(np.log10(T_bubble))) * cvt.Lambda_cgs2au

    integrand_bubble = params['chi_e'].value * n_bubble**2 * Lambda_bubble * 4 * np.pi * r_bubble**2
    # calculate power loss due to cooling
    L_bubble = np.abs(_trapezoid(integrand_bubble, x=r_bubble))
    # intermediate result for calculation of average temperature [K pc3]
    Tavg_bubble = np.abs(_trapezoid(r_bubble**2 * T_bubble, x=r_bubble))

    #---------------- 2. Conduction zone. High resolution region, 10**4 < T < 10**5.5 K. 
    L_conduction = 0.0
    Tavg_conduction = 0.0
    dTdR_coolingswitch = dTdr_bubble[0]

    if index_cooling_switch != index_CIE_switch:
        # if this zone is not well resolved, solve ODE again with high resolution (IMPROVE BY ALWAYS INTERPOLATING)
        if index_CIE_switch - index_cooling_switch < 100:
             # This is the original array that is too short
            lowres_r = r_array[:index_CIE_switch + 1]
            _highres = 1e2
             # how many intervales in high-res version? [::-1] included because r is reversed.
            r_conduction = np.arange(
                min(lowres_r), max(lowres_r),
                (max(lowres_r) - min(lowres_r)) / _highres
            )[::-1]
            # rerun structure with greater precision
            psoln_cond = scipy.integrate.odeint(
                _get_bubble_ODE,
                [v_array[index_cooling_switch], T_array[index_cooling_switch], dTdr_array[index_cooling_switch]],
                r_conduction,
                args=(params, Pb),
                tfirst=True
            )

            # Here, something needs to be done. Because of the precision of the solver, 
            # it may return temperature with values > 10**5.5K eventhough that was the maximum limit (i.e., 10**5.500001).
            # This will crash the interpolator. To fix this, we simple shave away values in the array where T > 10**5.5, 
            # and concatenate to the low-rez limit. 
            
            # Actually, the final value may not be required; the value is already included
            # in the first zone, so we don't have to worry about them here.
            
            v_cond = psoln_cond[:, 0]
            T_cond = psoln_cond[:, 1]
            dTdr_cond = psoln_cond[:, 2]

            mask = T_cond < _CIEswitch
            r_conduction = r_conduction[mask]
            T_cond = T_cond[mask]
            dTdr_cond = dTdr_cond[mask]
            dTdR_coolingswitch = dTdr_cond[0] if len(dTdr_cond) > 0 else dTdr_bubble[0]
        else:
            r_conduction = r_array[:index_CIE_switch + 1]
            T_cond = T_array[:index_CIE_switch + 1]
            dTdr_cond = dTdr_array[:index_CIE_switch + 1]
            dTdR_coolingswitch = dTdr_cond[0]
        # calculate array [au]
        n_cond = Pb / ((params['mu_convert'].value / params['mu_ion'].value) * params['k_B'].value * T_cond)
        phi_cond = params['Qi'].value / (4 * np.pi * r_conduction**2)
        # import values from two cooling curves
        cooling_nonCIE = params['cStruc_cooling_nonCIE'].value
        heating_nonCIE = params['cStruc_heating_nonCIE'].value
        # cooling rate [cgs]
        cool_cond = 10 ** cooling_nonCIE.interp(
            np.transpose(np.log10([n_cond / cvt.ndens_cgs2au, T_cond, phi_cond / cvt.phi_cgs2au]))
        )
        heat_cond = 10 ** heating_nonCIE.interp(
            np.transpose(np.log10([n_cond / cvt.ndens_cgs2au, T_cond, phi_cond / cvt.phi_cgs2au]))
        )
        # net cooling rate [au]
        dudt_cond = (heat_cond - cool_cond) * cvt.dudt_cgs2au
        # integrand [au]
        integrand_cond = dudt_cond * 4 * np.pi * r_conduction**2
        # calculate power loss due to cooling [au]
        L_conduction = np.abs(_trapezoid(integrand_cond, x=r_conduction))
        # intermediate result for calculation of average temperature
        Tavg_conduction = np.abs(_trapezoid(r_conduction**2 * T_cond, x=r_conduction))

    #---------------- 3. Region between 1e4 K and T_array[index_cooling_switch]
    # If R2_prime is very close to R2 (i.e., where T ~ 1e4K), then this region is tiny (or non-existent)
    R2_coolingswitch = (_coolingswitch - T_array[index_cooling_switch]) / dTdR_coolingswitch + r_array[index_cooling_switch]
    # interpolate between R2_prime and R2_1e4, important because the cooling function varies a lot between 1e4 and 1e5K (R2_prime is above 1e4)
    fT_interp_interm = interp1d(
        np.array([r_array[index_cooling_switch], R2_coolingswitch]),
        np.array([T_array[index_cooling_switch], _coolingswitch]),
        kind='linear'
    )

    r_interm = np.linspace(r_array[index_cooling_switch], R2_coolingswitch, num=1000, endpoint=True)
    T_interm = fT_interp_interm(r_interm)
    n_interm = Pb / ((params['mu_convert'].value / params['mu_ion'].value) * params['k_B'].value * T_interm)
    phi_interm = params['Qi'].value / (4 * np.pi * r_interm**2)
    # get cooling, taking into account for both CIE and non-CIE regimes
    L_intermediate = 0.0
    for regime in ['non-CIE', 'CIE']:
        mask = T_interm < _CIEswitch if regime == 'non-CIE' else T_interm >= _CIEswitch
        if not np.any(mask):
            continue

        if regime == 'non-CIE':
            cooling_nonCIE = params['cStruc_cooling_nonCIE'].value
            heating_nonCIE = params['cStruc_heating_nonCIE'].value
            cool_int = 10 ** cooling_nonCIE.interp(
                np.transpose(np.log10([n_interm[mask] / cvt.ndens_cgs2au, T_interm[mask], phi_interm[mask] / cvt.phi_cgs2au]))
            )
            heat_int = 10 ** heating_nonCIE.interp(
                np.transpose(np.log10([n_interm[mask] / cvt.ndens_cgs2au, T_interm[mask], phi_interm[mask] / cvt.phi_cgs2au]))
            )
            dudt_int = (heat_int - cool_int) * cvt.dudt_cgs2au
            integrand_int = dudt_int * 4 * np.pi * r_interm[mask]**2
        else:
            Lambda_int = 10**(cooling_CIE(np.log10(T_interm[mask]))) * cvt.Lambda_cgs2au
            integrand_int = params['chi_e'].value * n_interm[mask]**2 * Lambda_int * 4 * np.pi * r_interm[mask]**2
        # calculate power loss due to cooling
        L_intermediate += np.abs(_trapezoid(integrand_int, x=r_interm[mask]))

    Tavg_intermediate = np.abs(_trapezoid(r_interm**2 * T_interm, x=r_interm))

    # Total luminosity
    L_total = L_bubble + L_conduction + L_intermediate

    # Average temperature (volume-weighted)
    # <T> = ∫T dV / ∫dV = 3 × Σ(∫T r² dr) / Σ|r_outer³ - r_inner³|
    # abs(): r_bubble/r_conduction are descending slices of the grid while
    # r_interm = linspace(r2Prime, R2_coolingswitch) is ascending, so the
    # intermediate term would otherwise carry the wrong sign and (incorrectly)
    # subtract its volume. With abs() the three terms telescope to the true
    # full-domain volume R2_coolingswitch³ - R1³.
    if index_cooling_switch != index_CIE_switch:
        total_Tr2_integral = Tavg_bubble + Tavg_conduction + Tavg_intermediate
        total_volume = (abs(r_bubble[0]**3 - r_bubble[-1]**3) +
                        abs(r_conduction[0]**3 - r_conduction[-1]**3) +
                        abs(r_interm[0]**3 - r_interm[-1]**3))
        Tavg = 3 * total_Tr2_integral / total_volume
    else:
        total_Tr2_integral = Tavg_bubble + Tavg_intermediate
        total_volume = (abs(r_bubble[0]**3 - r_bubble[-1]**3) +
                        abs(r_interm[0]**3 - r_interm[-1]**3))
        Tavg = 3 * total_Tr2_integral / total_volume

    # Temperature at r_Tb
    # If rgoal is smaller than the radius of cooling threshold, i.e., larger than the index,
    if bubble_r_Tb > r_array[index_cooling_switch]:# looking for the smallest value in r_cz
        T_rgoal = fT_interp_interm(bubble_r_Tb)
    elif bubble_r_Tb > r_array[index_CIE_switch]:
        if index_cooling_switch != index_CIE_switch: # looking for the largest value in r_cz
            idx = operations.find_nearest(r_conduction, bubble_r_Tb)
            T_rgoal = T_cond[idx] + dTdr_cond[idx] * (bubble_r_Tb - r_conduction[idx])
        else:
            T_rgoal = T_bubble[0]
    # otherwise, interpolate.
    else:
        idx = operations.find_nearest(r_bubble, bubble_r_Tb)
        T_rgoal = T_bubble[idx] + dTdr_bubble[idx] * (bubble_r_Tb - r_bubble[idx])

    # =============================================================================
    # Step 5: Calculate mass
    # =============================================================================

    m_cumulative, _, _ = _get_mass_and_grav(n_array, r_array, params)
    mBubble = m_cumulative[-1]

    # Return dataclass
    return BubbleProperties(
        bubble_LTotal=L_total,
        bubble_T_r_Tb=T_rgoal,
        bubble_Tavg=Tavg,
        bubble_mass=mBubble,
        bubble_L1Bubble=L_bubble,
        bubble_L2Conduction=L_conduction,
        bubble_L3Intermediate=L_intermediate,
        bubble_v_arr=v_array,
        bubble_T_arr=T_array,
        bubble_dTdr_arr=dTdr_array,
        bubble_r_arr=r_array,
        bubble_n_arr=n_array,
        bubble_dMdt=bubble_dMdt,
        R1=R1,
        Pb=Pb,
        bubble_r_Tb=bubble_r_Tb,
    )


def _get_init_dMdt(params, Pb: float) -> float:
    """Initial guess for dMdt (Equation 33 in Weaver+77)."""
    dMdt_factor = 1.646
    R2 = params['R2'].value
    t_now = params['t_now'].value
    mu_ion = params['mu_ion'].value
    k_B = params['k_B'].value
    C_thermal = params['C_thermal'].value
    return (12 / 75 * dMdt_factor**(5/2) * 4 * np.pi * R2**3 / t_now
            * mu_ion / k_B
            * (t_now * C_thermal / R2**2)**(2/7)
            * Pb**(5/7))


def _clean_radius_grid(r_array: np.ndarray, min_relative_spacing: float = MIN_SPACING) -> np.ndarray:
    """
    Remove near-duplicate points from a radius grid.

    The np.insert() operations used to build the radius grid can create
    near-duplicate points at join boundaries (differences of ~1e-8 to 1e-9).
    With backward integration + dense output grid, LSODA hits situations
    where the next requested output radius is numerically just outside its
    valid interpolation interval [tcur - hu, tcur], causing "intdy-- t illegal"
    warnings.

    This function removes near-duplicates by enforcing a minimum relative
    spacing between consecutive points.

    Parameters
    ----------
    r_array : np.ndarray
        Radius array (typically in decreasing order for backward integration)
    min_relative_spacing : float
        Minimum allowed relative difference between consecutive points.
        Points closer than this (relative to their magnitude) are removed.
        Default is 1e-12 (MIN_SPACING).

    Returns
    -------
    np.ndarray
        Cleaned radius array with near-duplicates removed
    """
    if len(r_array) < 2:
        return r_array

    # Calculate relative differences between consecutive points
    # Use the average magnitude of consecutive points as reference
    avg_magnitude = 0.5 * (np.abs(r_array[:-1]) + np.abs(r_array[1:]))
    # Avoid division by zero for very small values
    avg_magnitude = np.maximum(avg_magnitude, 1e-30)
    relative_diff = np.abs(np.diff(r_array)) / avg_magnitude

    # Keep points that have sufficient spacing from previous point
    # First point is always kept
    keep_mask = np.concatenate([[True], relative_diff >= min_relative_spacing])

    cleaned = r_array[keep_mask]

    n_removed = len(r_array) - len(cleaned)
    if n_removed > 0:
        logger.debug(f'_clean_radius_grid: removed {n_removed} near-duplicate points '
                     f'({len(r_array)} -> {len(cleaned)})')

    return cleaned


def _create_legacy_radius_grid(R1: float, r2Prime: float) -> np.ndarray:
    """
    Create the legacy 60k-point radius grid with cleaning.

    This wraps the original grid construction logic (three np.logspace chunks
    stitched together with np.insert) and applies cleaning to remove
    near-duplicate points that cause LSODA interpolation warnings.

    Parameters
    ----------
    R1 : float
        Inner bubble radius [pc]
    r2Prime : float
        Outer integration boundary (R2 - dR2) [pc]

    Returns
    -------
    np.ndarray
        Cleaned radius array in decreasing order (for backward integration)
    """
    # Step 1: create array sampled at higher density at larger radius
    # i.e., more datapoints near bubble's outer edge (reverse logspace)
    r_array = (r2Prime + R1) - np.logspace(np.log10(R1), np.log10(r2Prime), int(2e4))

    # Step 2: add front-heavy resolution at high r
    r_improve = np.logspace(np.log10(r_array[0]), np.log10(r_array[2]), int(2e4))
    r_array = np.insert(r_array[3:], 0, r_improve)

    # Step 3: further front-heavy for end of array
    r_further = (r_array[-1] + r_array[-5]) - np.logspace(
        np.log10(r_array[-1]), np.log10(r_array[-5]), int(2e4)
    )
    r_array = np.insert(r_array[:-5], len(r_array[:-5]), r_further)

    # Clean near-duplicates to avoid LSODA interpolation warnings
    return _clean_radius_grid(r_array)


def _create_adaptive_radius_grid(R1: float, r2Prime: float,
                                  initial_conditions: list,
                                  params, Pb: float,
                                  coarse_points: int = 2000,
                                  shock_percentile: float = 85.0):
    """
    Create an adaptive radius grid concentrated around shock regions.

    Uses solve_ivp() with dense_output=True to solve ODE once, then evaluates
    on a coarse grid to find shock regions via |dT/dr|. Builds a refined grid
    concentrated only around shocks, resulting in ~3-5k points vs 60k with
    accuracy preserved at shock fronts.

    Parameters
    ----------
    R1 : float
        Inner bubble radius [pc]
    r2Prime : float
        Outer integration boundary (R2 - dR2) [pc]
    initial_conditions : list
        [v_r2Prime, T_r2Prime, dTdr_r2Prime] initial values for ODE
    params : dict
        Parameter dictionary
    Pb : float
        Bubble pressure
    coarse_points : int
        Number of points in initial coarse grid (default 2000)
    shock_percentile : float
        Percentile threshold for identifying shock regions (default 85)

    Returns
    -------
    tuple (r_array, psoln) or (None, None)
        On success: radius array and ODE solution array [v, T, dTdr]
        On failure: (None, None) - caller should fall back to legacy method
    """
    try:
        # Solve ODE with dense output using backward integration
        # r goes from r2Prime (high) down to R1 (low)
        result = scipy.integrate.solve_ivp(
            fun=lambda r, y: _get_bubble_ODE(r, y, params, Pb),
            t_span=(r2Prime, R1),
            y0=initial_conditions,
            method='LSODA',
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )

        if not result.success:
            logger.debug(f'solve_ivp failed: {result.message}')
            return None, None

        # Evaluate on coarse grid to find shock regions
        r_coarse = np.linspace(r2Prime, R1, coarse_points)
        sol_coarse = result.sol(r_coarse)
        T_coarse = sol_coarse[1, :]

        # Check for negative temperatures
        if np.any(T_coarse <= 0):
            logger.debug('Negative temperature detected in adaptive solve')
            return None, None

        # Calculate |dT/dr| to find shock regions
        dT_coarse = np.abs(np.diff(T_coarse))
        dr_coarse = np.abs(np.diff(r_coarse))
        dTdr_magnitude = dT_coarse / dr_coarse

        # Identify shock regions using percentile threshold
        shock_threshold = np.percentile(dTdr_magnitude, shock_percentile)
        shock_mask = dTdr_magnitude >= shock_threshold

        # Build refined grid: high resolution in shock regions, coarse elsewhere
        r_fine_segments = []

        # Always include boundaries
        r_fine_segments.append(np.array([r2Prime]))

        # Add points based on gradient magnitude
        for i in range(len(shock_mask)):
            r_start = r_coarse[i]
            r_end = r_coarse[i + 1]

            if shock_mask[i]:
                # High resolution in shock region (10 points per coarse interval)
                segment = np.linspace(r_start, r_end, 10, endpoint=False)
            else:
                # Low resolution in smooth region
                segment = np.array([r_start])

            r_fine_segments.append(segment)

        # Add final point
        r_fine_segments.append(np.array([R1]))

        # Concatenate and clean
        r_adaptive = np.concatenate(r_fine_segments)
        r_adaptive = np.unique(r_adaptive)[::-1]  # Ensure decreasing order
        r_adaptive = _clean_radius_grid(r_adaptive)

        # Evaluate dense output on adaptive grid
        sol_adaptive = result.sol(r_adaptive)

        # Build solution array in same format as odeint output
        psoln = np.column_stack([
            sol_adaptive[0, :],  # v
            sol_adaptive[1, :],  # T
            sol_adaptive[2, :]   # dTdr
        ])

        # Final validation
        T_adaptive = psoln[:, 1]
        if np.any(T_adaptive <= 0):
            logger.debug('Negative temperature in adaptive grid evaluation')
            return None, None

        logger.debug(f'Adaptive grid: {len(r_adaptive)} points '
                     f'(vs ~60k legacy), {np.sum(shock_mask)} shock intervals')

        return r_adaptive, psoln

    except Exception as e:
        logger.debug(f'Adaptive grid creation failed: {e}')
        return None, None


def _solve_bubble_ode_with_ivp(r_array: np.ndarray, initial_conditions: list,
                                params, Pb: float):
    """
    Alternative ODE solver using solve_ivp instead of odeint.

    Kept for future experimentation and as a reference implementation.
    Uses LSODA method with solve_ivp for potentially better error handling
    and modern interface.

    Parameters
    ----------
    r_array : np.ndarray
        Radius array at which to evaluate solution (decreasing order)
    initial_conditions : list
        [v, T, dTdr] initial values
    params : dict
        Parameter dictionary
    Pb : float
        Bubble pressure

    Returns
    -------
    np.ndarray or None
        Solution array [n_points, 3] with [v, T, dTdr] at each radius,
        or None if solver fails
    """
    try:
        # solve_ivp expects t_span to be (t_start, t_end)
        # For backward integration with decreasing r_array: t_span = (r_array[0], r_array[-1])
        result = scipy.integrate.solve_ivp(
            fun=lambda r, y: _get_bubble_ODE(r, y, params, Pb),
            t_span=(r_array[0], r_array[-1]),
            y0=initial_conditions,
            method='LSODA',
            t_eval=r_array,
            rtol=1e-8,
            atol=1e-10
        )

        if result.success:
            # Transpose to match odeint output shape [n_points, 3]
            return result.y.T
        else:
            logger.debug(f'solve_ivp failed: {result.message}')
            return None

    except Exception as e:
        logger.debug(f'solve_ivp exception: {e}')
        return None


def _get_velocity_residuals(dMdt_init, params, Pb: float, R1: float) -> float:
    """Calculate velocity residual for dMdt solver."""
    # =============================================================================
    # Get initial bubble values for integration
    # =============================================================================
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = _get_bubble_ODE_initial_conditions(
        dMdt_init, params, Pb, R1
    )

    # numpy 2.x: float(size-1 1-d array) errors, so coerce through .item()
    r2Prime_val = np.asarray(r2Prime).item()
    v_init      = np.asarray(v_r2Prime).item()
    T_init      = np.asarray(T_r2Prime).item()
    dTdr_init   = np.asarray(dTdr_r2Prime).item()

    # =============================================================================
    # radius array at which bubble structure is being evaluated.
    # =============================================================================
    # Use cleaned legacy grid (called many times during fsolve, so keep it simple)
    r_array = _create_legacy_radius_grid(R1, r2Prime_val)

    # tfirst = True because get_bubble_ODE() is defined as f(t, y).
    psoln = scipy.integrate.odeint(
        _get_bubble_ODE,
        [v_init, T_init, dTdr_init],
        r_array,
        args=(params, Pb),
        tfirst=True
    )

    v_array = psoln[:, 0]
    T_array = psoln[:, 1]

    residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)

    min_T = np.min(T_array)
    if min_T < 3e4:
        logger.debug(f'Rejected. min T: {min_T}')
        return residual * (3e4 / (min_T + 1e-1))**2

    if np.isnan(min_T):
        logger.debug('Rejected. nan temperature')
        return -1e3

    if not operations.monotonic(T_array):
        logger.debug('Temperature not monotonic')
        return 1e2

    return residual


def _get_bubble_ODE_initial_conditions(dMdt, params, Pb: float, R1: float):
    """Get initial conditions for bubble ODE (Eq 44 in Weaver+77)."""
    T_init = 3e4

    k_B = params['k_B'].value
    mu_ion = params['mu_ion'].value
    C_thermal = params['C_thermal'].value
    R2 = params['R2'].value

    constant = (25/4 * k_B / mu_ion / C_thermal)
    dR2 = T_init**(5/2) / (constant * dMdt / (4 * np.pi * R2**2))

    T = (constant * dMdt * dR2 / (4 * np.pi * R2**2))**(2/5)
    v = (params['cool_alpha'].value * R2 / params['t_now'].value
         - dMdt / (4 * np.pi * R2**2)
         * k_B * T / mu_ion / Pb)
    dTdr = -2/5 * T / dR2
    r2_prime = R2 - dR2

    return r2_prime, T, dTdr, v


def _get_bubble_ODE(r_arr, initial_ODEs, params, Pb: float):
    """Bubble structure ODE (Equations 42-43 in Weaver+77)."""
    v, T, dTdr = initial_ODEs

    if np.abs(T - 0) < 1e-5:
        logger.critical('T is zero in bubble ODE')
        sys.exit()

    ndens = Pb / ((params['mu_convert'].value / params['mu_ion'].value) * params['k_B'].value * T)
    phi = params['Qi'].value / (4 * np.pi * r_arr**2)

    dudt = net_coolingcurve.get_dudt(params['t_now'].value, ndens, T, phi, params)

    v_term = params['cool_alpha'].value * r_arr / params['t_now'].value

    dTdrr = (Pb / (params['C_thermal'].value * T**(5/2)) * (
        (params['cool_beta'].value + 2.5 * params['cool_delta'].value) / params['t_now'].value
        + 2.5 * (v - v_term) * dTdr / T - dudt / Pb
    ) - 2.5 * dTdr**2 / T - 2 * dTdr / r_arr)

    dvdr = ((params['cool_beta'].value + params['cool_delta'].value) / params['t_now'].value
            + (v - v_term) * dTdr / T - 2 * v / r_arr)

    return [dvdr, dTdr, dTdrr]


def _get_mass_and_grav(n, r, params):
    """Calculate cumulative mass and gravitational potential."""
    # Flip arrays to be monotonically increasing
    r_new = r[::-1]
    rho_new = n[::-1] * params['mu_convert'].value # Mass density [Msun/pc³] (rho = mu_H * n_H)

    # Calculate cumulative mass using O(n) cumulative integration
    # instead of O(n²) loop with simps
    # M(r) = ∫[0 to r] 4πr'² ρ(r') dr'
    integrand = rho_new * r_new**2
    m_cumulative = 4 * np.pi * scipy.integrate.cumulative_trapezoid(
        integrand, x=r_new, initial=0
    )
    # Gravitational potential [pc²/Myr²]
    grav_phi = -4 * np.pi * params['G'].value * scipy.integrate.simpson(
        r_new * rho_new, x=r_new
    )
    # Gravitational force per unit mass [pc/Myr²]
    # Add small number to avoid division by zero at r=0
    grav_force_m = params['G'].value * m_cumulative / (r_new**2 + 1e-10)

    return m_cumulative, grav_phi, grav_force_m
