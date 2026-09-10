#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shell structure with dataclass returns.

This module provides shell structure calculations that return a dataclass
instead of mutating the params dictionary. This is essential for use with
adaptive ODE solvers.

Why pure (non-mutating) functions:
- shell_structure_pure() returns a ShellProperties dataclass
- No dictionary mutations during calculation
- Use updateDict(params, shell_data) after call returns

@author: Jia Wei Teh
"""

import numpy as np
import scipy.integrate

# numpy 2.0 renamed np.trapz -> np.trapezoid. The repo PINS numpy<2 (pyproject/requirements),
# so the new spelling alone crashes on the cluster env: it killed all 8100 tasks of the
# 2026-09-07 rosette sweep at 12 s (AttributeError, exit 1). Same idiom as rt3d.py / powr_fi.py.
_trapz = getattr(np, "trapezoid", None) or np.trapz
from dataclasses import dataclass
from typing import Optional, Union
import logging

from trinity.shell_structure import get_shellODE

logger = logging.getLogger(__name__)

# odeint's default internal step ceiling (mxstep=500) is exhausted in the
# degenerate code-unit-overflow regime (simple_cluster), where it emits
# "Excess work done on this call" and silently truncates the shell integration.
# Raising the ceiling silences the warning and lets the solve complete; where the
# ceiling was never hit the result is bit-identical (verified across 6 configs in
# docs/dev/shell-solver: the odeint(mxstep=50k) variant is 1.00x speed with
# rel_n=0 in the realistic regimes). Robustness fix only -- same LSODA solver.
_SHELL_ODE_MXSTEP = 50000
# The one threshold for 'photons exhausted'. Used by BOTH the loop's phiCondition
# and f_esc_ion, so the flag and the fraction cannot disagree. PLAN.md W33.
_PHI_DEPLETED_EPS = 1e-9

# ---------------------------------------------------------------------------
# Adaptive marching. PLAN.md W44 (the shipped slice was a filled-sphere Stroemgren
# radius at the profile's MINIMUM density -- an ionisation length, while the march
# terminates on MASS: p50 175x too coarse against the real thickness), W46 (so the
# exit was overshot by up to one step: m/M reached 22.5x on an archived row), W52
# (this scheme, verified on 895 archived rows: never coarser than the legacy slice,
# every field within 1.1e-3 of a converged reference, m/M 0.9994-1.0000, and 10x
# less wall time because it stops at the exit instead of integrating the discarded
# post-front density pole). Full audit + the CSVs behind every number:
# docs/dev/phii-identity/shell-structure-test/AUDIT.md
#
# Set _SHELL_ADAPTIVE = False for the legacy fixed slice; that path is bit-identical
# to the pre-port solver (verified on 895 rows), so old-vs-new stays exact. Promoting
# this to a .param switch is a one-line ParamSpec when it is wanted.
_SHELL_ADAPTIVE = True
_SHELL_MASS_MARGIN = 1.05        # slice = this x the distance to the mass exit (a CEILING, see below)
_SHELL_EFOLDS_PER_SLICE = 5.0    # ... or this many phi e-folds, whichever is smaller
_SHELL_REFINE = 1                # terminal re-integrations to land the exit inside the last interval
_SHELL_MAX_SLICES = 100_000      # loud caps; the legacy loops have neither
_SHELL_MAX_POINTS = 20_000_000
_SHELL_MIN_STEP_ULP = 8          # a slice must survive being divided into `nsteps` points
_SHELL_REFINE_MIN_ULP = 256      # ... and a refinement STEP must clear lsoda's start test
# Stop the IONISED march on its domain boundary (phi -> 0) with a terminal event rather than
# integrating through it. PLAN.md W51/W57. odeint's defaults, kept so the two paths compare.
_SHELL_PHI_EVENT = True
_SHELL_IVP_RTOL = 1.49012e-8
_SHELL_IVP_ATOL = 1.49012e-8


def _integrate_ionised(y0, r_start, r_stop, nsteps, f_cover, params):
    """Integrate the ionised system over [r_start, r_stop], stopping early at phi = eps.

    Returns (r_arr, n_arr, phi_arr, tau_arr, phi_event_fired). The grid is uniform over
    whatever span was actually integrated, so a slice truncated by the event is still
    resolved by `nsteps` points -- the resolution follows the domain, not the request.
    """
    def _rhs(r, y):
        return get_shellODE.get_shellODE(y, r, f_cover, True, params)

    def _phi_floor(r, y):
        return y[1] - _PHI_DEPLETED_EPS
    _phi_floor.terminal = True
    _phi_floor.direction = -1          # only a DOWNWARD crossing ends the ionised region

    sol = scipy.integrate.solve_ivp(
        _rhs, (r_start, r_stop), y0, method='LSODA', events=_phi_floor,
        dense_output=True, rtol=_SHELL_IVP_RTOL, atol=_SHELL_IVP_ATOL)
    if not sol.success and sol.t.size < 2:
        raise RuntimeError(f"shell_structure: ionised integration failed at r={r_start:.6g}: {sol.message}")

    fired = bool(sol.t_events and len(sol.t_events[0]) > 0)
    r_end = float(sol.t_events[0][0]) if fired else float(sol.t[-1])
    if not (r_end > r_start):
        r_end = r_start + (r_stop - r_start) * 1e-12      # degenerate: keep a positive span
        fired = False
    r_arr = np.linspace(r_start, r_end, int(nsteps))
    y = sol.sol(r_arr)
    n_arr, phi_arr, tau_arr = y[0], y[1], y[2]
    n_arr[0], phi_arr[0], tau_arr[0] = y0                 # pin the boundary condition exactly
    if fired:
        phi_arr[-1] = _PHI_DEPLETED_EPS                   # the event IS the crossing
    return r_arr, n_arr, phi_arr, tau_arr, fired


def _adaptive_slice(slice_default, y, r, m_carry, m_end, is_ionised, f_cover, params, mu, nsteps):
    """Slice width from the LOCAL scales, never wider than the legacy slice.

    Two kinds of scale, used differently:

    * mass -- the distance to sweep the REMAINING mass at the current density. Every
      dn/dr term in get_shellODE is >= 0, so n(r) is non-decreasing and the true exit
      lies at or before that distance: a slice of _SHELL_MASS_MARGIN x it therefore
      CONTAINS the exit, which is then resolved by `nsteps` points. It is a ceiling,
      never a fraction -- taking a fraction of the remaining-mass distance each slice
      is a Zeno trap that never terminates.
    * phi -- the local e-folding length of the photon budget. Spanning at most
      _SHELL_EFOLDS_PER_SLICE of them keeps >= nsteps/efolds points per e-fold, so the
      ionisation front is never jumped in a single step (the density pole lies past it).
    """
    d = get_shellODE.get_shellODE(y, r, f_cover, is_ionised, params)
    n_here = y[0]
    cands = [slice_default]
    m_rem = m_end - m_carry
    if m_rem > 0 and n_here > 0:
        cands.append(_SHELL_MASS_MARGIN * m_rem / (4 * np.pi * r**2 * mu * n_here))
    if is_ionised and d[1] < 0 and y[1] > 0:
        cands.append(_SHELL_EFOLDS_PER_SLICE * y[1] / (-d[1]))
    out = min(cands)
    # A slice is divided into `nsteps` points, so it cannot be narrower than nsteps steps
    # of a few ulp without np.arange collapsing. States exist whose remaining shell really
    # is thinner than that at this radius (a shell thinner than ~1e-13 pc at r ~ 1 pc); the
    # honest answer there is the narrowest resolvable slice, not a refusal -- the terminal
    # refinement (linspace, which stays exact to 1 ulp) then lands the exit inside it.
    floor = _SHELL_MIN_STEP_ULP * nsteps * np.spacing(r)
    if out < floor:
        out = min(slice_default, floor)
    return out


def _refine_terminal(r_lo, y_lo, m_lo, r_hi, nsteps, is_ionised, f_cover, params, mu,
                     m_end, phi_exit, refine):
    """Land the exit inside (r_lo, r_hi] by re-integrating it on a fine grid.

    The march keeps the first grid point at or past the exit, so without this the shell
    carries m_end PLUS up to one step's mass (W46). Re-integrating from the state at
    r_lo places the exit to step/nsteps**refine.

    Returns (r, n, phi, tau, m_cum, dr, mass_fired, phi_fired) at the landed exit, or
    None if the interval could not be refined -- in which case the caller keeps the
    coarse exit point, which is exactly the legacy behaviour.
    """
    ra, rb, ya, ma = r_lo, r_hi, list(y_lo), m_lo
    out = None
    for _ in range(int(refine)):
        # lsoda refuses to start when its first step is below ~100*uround*max(|t|,|tout|)
        # and prints "tout too close to t". The grid puts `nsteps` steps across this
        # interval, so testing the SPAN against a few ulp was wrong by a factor of nsteps:
        # 40 of 1555 refinements over the 895-row archive passed the old test and then
        # tripped lsoda (PLAN.md W79). np.spacing(x) is uround*x to within 2x, so 256 ulp
        # per step clears lsoda's 100-200 ulp band with margin. Nothing is lost: an
        # interval this narrow already places the exit to ~1e-15 pc.
        if (rb - ra) <= _SHELL_REFINE_MIN_ULP * nsteps * np.spacing(rb):
            break                                    # refinement step would be below lsoda's floor
        rf = np.linspace(ra, rb, int(nsteps) + 1)
        sf = scipy.integrate.odeint(get_shellODE.get_shellODE, ya, rf,
                                    args=(f_cover, is_ionised, params), mxstep=_SHELL_ODE_MXSTEP)
        nf = sf[:, 0]
        pf = sf[:, 1] if is_ionised else np.zeros_like(nf)
        tf = sf[:, 2] if is_ionised else sf[:, 1]
        # odeint can fail on a near-ulp interval and return zeros, which ARE finite.
        # Accept the refinement only as a physical continuation of the state at r_lo.
        if not (np.all(np.isfinite(sf)) and np.all(nf > 0) and nf[-1] >= nf[0] * (1 - 1e-9)):
            break
        mf = np.empty_like(rf)
        mf[0] = ma
        mf[1:] = nf[1:] * mu * 4 * np.pi * rf[1:]**2 * np.diff(rf)
        cf = np.cumsum(mf)
        hit = cf >= m_end
        if phi_exit:
            hit = hit | (pf <= _PHI_DEPLETED_EPS)
        j_arr = np.nonzero(hit)[0]
        if len(j_arr) == 0 or j_arr[0] == 0:
            break
        j = int(j_arr[0])
        out = (rf[j], nf[j], pf[j], tf[j], cf[j], rf[j] - rf[j - 1],
               bool(cf[j] >= m_end),
               bool(pf[j] <= _PHI_DEPLETED_EPS) if phi_exit else False)
        ra, rb = rf[j - 1], rf[j]
        ya = [nf[j - 1], pf[j - 1], tf[j - 1]] if is_ionised else [nf[j - 1], tf[j - 1]]
        ma = cf[j - 1]
    return out


@dataclass
class ShellProperties:
    """
    Dataclass containing all shell structure properties.

    This can be used with updateDict(params, shell_properties) to
    update the params dictionary after shell calculation completes.
    """
    # Shell density
    shell_n0: float  # Density at inner edge of shell

    # Shell geometry
    rShell: float  # Outer radius of shell
    shell_thickness: float  # Thickness of shell

    # Absorption fractions
    shell_fAbsorbedIon: float  # Fraction of ionizing radiation absorbed (gas AND dust) = 1 - f_esc
    shell_fAbsorbedIonGas: float  # LyC absorbed by GAS alone (eq:fgas_LyC). Use this, not the
                                  # total, in any recombination-balance density. PLAN.md W69.
    shell_fAbsorbedNeu: float  # Fraction of non-ionizing radiation absorbed
    shell_fAbsorbedWeightedTotal: float  # Luminosity-weighted total absorption
    shell_fIonisedDust: float  # Fraction of ionizing radiation absorbed by dust

    # Shell properties
    shell_nMax: float  # Maximum density in shell
    shell_tauKappaRatio: float  # tau_IR / kappa_IR = integral(rho dr)

    # Gravitational properties
    shell_grav_r: Union[np.ndarray, float]  # Radius array for gravity
    shell_grav_phi: float  # Gravitational potential
    shell_grav_force_m: Union[np.ndarray, float]  # Gravitational force per unit mass

    # State flags
    isDissolved: bool  # Is the shell dissolved?
    # The two exit reasons of the IONISED loop. Both are read at the termination
    # index, so both describe the SAME radius R_IF. They are not exclusive: when
    # the two conditions fire together, both are True.
    is_phiDepleted: bool  # Did the ionised integration stop because φ ran out (φ ≤ eps at R_IF)?
                          # i.e. the photon budget was spent at or before the shell's outer edge.
                          # NOT 'at the shell's outer edge': when a neutral region follows,
                          # R_IF is INSIDE the shell and the shell continues past it.
    is_allMassSwept: bool  # Did the IONISED integration terminate on the mass condition?
    has_neutral: bool  # Was a neutral region integrated beyond the ionisation front?
    diss_condition_met: bool  # Is shell_nMax < nISM this timestep?

    # Ionization front properties
    n_IF: float  # Density at ionization front from shell ODE (code units)
    n_IF_ODE: float  # Same as n_IF (raw ODE value, kept for diagnostics)
    R_IF: float  # Radius of ionization front (pc)
    n_IF_Str: float  # Strömgren ionization balance density (Lancaster+2025); gates P_HII, no longer its source (C3c)
    n_IF_Str_raw: float  # Pre-cap n_IF_Str (diagnostic only; see docs/dev/phii-identity/)

    # Shell density profile arrays (ionized + neutral)
    shell_r_arr: Union[np.ndarray, float]  # Radial grid through shell [pc]
    shell_n_arr: Union[np.ndarray, float]  # Number density through shell [1/pc^3]
    shell_ion_idx: int  # Last index of ionized region in shell_r/n_arr (-1 if empty)


def shell_structure_pure(params) -> ShellProperties:
    """
    Evaluate shell structure and return properties as a dataclass.

    This is the pure version that does NOT mutate params.
    All calculated values are returned in a ShellProperties dataclass.

    Parameters
    ----------
    params : DescribedDict
        Parameter dictionary (read-only access)

    Returns
    -------
    ShellProperties
        Dataclass containing all shell properties
    """
    # Read input parameters (no mutations)
    mBubble = params['bubble_mass'].value
    pBubble = params['Pb'].value
    rShell0 = params['R2'].value
    mShell_end = params['shell_mass'].value
    Qi = params['Qi'].value
    Li = params['Li'].value
    Ln = params['Ln'].value

    # Capture previous rShell for dissolved case (original doesn't update rShell when dissolved)
    rShell_previous = params['rShell'].value

    # TODO: Add f_cover from fragmentation mechanics
    f_cover = 1

    # Initialize values at r = rShell0 = inner edge of shell
    rShell_start = rShell0
    phi0 = 1  # Attenuation function for ionizing flux (unitless)
    tau0_ion = 0  # tau(r) at ionized region
    mShell0 = 0

    # Density at inner edge of shell
    nShell0 = (params['mu_ion_shell'].value / params['mu_convert'].value /
               (params['k_B'].value * params['TShell_ion'].value) * params['Pb'].value)
    shell_n0 = nShell0  # Store for output

    # Initialize logic gates
    is_allMassSwept = False
    has_neutral = False
    is_shellDissolved = params.get('isDissolved', False)
    if hasattr(is_shellDissolved, 'value'):
        is_shellDissolved = is_shellDissolved.value
    is_phiDepleted = False

    # Arrays for ionized region
    mShell_arr_ion = np.array([])
    mShell_arr_cum_ion = np.array([])
    phiShell_arr_ion = np.array([])
    tauShell_arr_ion = np.array([])
    nShell_arr_ion = np.array([])
    rShell_arr_ion = np.array([])

    # No ionising photons => no ionised region at all. PLAN.md W47.
    _has_ionising = (Qi > 0.0) and (Li > 0.0)

    # Maximum shell radius (for integration bounds)
    max_shellRadius = (3 * Qi / (4 * np.pi * params['chi_e_shell'].value * params['caseB_alpha'].value * nShell0**2))**(1/3) + rShell_start
    if not _has_ionising:
        # That expression is a Stroemgren radius; with Qi = 0 it collapses onto R2 and no
        # longer sizes anything. The neutral march needs some default ceiling, and the
        # adaptive mass scale narrows it to the shell anyway.
        max_shellRadius = rShell_start + 1.0

    # Integration parameters
    nsteps = 1e3
    sliceSize = np.min([1, (max_shellRadius - rShell_start) / 10])
    rShell_step = sliceSize / nsteps
    sliceSize_default = sliceSize      # the legacy slice; the adaptive one never exceeds it
    _mu_H = params['mu_convert'].value
    # Qi <= 0 collapses max_shellRadius onto R2 and the slice to zero, which used to die
    # inside np.arange with 'cannot compute length' and no physics message. PLAN.md W47.
    if _has_ionising and not (np.isfinite(sliceSize) and sliceSize > 0):
        raise ValueError(
            f"shell_structure: non-positive integration slice ({sliceSize!r}) at R2={rShell0:.6g} "
            f"with Qi={Qi:.6g}, Pb={pBubble:.6g}, Li={Li:.6g}. Qi and Li are both positive here, so "
            f"this is not the no-ionising-photon case (PLAN.md W47) — the ionised solve is genuinely "
            f"undefined at this state."
        )

    # W47. Seed the ionised arrays with the single point at the inner edge and let the loop
    # fall through: is_phiDepleted True and is_allMassSwept False give has_neutral True, so
    # the neutral branch below integrates the WHOLE shell from R2, starting at the jumped
    # (neutral, T_neu) density. Every ionised diagnostic then degenerates correctly --
    # R_IF = R2, f_esc = 0, ionised volume 0 => n_IF_Str = 0 => the P_HII gate closes.
    if not _has_ionising:
        is_phiDepleted = True
        is_allMassSwept = False
        idx = 0
        rShell_arr = np.array([rShell_start])
        nShell_arr = np.array([nShell0])
        phiShell_arr = np.array([0.0])
        tauShell_arr = np.array([0.0])
        mShell_arr = np.array([0.0])
        mShell_arr_cum = np.array([0.0])

    logger.debug(f'sliceSize={sliceSize}, max_shellRadius={max_shellRadius}, '
                 f'rShell_start={rShell_start}, rShell_step={rShell_step}')

    # =============================================================================
    # Ionized region integration
    # =============================================================================
    _n_slices = 0
    while not is_allMassSwept and not is_phiDepleted:
        logger.debug('Ionised shell loop: not is_allMassSwept and not is_phiDepleted (φ still >0)')

        _n_slices += 1
        if _n_slices > _SHELL_MAX_SLICES or _n_slices * nsteps > _SHELL_MAX_POINTS:
            raise RuntimeError(
                f"shell_structure: ionised march did not terminate after {_n_slices} slices at "
                f"r={rShell_start:.6g} (remaining mass {mShell_end - mShell0:.6g}). Neither loop "
                f"had a cap before this; see PLAN.md W44."
            )
        if _SHELL_ADAPTIVE:
            sliceSize = _adaptive_slice(sliceSize_default, [nShell0, phi0, tau0_ion],
                                        rShell_start, mShell0, mShell_end, True, f_cover,
                                        params, _mu_H, nsteps)
            rShell_step = sliceSize / nsteps
            if not (np.isfinite(sliceSize) and sliceSize > 0) or rShell_start + rShell_step == rShell_start:
                raise ValueError(
                    f"shell_structure: adaptive slice {sliceSize!r} (step {rShell_step!r}) is not "
                    f"resolvable at r={rShell_start:.6g}"
                )

        rShell_stop = rShell_start + sliceSize
        is_ionised = True
        y0 = [nShell0, phi0, tau0_ion]
        _phi_event_fired = False
        if _SHELL_PHI_EVENT:
            (rShell_arr, nShell_arr, phiShell_arr, tauShell_arr,
             _phi_event_fired) = _integrate_ionised(
                y0, rShell_start, rShell_stop, nsteps, f_cover, params)
            rShell_step = (rShell_arr[-1] - rShell_arr[0]) / max(1, rShell_arr.size - 1)
        else:
            rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)
            sol_ODE = scipy.integrate.odeint(
                get_shellODE.get_shellODE, y0, rShell_arr,
                args=(f_cover, is_ionised, params), mxstep=_SHELL_ODE_MXSTEP
            )
            nShell_arr = sol_ODE[:, 0]
            phiShell_arr = sol_ODE[:, 1]
            tauShell_arr = sol_ODE[:, 2]

        # Mass of spherical shell
        mShell_arr = np.empty_like(rShell_arr)
        mShell_arr[0] = mShell0
        # np.arange builds points as start + i*step, so consecutive spacings differ from
        # `step` in the last ulps. The interval a shell actually spans is np.diff, and on
        # the adaptive path that matters: a last-ulp drift in the cumulative mass can move
        # the termination index by one. The legacy branch keeps the constant-step form so
        # _SHELL_ADAPTIVE = False stays bit-identical to the pre-port solver.
        _dr_mass = np.diff(rShell_arr) if _SHELL_ADAPTIVE else rShell_step
        mShell_arr[1:] = (nShell_arr[1:] * params['mu_convert'].value *
                         4 * np.pi * rShell_arr[1:]**2 * _dr_mass)
        mShell_arr_cum = np.cumsum(mShell_arr)

        # Find termination index
        massCondition = mShell_arr_cum >= mShell_end
        phiCondition = phiShell_arr <= _PHI_DEPLETED_EPS
        idx_array = np.nonzero((massCondition | phiCondition))[0]

        if len(idx_array) == 0:
            idx = len(rShell_arr) - 1
        else:
            idx = idx_array[0]

        # The march keeps the first point at or past the exit; re-integrate the last
        # interval so the exit is LANDED rather than overshot. PLAN.md W46.
        # When the event fired AND phi is what terminated at idx, the exit is already exact --
        # refining it would only re-integrate the same point. Refinement still applies to a
        # MASS exit, which no event lands.
        _exit_is_exact = (_phi_event_fired and idx == len(rShell_arr) - 1
                          and phiCondition[idx] and not massCondition[idx])
        if _SHELL_ADAPTIVE and _SHELL_REFINE and len(idx_array) and idx >= 1 and not _exit_is_exact:
            _ref = _refine_terminal(
                rShell_arr[idx - 1],
                [nShell_arr[idx - 1], phiShell_arr[idx - 1], tauShell_arr[idx - 1]],
                mShell_arr_cum[idx - 1], rShell_arr[idx], nsteps, True, f_cover,
                params, _mu_H, mShell_end, True, _SHELL_REFINE)
            if _ref is not None:
                (rShell_arr[idx], nShell_arr[idx], phiShell_arr[idx], tauShell_arr[idx],
                 mShell_arr_cum[idx], _dr, _mass_fired, _phi_fired) = _ref
                mShell_arr[idx] = mShell_arr_cum[idx] - mShell_arr_cum[idx - 1]
                massCondition[idx] = _mass_fired
                phiCondition[idx] = _phi_fired

        mShell_arr_cum[idx + 1:] = 0.0
        # The shell TERMINATES at idx, so the flags must be read AT idx. `any(...)`
        # over the whole slice also sees radii PAST the termination point, which are
        # not part of the shell. Two ways that bit:
        #   mass fires at idx, phi crosses later  -> is_phiDepleted spuriously True
        #       (reporting only: it made is_phiDepleted True on rows whose f_esc was
        #        0.67, contradicting this flag's own docstring)
        #   phi fires at idx, mass crosses later  -> is_allMassSwept spuriously True,
        #       which sets has_neutral False and SKIPS the neutral-region integration
        #       entirely. That one changes the trajectory.
        # The neutral loop below is NOT affected: it has a single condition, so there
        # `any(massCondition)` and `massCondition[idx]` are identically equal.
        # docs/dev/phii-identity/PLAN.md W32.
        if len(idx_array) == 0:
            is_allMassSwept = False
            is_phiDepleted = False
        else:
            is_allMassSwept = bool(massCondition[idx])
            is_phiDepleted = bool(phiCondition[idx])

        # Store values
        mShell_arr_ion = np.concatenate((mShell_arr_ion, mShell_arr[:idx]))
        mShell_arr_cum_ion = np.concatenate((mShell_arr_cum_ion, mShell_arr_cum[:idx]))
        phiShell_arr_ion = np.concatenate((phiShell_arr_ion, phiShell_arr[:idx]))
        tauShell_arr_ion = np.concatenate((tauShell_arr_ion, tauShell_arr[:idx]))
        nShell_arr_ion = np.concatenate((nShell_arr_ion, nShell_arr[:idx]))
        rShell_arr_ion = np.concatenate((rShell_arr_ion, rShell_arr[:idx]))

        # Reinitialize for next iteration
        nShell0 = nShell_arr[idx]
        phi0 = max(0.0, phiShell_arr[idx])   # guard against sub-threshold negative phi
        tau0_ion = tauShell_arr[idx]
        mShell0 = mShell_arr_cum[idx]
        rShell_start = rShell_arr[idx]

        # Dissolution condition is now evaluated after shell structure is computed
        # (see diss_condition_met below); shell_structure_pure is stateless.

    # Append final values
    mShell_arr_ion = np.append(mShell_arr_ion, mShell_arr[idx])
    mShell_arr_cum_ion = np.append(mShell_arr_cum_ion, mShell_arr_cum[idx])
    phiShell_arr_ion = np.append(phiShell_arr_ion, phiShell_arr[idx])
    tauShell_arr_ion = np.append(tauShell_arr_ion, tauShell_arr[idx])
    nShell_arr_ion = np.append(nShell_arr_ion, nShell_arr[idx])
    rShell_arr_ion = np.append(rShell_arr_ion, rShell_arr[idx])

    # Neutral region exists only when photons are depleted AND mass remains
    has_neutral = is_phiDepleted and not is_allMassSwept

    # Extract ionization front properties
    n_IF = nShell_arr_ion[-1]  # Density at ionization front from shell ODE
    n_IF_ODE = n_IF            # Preserve raw ODE value for diagnostics
    R_IF = rShell_arr_ion[-1]  # Radius of ionization front

    # Ionizing photon escape fraction at shell outer edge.
    # Snapped to the SAME threshold the depletion test uses, so that
    #     is_phiDepleted  ==  (f_esc_ion <= _PHI_DEPLETED_EPS)
    # holds identically. Before this, a row the solver considered depleted could
    # still report f_esc = 1e-10 > 0, so a bare `f_esc > 0` test disagreed with the
    # flag. The numerical effect on Qi_abs is 1 part in 1e9; the classification
    # effect is what matters. PLAN.md W33.
    _phi_end = phiShell_arr_ion[-1]
    f_esc_ion = 0.0 if _phi_end <= _PHI_DEPLETED_EPS else float(_phi_end)

    # ------------------------------------------------------------------
    # LyC budget split, computed ONCE here (PLAN.md W69). aa61074-26.tex eq:fgas_LyC /
    # eq:fdust_LyC are the two sink terms of dphi/dr integrated over the ionised layer,
    # and eq:Qi_budget is f_gas + f_dust + f_esc = 1 (verified numerically to max 1.5e-7
    # on 895 replayed rows). Trapezoid, not the left-Riemann sum this used to be ~60
    # lines below: that one also dropped its last cell.
    # ------------------------------------------------------------------
    if rShell_arr_ion.size >= 2:
        _f_gas_ion = float(_trapz(
            4.0 * np.pi * rShell_arr_ion**2 * params['chi_e_shell'].value
            * params['caseB_alpha'].value * nShell_arr_ion**2 / Qi, rShell_arr_ion))
        _f_dust_ion = float(_trapz(
            nShell_arr_ion * params['dust_sigma'].value * phiShell_arr_ion, rShell_arr_ion))
    else:
        _f_gas_ion = 0.0          # W47: no ionised layer at all
        _f_dust_ion = 0.0
    _f_gas_ion = min(max(_f_gas_ion, 0.0), 1.0)
    _f_dust_ion = min(max(_f_dust_ion, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Strömgren ionization balance density (Lancaster+2025, generalised)
    #
    # n_IF_Str = sqrt(3 (1 - f_esc_ion) Qi / (4π χ_e αB ΔV))
    #
    # Continuous across regimes:
    #   is_phiDepleted=True  → ΔV = R_IF³ - R2³,  f_esc_ion ≈ 0
    #   is_phiDepleted=False → ΔV = R_sh³ - R2³,  f_esc_ion = phi(R_sh)
    # Cap: n_IF_Str ≤ shell_n0 (pressure equilibrium for thin skins)
    # ------------------------------------------------------------------
    # R_IF = rShell_arr_ion[-1] in both regimes (I-front or shell edge)
    _vol_ion = R_IF**3 - rShell0**3
    # W69: f_abs^gas, NOT (1 - f_esc) = f_gas + f_dust. eq:nIF_Str takes the gas term
    # alone -- it is a recombination balance, and dust does not recombine.
    _Qi_absorbed = _f_gas_ion * Qi

    if (_vol_ion > 0.0) and (_Qi_absorbed > 0.0):
        n_IF_Str = np.sqrt(
            3.0 * _Qi_absorbed /
            (4.0 * np.pi * params['chi_e_shell'].value * params['caseB_alpha'].value * _vol_ion)
        )
        n_IF_Str_raw = n_IF_Str  # kept: same value now, and the name is referenced downstream
        # ⛔ CAP REMOVED 2026-09-06 (PLAN.md W60), on the maintainer's instruction.
        # It was `n_IF_Str = min(n_IF_Str, shell_n0)` with the rationale "thin ionised skin →
        # P_HII cannot exceed P_b". It is NOT in the paper (aa61074-26.tex eq:nIF_Str has no
        # cap), and it BOUND ON 895/895 archived rows -- so it pinned n_IF_Str to shell_n0 =
        # (mu_p/mu_H) Pb/(k_B T_ion), i.e. it made the paper's P_HII exactly Pb and destroyed
        # the very Pb-decoupling eq:nIF_Str exists to provide. Uncapped the ratio is 1.08-5.98
        # (p50 1.77) on those rows.
        # Removal is provably trajectory-neutral TODAY: n_IF_Str is not read again inside this
        # function, and every downstream site is `if include_PHII and n_IF_Str > 0` -- a gate
        # whose outcome min() cannot change. The shipped P_HII is C3c, which does not read it.
        # ⛔ THAT IS EXACTLY WHY THIS MUST NOT BE LEFT HERE: the quantity is now unused, so the
        # cap question is dormant, not settled. See PLAN.md W60 / the phii-identity SSOT.
    else:
        n_IF_Str = 0.0
        n_IF_Str_raw = 0.0

    # =============================================================================
    # Continue computation if shell hasn't dissolved
    # =============================================================================
    if not is_shellDissolved:
        logger.debug('Shell not dissolved, computing gravitational potential')

        # Gravitational potential for ionized part
        grav_ion_rho = nShell_arr_ion * params['mu_convert'].value
        grav_ion_r = rShell_arr_ion
        # Under adaptive slicing the step VARIES between slices (and the terminal step is
        # shorter still), so the mass of each shell must come from its own interval. The
        # legacy branch keeps the constant-step form exactly, including its quirk of
        # giving r = R2 a full step's mass it has no preceding interval for.
        if _SHELL_ADAPTIVE:
            _dr_grav_ion = np.concatenate(([0.0], np.diff(grav_ion_r)))
        else:
            _dr_grav_ion = rShell_step
        grav_ion_m = grav_ion_rho * 4 * np.pi * grav_ion_r**2 * _dr_grav_ion
        grav_ion_m_cum = np.cumsum(grav_ion_m) + mBubble
        # simpson needs >= 3 samples; the W47 no-ionising path seeds a single inner-edge
        # point, whose ionised gravity contribution is zero by construction.
        if grav_ion_r.size >= 3:
            grav_ion_phi = -4 * np.pi * params['G'].value * scipy.integrate.simpson(
                grav_ion_r * grav_ion_rho, x=grav_ion_r
            )
        elif grav_ion_r.size == 2:
            grav_ion_phi = -4 * np.pi * params['G'].value * float(
                _trapz(grav_ion_r * grav_ion_rho, grav_ion_r))
        else:
            grav_ion_phi = 0.0
        grav_phi = grav_ion_phi
        grav_ion_force_m = params['G'].value * grav_ion_m_cum / grav_ion_r**2

        grav_force_m = grav_ion_force_m
        grav_r = grav_ion_r

        # Dust vs hydrogen absorption -- reuse the single trapezoid split computed above
        # (W69). This replaces a second, left-Riemann quadrature of the same two integrals.
        # dr_ion_arr stays: tau_kappa_IR reads it further down.
        dr_ion_arr = rShell_arr_ion[1:] - rShell_arr_ion[:-1]
        _tot_abs = _f_dust_ion + _f_gas_ion
        f_ionised_dust = (_f_dust_ion / _tot_abs) if _tot_abs > 0.0 else 0.0

        # Arrays for neutral region
        mShell_arr_neu = np.array([])
        mShell_arr_cum_neu = np.array([])
        tauShell_arr_neu = np.array([])
        nShell_arr_neu = np.array([])
        rShell_arr_neu = np.array([])
        rShell_start = rShell_arr_ion[-1]

        logger.debug('Ready to evaluate neutral shell region')

        # =============================================================================
        # Neutral region integration (if φ depleted and mass remains)
        # =============================================================================
        if has_neutral:
            logger.debug('φ depleted with mass remaining — integrating neutral region')

            # Temperature/density discontinuity at boundary
            nShell0 = (nShell0 * params['mu_atom'].value / params['mu_ion_shell'].value *
                      params['TShell_ion'].value / params['TShell_neu'].value)
            tau0_neu = tau0_ion

            tau_max = 100
            nsteps = 5e3
            sliceSize = np.min([1, (max_shellRadius - rShell_start) / 10])
            rShell_step = sliceSize / nsteps

            # The neutral loop gets its OWN termination variable. It used to reuse
            # `is_allMassSwept`, which meant the returned flag described the neutral
            # integration, not the ionised one: every row that grew a neutral region
            # came back (phi=True, mass=True, neutral=True), so the returned flags no
            # longer satisfied `has_neutral == is_phiDepleted and not is_allMassSwept`
            # and the stored column could not say where the ionised solve stopped.
            # Entry here requires has_neutral, i.e. is_allMassSwept is False, so
            # starting this at False is behaviour-identical. PLAN.md W32.
            neutral_massSwept = False
            sliceSize_default_neu = sliceSize
            if not (np.isfinite(sliceSize) and sliceSize > 0):
                raise ValueError(
                    f"shell_structure: non-positive neutral slice ({sliceSize!r}) at "
                    f"R_IF={rShell_start:.6g}; max_shellRadius does not bound the march (PLAN.md W44)."
                )
            _n_slices_neu = 0

            while not neutral_massSwept:
                logger.debug('Neutral shell loop: not neutral_massSwept')

                _n_slices_neu += 1
                if _n_slices_neu > _SHELL_MAX_SLICES or _n_slices_neu * nsteps > _SHELL_MAX_POINTS:
                    raise RuntimeError(
                        f"shell_structure: neutral march did not terminate after {_n_slices_neu} "
                        f"slices at r={rShell_start:.6g} (remaining mass {mShell_end - mShell0:.6g})."
                    )
                if _SHELL_ADAPTIVE:
                    sliceSize = _adaptive_slice(sliceSize_default_neu, [nShell0, tau0_neu],
                                                rShell_start, mShell0, mShell_end, False, f_cover,
                                                params, _mu_H, nsteps)
                    rShell_step = sliceSize / nsteps
                    if not (np.isfinite(sliceSize) and sliceSize > 0) or rShell_start + rShell_step == rShell_start:
                        raise ValueError(
                            f"shell_structure: adaptive neutral slice {sliceSize!r} is not "
                            f"resolvable at r={rShell_start:.6g}"
                        )

                rShell_stop = rShell_start + sliceSize
                rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)
                is_ionised = False

                y0 = [nShell0, tau0_neu]
                sol_ODE = scipy.integrate.odeint(
                    get_shellODE.get_shellODE, y0, rShell_arr,
                    args=(f_cover, is_ionised, params), mxstep=_SHELL_ODE_MXSTEP
                )
                nShell_arr = sol_ODE[:, 0]
                tauShell_arr = sol_ODE[:, 1]

                mShell_arr = np.empty_like(rShell_arr)
                mShell_arr[0] = mShell0
                _dr_mass = np.diff(rShell_arr) if _SHELL_ADAPTIVE else rShell_step
                mShell_arr[1:] = (nShell_arr[1:] * params['mu_convert'].value *
                                 4 * np.pi * rShell_arr[1:]**2 * _dr_mass)
                mShell_arr_cum = np.cumsum(mShell_arr)

                massCondition = mShell_arr_cum >= mShell_end
                idx_array = np.nonzero(massCondition)[0]

                if len(idx_array) == 0:
                    idx = len(rShell_arr) - 1
                else:
                    idx = idx_array[0]

                if _SHELL_ADAPTIVE and _SHELL_REFINE and len(idx_array) and idx >= 1:
                    _ref = _refine_terminal(
                        rShell_arr[idx - 1],
                        [nShell_arr[idx - 1], tauShell_arr[idx - 1]],
                        mShell_arr_cum[idx - 1], rShell_arr[idx], nsteps, False, f_cover,
                        params, _mu_H, mShell_end, False, _SHELL_REFINE)
                    if _ref is not None:
                        (rShell_arr[idx], nShell_arr[idx], _pf_unused, tauShell_arr[idx],
                         mShell_arr_cum[idx], _dr, _mass_fired, _unused) = _ref
                        mShell_arr[idx] = mShell_arr_cum[idx] - mShell_arr_cum[idx - 1]
                        massCondition[idx] = _mass_fired

                neutral_massSwept = any(massCondition)

                mShell_arr_neu = np.concatenate((mShell_arr_neu, mShell_arr[:idx]))
                mShell_arr_cum_neu = np.concatenate((mShell_arr_cum_neu, mShell_arr_cum[:idx]))
                tauShell_arr_neu = np.concatenate((tauShell_arr_neu, tauShell_arr[:idx]))
                nShell_arr_neu = np.concatenate((nShell_arr_neu, nShell_arr[:idx]))
                rShell_arr_neu = np.concatenate((rShell_arr_neu, rShell_arr[:idx]))

                nShell0 = nShell_arr[idx]
                tau0_neu = tauShell_arr[idx]
                mShell0 = mShell_arr_cum[idx]
                rShell_start = rShell_arr[idx]

            # Append final neutral values
            mShell_arr_neu = np.append(mShell_arr_neu, mShell_arr[idx])
            mShell_arr_cum_neu = np.append(mShell_arr_cum_neu, mShell_arr_cum[idx])
            tauShell_arr_neu = np.append(tauShell_arr_neu, tauShell_arr[idx])
            nShell_arr_neu = np.append(nShell_arr_neu, nShell_arr[idx])
            rShell_arr_neu = np.append(rShell_arr_neu, rShell_arr[idx])

            # Gravitational potential for neutral part
            grav_neu_rho = nShell_arr_neu * params['mu_convert'].value
            grav_neu_r = rShell_arr_neu
            if _SHELL_ADAPTIVE:
                _dr_grav_neu = np.concatenate(([0.0], np.diff(grav_neu_r)))
            else:
                _dr_grav_neu = rShell_step
            grav_neu_m = grav_neu_rho * 4 * np.pi * grav_neu_r**2 * _dr_grav_neu
            grav_neu_m_cum = np.cumsum(grav_neu_m) + grav_ion_m_cum[-1]
            grav_neu_phi = -4 * np.pi * params['G'].value * scipy.integrate.simpson(
                grav_neu_r * grav_neu_rho, x=grav_neu_r
            )
            grav_phi = grav_neu_phi + grav_ion_phi
            grav_neu_force_m = params['G'].value * grav_neu_m_cum / grav_neu_r**2

            grav_force_m = np.concatenate([grav_force_m, grav_neu_force_m])
            grav_r = np.concatenate([grav_r, grav_neu_r])

        logger.debug(f'Checking shell phiShell_arr_ion[:10]: {phiShell_arr_ion[:10]}')

        # =============================================================================
        # Compute final shell properties
        # =============================================================================
        if has_neutral:
            shellThickness = rShell_arr_neu[-1] - rShell0
            tau_rEnd = tauShell_arr_neu[-1]
            nShell_max = max(np.max(nShell_arr_ion), np.max(nShell_arr_neu))
            dr_neu_arr = rShell_arr_neu[1:] - rShell_arr_neu[:-1]
            tau_kappa_IR = (params['mu_convert'].value * np.sum(nShell_arr_ion[:-1] * dr_ion_arr) +
                params['mu_convert'].value * np.sum(nShell_arr_neu[:-1] * dr_neu_arr))
        else:
            shellThickness = rShell_arr_ion[-1] - rShell0
            tau_rEnd = tauShell_arr_ion[-1]
            nShell_max = np.max(nShell_arr_ion)
            tau_kappa_IR = params['mu_convert'].value * np.sum(nShell_arr_ion[:-1] * dr_ion_arr)

        # Absorption fractions (f_esc_ion computed above)
        f_absorbed_ion = 1.0 - f_esc_ion
        f_absorbed_neu = 1 - np.exp(-tau_rEnd)
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln) / (Li + Ln)

        rShell = grav_r[-1]

        # Combined shell density profile (ionized + neutral)
        # shell_ion_idx: last index belonging to the ionized region.
        # If shell_ion_idx == len(shell_r_arr)-1, the entire shell is ionized
        # (either is_phiDepleted with no neutral region, or all mass swept with photons leaking out).
        shell_ion_idx = len(rShell_arr_ion) - 1
        if has_neutral:
            shell_r_arr = np.concatenate([rShell_arr_ion, rShell_arr_neu])
            shell_n_arr = np.concatenate([nShell_arr_ion, nShell_arr_neu])
        else:
            shell_r_arr = rShell_arr_ion
            shell_n_arr = nShell_arr_ion

    elif is_shellDissolved:
        f_absorbed_ion = 0.0 # dissolved shell = no absorber; ionizing photons escape freely
        _f_gas_ion = 0.0     # W69: and none of them are absorbed by gas either
        f_absorbed_neu = 0.0
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln) / (Li + Ln)
        f_ionised_dust = np.nan
        is_phiDepleted = True
        has_neutral = False   # no shell, so no neutral region
        shellThickness = np.nan
        nShell_max = params['nISM'].value
        tau_kappa_IR = 0
        grav_r = np.nan
        grav_phi = np.nan
        grav_force_m = np.nan
        # Keep previous rShell value when dissolved (matches original behavior)
        rShell = rShell_previous
        # No ionization front when dissolved
        n_IF = 0.0
        n_IF_ODE = 0.0
        R_IF = 0.0
        n_IF_Str = 0.0
        n_IF_Str_raw = 0.0
        shell_r_arr = np.array([])
        shell_n_arr = np.array([])
        shell_ion_idx = -1

        logger.debug('Shell dissolved.')

    # Evaluate instantaneous dissolution condition: shell_nMax < nISM
    nISM = params['nISM'].value
    allow_dissolution = params.get('allowShellDissolution', True)
    if hasattr(allow_dissolution, 'value'):
        allow_dissolution = allow_dissolution.value
    diss_condition_met = bool(allow_dissolution and nShell_max < nISM)

    # Return dataclass with all properties
    return ShellProperties(
        shell_n0=shell_n0,
        rShell=rShell,
        shell_thickness=shellThickness,
        shell_fAbsorbedIon=f_absorbed_ion,
        shell_fAbsorbedIonGas=_f_gas_ion,
        shell_fAbsorbedNeu=f_absorbed_neu,
        shell_fAbsorbedWeightedTotal=f_absorbed,
        shell_fIonisedDust=f_ionised_dust,
        shell_nMax=nShell_max,
        shell_tauKappaRatio=tau_kappa_IR,
        shell_grav_r=grav_r,
        shell_grav_phi=grav_phi,
        shell_grav_force_m=grav_force_m,
        isDissolved=is_shellDissolved,
        is_phiDepleted=is_phiDepleted,
        is_allMassSwept=is_allMassSwept,
        has_neutral=has_neutral,
        diss_condition_met=diss_condition_met,
        n_IF=n_IF,
        n_IF_ODE=n_IF_ODE,
        R_IF=R_IF,
        n_IF_Str=n_IF_Str,
        n_IF_Str_raw=n_IF_Str_raw,
        shell_r_arr=shell_r_arr,
        shell_n_arr=shell_n_arr,
        shell_ion_idx=shell_ion_idx,
    )

