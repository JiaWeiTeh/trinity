#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bubble_luminosity.py

Compute hot-bubble structure and radiative losses for a wind-driven bubble.

This module is used as a *diagnostic* step during TRINITY evolution. It updates
the `params` container with:

- Inner wind-shock radius:        R1  [pc]
- Bubble pressure:               Pb  [Msun / (pc * Myr^2)]
- Conductive/evaporative mass flux from shell into bubble:
                                 bubble_dMdt  [Msun / Myr]
- Bubble structure arrays:
    bubble_r_arr     [pc]    (decreasing: near R2 -> near R1)
    bubble_T_arr     [K]     (increasing inward)
    bubble_dTdr_arr  [K/pc]
    bubble_n_arr     [1/pc^3]   (consistent with Pb = 2 n k_B T)
    bubble_v_arr     [pc/Myr]   (from continuity with dMdt)
- Cooling luminosities (positive):
    bubble_L1Bubble       (T >= T_CIEswitch)
    bubble_L2Conduction   (T_goal <= T < T_CIEswitch)
    bubble_L3Intermediate (T_min <= T < T_goal)  [analytic extension]
    bubble_LTotal
- Volume-weighted average temperature and hot-gas mass:
    bubble_Tavg, bubble_mass

Physics model (explicit, minimal)
---------------------------------
Assume the hot region (between R1 and R2) is approximately isobaric at Pb.

Equation of state convention used here:
    Pb = 2 n k_B T

Conductive evaporation mass flux dMdt is assumed constant.

Energy equation in quasi-steady flow with conduction and radiative term dudt:
    1/r^2 d/dr [ r^2 ( (5/2) Pb v + q ) ] = - dudt

Spitzer-like conduction:
    q = - C_thermal T^{5/2} dT/dr

This yields a 2nd order ODE in T(r) which we solve as a first-order system in [T, dT/dr].
Radiative term `dudt` is obtained from net_coolingcurve.get_dudt which returns negative
values for net cooling (TRINITY convention).

If you need Weaver self-similar (alpha/beta/delta) dynamics, implement it as a separate
model and keep this file strictly diagnostic.
"""

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Tuple

import math
import numpy as np
import scipy.optimize
from scipy.integrate import solve_ivp


# -----------------------------------------------------------------------------
# Optional imports (match your repository layout)
# -----------------------------------------------------------------------------
try:
    from src.cooling import net_coolingcurve
except Exception:  # pragma: no cover
    import net_coolingcurve  # type: ignore

try:
    from src._input.dictionary import DescribedItem  # type: ignore
except Exception:  # pragma: no cover
    try:
        from dictionary import DescribedItem  # type: ignore
    except Exception:  # pragma: no cover
        DescribedItem = None  # type: ignore


# -----------------------------------------------------------------------------
# Small helpers to work with DescribedDict / plain dict
# -----------------------------------------------------------------------------
def _get(params: Mapping[str, Any], key: str) -> Any:
    """Return params[key].value if present, else params[key]."""
    obj = params[key]
    return getattr(obj, "value", obj)


def _set(
    params: MutableMapping[str, Any],
    key: str,
    value: Any,
    info: Optional[str] = None,
    ori_units: Optional[str] = None,
    exclude_from_snapshot: bool = False,
) -> None:
    """
    Set params[key].value if key exists and has .value, else overwrite/insert.

    If DescribedItem is available and key does not exist, we create a DescribedItem.
    """
    if key in params and hasattr(params[key], "value"):
        params[key].value = value
        return

    if DescribedItem is not None:
        params[key] = DescribedItem(
            value=value,
            info=info,
            ori_units=ori_units,
            exclude_from_snapshot=exclude_from_snapshot,
        )
    else:
        params[key] = value


def _safe_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception as e:  # pragma: no cover
        raise TypeError(f"Parameter '{name}' must be convertible to float, got {type(x)}") from e


def _ensure_positive(x: float, name: str, floor: float = 0.0) -> float:
    if not np.isfinite(x):
        raise ValueError(f"Parameter '{name}' must be finite, got {x}")
    if x <= floor:
        raise ValueError(f"Parameter '{name}' must be > {floor}, got {x}")
    return x


# -----------------------------------------------------------------------------
# Wind termination shock + bubble pressure closure
# -----------------------------------------------------------------------------
def bubble_pressure_from_energy(Eb: float, R2: float, R1: float, gamma: float) -> float:
    """
    Pb = (gamma - 1) Eb / V,  V = 4π/3 (R2^3 - R1^3)
    """
    V = (4.0 * math.pi / 3.0) * (R2**3 - R1**3)
    if V <= 0.0:
        raise ValueError(f"Non-positive bubble volume for R1={R1}, R2={R2}")
    return (gamma - 1.0) * Eb / V


def wind_postshock_pressure(Lwind: float, vwind: float, R1: float, gamma: float) -> float:
    """
    Strong shock with negligible upstream pressure:
      P2 = 2/(gamma+1) rho_w vwind^2

    With rho_w = dMdt_w / (4π R1^2 vwind) and dMdt_w = 2 Lwind / vwind^2:
      P2 = Lwind / (π (gamma+1) R1^2 vwind)
    """
    if R1 <= 0.0:
        raise ValueError("R1 must be > 0.")
    return Lwind / (math.pi * (gamma + 1.0) * (R1**2) * vwind)


def solve_R1_and_Pb(
    Lwind: float,
    vwind: float,
    Eb: float,
    R2: float,
    gamma: float,
    rmin_frac: float = 1e-6,
    rmax_frac: float = 0.999,
) -> Tuple[float, float]:
    """Solve for R1 by equating wind post-shock pressure and Pb(Eb,R1,R2)."""
    R2 = _ensure_positive(R2, "R2")
    gamma = _ensure_positive(gamma, "gamma_adia", floor=1.0)

    if not (np.isfinite(Lwind) and np.isfinite(vwind)) or Lwind <= 0.0 or vwind <= 0.0:
        R1 = rmin_frac * R2
        Pb = bubble_pressure_from_energy(Eb, R2, R1, gamma)
        return R1, Pb

    if not np.isfinite(Eb) or Eb <= 0.0:
        R1 = rmin_frac * R2
        Pb = 0.0
        return R1, Pb

    r_lo = max(rmin_frac * R2, 1e-30)
    r_hi = rmax_frac * R2

    def f(r1: float) -> float:
        pb_e = bubble_pressure_from_energy(Eb, R2, r1, gamma)
        pb_w = wind_postshock_pressure(Lwind, vwind, r1, gamma)
        return pb_e - pb_w

    f_lo = f(r_lo)
    f_hi = f(r_hi)

    if np.sign(f_lo) == np.sign(f_hi):
        Pb0 = bubble_pressure_from_energy(Eb, R2, r_lo, gamma)
        R1_approx = math.sqrt(max(Lwind, 0.0) / (math.pi * (gamma + 1.0) * max(vwind, 1e-30) * max(Pb0, 1e-99)))
        R1 = float(np.clip(R1_approx, r_lo, r_hi))
        Pb = bubble_pressure_from_energy(Eb, R2, R1, gamma)
        return R1, Pb

    R1 = float(scipy.optimize.brentq(f, r_lo, r_hi, maxiter=200, xtol=1e-12))
    Pb = bubble_pressure_from_energy(Eb, R2, R1, gamma)
    return R1, Pb


# -----------------------------------------------------------------------------
# dMdt initial guess (Weaver-like scaling used in the legacy code)
# -----------------------------------------------------------------------------
def get_init_dMdt(params: Mapping[str, Any]) -> float:
    """
    Initial guess for evaporation mass flux dMdt (Weaver+77-like scaling).

    Depends on: R2, t_now, Pb, C_thermal, mu_neu, k_B.
    """
    R2 = _safe_float(_get(params, "R2"), "R2")
    t = _safe_float(_get(params, "t_now"), "t_now")
    Pb = _safe_float(_get(params, "Pb"), "Pb")
    C_th = _safe_float(_get(params, "C_thermal"), "C_thermal")
    mu_neu = _safe_float(_get(params, "mu_neu"), "mu_neu")
    k_B = _safe_float(_get(params, "k_B"), "k_B")

    R2 = _ensure_positive(R2, "R2")
    t = max(t, 1e-12)
    Pb = max(Pb, 0.0)
    C_th = _ensure_positive(C_th, "C_thermal")
    k_B = _ensure_positive(k_B, "k_B")

    dMdt_factor = 1.646  # legacy constant
    dMdt = (
        (12.0 / 75.0)
        * (dMdt_factor ** 2.5)
        * (4.0 * math.pi * R2**3 / t)
        * (mu_neu / k_B)
        * ((t * C_th / (R2**2)) ** (2.0 / 7.0))
        * (Pb ** (5.0 / 7.0))
    )
    return float(max(dMdt, 0.0))


# -----------------------------------------------------------------------------
# Temperature profile ODE (isobaric conduction + net cooling)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BubbleODEContext:
    t_now: float
    Pb: float
    Qi: float
    k_B: float
    mu_ion: float
    C_thermal: float
    dMdt: float
    params: Mapping[str, Any]
    phi_floor: float = 1e-99
    T_floor: float = 1.0


def _bubble_rhs(r: float, y: np.ndarray, ctx: BubbleODEContext) -> np.ndarray:
    """RHS for y = [T, g] where g = dT/dr."""
    T = float(max(y[0], ctx.T_floor))
    g = float(y[1])

    n = ctx.Pb / (2.0 * ctx.k_B * T)
    n = max(n, 0.0)

    phi = ctx.Qi / (4.0 * math.pi * max(r, 1e-30) ** 2)
    phi = max(phi, ctx.phi_floor)

    dudt = float(net_coolingcurve.get_dudt(ctx.t_now, n, T, phi, ctx.params))  # negative for cooling

    A = 5.0 * ctx.k_B * ctx.dMdt / (4.0 * math.pi * ctx.mu_ion)
    C = ctx.C_thermal

    r2 = max(r, 1e-30) ** 2
    T52 = T ** 2.5

    dgdr = (
        dudt / (C * T52)
        + (A * g) / (r2 * C * T52)
        - 2.0 * g / max(r, 1e-30)
        - 2.5 * (g * g) / T
    )

    return np.array([g, dgdr], dtype=float)


def _trapz_on_increasing_r(r: np.ndarray, f: np.ndarray) -> float:
    """Trapezoid integral ∫ f(r) dr, robust to r being decreasing."""
    if r.size < 2:
        return 0.0
    if r[0] > r[-1]:
        return float(np.trapz(f[::-1], x=r[::-1]))
    return float(np.trapz(f, x=r))


def _compute_dudt_array(
    params: Mapping[str, Any],
    r: np.ndarray,
    T: np.ndarray,
    Pb: float,
    *,
    T_floor: float,
    phi_floor: float = 1e-99,
) -> np.ndarray:
    """Compute dudt(r) by looping scalars (net_coolingcurve is scalar-branching)."""
    t_now = _safe_float(_get(params, "t_now"), "t_now")
    Qi = _safe_float(_get(params, "Qi"), "Qi")
    k_B = _safe_float(_get(params, "k_B"), "k_B")

    dudt = np.empty_like(T, dtype=float)
    for i in range(T.size):
        Ti = float(max(T[i], T_floor))
        ri = float(max(r[i], 1e-30))
        ni = Pb / (2.0 * k_B * Ti)
        ni = max(ni, 0.0)
        phi = Qi / (4.0 * math.pi * ri * ri)
        phi = max(phi, phi_floor)
        dudt[i] = float(net_coolingcurve.get_dudt(t_now, ni, Ti, phi, params))
    return dudt


def _compute_mass(params: Mapping[str, Any], r: np.ndarray, n: np.ndarray) -> float:
    """Compute hot-gas mass in Msun by integrating ρ dV, with ρ = n * mu_ion."""
    mu_ion = _safe_float(_get(params, "mu_ion"), "mu_ion")
    rho = n * mu_ion
    integrand = 4.0 * math.pi * r * r * rho
    return _trapz_on_increasing_r(r, integrand)


def _integrate_temperature_profile(
    params: Mapping[str, Any],
    R1: float,
    R2: float,
    Pb: float,
    dMdt: float,
    *,
    T_goal: float,
    T_min: float,
    n_points: int = 2000,
    n_points_intermediate: int = 200,
    method: str = "Radau",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays (r, T, dTdr) with r decreasing (outer->inner).
    Analytically extends a thin layer from T_min to T_goal using T ∝ (R2-r)^{2/5}.
    """
    t_now = _safe_float(_get(params, "t_now"), "t_now")
    Qi = _safe_float(_get(params, "Qi"), "Qi")
    k_B = _safe_float(_get(params, "k_B"), "k_B")
    mu_ion = _safe_float(_get(params, "mu_ion"), "mu_ion")
    C_th = _safe_float(_get(params, "C_thermal"), "C_thermal")

    t_now = max(t_now, 1e-12)
    k_B = _ensure_positive(k_B, "k_B")
    mu_ion = _ensure_positive(mu_ion, "mu_ion")
    C_th = _ensure_positive(C_th, "C_thermal")

    T_goal = _ensure_positive(float(T_goal), "T_goal")
    T_min = _ensure_positive(float(T_min), "T_min")
    if T_goal <= T_min:
        raise ValueError(f"T_goal must be > T_min. Got T_goal={T_goal}, T_min={T_min}")

    R1 = _ensure_positive(R1, "R1")
    R2 = _ensure_positive(R2, "R2")
    if R1 >= R2:
        raise ValueError(f"Need R1 < R2, got R1={R1}, R2={R2}")

    constant = 25.0 / 4.0 * k_B / (mu_ion * C_th)
    mdot_area = dMdt / (4.0 * math.pi * R2**2)
    mdot_area = max(mdot_area, 1e-99)

    dR2 = (T_goal ** 2.5) / (constant * mdot_area)

    dR2_min = 1e-12 * R2
    dR2_max = 0.2 * (R2 - R1)
    dR2 = float(np.clip(dR2, dR2_min, dR2_max))

    r_start = R2 - dR2
    if r_start <= R1:
        r_start = 0.5 * (R1 + R2)
        dR2 = R2 - r_start

    g_start = -(2.0 / 5.0) * T_goal / dR2

    ctx = BubbleODEContext(
        t_now=t_now,
        Pb=Pb,
        Qi=Qi,
        k_B=k_B,
        mu_ion=mu_ion,
        C_thermal=C_th,
        dMdt=dMdt,
        params=params,
    )

    sol = solve_ivp(
        fun=lambda rr, yy: _bubble_rhs(rr, yy, ctx),
        t_span=(r_start, R1),
        y0=np.array([T_goal, g_start], dtype=float),
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=(r_start - R1) / 200.0,
    )

    if not sol.success:
        raise RuntimeError(f"Bubble temperature integration failed: {sol.message}")

    n_points = int(max(n_points, 200))
    x_max = max(R2 - R1, dR2)
    x_solved = np.geomspace(dR2, x_max, n_points)
    r_solved = R2 - x_solved
    r_solved = np.clip(r_solved, R1, r_start)

    TT = sol.sol(r_solved)[0]
    gg = sol.sol(r_solved)[1]

    if np.any(~np.isfinite(TT)) or np.any(TT <= 0.0):
        raise RuntimeError("Non-finite or non-positive temperatures encountered in the bubble solution.")

    x_cool = dR2 * (T_min / T_goal) ** 2.5
    x_cool = float(np.clip(x_cool, 1e-30, dR2))

    if x_cool < dR2 * 0.999999:
        n_points_intermediate = int(max(n_points_intermediate, 50))
        x_int = np.geomspace(x_cool, dR2, n_points_intermediate)
        r_int = R2 - x_int
        T_int = T_goal * (x_int / dR2) ** (2.0 / 5.0)
        g_int = -(2.0 / 5.0) * T_int / x_int

        r = np.concatenate([r_int, r_solved[1:]])
        T = np.concatenate([T_int, TT[1:]])
        g = np.concatenate([g_int, gg[1:]])
    else:
        r, T, g = r_solved, TT, gg

    T = np.maximum(T, T_min)
    return r, T, g


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_bubbleproperties(params: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Main entry point: compute bubble structure + luminosities and update params.

    Required keys (minimum):
      R2, Eb, LWind, vWind, gamma_adia, Qi, t_now,
      k_B, mu_ion, mu_neu, C_thermal,
      bubble_xi_Tb
    """
    R2 = _safe_float(_get(params, "R2"), "R2")
    Eb = _safe_float(_get(params, "Eb"), "Eb")
    Lwind = _safe_float(_get(params, "LWind"), "LWind")
    vwind = _safe_float(_get(params, "vWind"), "vWind")
    gamma = _safe_float(_get(params, "gamma_adia"), "gamma_adia")

    R2 = _ensure_positive(R2, "R2")
    gamma = _ensure_positive(gamma, "gamma_adia", floor=1.0)

    # Step 1: R1 + Pb closure (fixes legacy bug)
    R1, Pb = solve_R1_and_Pb(Lwind, vwind, Eb, R2, gamma)
    _set(params, "R1", R1, info="Inner wind-shock radius.", ori_units="pc")
    _set(params, "Pb", Pb, info="Hot-bubble pressure.", ori_units="Msun*pc**-1*Myr**-2")

    if Pb <= 0.0:
        _set(params, "bubble_L1Bubble", 0.0)
        _set(params, "bubble_L2Conduction", 0.0)
        _set(params, "bubble_L3Intermediate", 0.0)
        _set(params, "bubble_LTotal", 0.0)
        _set(params, "bubble_Tavg", np.nan)
        _set(params, "bubble_mass", 0.0)
        _set(params, "bubble_r_arr", np.array([], dtype=float), exclude_from_snapshot=True)
        _set(params, "bubble_T_arr", np.array([], dtype=float), exclude_from_snapshot=True)
        _set(params, "bubble_dTdr_arr", np.array([], dtype=float), exclude_from_snapshot=True)
        _set(params, "bubble_n_arr", np.array([], dtype=float), exclude_from_snapshot=True)
        _set(params, "bubble_v_arr", np.array([], dtype=float), exclude_from_snapshot=True)
        return params

    # Step 2: dMdt
    dMdt_obj = _get(params, "bubble_dMdt") if "bubble_dMdt" in params else np.nan
    dMdt_val = float(getattr(dMdt_obj, "value", dMdt_obj)) if dMdt_obj is not None else float("nan")
    if not np.isfinite(dMdt_val) or dMdt_val <= 0.0:
        dMdt_val = get_init_dMdt(params)
    _set(params, "bubble_dMdt", dMdt_val, info="Conductive evaporation mass flux.", ori_units="Msun/Myr")
    if "dMdt" in params:
        _set(params, "dMdt", dMdt_val)

    # Step 3: r_Tb
    xi_Tb = float(_get(params, "bubble_xi_Tb"))
    xi_Tb = float(np.clip(xi_Tb if np.isfinite(xi_Tb) else 0.5, 0.0, 1.0))
    r_Tb = R1 + xi_Tb * (R2 - R1)
    _set(params, "bubble_r_Tb", r_Tb, info="Radius where bubble temperature is sampled.", ori_units="pc")

    # Defaults
    T_goal = float(_get(params, "T_goal")) if "T_goal" in params else 3.0e4
    T_min = 1.0e4
    T_CIEswitch = 10.0 ** 5.5

    n_points = int(_get(params, "bubble_n_points")) if "bubble_n_points" in params else 2000
    n_points_int = int(_get(params, "bubble_n_points_intermediate")) if "bubble_n_points_intermediate" in params else 200

    # Step 4: integrate
    r_arr, T_arr, dTdr_arr = _integrate_temperature_profile(
        params=params,
        R1=R1,
        R2=R2,
        Pb=Pb,
        dMdt=dMdt_val,
        T_goal=T_goal,
        T_min=T_min,
        n_points=n_points,
        n_points_intermediate=n_points_int,
    )

    k_B = _safe_float(_get(params, "k_B"), "k_B")
    mu_ion = _safe_float(_get(params, "mu_ion"), "mu_ion")

    n_arr = Pb / (2.0 * k_B * T_arr)
    rho_arr = n_arr * mu_ion
    v_arr = dMdt_val / (4.0 * math.pi * r_arr * r_arr * rho_arr)

    _set(params, "bubble_r_arr", r_arr, exclude_from_snapshot=True)
    _set(params, "bubble_T_arr", T_arr, exclude_from_snapshot=True)
    _set(params, "bubble_dTdr_arr", dTdr_arr, exclude_from_snapshot=True)
    _set(params, "bubble_n_arr", n_arr, exclude_from_snapshot=True)
    _set(params, "bubble_v_arr", v_arr, exclude_from_snapshot=True)

    # Step 5: T(r_Tb)
    if r_Tb > r_arr[0]:
        x0 = R2 - float(r_arr[0])
        T0 = float(T_arr[0])
        x = max(R2 - r_Tb, 1e-30)
        T_rTb = T0 * (x / x0) ** (2.0 / 5.0)
    else:
        r_inc = r_arr[::-1]
        T_inc = T_arr[::-1]
        T_rTb = float(np.interp(r_Tb, r_inc, T_inc))

    _set(params, "bubble_T_r_Tb", T_rTb, info="Bubble temperature at bubble_r_Tb.", ori_units="K")
    if "T0" in params:
        _set(params, "T0", T_rTb)

    # Step 6: luminosities
    dudt_arr = _compute_dudt_array(params, r_arr, T_arr, Pb, T_floor=T_min)
    integrand = (-dudt_arr) * 4.0 * math.pi * r_arr * r_arr

    mask_bubble = T_arr >= T_CIEswitch
    mask_cond = (T_arr >= T_goal) & (T_arr < T_CIEswitch)
    mask_int = (T_arr >= T_min) & (T_arr < T_goal)

    def integrate_mask(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return _trapz_on_increasing_r(r_arr[mask], integrand[mask])

    L_bubble = integrate_mask(mask_bubble)
    L_cond = integrate_mask(mask_cond)
    L_int = integrate_mask(mask_int)
    L_total = L_bubble + L_cond + L_int

    _set(params, "bubble_L1Bubble", L_bubble, info="Cooling luminosity in CIE hot zone.")
    _set(params, "bubble_L2Conduction", L_cond, info="Cooling luminosity in conduction zone.")
    _set(params, "bubble_L3Intermediate", L_int, info="Cooling luminosity in intermediate zone.")
    _set(params, "bubble_LTotal", L_total, info="Total bubble cooling luminosity.")

    # Step 7: averages
    V_hot = _trapz_on_increasing_r(r_arr, 4.0 * math.pi * r_arr * r_arr)
    Tavg = _trapz_on_increasing_r(r_arr, 4.0 * math.pi * r_arr * r_arr * T_arr) / max(V_hot, 1e-99)
    mBubble = _compute_mass(params, r_arr, n_arr)

    _set(params, "bubble_Tavg", Tavg, info="Volume-weighted bubble temperature (T>=1e4 K).", ori_units="K")
    _set(params, "bubble_mass", mBubble, info="Hot bubble gas mass (T>=1e4 K).", ori_units="Msun")

    return params
