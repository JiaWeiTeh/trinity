#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMC Parameter Validation
========================

Standalone validation for Giant Molecular Cloud parameters.
Checks physical plausibility before running a simulation.

Constraints checked:
1. rCloud <= r_max (default 200 pc, typical GMC limit)
2. nEdge >= nISM (edge density above ISM background)
3. Mass error <= tolerance (self-consistent parameters)

Supports both power-law and Bonnor-Ebert density profiles.

Usage
-----
High-level (from params dict)::

    result = validate_gmc_from_params(params)
    if not result.valid:
        print(result.summary())
        sys.exit(1)

Low-level (explicit values)::

    result = validate_gmc_params(
        mCloud=1e5, nCore=1e3, mu=1.4, nISM=1.0,
        dens_profile='densPL', alpha=-2, rCore=1.0,
    )

Constraint-only (pre-computed rCloud/nEdge)::

    issues = check_gmc_constraints(
        rCloud=150.0, nEdge=0.5, mCloud=1e5, M_computed=1.001e5,
    )

@author: TRINITY Team
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.cloud_properties.powerLawSphere import (
    compute_rCloud_powerlaw,
    compute_rCloud_homogeneous,
)
from src.cloud_properties.bonnorEbertSphere import (
    create_BE_sphere,
    solve_lane_emden,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Default physical constraints
# ============================================================================

R_CLOUD_MAX = 200.0     # pc — typical single-GMC limit
MASS_TOLERANCE = 0.001   # 0.1% relative mass error


# ============================================================================
# Result data structure
# ============================================================================

@dataclass
class GMCValidationResult:
    """
    Result of GMC parameter validation.

    Attributes
    ----------
    valid : bool
        True if all constraints pass (no errors).
    errors : list[str]
        Critical issues — simulation should not proceed.
    warnings : list[str]
        Non-critical notes (e.g. rCloud near limit).
    rCloud : float
        Computed cloud radius [pc].
    nEdge : float
        Computed edge density (same units as nISM input).
    mass_error : float
        Relative mass error |M_computed - mCloud| / mCloud.
    M_computed : float
        Enclosed mass recomputed at rCloud.
    suggestions : list[dict]
        Valid alternative parameter sets (when invalid).
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rCloud: float = np.nan
    nEdge: float = np.nan
    mass_error: float = np.nan
    M_computed: float = np.nan
    suggestions: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Format a human-readable summary string."""
        lines = []
        if self.valid:
            lines.append("GMC parameters are VALID.")
        else:
            lines.append("GMC parameters are INVALID.")

        lines.append(f"  rCloud      = {self.rCloud:.4f} pc")
        lines.append(f"  nEdge       = {self.nEdge:.4e}")
        lines.append(f"  mass error  = {self.mass_error * 100:.4f}%")

        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")

        if self.suggestions:
            lines.append("  Suggested valid alternatives:")
            for i, s in enumerate(self.suggestions, 1):
                parts = ", ".join(f"{k}={v}" for k, v in s.items())
                lines.append(f"    {i}. {parts}")

        return "\n".join(lines)


# ============================================================================
# Low-level: constraint check on pre-computed values
# ============================================================================

def check_gmc_constraints(
    rCloud,
    nEdge,
    mCloud,
    M_computed,
    nISM=1.0,
    r_max=R_CLOUD_MAX,
    mass_tolerance=MASS_TOLERANCE,
):
    """
    Check the three GMC plausibility constraints on pre-computed values.

    This is the lightweight helper used by plotting scripts that already
    compute rCloud and nEdge themselves.

    Parameters
    ----------
    rCloud : float
        Cloud radius [pc].
    nEdge : float
        Edge density (same unit system as nISM).
    mCloud : float
        Target cloud mass.
    M_computed : float
        Enclosed mass recomputed at rCloud.
    nISM : float
        ISM background density (same unit system as nEdge).
    r_max : float
        Maximum cloud radius [pc] (default 200).
    mass_tolerance : float
        Maximum relative mass error (default 0.001 = 0.1%).

    Returns
    -------
    dict
        ``errors`` : list[str] — critical failures.
        ``warnings`` : list[str] — non-critical notes.
        ``mass_error`` : float — relative mass error.
    """
    errors = []
    warnings = []

    # 1. Cloud radius
    if rCloud > r_max:
        errors.append(
            f"rCloud ({rCloud:.1f} pc) exceeds maximum GMC size "
            f"({r_max:.0f} pc). Cloud is too diffuse for the given mass."
        )

    # 2. Edge density
    if nEdge < nISM:
        errors.append(
            f"Edge density ({nEdge:.2e}) < ISM density ({nISM:.2e}). "
            f"Consider: increasing nCore, increasing rCore, or reducing |alpha|."
        )

    # 3. Mass consistency
    mass_error = abs(M_computed - mCloud) / mCloud if mCloud > 0 else 0.0
    if mass_error > mass_tolerance:
        errors.append(
            f"Mass inconsistency: M(rCloud) = {M_computed:.4e} vs "
            f"mCloud = {mCloud:.4e} (error {mass_error * 100:.4f}%, "
            f"tolerance {mass_tolerance * 100:.3f}%)."
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "mass_error": mass_error,
    }


# ============================================================================
# High-level: validate from explicit physical values
# ============================================================================

def validate_gmc_params(
    mCloud,
    nCore,
    mu,
    nISM,
    dens_profile,
    # Power-law specific
    alpha=None,
    rCore=None,
    # Bonnor-Ebert specific
    Omega=None,
    gamma=5.0 / 3.0,
    # Constraint thresholds
    r_max=R_CLOUD_MAX,
    mass_tolerance=MASS_TOLERANCE,
    # Optional cached Lane-Emden solution
    lane_emden_solution=None,
):
    """
    Validate GMC parameters for physical plausibility.

    Computes the derived cloud properties (rCloud, nEdge, enclosed mass)
    from the given inputs and checks the three standard constraints.

    Parameters
    ----------
    mCloud : float
        Cloud mass [Msun].
    nCore : float
        Core number density [1/pc³] (code units).
    mu : float
        Mean molecular weight [Msun] (code units, typically 1.4).
    nISM : float
        ISM density [1/pc³] (code units).
    dens_profile : str
        ``'densPL'`` for power-law, ``'densBE'`` for Bonnor-Ebert.
    alpha : float, optional
        Power-law exponent (required for ``densPL``).
    rCore : float, optional
        Core radius [pc] (required for ``densPL`` with alpha != 0).
    Omega : float, optional
        Density contrast rho_core/rho_edge (required for ``densBE``).
    gamma : float, optional
        Adiabatic index (default 5/3, used for ``densBE``).
    r_max : float
        Maximum cloud radius [pc] (default 200).
    mass_tolerance : float
        Maximum relative mass error (default 0.001).
    lane_emden_solution : LaneEmdenSolution, optional
        Pre-computed Lane-Emden solution (avoids re-solving for ``densBE``).

    Returns
    -------
    GMCValidationResult
        Contains ``valid``, ``errors``, ``warnings``, ``rCloud``, ``nEdge``,
        ``mass_error``, ``M_computed``, and ``suggestions``.
    """
    if dens_profile == "densPL":
        return _validate_powerlaw(
            mCloud, nCore, mu, nISM, alpha, rCore,
            r_max, mass_tolerance,
        )
    elif dens_profile == "densBE":
        return _validate_bonnor_ebert(
            mCloud, nCore, mu, nISM, Omega, gamma,
            r_max, mass_tolerance, lane_emden_solution,
        )
    else:
        raise ValueError(f"Unknown density profile: {dens_profile!r}")


# ============================================================================
# High-level: validate from TRINITY params dict
# ============================================================================

def validate_gmc_from_params(params, r_max=R_CLOUD_MAX, mass_tolerance=MASS_TOLERANCE):
    """
    Validate GMC parameters extracted from a TRINITY params dictionary.

    Convenience wrapper around :func:`validate_gmc_params` for use in
    ``run.py``.

    Parameters
    ----------
    params : dict-like
        TRINITY parameter dictionary with ``.value`` attribute access.
    r_max : float
        Maximum cloud radius [pc] (default 200).
    mass_tolerance : float
        Maximum relative mass error (default 0.001).

    Returns
    -------
    GMCValidationResult
    """
    mCloud = params["mCloud"].value
    nCore = params["nCore"].value
    mu = params["mu_convert"].value
    nISM = params["nISM"].value
    dens_profile = params["dens_profile"].value

    kwargs = dict(
        mCloud=mCloud,
        nCore=nCore,
        mu=mu,
        nISM=nISM,
        dens_profile=dens_profile,
        r_max=r_max,
        mass_tolerance=mass_tolerance,
    )

    if dens_profile == "densPL":
        kwargs["alpha"] = params["densPL_alpha"].value
        kwargs["rCore"] = params["rCore"].value
    elif dens_profile == "densBE":
        kwargs["Omega"] = params["densBE_Omega"].value
        kwargs["gamma"] = params["gamma_adia"].value

    return validate_gmc_params(**kwargs)


# ============================================================================
# Internal: power-law validation
# ============================================================================

def _validate_powerlaw(mCloud, nCore, mu, nISM, alpha, rCore,
                       r_max, mass_tolerance):
    """Validate power-law density profile parameters."""
    if alpha is None:
        raise ValueError("alpha is required for densPL profile")

    # Compute rCloud
    try:
        if alpha == 0:
            rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu)
            nEdge = nCore
        else:
            if rCore is None:
                raise ValueError(
                    "rCore is required for densPL profile with alpha != 0"
                )
            rCloud, _ = compute_rCloud_powerlaw(
                mCloud, nCore, alpha, rCore=rCore, mu=mu,
            )
            nEdge = nCore * (rCloud / rCore) ** alpha
    except (ValueError, ZeroDivisionError, RuntimeError) as exc:
        return GMCValidationResult(
            valid=False,
            rCloud=np.nan,
            nEdge=np.nan,
            mass_error=np.nan,
            M_computed=np.nan,
            errors=[f"Cannot compute rCloud: {exc}"],
        )

    # Compute enclosed mass for consistency check
    rhoCore = nCore * mu
    if alpha == 0:
        M_computed = (4.0 / 3.0) * np.pi * rCloud**3 * rhoCore
    else:
        M_computed = 4.0 * np.pi * rhoCore * (
            rCore**3 / 3.0
            + (rCloud ** (3.0 + alpha) - rCore ** (3.0 + alpha))
            / ((3.0 + alpha) * rCore**alpha)
        )

    # Check constraints
    checks = check_gmc_constraints(
        rCloud, nEdge, mCloud, M_computed, nISM, r_max, mass_tolerance,
    )

    errors = checks["errors"]
    warnings = checks["warnings"]

    # Find suggestions when invalid
    suggestions = []
    if errors:
        suggestions = _suggest_powerlaw_alternatives(
            mCloud, nCore, rCore, alpha, nISM, mu, r_max, mass_tolerance,
        )

    return GMCValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        rCloud=rCloud,
        nEdge=nEdge,
        mass_error=checks["mass_error"],
        M_computed=M_computed,
        suggestions=suggestions,
    )


# ============================================================================
# Internal: Bonnor-Ebert validation
# ============================================================================

def _validate_bonnor_ebert(mCloud, nCore, mu, nISM, Omega, gamma,
                           r_max, mass_tolerance, lane_emden_solution):
    """Validate Bonnor-Ebert sphere parameters."""
    if Omega is None:
        raise ValueError("Omega is required for densBE profile")

    # Create BE sphere to get derived quantities
    try:
        be_result = create_BE_sphere(
            M_cloud=mCloud,
            n_core=nCore,
            Omega=Omega,
            mu=mu,
            gamma=gamma,
            validate=True,
            lane_emden_solution=lane_emden_solution,
        )
    except (ValueError, RuntimeError) as exc:
        return GMCValidationResult(
            valid=False,
            rCloud=np.nan,
            nEdge=np.nan,
            mass_error=np.nan,
            M_computed=np.nan,
            errors=[f"Cannot create BE sphere: {exc}"],
        )

    rCloud = be_result.r_out
    nEdge = be_result.n_out  # = nCore / Omega

    # Mass consistency check: recompute M from derived quantities.
    # M = 4π × m(ξ_out) × ρc × a³  where a = rCloud / ξ_out
    rhoCore = nCore * mu  # [Msun/pc³]
    a_pc = rCloud / be_result.xi_out  # [pc]
    M_computed = 4.0 * np.pi * be_result.m_dim * rhoCore * a_pc**3

    # Check constraints
    checks = check_gmc_constraints(
        rCloud, nEdge, mCloud, M_computed, nISM, r_max, mass_tolerance,
    )

    errors = checks["errors"]
    warnings = checks["warnings"]

    # Stability warning
    if not be_result.is_stable:
        warnings.append(
            f"Omega={Omega:.2f} exceeds critical value (~14.04). "
            f"BE sphere is gravitationally UNSTABLE."
        )

    # Find suggestions when invalid
    suggestions = []
    if errors:
        suggestions = _suggest_bonnor_ebert_alternatives(
            mCloud, nCore, mu, nISM, Omega, gamma,
            r_max, mass_tolerance, lane_emden_solution,
        )

    return GMCValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        rCloud=rCloud,
        nEdge=nEdge,
        mass_error=checks["mass_error"],
        M_computed=M_computed,
        suggestions=suggestions,
    )


# ============================================================================
# Suggestion helpers
# ============================================================================

def _suggest_powerlaw_alternatives(
    mCloud, nCore, rCore, alpha, nISM, mu, r_max, mass_tolerance,
    n_suggestions=3, search_range=0.5,
):
    """
    Search nearby parameter space for valid power-law GMC configurations.

    Varies mCloud, nCore, rCore by ±search_range and returns the closest
    valid combinations sorted by distance from the original.
    """
    mCloud_factors = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
    nCore_factors = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
    # For alpha=0 (homogeneous), rCore is irrelevant — only vary mCloud/nCore
    if alpha == 0 or rCore is None:
        rCore_factors = np.array([1.0])
    else:
        rCore_factors = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

    valid = []
    for mf in mCloud_factors:
        for nf in nCore_factors:
            for rf in rCore_factors:
                if mf == 1.0 and nf == 1.0 and rf == 1.0:
                    continue
                mc = mCloud * mf
                nc = nCore * nf
                rc = rCore * rf if rCore is not None else None

                try:
                    if alpha == 0:
                        rcl = compute_rCloud_homogeneous(mc, nc, mu)
                        ne = nc
                    else:
                        rcl, _ = compute_rCloud_powerlaw(mc, nc, alpha, rCore=rc, mu=mu)
                        ne = nc * (rcl / rc) ** alpha
                except Exception:
                    continue

                if rcl > r_max or ne < nISM:
                    continue

                rhoCore = nc * mu
                if alpha == 0:
                    M_c = (4.0 / 3.0) * np.pi * rcl**3 * rhoCore
                else:
                    M_c = 4.0 * np.pi * rhoCore * (
                        rc**3 / 3.0
                        + (rcl ** (3.0 + alpha) - rc ** (3.0 + alpha))
                        / ((3.0 + alpha) * rc**alpha)
                    )
                merr = abs(M_c - mc) / mc if mc > 0 else 0.0
                if merr > mass_tolerance:
                    continue

                combo = {
                    "mCloud": mc,
                    "nCore": nc,
                    "rCloud": rcl,
                    "nEdge": ne,
                    "mass_error": merr,
                }
                if rc is not None:
                    combo["rCore"] = rc
                valid.append(combo)

    def _distance(combo):
        d_m = abs(np.log10(combo["mCloud"] / mCloud)) if mCloud > 0 else 0
        d_n = abs(np.log10(combo["nCore"] / nCore)) if nCore > 0 else 0
        d_r = abs(combo["rCore"] / rCore - 1) if rCore and rCore > 0 and "rCore" in combo else 0
        return d_m + d_n + d_r

    valid.sort(key=_distance)
    return valid[:n_suggestions]


def _suggest_bonnor_ebert_alternatives(
    mCloud, nCore, mu, nISM, Omega, gamma,
    r_max, mass_tolerance, lane_emden_solution,
    n_suggestions=3,
):
    """
    Search nearby parameter space for valid BE sphere configurations.

    Varies mCloud and nCore (Omega is kept fixed as a profile shape choice).
    """
    mCloud_factors = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
    nCore_factors = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 5.0])

    if lane_emden_solution is None:
        lane_emden_solution = solve_lane_emden()

    valid = []
    for mf in mCloud_factors:
        for nf in nCore_factors:
            if mf == 1.0 and nf == 1.0:
                continue
            mc = mCloud * mf
            nc = nCore * nf

            try:
                be = create_BE_sphere(
                    M_cloud=mc, n_core=nc, Omega=Omega,
                    mu=mu, gamma=gamma, validate=False,
                    lane_emden_solution=lane_emden_solution,
                )
            except Exception:
                continue

            rcl = be.r_out
            ne = be.n_out

            if rcl > r_max or ne < nISM:
                continue

            # Mass check
            rhoCore = nc * mu
            a_pc = rcl / be.xi_out
            M_c = 4.0 * np.pi * be.m_dim * rhoCore * a_pc**3
            merr = abs(M_c - mc) / mc if mc > 0 else 0.0
            if merr > mass_tolerance:
                continue

            valid.append({
                "mCloud": mc,
                "nCore": nc,
                "Omega": Omega,
                "rCloud": rcl,
                "nEdge": ne,
                "mass_error": merr,
            })

    def _distance(combo):
        d_m = abs(np.log10(combo["mCloud"] / mCloud)) if mCloud > 0 else 0
        d_n = abs(np.log10(combo["nCore"] / nCore)) if nCore > 0 else 0
        return d_m + d_n

    valid.sort(key=_distance)
    return valid[:n_suggestions]
