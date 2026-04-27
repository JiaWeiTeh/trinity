#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figure: cloud-edge smoothing comparison.

Compares the discontinuous step (previous behaviour) against the tanh
``hyperbolic blend`` used in ``src/cloud_properties/density_profile.py``
for a fiducial cloud ``mCloud = 1e6 Msun``, ``nCore = 1e3 cm^-3``.

Layout (2 rows x 2 columns):
    - top row   : number density n(r)  [cm^-3]
    - bottom row: enclosed mass M(<r)  [Msun]
    - left col  : previous discontinuous step at r = rCloud
    - right col : tanh blend at SMOOTH_FRAC values bracketing the
                  default 0.01 used in density_profile.py

The enclosed mass is obtained by direct cumulative integration of
``4*pi*r^2*rho(r)`` so that M(r) reflects exactly the n(r) shown above
(and visualises the negligible mass shift produced by the blend).

Cloud setup follows TRINITY's default.param convention: rCore is a
standalone input (pc), not a fraction of rCloud, and the homogeneous
power-law (densPL_alpha = 0) is used so the cloud edge is the full
nCore -> nISM drop and the smoothing band is visually unambiguous.
For alpha != 0 the same blend logic applies; only the amplitude of
the edge jump changes.

Output: ``fig/paper_rcloud_smooth.pdf``
"""

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR
import src._functions.unit_conversions as cvt
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous,
    compute_rCloud_powerlaw,
)


# =============================================================================
# Fiducial cloud parameters (matches default.param conventions)
# =============================================================================
M_CLOUD = 1e6           # Msun
N_CORE_CGS = 1e3        # cm^-3
N_ISM_CGS = 1.0         # cm^-3
ALPHA = 0               # power-law slope (densPL_alpha); 0 = homogeneous
R_CORE = 0.1            # standalone core radius [pc] (NOT a fraction of rCloud)

# Default smoothing width used in src/cloud_properties/density_profile.py
SMOOTH_FRAC_DEFAULT = 0.01
# Two values below the default and two above (bracketed log-spacing)
SMOOTH_FRACS = [0.002, 0.005, 0.01, 0.02, 0.05]

# Code-unit conversions (matches read_param.py conventions)
MU_CONVERT = 1.4 * cvt.M_H_CGS * cvt.g2Msun     # [Msun]
N_CORE_AU = N_CORE_CGS * cvt.ndens_cgs2au        # [1/pc^3]
N_ISM_AU = N_ISM_CGS * cvt.ndens_cgs2au          # [1/pc^3]


# =============================================================================
# Resolve rCloud from (mCloud, nCore, alpha, rCore)
# =============================================================================
if ALPHA == 0:
    # Homogeneous: rCore does not enter the rCloud calculation
    R_CLOUD = compute_rCloud_homogeneous(M_CLOUD, N_CORE_AU, mu=MU_CONVERT)
else:
    R_CLOUD, _ = compute_rCloud_powerlaw(
        M_CLOUD, N_CORE_AU, ALPHA,
        rCore=R_CORE, mu=MU_CONVERT,
    )

print(f"Fiducial cloud: M={M_CLOUD:.0e} Msun, "
      f"nCore={N_CORE_CGS:.0e} cm^-3, alpha={ALPHA}, rCore={R_CORE} pc")
print(f"  rCloud = {R_CLOUD:.3f} pc")


# =============================================================================
# Density helpers (number density in code units 1/pc^3)
# =============================================================================
def density_inside(r):
    """Power-law / homogeneous density valid inside the cloud."""
    if ALPHA == 0:
        return np.full_like(r, N_CORE_AU)
    n = N_CORE_AU * (r / R_CORE) ** ALPHA
    return np.where(r <= R_CORE, N_CORE_AU, n)


def density_jump(r):
    """Previous behaviour: hard step at r = rCloud."""
    return np.where(r <= R_CLOUD, density_inside(r), N_ISM_AU)


def density_blend(r, smooth_frac):
    """tanh-blended density (matches density_profile.py with SMOOTH_FRAC=f)."""
    delta = smooth_frac * R_CLOUD
    w_out = 0.5 * (1.0 + np.tanh((r - R_CLOUD) / delta))
    n_in = density_inside(r)
    return n_in * (1.0 - w_out) + N_ISM_AU * w_out


def cumulative_trapz(y, x):
    """Cumulative trapezoidal integral; out[0]=0, len(out)=len(x)."""
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return np.concatenate([[0.0], np.cumsum(avg * dx)])


def enclosed_mass(r, n_au):
    """M(<r) = int_0^r 4 pi r'^2 rho(r') dr'  [Msun].

    The grid starts at r[0] > 0; assume the density is approximately
    constant (= n_au[0]) over (0, r[0]] and add the analytical
    (4/3) pi r[0]^3 rho[0] offset so that M(r[0]) is the true cumulative
    mass and the log-axis plot has no zero values.
    """
    rho = n_au * MU_CONVERT
    M_inner = (4.0 / 3.0) * np.pi * r[0] ** 3 * rho[0]
    return M_inner + cumulative_trapz(4.0 * np.pi * r ** 2 * rho, r)


# =============================================================================
# Radius grid: log-spaced overall, with a fine linear band around rCloud so
# that the narrow tanh transition is well resolved for every SMOOTH_FRAC.
# =============================================================================
r_min = 1e-2 * R_CLOUD
r_max = 3.0 * R_CLOUD
r_log = np.geomspace(r_min, r_max, 4000)
r_band = np.linspace(0.7 * R_CLOUD, 1.3 * R_CLOUD, 2000)
r = np.unique(np.concatenate([r_log, r_band]))


# =============================================================================
# Plot
# =============================================================================
plt.rcParams.update({
    'font.size':       16,
    'axes.labelsize':  16,
    'axes.titlesize':  16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
})

fig, axes = plt.subplots(
    2, 2, figsize=(12, 9), sharex=True, sharey='row', dpi=150,
)
(ax_n_jump, ax_n_blend), (ax_m_jump, ax_m_blend) = axes

# --- previous discontinuous step ---
n_jump = density_jump(r)
M_jump = enclosed_mass(r, n_jump)
ax_n_jump.plot(r, n_jump * cvt.ndens_au2cgs, color='k', lw=2.0, label='step')
ax_m_jump.plot(r, M_jump, color='k', lw=2.0, label='step')

# --- tanh blend at several smoothing widths ---
cmap = plt.get_cmap('viridis')
log_fracs = np.log10(SMOOTH_FRACS)
norm = plt.Normalize(log_fracs.min(), log_fracs.max())
for sf in SMOOTH_FRACS:
    n_b = density_blend(r, sf)
    M_b = enclosed_mass(r, n_b)
    is_default = np.isclose(sf, SMOOTH_FRAC_DEFAULT)
    color = cmap(norm(np.log10(sf)))
    lw = 2.6 if is_default else 1.6
    ls = '-' if is_default else '--'
    label = rf'$f={sf:g}$' + (' (default)' if is_default else '')
    ax_n_blend.plot(r, n_b * cvt.ndens_au2cgs,
                    color=color, lw=lw, ls=ls, label=label)
    ax_m_blend.plot(r, M_b, color=color, lw=lw, ls=ls, label=label)

# Reference markers: rCloud and target cloud mass
for ax in (ax_n_jump, ax_n_blend, ax_m_jump, ax_m_blend):
    ax.axvline(R_CLOUD, color='red', ls=':', lw=1.0, alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(r_min, r_max)

for ax in (ax_m_jump, ax_m_blend):
    ax.axhline(M_CLOUD, color='red', ls=':', lw=1.0, alpha=0.6)
    ax.set_xlabel(r'$r\ [\mathrm{pc}]$')
ax_m_jump.set_ylabel(r'$M(<r)\ [\mathrm{M}_\odot]$')

ax_n_jump.set_ylabel(r'$n(r)\ [\mathrm{cm}^{-3}]$')
ax_n_jump.set_ylim(0.3 * N_ISM_CGS, 5 * N_CORE_CGS)

ax_n_jump.set_title('previous: discontinuous step')
ax_n_blend.set_title(r'hyperbolic blend ($\delta = f\,r_\mathrm{cloud}$)')

ax_n_jump.legend(loc='lower left', frameon=False)
ax_m_jump.legend(loc='lower right', frameon=False)
ax_n_blend.legend(loc='lower left', frameon=False)
ax_m_blend.legend(loc='lower right', frameon=False)

# Annotate rCloud line on every panel (sharey='row' has set m-axis ylim)
for ax in (ax_n_jump, ax_n_blend, ax_m_jump, ax_m_blend):
    y_top = ax.get_ylim()[1]
    ax.text(R_CLOUD, y_top, r' $r_\mathrm{cloud}$',
            color='red', va='top', ha='left', fontsize=11)

fig.suptitle(
    rf'Fiducial cloud: $M_\mathrm{{cl}}=10^{{6}}\,\mathrm{{M}}_\odot$, '
    rf'$n_\mathrm{{core}}=10^{{3}}\,\mathrm{{cm}}^{{-3}}$, '
    rf'$\alpha={ALPHA}$, $r_\mathrm{{core}}={R_CORE}\,\mathrm{{pc}}$',
    fontsize=16,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = FIG_DIR / 'paper_rcloud_smooth.pdf'
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
