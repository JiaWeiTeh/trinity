#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figure: cloud-edge smoothing comparison.

Single panel n(r) for a fiducial cloud (mCloud = 1e6 Msun,
nCore = 1e3 cm^-3, alpha = 0, rCore = 0.1 pc) showing four curves:

    - the original discontinuous step at r = rCloud
    - the tanh ``hyperbolic blend`` (SMOOTH_FRAC = 0.01) currently used
      in src/cloud_properties/density_profile.py
    - one smaller smoothing width (SMOOTH_FRAC = 0.005)
    - one larger smoothing width (SMOOTH_FRAC = 0.02)

Both axes are linear so the rotational symmetry of the blend about
(rCloud, (n_core + n_ISM)/2) is visible by eye -- the same Delta n
antisymmetry about r = rCloud that buys O(delta^2) mass conservation.
On a log y axis a referee would (correctly) read equal absolute
deviations as wildly unequal log shifts, because the outside asymptote
n_ISM is 1000x smaller than n_core.

Sized for an A&A ``\\columnwidth`` figure; the figure size is taken
from ``trinity.mplstyle`` (3.5 x 2.8 in).

Output: ``fig/paper_rcloud_smooth.pdf``
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")        # save-only; no GUI window even if a backend exists
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
# One value below the default and one above
SMOOTH_FRAC_LOW = 0.005
SMOOTH_FRAC_HIGH = 0.02

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
    """Original discontinuous step at r = rCloud."""
    return np.where(r <= R_CLOUD, density_inside(r), N_ISM_AU)


def density_blend(r, smooth_frac):
    """tanh-blended density (matches density_profile.py with SMOOTH_FRAC=f)."""
    delta = smooth_frac * R_CLOUD
    w_out = 0.5 * (1.0 + np.tanh((r - R_CLOUD) / delta))
    n_in = density_inside(r)
    return n_in * (1.0 - w_out) + N_ISM_AU * w_out


# =============================================================================
# Radius grid: log-spaced overall, with a fine linear band around rCloud so
# that the narrow tanh transition is well resolved for every SMOOTH_FRAC.
# =============================================================================
r_min = 1e-2 * R_CLOUD
r_max = 1.5 * R_CLOUD
r_log = np.geomspace(r_min, r_max, 4000)
r_band = np.linspace(0.7 * R_CLOUD, 1.3 * R_CLOUD, 2000)
r = np.unique(np.concatenate([r_log, r_band]))


# =============================================================================
# Plot (style and figsize set by trinity.mplstyle via the plot_base import)
# =============================================================================
fig, ax = plt.subplots()

# Original discontinuous step
n_jump = density_jump(r)
ax.plot(r, n_jump * cvt.ndens_au2cgs,
        color='k', ls='-', lw=1.4, label='step (original)')

# Three blend widths: below default, default, above default
blend_specs = [
    (SMOOTH_FRAC_LOW,     '#0072B2', '--', 1.0),  # below
    (SMOOTH_FRAC_DEFAULT, '#009E73', '-',  1.6),  # default (highlighted)
    (SMOOTH_FRAC_HIGH,    '#D55E00', '--', 1.0),  # above
]
for sf, color, ls, lw in blend_specs:
    n_b = density_blend(r, sf)
    is_default = np.isclose(sf, SMOOTH_FRAC_DEFAULT)
    label = rf'$f={sf:g}$' + (' (default)' if is_default else '')
    ax.plot(r, n_b * cvt.ndens_au2cgs, color=color, ls=ls, lw=lw, label=label)

# r_cloud reference line (kept thin; the x-tick below also marks it)
ax.axvline(R_CLOUD, color='red', ls=':', lw=1.0, alpha=0.6)

# Both axes linear so the antisymmetry of (n_blend - n_step) about rCloud
# -- i.e. the rotational symmetry around (rCloud, (n_core + n_ISM)/2) --
# is visible by eye.
ax.set_xlim(0.0, 1.3 * R_CLOUD)
ax.set_ylim(-0.05 * N_CORE_CGS, 1.15 * N_CORE_CGS)

# Schematic axes: a single x-tick at rCloud, no numeric values.
ax.set_xticks([R_CLOUD])
ax.set_xticklabels([r'$R_\mathrm{cloud}$'])
ax.set_yticks([])
ax.minorticks_off()  # only one major xtick -> auto minors are nonsensical
ax.set_ylabel(r'$n(r)$')

# Inline asymptote labels in lieu of numeric y-ticks
ax.text(0.04 * R_CLOUD, N_CORE_CGS, r'$n_\mathrm{core}$',
        va='bottom', ha='left', color='0.25')
ax.text(1.28 * R_CLOUD, N_ISM_CGS, r'$n_\mathrm{ISM}$',
        va='bottom', ha='right', color='0.25')

# Mark the symmetry point of the blend (every s-curve crosses here)
ax.axhline(0.5 * (N_CORE_CGS + N_ISM_CGS),
           color='gray', ls=':', lw=0.8, alpha=0.5)

ax.legend(loc='upper right', handlelength=1.6, labelspacing=0.3)

fig.tight_layout()

out_path = FIG_DIR / 'paper_rcloud_smooth.pdf'
fig.savefig(out_path)
plt.close(fig)
print(f"Saved: {out_path}")
