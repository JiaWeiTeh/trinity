"""The photodissociation region behind the ionisation front: its far-UV field, its temperature and its H2 transition.

Step (b) of the two-zone neutral layer (docs/dev/phii-identity/weak-wind/neutral-shell/PDR_DEFAULT_PLAN.md; rulings D-B1 two
switches, D-B2 the PDR table, D-B3 fFUV_Ln; the table is wk2020's since 2026-11-08, pdr-table/WK2020_PLAN.md). Pure functions in
CGS units; shell_structure converts at its boundary.

g0_habing      the 6-13.6 eV field at radius r from a point source, G0 = L_FUV e^-tau / (4 pi r^2 1.6e-3 erg cm^-2 s^-1), in
               Habing units (Bakes & Tielens 1994 sect. 5.1; Kaufman et al. 1999 sect. 1).
t_surface      the PDR surface temperature: the gas temperature at A_V = 0.01 of the Wolfire-Kaufman 2020 models, as the PDR
               Toolbox distributes them (Pound & Wolfire 2023, AJ 165, 25: wk2020, Z = 1, constant density, TSAV0p01sm.fits),
               on their 0.125-dex grid in (log n, log G0), n = 1e1-1e7 cm^-3, G0 = F_FUV/1.6e-3 = 10^-0.5-10^6.5, 9.46-7860 K
               (lib/default/pdr/wk2020_ts_grid.csv, made by docs/dev/phii-identity/weak-wind/neutral-shell/pdr-table/harness/
               wk2020_table.py; Kaufman et al. 1999's grid until 2026-11-08). Bilinear in log T; clamped at the grid's edges.
t_pdr          the atomic zone's temperature behind the front: a root of T = T_S(G0, n(T)), n(T) = n_ion jump / T with
               jump = (mu_atom/mu_ion_shell) T_ion (pressure continuity across the front). Where the table falls faster than
               n^-1 there can be more than one root (13 points of a 0.1-dex mesh over log G0 -1-12, log n_ion 0-9 at jump
               16000 K: pdr-table/WK2020_PLAN.md), the multiphase equilibria of gas at fixed pressure; the warmest is taken
               (the heated skin behind the front).
n_tran_bs16    the column from the front to the H/H2 transition (N_1 + 2 N_2, Bialy & Sternberg 2016 eq. 31), tau_tran / sigma_g
               with tau_tran = 0.7 ln[(alphaG/2)^(1/0.7) + 1] (their eq. 39), alphaG = 0.59 I_UV (100 cm^-3/n)
               [9.9/(1 + 8.9 Z')]^0.37 (their eq. 22), sigma_g = 1.9e-21 Z' cm^2 (Sternberg et al. 2014 eq. 20); floored at
               N_TRAN_MIN. The star is a beam on the shell, and a beamed field of strength I_UV delivers the flux F0/2 =
               1.03e7 I_UV photons cm^-2 s^-1 (Sternberg et al. 2014 sect. 2.1), half the free-space Draine field's F0, whose
               energy density is 1.7 Habing (their footnote 4); G0 here is the incident field, so I_UV = 2 G0/1.7.
"""
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.optimize

HABING = 1.6e-3            # erg cm^-2 s^-1 over 6-13.6 eV
SIGMA_G = 1.9e-21          # cm^2 per H nucleus at Z' = 1, the dust's Lyman-Werner absorption
N_TRAN_MIN = 1e18          # cm^-2; Bialy & Sternberg's worked example (I_UV 1, n 1e3) gives 2.3e18
_GRID = Path(__file__).resolve().parents[2] / "lib" / "default" / "pdr" / "wk2020_ts_grid.csv"


@lru_cache(maxsize=1)
def _table():
    """(log n nodes, log G0 nodes, log T on the grid [n, G0]) from the committed table."""
    rows = [ln.split(",") for ln in _GRID.read_text().splitlines() if ln and not ln.startswith("#")][1:]
    ln_n = np.array(sorted({float(r[0]) for r in rows}))
    ln_g = np.array(sorted({float(r[1]) for r in rows}))
    if len(rows) != ln_n.size * ln_g.size:
        raise ValueError(f"{_GRID}: {len(rows)} rows for a {ln_n.size} x {ln_g.size} grid")
    lt = np.full((ln_n.size, ln_g.size), np.nan)
    for r in rows:
        lt[np.searchsorted(ln_n, float(r[0])), np.searchsorted(ln_g, float(r[1]))] = math.log10(float(r[2]))
    if not np.all(np.isfinite(lt)):
        raise ValueError(f"{_GRID}: the grid has holes")
    return ln_n, ln_g, lt


def g0_habing(l_fuv, r, tau):
    """Far-UV field [Habing] at r [cm] from l_fuv [erg/s] behind a dust depth tau."""
    return l_fuv * math.exp(-tau) / (4.0 * math.pi * r * r * HABING)


def t_surface(g0, n):
    """The PDR surface temperature [K] (wk2020, module docstring) at G0 [Habing] and n [cm^-3] (scalars or arrays)."""
    ln_n, ln_g, lt = _table()
    x = np.clip(np.log10(np.maximum(n, 1e-300)), ln_n[0], ln_n[-1])
    y = np.clip(np.log10(np.maximum(g0, 1e-300)), ln_g[0], ln_g[-1])
    i = np.minimum(np.searchsorted(ln_n, x, side="right") - 1, ln_n.size - 2)
    j = np.minimum(np.searchsorted(ln_g, y, side="right") - 1, ln_g.size - 2)
    fx = (x - ln_n[i]) / (ln_n[i + 1] - ln_n[i])
    fy = (y - ln_g[j]) / (ln_g[j + 1] - ln_g[j])
    v = ((1 - fx) * (1 - fy) * lt[i, j] + fx * (1 - fy) * lt[i + 1, j]
         + (1 - fx) * fy * lt[i, j + 1] + fx * fy * lt[i + 1, j + 1])
    return 10.0 ** v


def t_pdr(g0, n_ion, jump):
    """The atomic zone's temperature [K] behind the front (module docstring): the warmest root of
    log T - log T_S(G0, n_ion jump / T), which is positive at 1e5 K and negative at 1 K (the table spans 9.46-7860 K)."""
    lt = np.linspace(5.0, 0.0, 501)
    fv = lt - np.log10(t_surface(g0, n_ion * jump / 10.0 ** lt))
    k = int(np.argmax((fv[:-1] > 0.0) & (fv[1:] <= 0.0)))      # the first sign change from the top
    f = lambda x: x - math.log10(float(t_surface(g0, n_ion * jump / 10.0 ** x)))  # noqa: E731
    return 10.0 ** scipy.optimize.brentq(f, lt[k + 1], lt[k], xtol=1e-12)


def n_tran_bs16(g0, n, z):
    """Column [cm^-2] from the front to the H/H2 transition at an incident G0 [Habing], n [cm^-3] and Z' (module docstring)."""
    if not (g0 > 0.0 and z > 0.0):
        return N_TRAN_MIN
    alpha_g = 0.59 * (2.0 * g0 / 1.7) * (100.0 / n) * (9.9 / (1.0 + 8.9 * z)) ** 0.37
    tau = 0.7 * math.log((alpha_g / 2.0) ** (1.0 / 0.7) + 1.0)
    return max(tau / (SIGMA_G * z), N_TRAN_MIN)
