#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the Barnes 2026 (PHANGS) comparison scripts.

Everything here is *TRINITY-only*: it reads finished TRINITY runs and puts
the relevant pressures on Barnes' ``P/k`` [K cm^-3] basis. No Barnes data is
overlaid yet — these helpers just expose TRINITY's own quantities in the
units a Barnes comparison will eventually need.

Pressure basis
--------------
``PISM`` is stored as ``P/k`` [K cm^-3]; the runtime pressures (``P_HII``,
``Pb``, ``P_ram``, ``P_drive``) are in code units [Msun/Myr^2/pc]. The single
constant ``Pb_au2_KcmInv`` from ``trinity._functions.unit_conversions``
converts the latter onto the former, so everything lands on one axis.

Radiation pressure
------------------
TRINITY does not store a ``P_rad`` scalar; it stores a radiation *force*
``F_rad = f_abs_weighted * Lbol / c * (1 + tau_IR * kappa_IR)`` (single
weighted absorbed fraction, IR-trapping boosted). Two definitions are
exposed:

* :func:`p_rad_native` — ``F_rad / (4 pi R2^2)``: the pressure TRINITY
  actually exerts on the shell, converted to K cm^-3 with the same factor
  as the other pressures.
* :func:`p_rad_barnes` — Barnes' analytic ``prefactor * L / (4 pi r^2 c)``
  recomputed on TRINITY's luminosities (single-scattering, no IR boost).
  The prefactor (the "3") is UNVERIFIED against the paper; see
  :data:`BARNES_PRAD_PREFACTOR`.
"""

import math
from pathlib import Path

import numpy as np

from trinity._output.trinity_reader import load_output, find_all_simulations
from trinity._functions.unit_conversions import (
    Pb_au2_KcmInv,   # code pressure [Msun/Myr^2/pc] -> P/k [K cm^-3]
    L_au2cgs,        # code luminosity [Msun pc^2/Myr^3] -> erg/s
    pc2cm,           # pc -> cm
    K_B_CGS,         # Boltzmann constant [erg/K]
    ndens_au2cgs,    # number density [pc^-3] -> [cm^-3]
)

# Universal constant (does not vary per run); avoids depending on whether a
# given run's metadata.json carries c_light.
C_LIGHT_CGS = 2.99792458e10  # speed of light [cm/s]

FOUR_PI = 4.0 * math.pi

# Default stellar ages [Myr] at which each run is sampled (one plot row each).
DEFAULT_AGES_MYR = (0.5, 1.0, 3.0)

# Prefactor in Barnes' P_rad = prefactor * L / (4 pi r^2 c). The pasted
# analysis cites 3; this is UNVERIFIED against the Barnes paper and is kept
# as a named constant so it is trivial to correct once the paper is in hand.
BARNES_PRAD_PREFACTOR = 3.0


# ---------------------------------------------------------------------------
# Pressure conversions
# ---------------------------------------------------------------------------
def to_Pk(P_code):
    """Code pressure [Msun/Myr^2/pc] -> Barnes' ``P/k`` basis [K cm^-3]."""
    return np.asarray(P_code, dtype=float) * Pb_au2_KcmInv


def pism_to_Pk(PISM_code):
    """``PISM`` (stored as P/k with density in pc^-3, i.e. K pc^-3) -> K cm^-3.

    PISM is declared in the schema as P/k [K cm^-3] but persisted internally
    with the density part in pc^-3, so it needs ``ndens_au2cgs`` (pc^-3 ->
    cm^-3) rather than the code-pressure factor used by :func:`to_Pk`. This
    puts PISM on the same K cm^-3 axis as the converted runtime pressures.
    """
    return np.asarray(PISM_code, dtype=float) * ndens_au2cgs


def p_rad_native(F_rad, R2):
    """TRINITY's native radiation pressure on the shell, in K cm^-3.

    ``P_rad = F_rad / (4 pi R2^2)`` using the radiation force TRINITY itself
    applies (IR-trapping boost included). Inputs in code units.
    """
    F_rad = np.asarray(F_rad, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_code = F_rad / (FOUR_PI * R2 ** 2)
    return to_Pk(P_code)


def p_rad_barnes(Lbol, Li, R_pc, f_neu, f_ion,
                 prefactor=BARNES_PRAD_PREFACTOR):
    """Barnes-formula radiation pressure recomputed on TRINITY luminosities.

    ``P_rad = prefactor * (f_neu*Lbol + f_ion*Li) / (4 pi R^2 c) / k_B``
    (single-scattering, no IR boost). ``Lbol``/``Li`` in code units, ``R_pc``
    in pc; output in K cm^-3. The direct term uses the non-ionizing absorbed
    fraction (Barnes' ``1 - e^-tau_UV`` analog) and the ionizing term the
    ionizing absorbed fraction.
    """
    Lbol = np.asarray(Lbol, dtype=float) * L_au2cgs     # erg/s
    Li = np.asarray(Li, dtype=float) * L_au2cgs         # erg/s
    R_cm = np.asarray(R_pc, dtype=float) * pc2cm
    f_neu = np.asarray(f_neu, dtype=float)
    f_ion = np.asarray(f_ion, dtype=float)
    L_abs = f_neu * Lbol + f_ion * Li
    with np.errstate(divide="ignore", invalid="ignore"):
        P_cgs = prefactor * L_abs / (FOUR_PI * R_cm ** 2 * C_LIGHT_CGS)  # dyn/cm^2
    return P_cgs / K_B_CGS   # K cm^-3


def sigma_gas(mCloud, rCloud):
    """Cloud gas surface density ``mCloud / (pi rCloud^2)`` [Msun/pc^2].

    ``mCloud`` is the post-SFE cloud *gas* mass (mCloud_input = mCloud +
    mCluster) and ``rCloud`` the cloud radius — both run-constants in Msun
    and pc — so this is a per-run constant, the environmental Sigma_gas.
    """
    mCloud = np.asarray(mCloud, dtype=float)
    rCloud = np.asarray(rCloud, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mCloud / (math.pi * rCloud ** 2)


def ir_fraction(records):
    """Fraction of the *true total* radiation pressure that is dust-reprocessed IR.

    TRINITY's radiation force carries an IR-trapping boost (1 + tau_IR), where
    ``tau_IR`` (the dimensionless IR optical depth carried on each record) is
    shell_tauKappaRatio * dust_KappaIR in code units. Hence

        P_rad,total = P_rad,(dir+ion) * (1 + tau_IR),

    and the share Barnes omits (it keeps only dir+ion) is
    f_IR = P_IR / P_rad,total = tau_IR / (1 + tau_IR), in [0, 1). Returns an
    array over *records* (NaN where tau_IR is non-finite, e.g. dust_KappaIR
    absent).
    """
    tau = np.array([r["tau_IR"] for r in records], dtype=float)
    with np.errstate(invalid="ignore"):
        return tau / (1.0 + tau)


# ---------------------------------------------------------------------------
# Run loading and sampling
# ---------------------------------------------------------------------------
def load_runs(folder):
    """Return a ``TrinityOutput`` for every run under *folder* (sorted)."""
    return [load_output(p) for p in find_all_simulations(folder)]


def _f(value):
    """Coerce to float, mapping ``None`` to NaN."""
    return float(value) if isinstance(value, (int, float)) else float("nan")


def sample_run_at_age(output, age_myr):
    """Sample one run at *age_myr* (closest snapshot).

    Returns a dict of the quantities the Barnes plots need (code units for
    luminosities/forces, pc for radii, plus the ``PISM``/``mCluster``
    run-constants), or ``None`` if the run never reaches *age_myr*.
    """
    if age_myr < output.t_min or age_myr > output.t_max:
        return None
    snap = output.get_at_time(age_myr, mode="closest", quiet=True)
    md = output.metadata
    return dict(
        name=output.filepath.parent.name,
        t=_f(snap.t_now),
        R2=_f(snap["R2"]),
        R_IF=_f(snap["R_IF"]),
        F_rad=_f(snap["F_rad"]),
        P_HII=_f(snap["P_HII"]),
        Lbol=_f(snap["Lbol"]),
        Li=_f(snap["Li"]),
        f_neu=_f(snap["shell_fAbsorbedNeu"]),
        f_ion=_f(snap["shell_fAbsorbedIon"]),
        mCluster=_f(md.get("mCluster")),
        PISM=_f(md.get("PISM")),
        mCloud=_f(md.get("mCloud")),
        rCloud=_f(md.get("rCloud")),
        # IR optical depth (dimensionless): shell column * Rosseland opacity,
        # both in code units. NaN if dust_KappaIR is absent.
        tau_IR=(_f(snap["shell_tauKappaRatio"]) * float(md["dust_KappaIR"])
                if md.get("dust_KappaIR") is not None else float("nan")),
    )


def collect_age_records(outputs, ages):
    """Map each age -> list of per-run records that reach it."""
    by_age = {}
    for age in ages:
        recs = [sample_run_at_age(o, age) for o in outputs]
        by_age[age] = [r for r in recs if r is not None]
    return by_age


def project_root():
    """Repository root (two levels above this file: barnes26 -> paper -> root)."""
    return Path(__file__).resolve().parents[2]


def apply_trinity_style():
    """Apply the shared ``trinity.mplstyle`` to matplotlib.

    Used instead of importing ``paper._lib.plot_base`` (whose module-level
    ``FIG_DIR.mkdir`` materialises a stray ``fig/`` directory as an import
    side effect). matplotlib is imported lazily so this module stays
    import-light.
    """
    import matplotlib.pyplot as plt
    style = Path(__file__).resolve().parents[1] / "_lib" / "trinity.mplstyle"
    if style.exists():
        plt.style.use(str(style))


# ---------------------------------------------------------------------------
# Population rendering helpers (used by the figure scripts in --population mode)
# ---------------------------------------------------------------------------
# These take a matplotlib Axes and use only its methods + numpy, so this module
# still pulls in no matplotlib at import time.
def binned_median(x, y, *, xscale="log", nbins=12, min_count=5):
    """Binned-median trend of ``y`` vs ``x``.

    Returns ``(centres, medians)`` over ``nbins`` x-bins (log-spaced when
    ``xscale == 'log'``), keeping only bins with at least ``min_count`` points.
    ``y`` may be negative (e.g. a log-ratio); only ``x`` is restricted (to > 0
    for log binning).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if xscale == "log":
        m &= x > 0
    x, y = x[m], y[m]
    if x.size == 0:
        return np.array([]), np.array([])
    if xscale == "log":
        edges = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
        centres = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(x.min(), x.max(), nbins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges)
    cx, cy = [], []
    for b in range(1, nbins + 1):
        sel = idx == b
        if int(sel.sum()) >= min_count:
            cx.append(float(centres[b - 1]))
            cy.append(float(np.median(y[sel])))
    return np.array(cx), np.array(cy)


# Okabe-Ito qualitative palette for discrete environments (the swept P_DE/PISM).
ENV_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000")


def pde_env_labels(records):
    """Map each distinct ``PISM`` (code units) in *records* to a P_DE legend label."""
    labels = {}
    for v in sorted({r["PISM"] for r in records if np.isfinite(r["PISM"])}):
        pk = float(pism_to_Pk(v))
        labels[v] = (rf"$P_{{\rm DE}}=10^{{{np.log10(pk):.1f}}}$ K cm$^{{-3}}$"
                     if pk > 0 else r"$P_{\rm DE}$ n/a")
    return labels


def scatter_median_by_env(ax, x, y, env, *, xscale="log", yscale="log",
                          env_labels=None, median=True, s=7, alpha=0.18,
                          nbins=12, min_count=20):
    """Scatter ``(x, y)`` coloured by environment + a per-environment median line.

    ``env`` is a per-point key (e.g. PISM). One Okabe-Ito colour per distinct
    environment (sorted ascending); the median line gets a white halo for
    contrast over the scatter. Returns ``[(label, colour), ...]`` for the
    legend. Uses only Axes methods + numpy (no matplotlib import here).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    env = np.asarray(env)
    finite = np.isfinite(x) & np.isfinite(y)
    if xscale == "log":
        finite &= x > 0
    if yscale == "log":
        finite &= y > 0
    handles = []
    for i, e in enumerate(sorted(set(env[finite].tolist()))):
        c = ENV_COLORS[i % len(ENV_COLORS)]
        m = finite & (env == e)
        ax.scatter(x[m], y[m], s=s, color=c, alpha=alpha, linewidths=0, rasterized=True)
        if median:
            cx, cy = binned_median(x[m], y[m], xscale=xscale, nbins=nbins, min_count=min_count)
            if cx.size:
                ax.plot(cx, cy, color="white", lw=4.0, zorder=5, solid_capstyle="round")
                ax.plot(cx, cy, color=c, lw=2.2, zorder=6, solid_capstyle="round")
        label = env_labels.get(e) if env_labels else str(e)
        handles.append((label, c))
    return handles
