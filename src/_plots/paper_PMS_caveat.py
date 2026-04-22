#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMS caveat figure for TRINITY Paper I appendix.

Panel A (top): individual-star L_bol(t) tracks for six initial masses
(0.5, 1, 3, 10, 30, 60 M_sun), from first MIST EEP through end of
track, with ZAMS arrival marked on each curve.  Coloured by initial
mass (plasma colourmap).

Panel B (bottom): Kroupa-IMF-weighted dL_bol/d(log M) at five ages
(0.1, 0.3, 1, 3, 10 Myr) for a 10^6 M_sun cluster, using MIST basic
isochrones.  Coloured by age (viridis colourmap).  Shows that the
bolometric budget is concentrated above ~10 M_sun at all plotted ages.

Each panel has its own colourbar; no legends.  Uses MIST v1.2 data at
[Fe/H] = 0.00, v/v_crit = 0.0 (non-rotating).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from scipy.integrate import quad

# Bootstrap project root and load trinity style via plot_base.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src._plots.plot_base import FIG_DIR, PROJECT_ROOT  # noqa: E402
from src._functions.read_mist_models import EEP, ISO  # noqa: E402

# numpy.trapezoid is the 2.0+ name; numpy 1.x has numpy.trapz instead.
# numpy 2.x has removed trapz, so evaluate lazily.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MIST grid: solar metallicity, non-rotating (vvcrit=0.0).
# vvcrit=0.0 is the conservative choice; rotating models give slightly
# shorter PMS lifetimes and brighter massive-star tracks, but the
# difference is small for the caveat plot.
MIST_DIR = PROJECT_ROOT / "lib" / "sps" / "mist" / \
    "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS"

# MIST basic isochrone file (Panel B).  If the user has reorganised the
# download, edit this constant only.
ISO_FILE = PROJECT_ROOT / "lib" / "sps" / "mist" / \
    "MIST_v1.2_vvcrit0.0_basic_isos" / \
    "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_basic.iso"

MASSES = [0.5, 1.0, 3.0, 10.0, 30.0, 60.0]

MASS_LABELS = {
    0.5:  r"$0.5\,M_\odot$",
    1.0:  r"$1\,M_\odot$",
    3.0:  r"$3\,M_\odot$",
    10.0: r"$10\,M_\odot$",
    30.0: r"$30\,M_\odot$",
    60.0: r"$60\,M_\odot$",
}

# Panel A: mass -> colour via plasma colourmap on a log-mass axis.
# Distinct colourmap from Panel B so the two colourbars are not confused
# for representing the same quantity.
MASS_CMAP = plt.get_cmap("plasma")
MASS_NORM = LogNorm(vmin=min(MASSES), vmax=max(MASSES))

# Ages for Panel B, in log10(t/yr).
LOG_AGES = [5.0, 5.5, 6.0, 6.5, 7.0]

# Panel B: log_age -> colour via viridis on a linear log-age axis.
AGE_CMAP = plt.get_cmap("viridis")
AGE_NORM = Normalize(vmin=min(LOG_AGES), vmax=max(LOG_AGES))

# Tick labels for the Panel B age colourbar.
AGE_TICK_LABELS = {
    5.0: r"$0.1$",
    5.5: r"$0.3$",
    6.0: r"$1$",
    6.5: r"$3$",
    7.0: r"$10$",
}

# Kroupa (2001) broken power law.  IMF_M_MIN set to 0.1 so log10 = -1.0
# aligns exactly with the Panel B x-axis left edge.
KROUPA_BREAK = 0.5       # M_sun, slope break
KROUPA_ALPHA_LO = 1.3    # dN/dM propto M^-alpha for M < 0.5
KROUPA_ALPHA_HI = 2.3    # dN/dM propto M^-alpha for M >= 0.5
IMF_M_MIN = 0.1          # M_sun
IMF_M_MAX = 120.0        # M_sun
CLUSTER_MASS = 1e6       # M_sun

OUTFILE = FIG_DIR / "paper_PMS_caveat.pdf"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def mist_filename(mass):
    """MIST track filename: mass in units of 0.01 M_sun, zero-padded to 5 digits."""
    return f"{int(round(mass * 100)):05d}M.track.eep"


def check_mist_dir():
    """Verify MIST_DIR exists and contains the expected files. Exit 1 otherwise."""
    if not MIST_DIR.is_dir():
        print(
            f"ERROR: MIST track directory not found at {MIST_DIR}\n"
            "Expected layout:\n"
            f"    {MIST_DIR}/00050M.track.eep\n"
            f"    {MIST_DIR}/00100M.track.eep\n"
            "    ...\n"
            "See the MIST data acquisition doc (MIST_data_acquisition.md) "
            "for download instructions."
        )
        sys.exit(1)

    missing = [m for m in MASSES if not (MIST_DIR / mist_filename(m)).exists()]
    if missing:
        available = sorted(p.name for p in MIST_DIR.glob("*.track.eep"))
        print(
            f"ERROR: missing MIST track file(s) in {MIST_DIR}:\n"
            + "\n".join(f"    {mist_filename(m)}  (M = {m} M_sun)" for m in missing)
            + "\nAvailable .track.eep files:\n"
            + "\n".join(f"    {n}" for n in available)
        )
        sys.exit(1)


def load_mist_track(mass):
    """Load a single MIST EEP track."""
    path = MIST_DIR / mist_filename(mass)
    return EEP(str(path), verbose=False)


def find_zams_row(eep):
    """
    Return the row index of the ZAMS in the EEP track, or None if absent.

    MIST uses the FSPS phase convention: -1 = PMS, 0 = MS, ...
    ZAMS is taken as the first row where phase == 0.
    """
    phase = eep.eeps["phase"]
    idx = np.where(phase == 0)[0]
    return int(idx[0]) if idx.size else None


def load_mist_isochrone():
    """Load the MIST basic isochrone file. Exit 1 if missing."""
    if not ISO_FILE.is_file():
        print(
            f"ERROR: MIST isochrone file not found at {ISO_FILE}\n"
            "Expected layout:\n"
            f"    {ISO_FILE}\n"
            "See the MIST data acquisition doc (MIST_data_acquisition.md) "
            "for download instructions."
        )
        sys.exit(1)
    return ISO(str(ISO_FILE), verbose=False)


def _kroupa_xi_scalar(m):
    """Unnormalised Kroupa 2001 IMF evaluated on a scalar."""
    if m < KROUPA_BREAK:
        return m ** (-KROUPA_ALPHA_LO)
    return (KROUPA_BREAK ** (KROUPA_ALPHA_HI - KROUPA_ALPHA_LO)
            * m ** (-KROUPA_ALPHA_HI))


def kroupa_xi_unnormalised(M):
    """Kroupa 2001 broken power law, unnormalised.  dN/dM.

    Vectorised; accepts arrays or scalars and returns an array.
    """
    M = np.asarray(M, dtype=float)
    return np.where(
        M < KROUPA_BREAK,
        M ** (-KROUPA_ALPHA_LO),
        KROUPA_BREAK ** (KROUPA_ALPHA_HI - KROUPA_ALPHA_LO)
        * M ** (-KROUPA_ALPHA_HI),
    )


def _kroupa_norm():
    """Normalisation so int M*xi(M) dM = CLUSTER_MASS over [IMF_M_MIN, IMF_M_MAX].

    Integrand has a slope discontinuity at M = KROUPA_BREAK; split the
    interval there so `quad` does not warn about convergence.
    """
    integrand = lambda m: m * _kroupa_xi_scalar(m)
    lo, _ = quad(integrand, IMF_M_MIN, KROUPA_BREAK)
    hi, _ = quad(integrand, KROUPA_BREAK, IMF_M_MAX)
    return CLUSTER_MASS / (lo + hi)


KROUPA_NORM = _kroupa_norm()


def kroupa_xi(M):
    """Normalised Kroupa IMF in stars per unit M: int M*xi(M) dM = CLUSTER_MASS."""
    return KROUPA_NORM * kroupa_xi_unnormalised(M)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel_A(ax, tracks):
    """Plot L_bol(t) for each mass on ``ax``, marking ZAMS on each curve.

    Returns the ScalarMappable used for the mass colourbar so the caller
    can attach it to the figure.
    """
    for mass in MASSES:
        colour = MASS_CMAP(MASS_NORM(mass))
        eep = tracks[mass]
        age = np.asarray(eep.eeps["star_age"], dtype=float)
        log_L = np.asarray(eep.eeps["log_L"], dtype=float)

        # Guard against star_age == 0 at the first EEP (log10 undefined).
        start = 0
        if age.size and age[0] <= 0.0:
            start = 1
            print(
                f"  [{mass:>4} Msun] skipped first EEP row "
                f"(star_age = {age[0]:g})"
            )
        age = age[start:]
        log_L = log_L[start:]
        log_t = np.log10(age)

        ax.plot(log_t, log_L, color=colour, lw=1.2, zorder=3)

        zams_idx = find_zams_row(eep)
        if zams_idx is None:
            print(f"  [{mass:>4} Msun] no MS phase row found -- ZAMS marker skipped")
            continue
        if zams_idx < start:
            print(f"  [{mass:>4} Msun] ZAMS row was in the skipped prefix -- skipping marker")
            continue
        zi = zams_idx - start
        x_zams = log_t[zi]
        y_zams = log_L[zi]
        ax.plot(x_zams, y_zams, marker="o", color=colour,
                markersize=5, markeredgecolor="black",
                markeredgewidth=0.6, linestyle="None", zorder=5)
        print(f"  [{mass:>4} Msun] ZAMS at log10(t/yr) = {x_zams:.3f}")

    ax.set_xlim(3.0, 10.5)
    ax.set_ylim(-2.0, 6.5)
    ax.set_xlabel(r"$\log_{10}(t / \mathrm{yr})$")
    ax.set_ylabel(r"$\log_{10}(L_{\rm bol} / L_\odot)$")

    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes,
            ha="left", va="top")

    return ScalarMappable(norm=MASS_NORM, cmap=MASS_CMAP)


def plot_panel_B(ax, iso):
    """
    Plot dL_bol/d(log M) for a Kroupa cluster at five ages.

    Parameters
    ----------
    iso : read_mist_models.ISO
        Already-loaded basic isochrone set.
    """
    logM_hi_cut = 1.0  # log10(10 M_sun)

    for log_age in LOG_AGES:
        colour = AGE_CMAP(AGE_NORM(log_age))
        idx = iso.age_index(log_age)
        matched = iso.ages[idx]
        if abs(matched - log_age) > 0.01:
            print(
                f"  [log t = {log_age:.2f}] WARNING: snapped to {matched:.3f} "
                "(grid does not contain the requested value)"
            )
        slice_ = iso.isos[idx]

        M = np.asarray(slice_["initial_mass"], dtype=float)
        log_L = np.asarray(slice_["log_L"], dtype=float)

        # Keep only finite entries and masses within the IMF range.
        good = np.isfinite(M) & np.isfinite(log_L) & (M >= IMF_M_MIN)
        M = M[good]
        log_L = log_L[good]

        # dL/d(log M) = L(M) * xi(M) * M * ln(10)
        L = 10.0 ** log_L
        xi = kroupa_xi(M)
        dL_dlogM = L * xi * M * np.log(10.0)

        # Sort by log M so the curve is monotonic and trapezoid makes sense.
        logM = np.log10(M)
        order = np.argsort(logM)
        logM = logM[order]
        dL_dlogM = dL_dlogM[order]

        ax.plot(logM, dL_dlogM, color=colour, lw=1.2, zorder=3)

        # Diagnostics: total cluster L_bol and fraction from M > 10 M_sun.
        L_total = float(_trapezoid(dL_dlogM, logM))
        hi = logM >= logM_hi_cut
        L_hi = (
            float(_trapezoid(dL_dlogM[hi], logM[hi]))
            if hi.sum() >= 2 else 0.0
        )
        frac_hi = L_hi / L_total if L_total > 0 else float("nan")
        print(
            f"  [log t = {log_age:.2f} -> matched {matched:.3f}] "
            f"L_total = {L_total:.3e} L_sun, "
            f"f(M>10) = {frac_hi*100:.1f}%"
        )

    ax.set_xlim(-1.0, 2.1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_{\rm init} / M_\odot)$")
    ax.set_ylabel(r"$dL_{\rm bol} / d\log M \; [L_\odot]$")

    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes,
            ha="left", va="top")

    return ScalarMappable(norm=AGE_NORM, cmap=AGE_CMAP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_mist_dir()

    print(f"Loading MIST tracks from {MIST_DIR}")
    tracks = {m: load_mist_track(m) for m in MASSES}
    for m in MASSES:
        n = tracks[m].eeps.size
        print(f"  [{m:>4} Msun] minit={tracks[m].minit:.3f}, {n} EEP rows, "
              f"rot={tracks[m].rot}")

    print(f"\nLoading MIST isochrone from {ISO_FILE}")
    iso = load_mist_isochrone()
    print(f"  {iso.num_ages} isochrones, log_age range "
          f"[{min(iso.ages):.2f}, {max(iso.ages):.2f}]")
    print(f"  Kroupa normalisation: KROUPA_NORM = {KROUPA_NORM:.4e}")
    integrand = lambda m: m * KROUPA_NORM * _kroupa_xi_scalar(m)
    verify_lo, _ = quad(integrand, IMF_M_MIN, KROUPA_BREAK)
    verify_hi, _ = quad(integrand, KROUPA_BREAK, IMF_M_MAX)
    print(f"  Verified integral M*xi dM = {verify_lo + verify_hi:.4e} M_sun "
          f"(target {CLUSTER_MASS:.4e})")

    # A&A two-column figure, vertically stacked: Panel A on top, Panel B below.
    # Each panel gets its own colourbar on the right.
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(7.0, 5.5),
        gridspec_kw={"hspace": 0.35},
    )
    sm_A = plot_panel_A(axA, tracks)
    sm_B = plot_panel_B(axB, iso)

    # Panel A colourbar (mass).  Log-spaced ticks at the six plotted masses.
    cbar_A = fig.colorbar(sm_A, ax=axA, pad=0.02)
    cbar_A.set_label(r"$M_{\rm init} \; [M_\odot]$")
    cbar_A.set_ticks(MASSES)
    cbar_A.set_ticklabels([f"{m:g}" for m in MASSES])
    cbar_A.minorticks_off()

    # Panel B colourbar (age in Myr).  Linear in log_age; five ticks.
    cbar_B = fig.colorbar(sm_B, ax=axB, pad=0.02)
    cbar_B.set_label(r"$t \; [\mathrm{Myr}]$")
    cbar_B.set_ticks(LOG_AGES)
    cbar_B.set_ticklabels([AGE_TICK_LABELS[a] for a in LOG_AGES])
    cbar_B.minorticks_off()

    fig.savefig(OUTFILE, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTFILE}")


if __name__ == "__main__":
    main()
