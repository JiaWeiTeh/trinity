#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMS caveat figure for TRINITY Paper I appendix.

Panel A (top): individual-star L_bol(t) tracks for six initial masses
(0.5, 1, 3, 10, 30, 60 M_sun), from first MIST EEP through end of
track, with ZAMS arrival marked on each curve.  Coloured by initial
mass (plasma colourmap).

Panel B (bottom): cumulative luminosity fraction F(>M) = L_bol(>M) /
L_bol,tot for a Kroupa 10^6 M_sun cluster at five ages (0.1, 0.3, 1,
3, 10 Myr), using MIST basic isochrones.  Coloured by age (viridis
colourmap).  A dashed reference line at F = 0.98 marks the 98%-of-
light threshold the appendix quotes.

Each panel has its own legend.  Uses MIST v1.2 data at
[Fe/H] = 0.00, v/v_crit = 0.0 (non-rotating).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import PercentFormatter
from scipy.integrate import quad, cumulative_trapezoid

# Bootstrap project root and load trinity style via plot_base.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src._plots.plot_base import FIG_DIR, PROJECT_ROOT  # noqa: E402
from src._functions.read_mist_models import EEP, ISO  # noqa: E402

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

# Legend labels for the Panel B ages.
AGE_LABELS = {
    5.0: r"$0.1\,\mathrm{Myr}$",
    5.5: r"$0.3\,\mathrm{Myr}$",
    6.0: r"$1\,\mathrm{Myr}$",
    6.5: r"$3\,\mathrm{Myr}$",
    7.0: r"$10\,\mathrm{Myr}$",
}

# Panel A per-curve PMS styling: fill_between each curve and the axis
# bottom over its own PMS segment, and draw the PMS portion of the line
# itself as a low-alpha dashed curve (MS stays solid at full alpha).
PMS_FILL_ALPHA = 0.18
PMS_LINE_ALPHA = 0.6
PMS_LINE_STYLE = "--"

# Kroupa (2001) broken power law.  IMF_M_MIN set to 0.1 so log10 = -1.0
# aligns exactly with the Panel B x-axis left edge.
KROUPA_BREAK = 0.5       # M_sun, slope break
KROUPA_ALPHA_LO = 1.3    # dN/dM propto M^-alpha for M < 0.5
KROUPA_ALPHA_HI = 2.3    # dN/dM propto M^-alpha for M >= 0.5
IMF_M_MIN = 0.08         # M_sun, hydrogen-burning limit
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


def _interp_F_at_logM(logM_sorted, F, logM_target):
    """Linear-interp F at a given log M; NaN if target is outside the range."""
    if logM_target < logM_sorted[0] or logM_target > logM_sorted[-1]:
        return float("nan")
    i = int(np.searchsorted(logM_sorted, logM_target))
    if i == 0:
        return float(F[0])
    x0, x1 = logM_sorted[i - 1], logM_sorted[i]
    y0, y1 = F[i - 1], F[i]
    return float(y0 + (logM_target - x0) * (y1 - y0) / (x1 - x0))


def _logM_at_F(logM_sorted, F, F_target):
    """Linear-interp log M at a given F, where F is monotone decreasing in log M."""
    below = np.where(F < F_target)[0]
    if below.size == 0:
        return float("nan")  # F never drops below the target
    i = int(below[0])
    if i == 0:
        return float(logM_sorted[0])
    x0, x1 = logM_sorted[i - 1], logM_sorted[i]
    y0, y1 = F[i - 1], F[i]
    return float(x0 + (F_target - y0) * (x1 - x0) / (y1 - y0))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel_A(ax, tracks):
    """Plot L_bol(log t) for each mass on ``ax``.

    Absolute log10(star_age / yr) on the x-axis so the mass-dependent
    spread of ZAMS arrival times (~10^5 yr for 60 M_sun vs ~10^8 yr for
    0.5 M_sun) is preserved -- that spread is the point of the figure.
    Each curve gets its PMS segment shaded in its own colour at low
    alpha so the "absent-from-SB99" portion is visible per curve.
    """
    y_min, y_max = -2.0, 6.5

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

        zams_idx = find_zams_row(eep)
        if zams_idx is None or zams_idx < start:
            print(f"  [{mass:>4} Msun] no usable ZAMS row; plotted solid, no shading")
            ax.plot(log_t, log_L, color=colour, lw=1.2,
                    label=MASS_LABELS[mass], zorder=3)
            continue

        zi = zams_idx - start

        # PMS segment: dashed, lower alpha, no legend entry.
        ax.plot(log_t[: zi + 1], log_L[: zi + 1], color=colour, lw=1.2,
                linestyle=PMS_LINE_STYLE, alpha=PMS_LINE_ALPHA, zorder=3)
        # MS segment: solid, full alpha, carries the legend entry.  Start at
        # the ZAMS row so the two segments share a common endpoint.
        ax.plot(log_t[zi:], log_L[zi:], color=colour, lw=1.2,
                label=MASS_LABELS[mass], zorder=3)

        # Fill this curve's PMS segment down to the y-axis bottom.
        ax.fill_between(log_t[: zi + 1], log_L[: zi + 1], y_min,
                        color=colour, alpha=PMS_FILL_ALPHA,
                        linewidth=0, zorder=2)

        ax.plot(log_t[zi], log_L[zi], marker="o", color=colour,
                markersize=5, markeredgecolor="black",
                markeredgewidth=0.6, linestyle="None", zorder=5)
        print(f"  [{mass:>4} Msun] ZAMS at log10(t/yr) = {log_t[zi]:.3f}")

    ax.set_xlim(3.0, 10.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\log_{10}(t) \; [\mathrm{yr}]$")
    ax.set_ylabel(r"$\log_{10}(L_{\rm bol} / L_\odot)$")

    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)


def _panel_B_style(log_age):
    """Return (linestyle, alpha, colour_override) for Panel B.

    Late ages (>= 10 Myr) are drawn as a dashed grey, low-alpha line to
    read visually as a dying population and recede behind the young
    ages.  ``colour_override`` is ``None`` for the young curves, which
    use the viridis colour; a string like "0.5" for the late curves.
    """
    if log_age >= 7.0:
        return ("--", 0.6, "0.5")
    return ("-", 1.0, None)


def plot_panel_B(ax, iso):
    """
    Plot the cumulative luminosity fraction F(>M) at five ages.

    F(>M) = L_bol from stars with mass > M, divided by the cluster's
    total L_bol.  The y-axis is zoomed on F in [0.95, 1.005] so the
    98%-of-light regime is readable; curves fall off the bottom at
    higher masses.  A dotted horizontal line at F = 0.98 marks the
    "98% of light" threshold the
    appendix quotes; the mass where each curve crosses it is printed as
    log M(F=0.98).  Young-age curves (< 10 Myr) are drawn solid as the
    argument regime; the 10 Myr curve is drawn dashed to mark the
    narrative break where massive stars have died.

    Parameters
    ----------
    iso : read_mist_models.ISO
        Already-loaded basic isochrone set.
    """
    for log_age in LOG_AGES:
        ls, alpha, colour_override = _panel_B_style(log_age)
        colour = colour_override if colour_override is not None \
            else AGE_CMAP(AGE_NORM(log_age))
        idx = iso.age_index(log_age)
        matched = iso.ages[idx]
        if abs(matched - log_age) > 0.025:
            print(
                f"  [log t = {log_age:.2f}] WARNING: snapped to {matched:.3f} "
                "(grid does not contain the requested value)"
            )
        slice_ = iso.isos[idx]

        M = np.asarray(slice_["initial_mass"], dtype=float)
        log_L = np.asarray(slice_["log_L"], dtype=float)

        good = np.isfinite(M) & np.isfinite(log_L) & (M >= IMF_M_MIN)
        M = M[good]
        log_L = log_L[good]

        # Sort ascending in M so the cumulative integral is monotonic.
        order = np.argsort(M)
        M = M[order]
        log_L = log_L[order]
        logM = np.log10(M)

        # Integrate in log M: the Kroupa integrand L*xi is a steep power
        # law in linear M but smooth in log M, so trapezoid error drops.
        # dL = L * xi * dM = L * xi * M * ln(10) * d(log10 M).
        L = 10.0 ** log_L
        xi = kroupa_xi(M)
        dL_dlogM = L * xi * M * np.log(10.0)

        cum_from_below = cumulative_trapezoid(dL_dlogM, logM, initial=0.0)
        L_total = float(cum_from_below[-1])
        F_above = 1.0 - cum_from_below / L_total
        ax.plot(logM, F_above, color=colour, lw=1.2, linestyle=ls,
                alpha=alpha, label=AGE_LABELS[log_age], zorder=3)

        f_above_10 = _interp_F_at_logM(logM, F_above, 1.0)
        log_M_98 = _logM_at_F(logM, F_above, 0.98)
        print(
            f"  [log t = {log_age:.2f} -> matched {matched:.3f}] "
            f"L_total = {L_total:.3e} L_sun, "
            f"f(M>10) = {f_above_10*100:.1f}%, "
            f"log M(F=0.98) = {log_M_98:.2f}"
        )

    ax.set_xlim(-1.0, 2.1)
    ax.set_ylim(0.95, 1.005)
    ax.set_xlabel(r"$\log_{10}(M_{\rm init} / M_\odot)$")
    ax.set_ylabel(r"$L_{\rm bol}(>M) / L_{\rm bol,\,tot}$")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # Dotted 98% reference line: dark grey, lowered alpha so the curves
    # stay visually primary.  Dotted (not dashed) so it does not clash
    # with the dashed 10 Myr curve.
    ax.axhline(0.98, linestyle=":", color="0.3", lw=0.8,
               alpha=0.5, zorder=2)
    ax.text(2.0, 0.982, r"$98\%$", ha="right", va="bottom",
            color="0.3", alpha=0.7, fontsize=9)

    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=9)


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

    # A&A wide figure, side-by-side: Panel A on the left, Panel B on the right.
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.0, 4.2),
        gridspec_kw={"wspace": 0.28},
    )
    plot_panel_A(axA, tracks)
    plot_panel_B(axB, iso)

    fig.savefig(OUTFILE, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTFILE}")


if __name__ == "__main__":
    main()
