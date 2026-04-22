#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMS caveat figure for TRINITY Paper I appendix -- Panel A only.

Panel A: individual-star L_bol(t) tracks for six initial masses
(0.5, 1, 3, 10, 30, 60 M_sun), from first MIST EEP through end of
track, with ZAMS arrival marked on each curve.

Uses MIST v1.2 EEP tracks at [Fe/H] = 0.00, v/v_crit = 0.0 (non-rotating).

Panel B (Kroupa-IMF-weighted dL/d(log M) at five ages) will be added
in a follow-up commit; the layout will be refactored into a two-panel
figure at that time.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Bootstrap project root and load trinity style via plot_base.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src._plots.plot_base import FIG_DIR, PROJECT_ROOT  # noqa: E402
from src._functions.read_mist_models import EEP  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MIST grid: solar metallicity, non-rotating (vvcrit=0.0).
# vvcrit=0.0 is the conservative choice; rotating models give slightly
# shorter PMS lifetimes and brighter massive-star tracks, but the
# difference is small for the caveat plot.
MIST_DIR = PROJECT_ROOT / "lib" / "sps" / "mist" / \
    "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS"

MASSES = [0.5, 1.0, 3.0, 10.0, 30.0, 60.0]

MASS_LABELS = {
    0.5:  r"$0.5\,M_\odot$",
    1.0:  r"$1\,M_\odot$",
    3.0:  r"$3\,M_\odot$",
    10.0: r"$10\,M_\odot$",
    30.0: r"$30\,M_\odot$",
    60.0: r"$60\,M_\odot$",
}

# Wong 2011 colourblind-safe palette, one colour per mass in
# increasing-mass order (0.5 M_sun gets the first entry).
WONG_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel_A(ax, tracks):
    """Plot L_bol(t) for each mass on ``ax``, marking ZAMS on each curve."""
    for mass, colour in zip(MASSES, WONG_PALETTE):
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

        ax.plot(log_t, log_L, color=colour, lw=1.2,
                label=MASS_LABELS[mass], zorder=3)

        zams_idx = find_zams_row(eep)
        if zams_idx is None:
            print(f"  [{mass:>4} Msun] no MS phase row found -- ZAMS marker skipped")
            continue
        # zams_idx is an index into the full table; account for skipped row.
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

    # Legend: 3 columns x 2 rows, placed above the axis (paper_Rosette style).
    handles, labels = ax.get_legend_handles_labels()
    ax.figure.legend(
        handles, labels,
        loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 1.12),
    )

    # Panel label (a)
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes,
            ha="left", va="top")


def plot_panel_B(ax, tracks):
    """TODO: Kroupa-IMF-weighted dL/d(log M) at five ages. Added in follow-up commit."""
    pass


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

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    plot_panel_A(ax, tracks)

    fig.savefig(OUTFILE, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
