#!/usr/bin/env python3
"""Regenerate current-run companion plots for the mock CLOUDY deck."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STYLE = ROOT / "paper" / "_lib" / "trinity.mplstyle"
CSV = HERE / "1e5_sfe001_n1e3_PL0_yesPHII_current_run_timeseries.csv"
PREFIX = "1e5_sfe001_n1e3_PL0_yesPHII"

sys.path.insert(0, str(ROOT / "paper" / "rosette" / "matching"))
from observables import AGE_WINDOW_MYR  # noqa: E402


plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = True


data = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding=None)
data = data[(data["age_myr"] >= 0.0) & (data["age_myr"] <= 2.5)]

age = data["age_myr"]
r_bubble = data["r_bubble_pc"]
r_shell = data["r_shell_pc"]
p_hii = data["P_HII_over_kB_K_cm3"]

blue = "#0072B2"
green = "#009E73"
vermillion = "#D55E00"
orange = "#E69F00"

# Paper-facing Rosette anchors, kept local so this diagnostic plot does not
# silently inherit the older 7 +/- 1 pc matching target.
CAVITY_PC = 6.2  # Bruhweiler+2010 wind cavity, adopted in paper/rosette/main.tex.
HII_OUTER_PC = (19.0, 2.0)  # Celnik 1985 radio continuum/RRL outer HII edge.
MOLECULAR_RING_PC = 11.0  # Dent+2009 CO ring; different structure from r_shell.
DUST_SHELL_PC = (18.0, 22.0)  # Planck XXXIV 353 GHz cross-check, not a radius target.


def save(fig, ax, name):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(HERE / name)
    plt.close(fig)


def add_age_window(ax):
    lo, hi = AGE_WINDOW_MYR
    ax.axvspan(lo, hi, color="0.5", alpha=0.12, zorder=0)
    ax.axvline(0.5 * (lo + hi), color="k", ls="--", lw=1.0, alpha=0.65, zorder=1)


def add_radius_anchor(ax, value, err, color, marker, name, *, linestyle="--"):
    t = 0.5 * sum(AGE_WINDOW_MYR)
    label = rf"{name}: ${value:g}$ pc"
    if err is not None:
        ax.axhspan(value - err, value + err, color=color, alpha=0.13, zorder=0)
        label = rf"{name}: ${value:g}\pm{err:g}$ pc"
    ax.axhline(value, color=color, lw=1.0, ls=linestyle, alpha=0.65, zorder=1)
    ax.errorbar(
        t,
        value,
        yerr=err,
        fmt=marker,
        color=color,
        markersize=8,
        capsize=3.5 if err is not None else 0.0,
        capthick=1.2,
        markeredgecolor="k",
        markeredgewidth=0.5,
        zorder=10,
        label=label,
    )


fig, ax = plt.subplots(figsize=(4.8, 3.35))
ok = np.isfinite(r_bubble) & np.isfinite(p_hii) & (p_hii > 0)
ax.plot(r_bubble[ok], p_hii[ok], color=vermillion)
ax.set_yscale("log")
ax.set_xlabel(r"$r_{\rm bubble}$ [pc]")
ax.set_ylabel(r"$P_{\rm HII}/k_{\rm B}$ [K cm$^{-3}$]")
save(fig, ax, f"{PREFIX}_P_HII_vs_r_bubble.pdf")

fig, ax = plt.subplots(figsize=(4.8, 3.35))
ax.plot(age, r_bubble, color=blue)
add_age_window(ax)
add_radius_anchor(ax, CAVITY_PC, None, green, "^", "Wind cavity")
ax.set_xlim(0.0, 2.5)
ax.set_xlabel(r"Age [Myr]")
ax.set_ylabel(r"$r_{\rm bubble}$ [pc]")
ax.legend(loc="upper left")
save(fig, ax, f"{PREFIX}_r_bubble_vs_age.pdf")

fig, ax = plt.subplots(figsize=(4.8, 3.35))
ax.plot(age, r_shell, color=green)
add_age_window(ax)
add_radius_anchor(ax, HII_OUTER_PC[0], HII_OUTER_PC[1], blue, "o", "HII outer")
add_radius_anchor(ax, MOLECULAR_RING_PC, None, "0.35", "s", "CO ring", linestyle=":")
ax.axhspan(
    DUST_SHELL_PC[0],
    DUST_SHELL_PC[1],
    color=orange,
    alpha=0.15,
    zorder=0,
    label=fr"Dust shell ({DUST_SHELL_PC[0]:.0f}--{DUST_SHELL_PC[1]:.0f} pc)",
)
ax.set_xlim(0.0, 2.5)
ax.set_xlabel(r"Age [Myr]")
ax.set_ylabel(r"$r_{\rm shell}$ [pc]")
ax.legend(loc="upper left")
save(fig, ax, f"{PREFIX}_r_shell_vs_age.pdf")

obsolete = HERE / f"{PREFIX}_radii_vs_age.pdf"
if obsolete.exists():
    obsolete.unlink()
