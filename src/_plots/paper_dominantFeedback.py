#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dominant Feedback Grid: 2D colormap showing which feedback mechanism
dominates at specific time snapshots (0.5, 1.0, 1.5, 2.0 Myr).

Grid: X = cloud mass, Y = SFE, Color = dominant force
White cells indicate no data (cloud collapsed/dissolved before that time).

Created for TRINITY project - A&A/MNRAS publication figures.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from load_snapshots import load_output, find_data_file

print("...plotting dominant feedback grid")

import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))

# ============== CONFIGURATION ==============
mCloud_list = ["1e5", "1e7", "1e8"]
sfe_list = ["001", "010", "020", "030", "050", "080"]
ndens = "1e4"
TARGET_TIMES = [0.5, 1.0, 1.5, 2.0]  # Myr

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"
# Output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Force field definitions (order matters for color indexing)
FORCE_FIELDS = [
    ("F_grav",     "Gravity",           "black"),
    ("F_ram",      "Ram",               "#1f77b4"),  # Matplotlib blue
    ("F_ion_out",  "Photoionised gas",  "#d62728"),  # Red
    ("F_rad",      "Radiation",         "#9467bd"),  # Purple
]

# Visualization modes:
# - "discrete": Sharp cell boundaries (original)
# - "interpolated": Bilinear interpolation (blurs categorical boundaries)
# - "smooth_contour": Upsampled grid with contour-like smooth boundaries
MESH_MODE = "smooth_contour"

# Upsampling factor for smooth_contour mode (higher = smoother)
UPSAMPLE_FACTOR = 40 #20

SAVE_PNG = False
SAVE_PDF = True

# ============== DATA LOADING ==============

def load_run(data_path: Path):
    """
    Load simulation data using TrinityOutput reader.

    Returns:
        t: Time array (Myr)
        forces: 2D array shape (n_forces, n_snapshots)
        None, None if file doesn't exist
    """
    if data_path is None or not data_path.exists():
        return None, None

    output = load_output(data_path)

    if len(output) == 0:
        return None, None

    # Extract time
    t = output.get('t_now')

    # Helper to get field with default
    def get_field(field, default=0.0):
        arr = output.get(field)
        if arr is None:
            return np.full(len(output), default)
        return np.nan_to_num(arr, nan=default)

    F_grav = get_field("F_grav", 0.0)
    F_ram = get_field("F_ram", 0.0)
    F_ion = get_field("F_ion_out", 0.0)
    F_rad = get_field("F_rad", 0.0)

    # Handle NaN in F_ram (reconstruct if possible)
    if np.all(np.isnan(output.get("F_ram"))):
        F_wind = get_field("F_ram_wind", np.nan)
        F_sn = get_field("F_ram_SN", np.nan)
        if not (np.all(np.isnan(F_wind)) and np.all(np.isnan(F_sn))):
            F_ram = np.nan_to_num(F_wind, nan=0.0) + np.nan_to_num(F_sn, nan=0.0)
        else:
            F_ram = np.zeros_like(t)

    # Stack forces: shape (4, n_snapshots)
    forces = np.vstack([F_grav, F_ram, F_ion, F_rad])

    # Ensure time is monotonically increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        forces = forces[:, order]

    return t, forces


# ============== INTERPOLATION & DOMINANCE ==============

def interp_finite(x, y, xnew):
    """
    Interpolate y at positions xnew using only finite values.
    Returns NaN if insufficient data or xnew is outside data range.
    """
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.full_like(xnew, np.nan, dtype=float)

    # Return NaN for extrapolation beyond data range
    result = np.interp(xnew, x[m], y[m], left=np.nan, right=np.nan)
    return result


def get_dominant_at_time(t, forces, target_time):
    """
    Find the dominant force at a specific time.

    Args:
        t: Time array (Myr)
        forces: Shape (n_forces, n_snapshots)
        target_time: Query time in Myr

    Returns:
        int: Index of dominant force (0-3)
        np.nan: If no data at that time
    """
    # Check if target_time is within simulation range
    if t is None or len(t) == 0:
        return np.nan

    if target_time > t.max() or target_time < t.min():
        return np.nan

    # Interpolate each force to target_time
    F_interp = np.array([
        interp_finite(t, forces[i], np.array([target_time]))[0]
        for i in range(forces.shape[0])
    ])

    # Check for valid interpolation
    if np.all(np.isnan(F_interp)) or np.all(F_interp == 0):
        return np.nan

    # Compute absolute force fractions
    F_abs = np.abs(F_interp)
    F_total = np.nansum(F_abs)

    if F_total == 0:
        return np.nan

    # Return index of maximum
    return np.nanargmax(F_abs)


def build_dominance_grid(target_time, mCloud_list, sfe_list, ndens, base_dir):
    """
    Build 2D array of dominant feedback indices for a given time.

    Returns:
        grid: Shape (len(sfe_list), len(mCloud_list))
              Values: 0-3 for force index, NaN for no data
    """
    n_sfe = len(sfe_list)
    n_mass = len(mCloud_list)
    grid = np.full((n_sfe, n_mass), np.nan, dtype=float)

    for j, mCloud in enumerate(mCloud_list):
        for i, sfe in enumerate(sfe_list):
            run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
            data_path = find_data_file(base_dir, run_name)

            t, forces = load_run(data_path)

            if t is not None:
                dominant = get_dominant_at_time(t, forces, target_time)
                grid[i, j] = dominant

    return grid


# ============== GRID SMOOTHING ==============

def smooth_categorical_grid(grid, upsample_factor=20, sigma=2.0):
    """
    Create smooth boundaries for categorical data using soft voting.

    For each category, we create a distance-based "membership" field,
    smooth it with Gaussian filter, then use argmax to get the final
    category at each point.

    Parameters:
        grid: 2D array of category indices (0-3), NaN for no data
        upsample_factor: How much to upsample the grid
        sigma: Gaussian smoothing sigma (in upsampled pixels)

    Returns:
        grid_smooth: Upsampled and smoothed grid
        extent: (x_min, x_max, y_min, y_max) for plotting
    """
    n_sfe, n_mass = grid.shape
    n_categories = 4  # Number of force types

    # Create binary masks for each category
    category_masks = np.zeros((n_categories, n_sfe, n_mass), dtype=float)
    for cat in range(n_categories):
        category_masks[cat] = (grid == cat).astype(float)

    # Upsample each category mask
    upsampled_masks = np.zeros((n_categories,
                                 n_sfe * upsample_factor,
                                 n_mass * upsample_factor), dtype=float)

    for cat in range(n_categories):
        # Use spline interpolation for smooth upsampling
        upsampled_masks[cat] = zoom(category_masks[cat], upsample_factor, order=1)

    # Apply Gaussian smoothing to each category mask
    smoothed_masks = np.zeros_like(upsampled_masks)
    for cat in range(n_categories):
        smoothed_masks[cat] = gaussian_filter(upsampled_masks[cat], sigma=sigma)

    # Get final category by argmax (soft voting)
    grid_smooth = np.argmax(smoothed_masks, axis=0).astype(float)

    # Handle NaN regions: find where original had NaN
    nan_mask = np.isnan(grid)
    nan_mask_upsampled = zoom(nan_mask.astype(float), upsample_factor, order=0)
    grid_smooth[nan_mask_upsampled > 0.5] = np.nan

    # Also mark as NaN where no category has significant probability
    max_prob = np.max(smoothed_masks, axis=0)
    grid_smooth[max_prob < 0.01] = np.nan

    # Compute extent for imshow
    extent = [-0.5, n_mass - 0.5, -0.5, n_sfe - 0.5]

    return grid_smooth, extent


def upsample_nearest(grid, upsample_factor=20):
    """
    Simple nearest-neighbor upsampling (keeps sharp boundaries but at higher resolution).
    """
    return zoom(grid, upsample_factor, order=0)


# ============== PLOTTING ==============

def create_colormap():
    """
    Create discrete colormap for 4 force types.
    NaN values will be rendered as white.
    """
    colors = [field[2] for field in FORCE_FIELDS]
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')  # For NaN values

    # Boundaries for 4 discrete values (0, 1, 2, 3)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def plot_dominance_grid(ax, grid, mCloud_list, sfe_list, target_time,
                        cmap, norm, mode="discrete", upsample_factor=20):
    """
    Plot a single dominance grid on an axis.

    Args:
        ax: Matplotlib axis
        grid: 2D array of dominant force indices
        mCloud_list: Cloud mass labels
        sfe_list: SFE labels
        target_time: Time in Myr for title
        cmap: Colormap
        norm: BoundaryNorm
        mode: "discrete", "interpolated", or "smooth_contour"
        upsample_factor: Factor for upsampling in smooth_contour mode
    """
    mass_indices = np.arange(len(mCloud_list))
    sfe_indices = np.arange(len(sfe_list))

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(
        np.arange(len(mCloud_list) + 1) - 0.5,
        np.arange(len(sfe_list) + 1) - 0.5
    )

    # Mask NaN values for proper white display
    grid_masked = np.ma.masked_invalid(grid)

    if mode == "smooth_contour":
        # Use soft voting with Gaussian smoothing for smooth region boundaries
        # sigma scales with upsample_factor to get consistent smoothness
        sigma = upsample_factor * 0.8
        grid_smooth, extent = smooth_categorical_grid(grid, upsample_factor, sigma)
        grid_smooth_masked = np.ma.masked_invalid(grid_smooth)

        im = ax.imshow(
            grid_smooth_masked,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest'  # Already smooth, don't blur more
        )

    elif mode == "interpolated":
        # Use imshow with interpolation for smooth transitions
        # Note: This blurs categorical boundaries (not ideal for categorical data)
        im = ax.imshow(
            grid_masked,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            origin='lower',
            extent=[-0.5, len(mCloud_list)-0.5, -0.5, len(sfe_list)-0.5],
            interpolation='bilinear'
        )
    else:
        # Discrete cells with clean edges (original mode)
        im = ax.pcolormesh(
            X, Y, grid_masked,
            cmap=cmap,
            norm=norm,
            edgecolors='white',
            linewidths=0.5
        )

    # Set axis labels and ticks
    ax.set_xticks(mass_indices)
    ax.set_xticklabels([rf"$10^{{{int(np.log10(float(m)))}}}$" for m in mCloud_list])

    ax.set_yticks(sfe_indices)
    ax.set_yticklabels([f"{int(s)/100:.2f}" for s in sfe_list])

    # Title with time
    ax.set_title(rf"$t = {target_time}$ Myr", fontsize=11)

    return im


def create_legend(force_fields):
    """
    Create legend handles showing force-color mapping.
    """
    handles = [
        Patch(facecolor=color, edgecolor='0.3', label=label)
        for _, label, color in force_fields
    ]
    # Add white for "no data"
    handles.append(
        Patch(facecolor='white', edgecolor='0.3', label='No data')
    )

    return handles


# ============== MAIN ==============

def main():
    """Main function to generate the dominant feedback grid plot."""

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Create figure: 2x2 grid
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(7.1, 6.0),  # Double column width for A&A
        constrained_layout=True,
        dpi=300
    )

    cmap, norm = create_colormap()

    # Plot each time snapshot
    for ax, target_time in zip(axes.flat, TARGET_TIMES):
        print(f"  Building grid for t = {target_time} Myr...")

        grid = build_dominance_grid(
            target_time, mCloud_list, sfe_list, ndens, BASE_DIR
        )

        plot_dominance_grid(
            ax, grid, mCloud_list, sfe_list, target_time,
            cmap, norm, mode=MESH_MODE, upsample_factor=UPSAMPLE_FACTOR
        )

    # Add shared axis labels
    fig.supxlabel(r"$M_{\rm cloud}$ [$M_\odot$]", fontsize=12)
    fig.supylabel(r"Star Formation Efficiency $\epsilon$", fontsize=12)

    # Add figure title
    nlog = int(np.log10(float(ndens)))
    fig.suptitle(
        rf"Dominant Feedback Mechanism ($n = 10^{{{nlog}}}$ cm$^{{-3}}$)",
        fontsize=12, y=1.02
    )

    # Add legend
    handles = create_legend(FORCE_FIELDS)
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        facecolor='white',
        edgecolor='0.3'
    )

    # Save figure
    tag = f"dominant_feedback_n{ndens}_{MESH_MODE}"

    if SAVE_PNG:
        out_png = FIG_DIR / f"{tag}.png"
        fig.savefig(out_png, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_png}")

    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
