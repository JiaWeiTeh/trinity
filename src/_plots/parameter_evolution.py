#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter evolution plotter for TRINITY simulation outputs.

Reads a dictionary.jsonl file and plots a grid of parameter evolution over time.

Usage:
    python parameter_evolution.py <path_to_jsonl>
    python parameter_evolution.py /path/to/output/dictionary.jsonl
    python parameter_evolution.py /path/to/output/dictionary.jsonl --save-pdf
    python parameter_evolution.py /path/to/output/dictionary.jsonl --params t_now,R2,Eb

Author: TRINITY Team
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# =============================================================================
# Configuration: Parameters to track
# =============================================================================
# Modify this list to add/remove parameters. Format: (key, label, unit)
# If unit is None, no unit will be shown.

DEFAULT_PARAMETERS = [
    # Time and radii
    ("t_now", r"$t$", "Myr"),
    ("R2", r"$R_2$", "pc"),
    ("R1", r"$R_1$", "pc"),
    ("rShell", r"$r_{\rm shell}$", "pc"),

    # Velocities
    ("v2", r"$v_2$", "pc/Myr"),
    ("v_mech_total", r"$v_{\rm mech}$", "pc/Myr"),

    # Energy and luminosities
    ("Eb", r"$E_b$", "erg"),
    ("Lbol", r"$L_{\rm bol}$", None),
    ("Li", r"$L_i$", None),
    ("L_mech_total", r"$L_{\rm mech}$", None),
    ("Ln", r"$L_n$", None),

    # Forces
    ("F_grav", r"$F_{\rm grav}$", None),
    ("F_ram", r"$F_{\rm ram}$", None),
    ("F_ram_wind", r"$F_{\rm ram,wind}$", None),
    ("F_ram_SN", r"$F_{\rm ram,SN}$", None),
    ("F_ion_in", r"$F_{\rm ion,in}$", None),
    ("F_ion_out", r"$F_{\rm ion,out}$", None),
    ("F_rad", r"$F_{\rm rad}$", None),
    ("F_ISM", r"$F_{\rm ISM}$", None),
    ("F_SN", r"$F_{\rm SN}$", None),

    # Pressures
    ("PISM", r"$P_{\rm ISM}$", None),
    ("Pb", r"$P_b$", None),

    # Ionization
    ("Qi", r"$Q_i$", "s$^{-1}$"),
    ("T0", r"$T_0$", "K"),

    # Shell properties
    ("shell_mass", r"$M_{\rm shell}$", r"$M_\odot$"),
    ("shell_massDot", r"$\dot{M}_{\rm shell}$", r"$M_\odot$/Myr"),
    ("shell_thickness", r"$\Delta r_{\rm shell}$", "pc"),
    ("shell_fAbsorbedIon", r"$f_{\rm abs,ion}$", None),
    ("shell_nMax", r"$n_{\rm max}$", "cm$^{-3}$"),
]


# =============================================================================
# Data loading functions
# =============================================================================

def load_jsonl(filepath: Path) -> list:
    """
    Load a JSONL file (line-delimited JSON).

    Each line is a JSON object representing one snapshot.

    Parameters
    ----------
    filepath : Path
        Path to the .jsonl file

    Returns
    -------
    list
        List of snapshot dictionaries, sorted by snapshot index
    """
    snapshots = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
                snapshots.append(snap)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    # Sort by snapshot index if available
    if snapshots and 'snap_id' in snapshots[0]:
        snapshots.sort(key=lambda s: s.get('snap_id', 0))

    return snapshots


def load_json(filepath: Path) -> list:
    """
    Load a regular JSON file (dictionary format).

    The file has structure: {"0": {...}, "1": {...}, ...}

    Parameters
    ----------
    filepath : Path
        Path to the .json file

    Returns
    -------
    list
        List of snapshot dictionaries, sorted by snapshot index
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get numeric keys and sort
    snap_keys = sorted(
        (k for k in data.keys() if str(k).isdigit()),
        key=lambda k: int(k)
    )

    return [data[k] for k in snap_keys]


def load_data(filepath: Path) -> list:
    """
    Auto-detect file format and load data.

    Supports both .jsonl (line-delimited) and .json (dictionary) formats.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try JSONL first (preferred new format)
    if filepath.suffix == '.jsonl':
        return load_jsonl(filepath)
    elif filepath.suffix == '.json':
        return load_json(filepath)
    else:
        # Try to auto-detect by reading first character
        with open(filepath, 'r') as f:
            first_char = f.read(1)
        if first_char == '{':
            # Could be JSONL (object per line) or JSON (single object)
            # Try JSONL first
            try:
                snaps = load_jsonl(filepath)
                if len(snaps) > 0:
                    return snaps
            except:
                pass
            return load_json(filepath)
        else:
            return load_jsonl(filepath)


def extract_time_series(snapshots: list, key: str) -> tuple:
    """
    Extract time and parameter arrays from snapshots.

    Parameters
    ----------
    snapshots : list
        List of snapshot dictionaries
    key : str
        Parameter key to extract

    Returns
    -------
    tuple
        (time_array, value_array) as numpy arrays
    """
    t = []
    values = []

    for snap in snapshots:
        t_val = snap.get('t_now', np.nan)
        val = snap.get(key, np.nan)

        # Handle None values
        if val is None:
            val = np.nan

        t.append(t_val)
        values.append(val)

    t = np.array(t, dtype=float)
    values = np.array(values, dtype=float)

    # Sort by time
    order = np.argsort(t)
    return t[order], values[order]


# =============================================================================
# Plotting functions
# =============================================================================

def plot_parameter_grid(
    filepath: Path,
    parameters: list = None,
    ncols: int = 4,
    figsize_per_panel: tuple = (3.5, 2.5),
    save_pdf: bool = False,
    save_png: bool = False,
    output_dir: Path = None,
    log_scale_params: set = None,
):
    """
    Plot a grid of parameter evolution over time.

    Parameters
    ----------
    filepath : Path
        Path to the data file (.jsonl or .json)
    parameters : list, optional
        List of (key, label, unit) tuples. If None, uses DEFAULT_PARAMETERS.
    ncols : int
        Number of columns in the grid
    figsize_per_panel : tuple
        (width, height) per panel in inches
    save_pdf : bool
        Save as PDF
    save_png : bool
        Save as PNG
    output_dir : Path, optional
        Output directory for saved figures. If None, uses current directory.
    log_scale_params : set, optional
        Set of parameter keys to plot with log scale on y-axis
    """
    filepath = Path(filepath)

    # Load mplstyle if available
    style_path = Path(__file__).parent / 'trinity.mplstyle'
    if style_path.exists():
        plt.style.use(str(style_path))

    # Load data
    print(f"Loading data from: {filepath}")
    snapshots = load_data(filepath)
    print(f"Loaded {len(snapshots)} snapshots")

    if len(snapshots) == 0:
        print("Error: No snapshots found in file")
        return

    # Use default parameters if not specified
    if parameters is None:
        parameters = DEFAULT_PARAMETERS

    # Default log scale parameters (typically energies, luminosities)
    if log_scale_params is None:
        log_scale_params = {
            'Eb', 'Lbol', 'Li', 'L_mech_total', 'Ln',
            'F_grav', 'F_ram', 'F_ram_wind', 'F_ram_SN',
            'F_ion_in', 'F_ion_out', 'F_rad', 'F_ISM', 'F_SN',
            'PISM', 'Pb', 'Qi', 'shell_mass', 'shell_massDot',
        }

    # Filter to only available parameters
    available_keys = set()
    for snap in snapshots:
        available_keys.update(snap.keys())

    valid_params = []
    for param in parameters:
        key = param[0]
        if key in available_keys:
            valid_params.append(param)
        else:
            print(f"  Skipping '{key}' (not found in data)")

    if len(valid_params) == 0:
        print("Error: No valid parameters found in data")
        print(f"Available keys: {sorted(available_keys)}")
        return

    print(f"Plotting {len(valid_params)} parameters")

    # Calculate grid dimensions
    nparams = len(valid_params)
    nrows = int(np.ceil(nparams / ncols))

    # Create figure
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=figsize,
        dpi=150,
        constrained_layout=True
    )

    # Flatten axes for easy iteration
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    # Plot each parameter
    for idx, (key, label, unit) in enumerate(valid_params):
        ax = axes_flat[idx]

        t, values = extract_time_series(snapshots, key)

        # Skip if all NaN
        if np.all(np.isnan(values)):
            ax.text(0.5, 0.5, f"{key}\n(no data)",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='gray')
            ax.set_axis_off()
            continue

        # Plot
        ax.plot(t, values, 'b-', lw=1.2)

        # Labels
        ylabel = label
        if unit:
            ylabel = f"{label} [{unit}]"
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel(r"$t$ [Myr]", fontsize=9)

        # Log scale for certain parameters
        if key in log_scale_params:
            # Only use log scale if values are positive
            finite_vals = values[np.isfinite(values)]
            if len(finite_vals) > 0 and np.all(finite_vals > 0):
                ax.set_yscale('log')

        # Title (parameter name)
        ax.set_title(key, fontsize=10, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3)

        # Tick parameters
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(len(valid_params), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Main title
    fig.suptitle(f"Parameter Evolution: {filepath.stem}", fontsize=12, fontweight='bold')

    # Save figures - default to same directory as input file
    if output_dir is None:
        output_dir = filepath.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"param_evolution_{filepath.stem}"

    if save_pdf:
        pdf_path = output_dir / f"{base_name}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved: {pdf_path}")

    if save_png:
        png_path = output_dir / f"{base_name}.png"
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {png_path}")

    plt.show()
    plt.close(fig)

    return fig


def list_available_keys(filepath: Path):
    """Print all available keys in the data file."""
    snapshots = load_data(filepath)

    if len(snapshots) == 0:
        print("No snapshots found")
        return

    # Collect all keys across all snapshots
    all_keys = set()
    for snap in snapshots:
        all_keys.update(snap.keys())

    print(f"\nAvailable keys in {filepath.name}:")
    print("-" * 40)
    for key in sorted(all_keys):
        print(f"  {key}")
    print(f"\nTotal: {len(all_keys)} keys")


# =============================================================================
# Command-line interface
# =============================================================================

def parse_params_string(params_str: str) -> list:
    """
    Parse a comma-separated parameter string into a list of tuples.

    Example: "t_now,R2,Eb" -> [("t_now", "t_now", None), ...]
    """
    keys = [k.strip() for k in params_str.split(',') if k.strip()]

    # Look up labels from DEFAULT_PARAMETERS
    default_dict = {p[0]: p for p in DEFAULT_PARAMETERS}

    result = []
    for key in keys:
        if key in default_dict:
            result.append(default_dict[key])
        else:
            # Use key as label if not in defaults
            result.append((key, key, None))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot parameter evolution from TRINITY simulation output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dictionary.jsonl
  %(prog)s /path/to/dictionary.jsonl --save-pdf
  %(prog)s /path/to/dictionary.jsonl --params R2,v2,Eb,shell_mass
  %(prog)s /path/to/dictionary.jsonl --list-keys
        """
    )

    parser.add_argument(
        'filepath',
        type=str,
        help='Path to the dictionary.jsonl (or .json) file'
    )

    parser.add_argument(
        '--params', '-p',
        type=str,
        default=None,
        help='Comma-separated list of parameters to plot (default: all available)'
    )

    parser.add_argument(
        '--ncols', '-c',
        type=int,
        default=4,
        help='Number of columns in the grid (default: 4)'
    )

    parser.add_argument(
        '--save-pdf',
        action='store_true',
        help='Save figure as PDF'
    )

    parser.add_argument(
        '--save-png',
        action='store_true',
        help='Save figure as PNG'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for saved figures (default: same as input file)'
    )

    parser.add_argument(
        '--list-keys', '-l',
        action='store_true',
        help='List all available keys in the file and exit'
    )

    args = parser.parse_args()

    filepath = Path(args.filepath)

    if args.list_keys:
        list_available_keys(filepath)
        return

    # Parse custom parameters if provided
    parameters = None
    if args.params:
        parameters = parse_params_string(args.params)
        print(f"Custom parameters: {[p[0] for p in parameters]}")

    # Plot
    plot_parameter_grid(
        filepath=filepath,
        parameters=parameters,
        ncols=args.ncols,
        save_pdf=args.save_pdf,
        save_png=args.save_png,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
