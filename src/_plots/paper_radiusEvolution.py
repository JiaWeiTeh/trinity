#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input

print("...plotting radius evolution grid (with r_Tb)")

# ---------------- configuration ----------------
# Set SINGLE_MODE = True to plot a single run instead of grid
SINGLE_MODE = True

# Single run configuration (used when SINGLE_MODE = True)
SINGLE_MCLOUD = "1e7"
SINGLE_SFE = "020"
SINGLE_NDENS = "1e4"

# Grid configuration (used when SINGLE_MODE = False)
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
ndens_list  = ["1e2", "1e3", "1e4"]                 # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

PHASE_LINE = True
CLOUD_LINE = True
SHOW_WEAVER = True  # Show Weaver-like R ∝ t^(3/5) solution
SMOOTH_WINDOW = None        # e.g. 7 to smooth radii; None/1 disables
SMOOTH_MODE = "edge"


# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True  # set True if you also want PDFs


# radius line styles/colors (added r_Tb)
RADIUS_FIELDS = [
    ("R1",     r"$R_1$",              "#9467bd", "-",  1.6),  # purple
    ("R2",     r"$R_2$",              "k",       "-",  2.0),  # black
    ("rShell", r"$r_{\rm shell}$",    "#ff7f0e", "-",  1.6),  # orange
    ("r_Tb",   r"$r_{T_b}=R_2\,\xi_{T_b}$", "0.35", ":",  1.8),  # grey dotted
]


def range_tag(prefix, values, key=float):
    """Return e.g. 'M1e5-1e8' or 'sfe001-080' (or single value if only one)."""
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"

def smooth_1d(y, window, mode="edge"):
    if window is None or window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")


def load_run_radii(data_path: Path):
    """Load one run and return arrays sorted by snapshot index.

    Uses TrinityOutput reader for clean data access.
    Supports both JSON and JSONL formats.
    """
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Use TrinityOutput.get() for clean array extraction
    t = output.get('t_now')
    phase = np.array(output.get('current_phase', as_array=False))

    R1 = output.get('R1')
    R2 = output.get('R2')
    rShell = output.get('rShell')

    # r_Tb = R2 * bubble_xi_Tb
    xi_Tb = output.get('bubble_xi_Tb')
    r_Tb = R2 * xi_Tb

    rcloud = float(output[0].get('rCloud', np.nan))

    # ensure increasing time
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        R1, R2, rShell, r_Tb = R1[order], R2[order], rShell[order], r_Tb[order]

    return t, phase, R1, R2, rShell, r_Tb, rcloud


def compute_weaver_solution(t, R2, t_ref_frac=0.1):
    """
    Compute Weaver-like solution R ∝ t^(3/5).

    The classic Weaver et al. (1977) wind-blown bubble solution gives
    R(t) ∝ (L_wind / ρ_0)^(1/5) * t^(3/5).

    We normalize to match R2 at an early reference time.

    Parameters
    ----------
    t : array
        Time array
    R2 : array
        R2 radius array (used for normalization)
    t_ref_frac : float
        Fraction of time range to use as reference point (default 0.1 = 10%)

    Returns
    -------
    R_weaver : array
        Weaver solution normalized to R2
    """
    # Find valid (finite) R2 values
    valid_mask = np.isfinite(R2) & (R2 > 0) & np.isfinite(t) & (t > 0)
    if not np.any(valid_mask):
        return np.full_like(t, np.nan)

    # Use early time as reference (at t_ref_frac of valid time range)
    t_valid = t[valid_mask]
    R2_valid = R2[valid_mask]

    t_min, t_max = t_valid.min(), t_valid.max()
    t_ref = t_min + t_ref_frac * (t_max - t_min)

    # Find closest time to t_ref
    ref_idx = np.argmin(np.abs(t_valid - t_ref))
    t_ref_actual = t_valid[ref_idx]
    R_ref = R2_valid[ref_idx]

    # Weaver solution: R(t) = R_ref * (t / t_ref)^(3/5)
    R_weaver = R_ref * (t / t_ref_actual) ** (3.0 / 5.0)

    return R_weaver


def plot_radii_on_ax(
    ax, t, phase, R1, R2, rShell, r_Tb, rcloud,
    phase_line=True, cloud_line=True, show_weaver=False,
    smooth_window=None, smooth_mode="edge",
    label_pad_points=4
):
    fig = ax.figure

    # optional smoothing
    R1s = smooth_1d(R1, smooth_window, mode=smooth_mode)
    R2s = smooth_1d(R2, smooth_window, mode=smooth_mode)
    rSs = smooth_1d(rShell, smooth_window, mode=smooth_mode)
    rTbs = smooth_1d(r_Tb, smooth_window, mode=smooth_mode)

    # --- phase lines with mini labels
    if phase_line:
        # energy/implicit -> transition (T)
        idx_T = np.flatnonzero(
            np.isin(phase[:-1], ["energy", "implicit"]) & (phase[1:] == "transition")
        ) + 1
        for x in t[idx_T]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.97, "T",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="r", alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )

        # transition -> momentum (M)
        idx_M = np.flatnonzero((phase[:-1] == "transition") & (phase[1:] == "momentum")) + 1
        for x in t[idx_M]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.97, "M",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="r", alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )

    # --- breakout line: first time R2 > rcloud
    if cloud_line and np.isfinite(rcloud):
        idx = np.flatnonzero(np.isfinite(R2s) & (R2s > rcloud))
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.25, zorder=0)

            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                label_pad_points/72, 0, fig.dpi_scale_trans
            )
            ax.text(
                x_rc, 0.05, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="bottom",
                fontsize=8, color="k", alpha=0.8,
                rotation=90, zorder=5
            )

    # --- radii lines
    ax.plot(t, R1s,    lw=RADIUS_FIELDS[0][4], ls=RADIUS_FIELDS[0][3], color=RADIUS_FIELDS[0][2], label=RADIUS_FIELDS[0][1], zorder=3)
    ax.plot(t, R2s,    lw=RADIUS_FIELDS[1][4], ls=RADIUS_FIELDS[1][3], color=RADIUS_FIELDS[1][2], label=RADIUS_FIELDS[1][1], zorder=4)
    ax.plot(t, rSs,    lw=RADIUS_FIELDS[2][4], ls=RADIUS_FIELDS[2][3], color=RADIUS_FIELDS[2][2], label=RADIUS_FIELDS[2][1], zorder=3)
    ax.plot(t, rTbs,   lw=RADIUS_FIELDS[3][4], ls=RADIUS_FIELDS[3][3], color=RADIUS_FIELDS[3][2], label=RADIUS_FIELDS[3][1], zorder=3)

    # --- Weaver-like solution: R ∝ t^(3/5)
    if show_weaver:
        R_weaver = compute_weaver_solution(t, R2s)
        ax.plot(t, R_weaver, lw=1.5, ls="--", color="k", alpha=0.6,
                label=r"Weaver: $R \propto t^{3/5}$", zorder=2)

    ax.set_xlim(t.min(), t.max())


# ---------------- run plotting ----------------
import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


def plot_single_run(mCloud, sfe, ndens):
    """Plot a single run's radius evolution."""
    run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
    data_path = find_data_file(BASE_DIR, run_name)

    if data_path is None:
        print(f"Data file not found for {run_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    try:
        t, phase, R1, R2, rShell, r_Tb, rcloud = load_run_radii(data_path)
        plot_radii_on_ax(
            ax, t, phase, R1, R2, rShell, r_Tb, rcloud,
            phase_line=PHASE_LINE,
            cloud_line=CLOUD_LINE,
            show_weaver=SHOW_WEAVER,
            smooth_window=SMOOTH_WINDOW,
            smooth_mode=SMOOTH_MODE
        )
    except Exception as e:
        print(f"Error loading {run_name}: {e}")
        plt.close(fig)
        return

    # Title with run parameters
    mlog = int(np.log10(float(mCloud)))
    nlog = int(np.log10(float(ndens)))
    eps = int(sfe) / 100.0
    ax.set_title(
        rf"$M_{{\rm cloud}}=10^{{{mlog}}}\,M_\odot$, "
        rf"$\epsilon={eps:.2f}$, "
        rf"$n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$"
    )
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel("Radius [pc]")

    # Legend
    handles = [
        Line2D([0], [0], color=RADIUS_FIELDS[0][2], lw=RADIUS_FIELDS[0][4], ls=RADIUS_FIELDS[0][3], label=RADIUS_FIELDS[0][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[1][2], lw=RADIUS_FIELDS[1][4], ls=RADIUS_FIELDS[1][3], label=RADIUS_FIELDS[1][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[2][2], lw=RADIUS_FIELDS[2][4], ls=RADIUS_FIELDS[2][3], label=RADIUS_FIELDS[2][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[3][2], lw=RADIUS_FIELDS[3][4], ls=RADIUS_FIELDS[3][3], label=RADIUS_FIELDS[3][1]),
    ]
    if SHOW_WEAVER:
        handles.append(Line2D([0], [0], color="k", ls="--", alpha=0.6, lw=1.5, label=r"Weaver: $R \propto t^{3/5}$"))
    handles.extend([
        Line2D([0], [0], color="k", ls="--", alpha=0.25, lw=1.6, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0], [0], color="r", lw=2, alpha=0.3, label="phase change"),
    ])
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    # Save
    tag = f"radiusEvolution_M{mCloud}_sfe{sfe}_n{ndens}"
    if SAVE_PNG:
        out_png = FIG_DIR / f"{tag}.png"
        fig.savefig(out_png, bbox_inches="tight")
        print(f"Saved: {out_png}")
    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_from_path(data_input: str, output_dir: str = None):
    """
    Plot radius evolution from a direct data path/folder.

    Parameters
    ----------
    data_input : str
        Can be: folder name, folder path, or file path
    output_dir : str, optional
        Base directory for output folders
    """
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loading data from: {data_path}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    try:
        t, phase, R1, R2, rShell, r_Tb, rcloud = load_run_radii(data_path)
        plot_radii_on_ax(
            ax, t, phase, R1, R2, rShell, r_Tb, rcloud,
            phase_line=PHASE_LINE,
            cloud_line=CLOUD_LINE,
            show_weaver=SHOW_WEAVER,
            smooth_window=SMOOTH_WINDOW,
            smooth_mode=SMOOTH_MODE
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        plt.close(fig)
        return

    ax.set_title(f"Radius Evolution: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel("Radius [pc]")

    # Legend
    handles = [
        Line2D([0], [0], color=RADIUS_FIELDS[0][2], lw=RADIUS_FIELDS[0][4], ls=RADIUS_FIELDS[0][3], label=RADIUS_FIELDS[0][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[1][2], lw=RADIUS_FIELDS[1][4], ls=RADIUS_FIELDS[1][3], label=RADIUS_FIELDS[1][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[2][2], lw=RADIUS_FIELDS[2][4], ls=RADIUS_FIELDS[2][3], label=RADIUS_FIELDS[2][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[3][2], lw=RADIUS_FIELDS[3][4], ls=RADIUS_FIELDS[3][3], label=RADIUS_FIELDS[3][1]),
    ]
    if SHOW_WEAVER:
        handles.append(Line2D([0], [0], color="k", ls="--", alpha=0.6, lw=1.5, label=r"Weaver: $R \propto t^{3/5}$"))
    handles.extend([
        Line2D([0], [0], color="k", ls="--", alpha=0.25, lw=1.6, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0], [0], color="r", lw=2, alpha=0.3, label="phase change"),
    ])
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY radius evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_radiusEvolution.py 1e7_sfe020_n1e4
  python paper_radiusEvolution.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_radiusEvolution.py /path/to/dictionary.jsonl
  python paper_radiusEvolution.py  # (uses config at top of file)
        """
    )
    parser.add_argument(
        'data', nargs='?', default=None,
        help='Data input: folder name, folder path, or file path'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Base directory for output folders (default: TRINITY_OUTPUT_DIR or "outputs")'
    )

    args = parser.parse_args()

    if args.data:
        # Command-line mode: plot from specified path
        plot_from_path(args.data, args.output_dir)
    elif SINGLE_MODE:
        # Config mode: plot single run
        plot_single_run(SINGLE_MCLOUD, SINGLE_SFE, SINGLE_NDENS)
    else:
        # Config mode: plot grid
        for ndens in ndens_list:
            nrows, ncols = len(mCloud_list), len(sfe_list)

            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(3.2 * ncols, 2.6 * nrows),
                sharex=False, sharey=False,
                dpi=500,
                constrained_layout=False
            )

            fig.subplots_adjust(top=0.90)
            nlog = int(np.log10(float(ndens)))
            fig.suptitle(rf"Radius evolution ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.05)

            m_tag   = range_tag("M",   mCloud_list, key=float)
            sfe_tag = range_tag("sfe", sfe_list,    key=int)
            n_tag   = f"n{ndens}"
            tag = f"radius_grid_{m_tag}_{sfe_tag}_{n_tag}"

            for i, mCloud in enumerate(mCloud_list):
                for j, sfe in enumerate(sfe_list):
                    ax = axes[i, j]
                    run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
                    data_path = find_data_file(BASE_DIR, run_name)

                    if data_path is None:
                        print(f"  {run_name}: missing")
                        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                        ax.set_axis_off()
                        continue

                    print(f"  Loading: {data_path}")
                    try:
                        t, phase, R1, R2, rShell, r_Tb, rcloud = load_run_radii(data_path)
                        plot_radii_on_ax(
                            ax, t, phase, R1, R2, rShell, r_Tb, rcloud,
                            phase_line=PHASE_LINE,
                            cloud_line=CLOUD_LINE,
                            show_weaver=SHOW_WEAVER,
                            smooth_window=SMOOTH_WINDOW,
                            smooth_mode=SMOOTH_MODE
                        )
                    except Exception as e:
                        print(f"Error in {run_name}: {e}")
                        ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                        ax.set_axis_off()
                        continue

                    if i == 0:
                        eps = int(sfe) / 100.0
                        ax.set_title(rf"$\epsilon={eps:.2f}$")

                    if j == 0:
                        mlog = int(np.log10(float(mCloud)))
                        ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"Radius [pc]")
                    else:
                        ax.tick_params(labelleft=False)

                    if i == nrows - 1:
                        ax.set_xlabel("t [Myr]")

            handles = [
                Line2D([0], [0], color=RADIUS_FIELDS[0][2], lw=RADIUS_FIELDS[0][4], ls=RADIUS_FIELDS[0][3], label=RADIUS_FIELDS[0][1]),
                Line2D([0], [0], color=RADIUS_FIELDS[1][2], lw=RADIUS_FIELDS[1][4], ls=RADIUS_FIELDS[1][3], label=RADIUS_FIELDS[1][1]),
                Line2D([0], [0], color=RADIUS_FIELDS[2][2], lw=RADIUS_FIELDS[2][4], ls=RADIUS_FIELDS[2][3], label=RADIUS_FIELDS[2][1]),
                Line2D([0], [0], color=RADIUS_FIELDS[3][2], lw=RADIUS_FIELDS[3][4], ls=RADIUS_FIELDS[3][3], label=RADIUS_FIELDS[3][1]),
            ]
            if SHOW_WEAVER:
                handles.append(Line2D([0], [0], color="k", ls="--", alpha=0.6, lw=1.5, label=r"Weaver: $R \propto t^{3/5}$"))
            handles.extend([
                Line2D([0], [0], color="k", ls="--", alpha=0.25, lw=1.6, label=r"$R_2>R_{\rm cloud}$"),
                Line2D([0], [0], color="r", lw=2, alpha=0.3, label=r"phase changes"),
            ])

            fig.subplots_adjust(top=0.9)

            leg = fig.legend(
                handles=handles,
                loc="upper center",
                ncol=3,
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                edgecolor="0.2",
                bbox_to_anchor=(0.5, 0.98),
                bbox_transform=fig.transFigure
            )
            leg.set_zorder(10)

            m_tag   = range_tag("M",   mCloud_list, key=float)
            sfe_tag = range_tag("sfe", sfe_list,    key=int)
            n_tag   = f"n{ndens}"
            tag = f"radiusEvolution_{m_tag}_{sfe_tag}_{n_tag}"

            if SAVE_PNG:
                out_png = FIG_DIR / f"{tag}.png"
                fig.savefig(out_png, bbox_inches="tight")
                print(f"Saved: {out_png}")
            if SAVE_PDF:
                out_pdf = FIG_DIR / f"{tag}.pdf"
                fig.savefig(out_pdf, bbox_inches="tight")
                print(f"Saved: {out_pdf}")

            plt.show()
            plt.close(fig)
