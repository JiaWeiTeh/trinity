#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRINITY Output Reader - Overview and Examples
==============================================

This script demonstrates the various utilities provided by the TrinityOutput
reader class for analyzing TRINITY simulation outputs.

The TrinityOutput reader (src/_output/trinity_reader.py) provides a clean,
Pythonic API for accessing simulation data, similar to astropy.io.fits.

Key Features Demonstrated
-------------------------
1. Opening output files with TrinityOutput.open()
2. Getting summary info with .info() and .info(verbose=True)
3. Extracting time series as numpy arrays with .get()
4. Indexing and slicing snapshots
5. Finding snapshots at specific times with .get_at_time()
6. Filtering by phase or time range with .filter()
7. Converting to pandas DataFrame with .to_dataframe()

Snapshot Consistency Note
-------------------------
As of January 2026, TRINITY snapshots are saved BEFORE ODE integration,
ensuring all values in a snapshot correspond to the same timestamp (t_now).
This includes dynamical variables, forces, and beta-delta residuals.

Run from the trinity root directory:
    python example_scripts/example_reader_overview.py

See Also
--------
- example_plot_radius_vs_time.py: Plotting examples
- src/_output/trinity_reader.py: Full reader implementation
- src/_plots/load_snapshots.py: Convenience wrapper for plotting scripts

@author: TRINITY Team
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src._output.trinity_reader import TrinityOutput
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE = 'example_scripts/example_dictionary_1e7_sfe001_n1e4.jsonl'


def print_section(title):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_section("TRINITY Output Reader - Overview Demo")

    # -------------------------------------------------------------------------
    # 1. Opening a file
    # -------------------------------------------------------------------------
    print_section("1. Opening a TRINITY output file")

    output = TrinityOutput.open(OUTPUT_FILE)
    print(f"Opened: {output.filepath}")
    print(f"Object: {repr(output)}")

    # -------------------------------------------------------------------------
    # 2. Basic info
    # -------------------------------------------------------------------------
    print_section("2. Basic information (.info())")

    output.info()

    # -------------------------------------------------------------------------
    # 3. Verbose info with parameter documentation
    # -------------------------------------------------------------------------
    print_section("3. Verbose info with documentation (.info(verbose=True))")
    print("(Showing first few parameter groups only...)")
    print()

    # Just show the header and first group
    print(f"  Use output.info(verbose=True) to see all {len(output.keys)} parameters")
    print(f"  with documentation grouped by category:")
    print()
    print("  Categories: Model, Time, State, Dynamics, Cooling, Feedback,")
    print("              Forces, Shell, Bubble, Residuals, Other")

    # -------------------------------------------------------------------------
    # 4. Accessing time series data
    # -------------------------------------------------------------------------
    print_section("4. Extracting time series data (.get())")

    times = output.get('t_now')
    radii = output.get('R2')
    velocities = output.get('v2')
    energies = output.get('Eb')

    print(f"  times = output.get('t_now')")
    print(f"    -> shape: {times.shape}, dtype: {times.dtype}")
    print(f"    -> range: [{times.min():.4e}, {times.max():.4e}] Myr")
    print()
    print(f"  radii = output.get('R2')")
    print(f"    -> shape: {radii.shape}")
    print(f"    -> range: [{radii.min():.4e}, {radii.max():.4e}] pc")
    print()
    print(f"  velocities = output.get('v2')")
    print(f"    -> range: [{velocities.min():.4e}, {velocities.max():.4e}] pc/Myr")
    print()
    print(f"  energies = output.get('Eb')")
    print(f"    -> range: [{energies.min():.4e}, {energies.max():.4e}] erg")

    # -------------------------------------------------------------------------
    # 5. Indexing and iteration
    # -------------------------------------------------------------------------
    print_section("5. Indexing and iteration")

    # Single snapshot
    snap = output[100]
    print(f"  Single snapshot: output[100]")
    print(f"    -> {snap}")
    print(f"    -> snap['R2'] = {snap['R2']:.4e} pc")
    print(f"    -> snap['v2'] = {snap['v2']:.4e} pc/Myr")
    print()

    # Slice
    snaps = output[10:15]
    print(f"  Slice: output[10:15]")
    print(f"    -> Returns {len(snaps)} Snapshot objects")
    for s in snaps:
        print(f"       {s}")
    print()

    # Iteration
    print(f"  Iteration: for snap in output[:5]:")
    for snap in output[:5]:
        print(f"    t={snap.t_now:.4e}, R2={snap['R2']:.4e}, phase={snap.phase}")

    # -------------------------------------------------------------------------
    # 6. Finding snapshots at specific times
    # -------------------------------------------------------------------------
    print_section("6. Finding snapshots at specific times (.get_at_time())")

    target_times = [0.01, 0.1, 0.5]
    for t in target_times:
        snap = output.get_at_time(t)
        r2 = output.get_at_time(t, 'R2')
        print(f"  t ~ {t:.2f} Myr:")
        print(f"    -> Actual time: {snap.t_now:.4e} Myr")
        print(f"    -> R2 = {r2:.4e} pc")
        print(f"    -> Phase: {snap.phase}")
        print()

    # -------------------------------------------------------------------------
    # 7. Filtering snapshots
    # -------------------------------------------------------------------------
    print_section("7. Filtering snapshots (.filter())")

    # By phase
    energy_phase = output.filter(phase='energy')
    implicit_phase = output.filter(phase='implicit')
    print(f"  By phase:")
    print(f"    energy phase:   {len(energy_phase):4d} snapshots")
    print(f"    implicit phase: {len(implicit_phase):4d} snapshots")
    print()

    # By time range
    early = output.filter(t_max=0.01)
    late = output.filter(t_min=0.1)
    middle = output.filter(t_min=0.01, t_max=0.1)
    print(f"  By time range:")
    print(f"    t < 0.01 Myr:        {len(early):4d} snapshots")
    print(f"    0.01 < t < 0.1 Myr:  {len(middle):4d} snapshots")
    print(f"    t > 0.1 Myr:         {len(late):4d} snapshots")
    print()

    # Combined filters
    implicit_late = output.filter(phase='implicit', t_min=0.1)
    print(f"  Combined (implicit AND t > 0.1):")
    print(f"    -> {len(implicit_late)} snapshots")

    # -------------------------------------------------------------------------
    # 8. Working with filtered data
    # -------------------------------------------------------------------------
    print_section("8. Working with filtered data")

    implicit = output.filter(phase='implicit')
    t_impl = implicit.get('t_now')
    r_impl = implicit.get('R2')
    v_impl = implicit.get('v2')

    print(f"  Implicit phase statistics:")
    print(f"    Time range: [{t_impl.min():.4e}, {t_impl.max():.4e}] Myr")
    print(f"    R2 range:   [{r_impl.min():.4e}, {r_impl.max():.4e}] pc")
    print(f"    v2 range:   [{v_impl.min():.4e}, {v_impl.max():.4e}] pc/Myr")
    print()

    # Calculate expansion rate
    dr_dt = np.gradient(r_impl, t_impl)
    print(f"  Calculated dR/dt:")
    print(f"    Mean: {np.mean(dr_dt):.4e} pc/Myr")
    print(f"    Range: [{dr_dt.min():.4e}, {dr_dt.max():.4e}] pc/Myr")

    # -------------------------------------------------------------------------
    # 9. Accessing properties
    # -------------------------------------------------------------------------
    print_section("9. Convenient properties")

    print(f"  output.model_name = '{output.model_name}'")
    print(f"  output.phases     = {output.phases}")
    print(f"  output.t_min      = {output.t_min:.4e} Myr")
    print(f"  output.t_max      = {output.t_max:.4e} Myr")
    print(f"  len(output)       = {len(output)} snapshots")
    print(f"  len(output.keys)  = {len(output.keys)} parameters")

    # -------------------------------------------------------------------------
    # 10. Converting to pandas DataFrame
    # -------------------------------------------------------------------------
    print_section("10. Converting to pandas DataFrame")

    try:
        import pandas as pd
        df = output.to_dataframe()
        print(f"  df = output.to_dataframe()")
        print(f"    -> Shape: {df.shape}")
        print(f"    -> Columns: {len(df.columns)}")
        print()
        print("  Sample (first 5 rows, selected columns):")
        cols = ['t_now', 'R2', 'v2', 'Eb', 'current_phase']
        print(df[cols].head().to_string(index=False))
    except ImportError:
        print("  (pandas not installed - skipping DataFrame example)")

    # -------------------------------------------------------------------------
    # 11. Quick data analysis example
    # -------------------------------------------------------------------------
    print_section("11. Quick analysis example: Shell momentum")

    # Calculate shell momentum: p = M_shell * v2
    shell_mass = output.get('shell_mass')
    v2 = output.get('v2')
    t = output.get('t_now')

    # Convert v2 from pc/Myr to km/s (1 pc/Myr ~ 0.978 km/s)
    v2_kms = v2 * 0.978

    # Shell momentum (in Msun * km/s)
    momentum = shell_mass * v2_kms

    print(f"  Shell momentum = M_shell * v2")
    print(f"    Initial: {momentum[0]:.4e} Msun km/s")
    print(f"    Final:   {momentum[-1]:.4e} Msun km/s")
    print(f"    Max:     {momentum.max():.4e} Msun km/s")

    # Find time of maximum momentum
    idx_max = np.argmax(momentum)
    print(f"    Time of max: {t[idx_max]:.4e} Myr")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Summary")
    print("""
  The TrinityOutput reader provides:

    1. Easy file opening:     TrinityOutput.open('file.jsonl')
    2. Quick overview:        output.info() / output.info(verbose=True)
    3. Time series access:    output.get('parameter_name')
    4. Snapshot access:       output[index] / output[start:stop]
    5. Time-based lookup:     output.get_at_time(t)
    6. Filtering:             output.filter(phase=, t_min=, t_max=)
    7. Iteration:             for snap in output: ...
    8. pandas integration:    output.to_dataframe()

  For plotting examples, see: example_plot_radius_vs_time.py
    """)


if __name__ == '__main__':
    main()
