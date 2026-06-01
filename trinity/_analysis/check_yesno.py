#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_yesno.py — diagnose why ``_yesPHII`` / ``_noPHII`` runs produce
identical R2(t) trajectories.

Hypothesis under test
---------------------
In the energy / implicit phases TRINITY uses

    P_drive = max(Pb, P_HII)

(see ``trinity/phase1_energy/energy_phase_ODEs.py:253-256``) and in
the transition phase

    P_drive = max(Pb, P_HII + P_ram).

Because P_HII enters through a ``max``, toggling ``include_PHII`` only
changes the trajectory when ``P_HII`` (or ``P_HII + P_ram`` during
transition) exceeds ``Pb``. For a fiducial massive cluster the
mechanical bubble pressure dominates by orders of magnitude, so the
yes/no runs integrate the same effective ODE and R2(t) is identical.

This script verifies that hypothesis per yes/no pair and prints a
diagnosis:

    EXPECTED   — trajectories match AND Pb dominates P_HII throughout
    BUG        — noPHII run has non-zero P_HII, or yesPHII run has no
                 P_HII despite include_PHII=True
    UNEXPECTED — trajectories match but P_HII > Pb somewhere (the
                 max()-coupling should have driven divergence there)
    DIVERGES   — trajectories differ as physically expected

Usage
-----
    python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno
    python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno --tol 1e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from trinity._output.trinity_reader import load_output, find_all_simulations


YES_SUFFIX = "_yesPHII"
NO_SUFFIX = "_noPHII"


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def pair_yes_no(folder: Path):
    """Return list of (base_name, yes_path, no_path); missing partner → None."""
    yes_by_base: dict[str, Path] = {}
    no_by_base: dict[str, Path] = {}
    for p in find_all_simulations(folder):
        name = p.parent.name
        if name.endswith(YES_SUFFIX):
            yes_by_base[name[: -len(YES_SUFFIX)]] = p
        elif name.endswith(NO_SUFFIX):
            no_by_base[name[: -len(NO_SUFFIX)]] = p

    bases = sorted(set(yes_by_base) | set(no_by_base))
    return [(b, yes_by_base.get(b), no_by_base.get(b)) for b in bases]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _get_field(output, name: str, default=np.nan):
    """Load a field as float array, replacing None with default."""
    arr = output.get(name)
    if arr is None:
        return np.full(len(output), default, dtype=float)
    arr = np.asarray(arr)
    if arr.dtype == object:
        arr = np.where(arr == None, default, arr)  # noqa: E711
    return arr.astype(float)


def load_run(path: Path):
    out = load_output(path)
    t = _get_field(out, "t_now")
    R2 = _get_field(out, "R2")
    Pb = _get_field(out, "Pb")
    P_HII = _get_field(out, "P_HII", default=0.0)
    P_ram = _get_field(out, "P_ram", default=0.0)
    P_drive = _get_field(out, "P_drive")

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, Pb, P_HII, P_ram, P_drive = (
            a[order] for a in (t, R2, Pb, P_HII, P_ram, P_drive)
        )

    return dict(t=t, R2=R2, Pb=Pb, P_HII=P_HII, P_ram=P_ram, P_drive=P_drive)


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------
def compare_trajectories(yes, no):
    """Interpolate R2 onto the overlapping time window, return max rel diff."""
    t_lo = max(yes["t"].min(), no["t"].min())
    t_hi = min(yes["t"].max(), no["t"].max())
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        return np.nan, np.nan, (np.nan, np.nan)

    t_grid = np.linspace(t_lo, t_hi, 512)
    R_yes = np.interp(t_grid, yes["t"], yes["R2"])
    R_no = np.interp(t_grid, no["t"], no["R2"])

    denom = np.maximum(np.abs(R_yes), 1e-30)
    rel = np.abs(R_yes - R_no) / denom
    return float(rel.max()), float(rel.mean()), (float(t_lo), float(t_hi))


def pressure_dominance(yes):
    """In the yesPHII run, when does P_HII actually matter?

    Returns
    -------
    frac_phii_wins : float
        Fraction of snapshots where P_HII > Pb (or P_HII+P_ram > Pb
        during the transition phase — we use the more permissive
        P_HII+P_ram comparison everywhere, which is a superset).
    max_ratio : float
        max over snapshots of P_HII / Pb.
    """
    Pb = yes["Pb"]
    P_HII = yes["P_HII"]
    P_ram = yes["P_ram"]

    valid = np.isfinite(Pb) & np.isfinite(P_HII) & (Pb > 0)
    if not np.any(valid):
        return np.nan, np.nan

    pb = Pb[valid]
    phii = P_HII[valid]
    pram = np.where(np.isfinite(P_ram[valid]), P_ram[valid], 0.0)

    frac = float(np.mean((phii + pram) > pb))
    max_ratio = float(np.max(phii / pb))
    return frac, max_ratio


# ---------------------------------------------------------------------------
# Per-pair report
# ---------------------------------------------------------------------------
def diagnose_pair(base, yes_path, no_path, r2_tol: float, phii_tol: float):
    print(f"\n=== {base} ===")
    print(f"  yesPHII: {yes_path}")
    print(f"  noPHII : {no_path}")

    if yes_path is None or no_path is None:
        print("  [SKIP] missing partner run")
        return "MISSING"

    try:
        yes = load_run(yes_path)
        no = load_run(no_path)
    except Exception as e:
        print(f"  [ERROR] load failed: {e}")
        return "ERROR"

    # Check B: noPHII must have P_HII ≈ 0 everywhere
    no_phii_max = float(np.nanmax(np.abs(no["P_HII"])))
    # Check C: yesPHII should have P_HII > 0 at some point
    yes_phii_max = float(np.nanmax(yes["P_HII"]))

    # Check A: trajectory identity
    rel_max, rel_mean, (t_lo, t_hi) = compare_trajectories(yes, no)
    trajectories_match = np.isfinite(rel_max) and rel_max < r2_tol

    # Check D: does Pb dominate in the yesPHII run?
    frac_phii_wins, phii_over_pb_max = pressure_dominance(yes)

    # Print metrics
    print(f"  t-overlap: [{t_lo:.4g}, {t_hi:.4g}] Myr")
    print(f"  R2 comparison:     max|ΔR/R| = {rel_max:.3e}   "
          f"mean|ΔR/R| = {rel_mean:.3e}   tol = {r2_tol:.1e}")
    print(f"  noPHII  max|P_HII|  = {no_phii_max:.3e}    (expect ≈ 0)")
    print(f"  yesPHII max P_HII   = {yes_phii_max:.3e}    "
          f"(expect > {phii_tol:.1e})")
    print(f"  yesPHII max(P_HII/Pb) = {phii_over_pb_max:.3e}")
    print(f"  yesPHII fraction of snapshots where "
          f"P_HII(+P_ram) > Pb: {frac_phii_wins*100:.2f}%")

    # Diagnosis
    bugs = []
    if no_phii_max > phii_tol:
        bugs.append(f"noPHII run has non-zero P_HII "
                    f"(max={no_phii_max:.3e} > {phii_tol:.1e}) — "
                    f"include_PHII=False gate is leaking")
    if yes_phii_max <= phii_tol:
        bugs.append(f"yesPHII run never produced P_HII "
                    f"(max={yes_phii_max:.3e}) — "
                    f"P_HII is not being computed or n_IF_Str=0 always")

    if bugs:
        status = "BUG"
        print(f"  >> DIAGNOSIS: BUG")
        for b in bugs:
            print(f"       - {b}")
    elif trajectories_match and frac_phii_wins == 0:
        status = "EXPECTED"
        print(f"  >> DIAGNOSIS: EXPECTED")
        print(f"       Pb dominates P_HII at every snapshot, so "
              f"P_drive=max(Pb,P_HII)=Pb identically in both runs.")
        print(f"       Identical R2(t) is the correct consequence of "
              f"max()-coupling in energy_phase_ODEs.py:253-256.")
    elif trajectories_match and frac_phii_wins > 0:
        status = "UNEXPECTED"
        print(f"  >> DIAGNOSIS: UNEXPECTED")
        print(f"       P_HII exceeds Pb in {frac_phii_wins*100:.2f}% of "
              f"snapshots, yet R2(t) matches to {rel_max:.1e}.")
        print(f"       Either the flag is being ignored despite the "
              f"folder suffix, or the P_HII field written to disk "
              f"doesn't reflect what the ODE actually saw.")
    else:
        status = "DIVERGES"
        print(f"  >> DIAGNOSIS: DIVERGES")
        print(f"       R2 differs by up to {rel_max*100:.2f}% — "
              f"P_HII is materially changing the dynamics.")

    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Diagnose _yesPHII vs _noPHII trajectory equivalence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-f", "--folder", required=True,
        help="Folder containing _yesPHII and _noPHII simulation subfolders "
             "(e.g. outputs/trinity_fiducial_yesno).",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3,
        help="Relative R2 tolerance below which trajectories count as "
             "identical (default: 1e-3).",
    )
    parser.add_argument(
        "--phii-tol", type=float, default=1e-20,
        help="Absolute P_HII floor below which values count as zero "
             "(default: 1e-20, in TRINITY AU pressure units).",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    pairs = pair_yes_no(folder)
    if not pairs:
        print(f"No _yesPHII/_noPHII simulations found in: {folder}")
        sys.exit(1)

    print(f"Scanning: {folder}")
    print(f"Found {len(pairs)} base name(s) with yes/no runs.")

    tally = {"EXPECTED": 0, "BUG": 0, "UNEXPECTED": 0,
             "DIVERGES": 0, "MISSING": 0, "ERROR": 0}
    for base, yes_path, no_path in pairs:
        status = diagnose_pair(base, yes_path, no_path,
                               r2_tol=args.tol, phii_tol=args.phii_tol)
        tally[status] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in tally.items():
        if v:
            print(f"  {k:10s}: {v}")

    # Non-zero exit only on BUG (data inconsistency). EXPECTED is a
    # physics conclusion, not a failure.
    sys.exit(1 if tally["BUG"] else 0)


if __name__ == "__main__":
    main()
