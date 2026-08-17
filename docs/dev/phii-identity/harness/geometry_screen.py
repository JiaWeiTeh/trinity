#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 9 scoping — is C3a's ionised gas where C3a says it is?

C3a computes the photoionised pressure from the density that balances recombination
over the WHOLE CAVITY, (4/3) pi R2^3:

    n_C3a = sqrt(3 Qi_abs / (4 pi chi_e alpha_B R2^3))

But trinity's own shell solve puts the photoionised gas in a thin layer at the INNER
EDGE OF THE SHELL (`shell_structure.py` integrates `nShell_arr_ion(r)` up to
`shell_ion_idx`); the cavity interior holds hot/wind gas. Those are different volumes
holding different gas, and C3a takes the photon budget of one (`Qi * shell_fAbsorbedIon`
-- the fraction absorbed IN THE SHELL) and spreads it over the other.

This screen measures the mismatch on committed run output. For a layer of thickness
dR at radius R2, recombination balance over 4 pi R2^2 dR instead of (4/3) pi R2^3 gives

    n_layer / n_cavity = sqrt( R2 / (3 dR) )

and pressure is linear in n, so that ratio is also the pressure ratio. It crosses 1 at
dR = R2/3.

⛔ The scoping version of this file claimed the correction is ONE-SIGNED (always makes the
photoionised pressure larger). **That was measured before the momentum phase was covered and
is retracted.** The shell is thin early (dR/R2 ~ 1e-3) and THICK in momentum (0.67-1.31), so
past dR = R2/3 the correction LOWERS the pressure: measured frac(ratio > 1) = 1.0000 /
1.0000 / 0.3810 / 0.0000 across energy / implicit / transition / momentum.

⚠️ The `dR` here is the FULL shell thickness, not the ionised layer — see `layer_thickness`
for why snapshot data cannot give the ionised layer, and use
`harness/layer_density_check.py` for the true value. That harness also shows this screen's
Stroemgren-balance framing overstates the real profile's density by up to 3.17x, because the
balance assumes every absorbed ionising photon recombines while the real shell loses 61-75%
of them to dust (G9.4, FALSIFIED). See PLAN.md §Batch 9.

    python docs/dev/phii-identity/harness/geometry_screen.py <run_dir> [<run_dir> ...] \
        --out docs/dev/phii-identity/data/b9_geometry_scope.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stamp import stamp  # noqa: E402


def rows_of(run_dir):
    """Yield parsed snapshots from a run's dictionary.jsonl."""
    dj = Path(run_dir) / "dictionary.jsonl"
    if not dj.exists():
        return
    with dj.open() as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except (ValueError, TypeError):
                continue


def layer_thickness(row):
    """FULL shell thickness from the snapshot. Returns (dR, index_clamped) or None.

    ⚠️ **This is the whole shell, ionised + neutral — NOT the ionised layer**, and it cannot
    be the ionised layer from snapshot data alone. `shell_ion_idx` indexes the ORIGINAL
    `shell_r/n_arr` (`registry.py:499`) while the snapshot's `shell_r_arr` is the
    DOWNSAMPLED grid paired with `log_shell_n_arr` (`dictionary.py:688-700`) — two different
    index spaces. On the B3M run `shell_ion_idx` reaches 26848 against a saved array of <=100,
    so the index clamps to the last element on 100% of rows in all four phases.

    Verified consequence (G9.4, `harness/layer_density_check.py`): because the shell is
    87-100% ionised, `dR_full` is within 0.5% of `dR_ion` in momentum and 13% in implicit, so
    the Batch 9 ratios stood. But the quantity is still the full thickness, and since
    `ratio = sqrt(R2/(3 dR))` falls with `dR`, any row where the shell is substantially
    neutral would have its ratio UNDERSTATED here.

    **For the true ionised-layer thickness, use `harness/layer_density_check.py`**, which
    replays `shell_structure_pure` so `shell_ion_idx` and `shell_r_arr` share an index space.

    `shell_ion_idx = -1` is a sentinel meaning there is no ionised region at all
    (`shell_structure.py:441`, the dissolved branch). It must not reach the arithmetic:
    `arr[-1] - arr[0]` is the whole shell, the exact opposite of "no ionised gas". Rows
    carrying the sentinel are excluded and counted separately.
    """
    arr, idx = row.get("shell_r_arr"), row.get("shell_ion_idx")
    if not arr or idx is None or len(arr) < 2:
        return None
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return None
    if i < 0:
        return "no_ionised_region"
    dR = arr[min(i, len(arr) - 1)] - arr[0]
    if dR <= 0:
        return None
    return dR, i >= len(arr) - 1


def screen(run_dir):
    """Per-phase summary of the geometry mismatch for one run."""
    per_phase = {}
    for row in rows_of(run_dir):
        phase = row.get("current_phase")
        R2, lt = row.get("R2"), layer_thickness(row)
        f_abs = row.get("shell_fAbsorbedIon")
        if not (phase and R2 and R2 > 0 and lt):
            continue
        acc = per_phase.setdefault(
            phase,
            dict(n=0, ratios=[], fabs=[], dR_over_R2=[], t_lo=None, t_hi=None, no_ion=0, whole=0),
        )
        if lt == "no_ionised_region":
            # Excluded from the ratio statistics — there is no ionised gas to place.
            acc["no_ion"] += 1
            continue
        dR, whole = lt
        acc["whole"] += 1 if whole else 0
        acc["n"] += 1
        acc["ratios"].append(math.sqrt(R2 / (3.0 * dR)))
        acc["dR_over_R2"].append(dR / R2)
        if isinstance(f_abs, (int, float)):
            acc["fabs"].append(float(f_abs))
        t = row.get("t_now")
        if isinstance(t, (int, float)):
            acc["t_lo"] = t if acc["t_lo"] is None else min(acc["t_lo"], t)
            acc["t_hi"] = t if acc["t_hi"] is None else max(acc["t_hi"], t)

    out = []
    for phase, a in per_phase.items():
        r, f = a["ratios"], a["fabs"]
        out.append(
            dict(
                config=Path(run_dir).name,
                phase=phase,
                rows=a["n"],
                rows_no_ionised_region=a["no_ion"],
                frac_index_clamped=a["whole"] / a["n"] if a["n"] else None,
                t_lo=a["t_lo"],
                t_hi=a["t_hi"],
                dR_over_R2_min=min(a["dR_over_R2"]),
                dR_over_R2_max=max(a["dR_over_R2"]),
                ratio_min=min(r),
                ratio_median=sorted(r)[len(r) // 2],
                ratio_max=max(r),
                # G9.2: the correction must be one-signed
                frac_ratio_gt_1=sum(1 for x in r if x > 1.0) / len(r),
                # G9.1: is any ionising photon left to maintain a cavity HII region?
                fabs_min=min(f) if f else None,
                fabs_max=max(f) if f else None,
                frac_fabs_ge_099=(sum(1 for x in f if x >= 0.99) / len(f)) if f else None,
            )
        )
    return sorted(out, key=lambda d: d["phase"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories containing dictionary.jsonl")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [r for run in args.runs for r in screen(run)]
    if not rows:
        sys.exit("no usable rows — need shell_r_arr + shell_ion_idx in dictionary.jsonl")

    print(
        f"{'config':10s}{'phase':11s}{'rows':>5}{'dR/R2 min':>11}{'ratio med':>11}"
        f"{'ratio max':>11}{'f_abs>=.99':>11}"
    )
    for r in rows:
        print(
            f"{r['config']:10s}{r['phase']:11s}{r['rows']:>5d}{r['dR_over_R2_min']:>11.2e}"
            f"{r['ratio_median']:>11.2f}{r['ratio_max']:>11.2f}"
            f"{(r['frac_fabs_ge_099'] if r['frac_fabs_ge_099'] is not None else float('nan')):>11.3f}"
        )

    one_signed = min(r["frac_ratio_gt_1"] for r in rows)
    print(f"\nG9.2 correction one-signed (ratio > 1): {one_signed:.4f} of rows  (must be 1.0000)")
    fa = [r["frac_fabs_ge_099"] for r in rows if r["frac_fabs_ge_099"] is not None]
    if fa:
        print(f"G9.1 shell absorbs ~all ionising photons: min {min(fa):.4f} of rows")
    phases = sorted({r["phase"] for r in rows})
    print(f"phases covered: {phases}")
    if "momentum" not in phases:
        print("⚠️  MOMENTUM NOT COVERED — G9.3 is not discharged by this run set.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            for run in args.runs:
                fh.write(f"# run {run}\n")
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
