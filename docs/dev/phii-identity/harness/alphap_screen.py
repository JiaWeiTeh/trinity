#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does TRINITY's momentum enhancement factor collapse discontinuously at 1c->2?

TRINITY already computes alpha_p; it just never names it. Identity I (see
`docs/dev/phii-identity/trinity_pressure_assessment.md` Sec. 1) is that substituting the
`get_r1` root into `bubble_E2P` gives Pb = pdot/(4 pi R1^2) at gamma = 5/3, so the wind
force reaching the shell is

    F = 4 pi R2^2 Pb = pdot * (R2/R1)^2 .

Lancaster Paper I `eq:alphap_shock` is alpha_p = (1/4)[3 x^2 + x^-2] with x = Rw/Rf, where
Rf is set by the POST-SHOCK pressure 3 pdot/(16 pi Rf^2). TRINITY's `get_r1` uses the
free-wind RAM value pdot/(4 pi R1^2), so at equal bubble pressure Rf = (sqrt(3)/2) R1,
x = (2/sqrt(3)) (R2/R1), and the 4/3 mismatch cancels the 3/4 in the leading term:

    alpha_p = (R2/R1)^2 + (3/16) (R1/R2)^2 .

⚠️ **Report `(R2/R1)^2`, not that inversion.** `(R2/R1)^2` is simultaneously TRINITY's exact
force ratio and Lancaster's exact 4 pi Rw^2 Phot / pdot, and it is what Paper II's tabulated
alpha_p approximates -- Lancaster themselves drop the x^-2 term "using the assumption
alpha_p >~ 1" (`eq:Phot_EC`). The full inversion reports 1.1875 where the force ratio is
exactly 1, which is precisely the regime the 1c->2 handover approaches. Both columns are
emitted; the gates use the force ratio.

**The question this screen answers.** In the momentum phase TRINITY applies F = pdot
identically (`pRam` with v_mech = 2 Lmech/pdot_total), i.e. alpha_p = 1 -- the idealised
momentum-driven limit. In 1a/1b/1c it carries (R2/R1)^2 >> 1. So: does (R2/R1)^2 fall
smoothly to ~1 by the handover, or does it drop discontinuously at a boundary governed by
`ENERGY_FLOOR = 1e3` (`run_transition_phase.py:97`) rather than by any physical criterion?

    python docs/dev/phii-identity/harness/alphap_screen.py <run_dir> [<run_dir> ...] \
        --out docs/dev/phii-identity/data-new/alphap_handover.csv

⚠️ **Validity.** Inside the `dt_switchon` ramp window the ODE does NOT integrate
Pb = pdot/(4 pi R1^2): `get_effective_bubble_pressure` pulls R1 -> 0 linearly for
t <= tSF + 1e-3 Myr (`get_bubbleParams.py:495-503`), and PLAN.md Sec. 1(3) measures the two
pressures differing by up to 3.31x inside it. Those rows are flagged and excluded from the
gates rather than silently reported.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stamp import stamp  # noqa: E402

# Source constants, quoted with the line that defines them so drift is detectable.
DT_SWITCHON = 1e-3        # get_bubbleParams.py:495
ENERGY_FLOOR = 1e3        # run_transition_phase.py:97
PAPER_II_BAND = (4.57, 6.82)   # lancaster2025b.tex tab:cem_comp, HWR + MWR across resolutions

ENERGY_PHASES = ("energy", "implicit", "transition")


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


def meta_of(run_dir):
    mj = Path(run_dir) / "metadata.json"
    if not mj.exists():
        return {}
    try:
        return json.loads(mj.read_text())
    except (ValueError, OSError):
        return {}


def screen(run_dir):
    """Per-row alpha_p plus the per-run handover summary."""
    meta = meta_of(run_dir)
    tSF = float(meta.get("tSF", 0.0) or 0.0)
    gamma = float(meta.get("gamma_adia", 5.0 / 3.0))

    per_row, per_phase = [], {}
    last_transition = None      # last VALID transition row -> the handover value
    momentum_r1_ne_r2 = 0
    momentum_rows = 0

    for row in rows_of(run_dir):
        phase, t = row.get("current_phase"), row.get("t_now")
        R1, R2, Eb = row.get("R1"), row.get("R2"), row.get("Eb")
        if phase is None or not isinstance(R1, (int, float)) or not isinstance(R2, (int, float)):
            continue

        if phase == "momentum":
            momentum_rows += 1
            # G-A4: run_momentum_phase.py:587-588 assigns params['R1'] = R2.
            if R1 != R2:
                momentum_r1_ne_r2 += 1

        if not (R1 > 0 and R2 > 0):
            continue

        ap_force = (R2 / R1) ** 2
        ap_shock = ap_force + (3.0 / 16.0) / ap_force      # == (R2/R1)^2 + (3/16)(R1/R2)^2
        # The ramp is tSF-relative and lives in phase 1a only.
        in_ramp = (phase == "energy") and isinstance(t, (int, float)) and t <= tSF + DT_SWITCHON
        valid = (phase in ENERGY_PHASES) and not in_ramp

        per_row.append(
            dict(config=Path(run_dir).name, t_now=t, phase=phase, R1=R1, R2=R2, Eb=Eb,
                 alpha_p_force=ap_force, alpha_p_shock=ap_shock,
                 in_ramp_window=in_ramp, valid=valid)
        )

        acc = per_phase.setdefault(phase, dict(rows=0, valid=0, ramp=0, ap=[]))
        acc["rows"] += 1
        acc["ramp"] += 1 if in_ramp else 0
        if valid:
            acc["valid"] += 1
            acc["ap"].append(ap_force)
        if valid and phase == "transition":
            last_transition = (t, ap_force, Eb)

    phases_seen = set(per_phase) | ({"momentum"} if momentum_rows else set())
    reached_handover = ("transition" in per_phase) and ("momentum" in phases_seen)

    ap_all = [v for a in per_phase.values() for v in a["ap"]]
    summary = dict(
        config=Path(run_dir).name,
        gamma_adia=gamma,
        tSF=tSF,
        phases=",".join(sorted(phases_seen)) or "none",
        rows_total=sum(a["rows"] for a in per_phase.values()) + momentum_rows,
        rows_ramp_flagged=sum(a["ramp"] for a in per_phase.values()),
        # G-A1
        alpha_p_max_1a1b1c=max(ap_all) if ap_all else None,
        reaches_paperII_band=(max(ap_all) >= PAPER_II_BAND[0]) if ap_all else None,
        # G-A2 -- the load-bearing number
        handover_reached=reached_handover,
        t_handover=last_transition[0] if last_transition else None,
        alpha_p_handover=last_transition[1] if last_transition else None,
        Eb_handover=last_transition[2] if last_transition else None,
        # G-A4
        momentum_rows=momentum_rows,
        momentum_rows_R1_ne_R2=momentum_r1_ne_r2,
    )
    for ph in ENERGY_PHASES:
        a = per_phase.get(ph)
        ap = sorted(a["ap"]) if a else []
        summary[f"{ph}_rows"] = a["rows"] if a else 0
        summary[f"{ph}_alpha_p_median"] = ap[len(ap) // 2] if ap else None
        summary[f"{ph}_alpha_p_min"] = ap[0] if ap else None
        summary[f"{ph}_alpha_p_max"] = ap[-1] if ap else None

    return summary, per_row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories containing dictionary.jsonl")
    ap.add_argument("--out", type=Path, help="per-run summary CSV")
    ap.add_argument("--per-row", type=Path, help="optional per-snapshot CSV")
    args = ap.parse_args()

    summaries, all_rows = [], []
    for run in args.runs:
        s, r = screen(run)
        if s["rows_total"] == 0:
            print(f"⚠️  {run}: no usable rows (need R1, R2, current_phase) -- skipped")
            continue
        summaries.append(s)
        all_rows.extend(r)
    if not summaries:
        sys.exit("no usable runs")

    print(f"{'config':38s}{'phases':34s}{'ap max':>9}{'ap @handover':>14}{'Eb @handover':>14}")
    for s in summaries:
        hv = s["alpha_p_handover"]
        print(f"{s['config'][:37]:38s}{s['phases'][:33]:34s}"
              f"{(s['alpha_p_max_1a1b1c'] or float('nan')):>9.2f}"
              f"{(hv if hv is not None else float('nan')):>14.3f}"
              f"{(s['Eb_handover'] if s['Eb_handover'] is not None else float('nan')):>14.3e}")

    # ---------------------------------------------------------------- gates --
    # Pre-registered 2026-08-18, BEFORE this script was run on any output.
    print("\n--- pre-registered gates ---")

    band_lo, band_hi = PAPER_II_BAND
    got = [s for s in summaries if s["alpha_p_max_1a1b1c"] is not None]
    n_band = sum(1 for s in got if s["reaches_paperII_band"])
    print(f"G-A1  alpha_p reaches Paper II's {band_lo}-{band_hi} somewhere in 1a/1b/1c: "
          f"{n_band}/{len(got)} configs."
          "\n      Prediction: PASS on essentially every config -- an energy-driven Weaver"
          "\n      bubble has R2/R1 of a few, so (R2/R1)^2 clears 6.82 comfortably."
          "\n      A miss here would mean TRINITY never carries a Lancaster-scale alpha_p at all.")

    live = [s for s in summaries if s["handover_reached"]]
    void = [s for s in summaries if not s["handover_reached"]]
    print(f"\nG-A2  alpha_p on the last valid transition row (the handover value): "
          f"{len(live)} live, {len(void)} VOID.")
    if void:
        print("      VOID (never reached both transition and momentum -- never a confirming null):")
        for s in void:
            print(f"        {s['config']}  phases={s['phases']}")
    if live:
        vals = sorted(s["alpha_p_handover"] for s in live)
        worst = max(vals)
        print(f"      measured: min {vals[0]:.3f}  median {vals[len(vals)//2]:.3f}  max {worst:.3f}")
        print("      Prediction: < 1.5 on every config. Because Pb -> pdot/(4 pi R1^2) and"
              "\n      R1 -> R2 as Eb -> 0, the drop to 1 should already have happened by the"
              "\n      time Eb crosses ENERGY_FLOOR, i.e. the collapse is smooth and alpha_p"
              "\n      comes off the list permanently."
              "\n      FALSIFIER: alpha_p_handover >= 2 on any config. That would make the"
              "\n      drop to 1 a genuine discontinuity in a measurable quantity, set by an"
              f"\n      energy floor (ENERGY_FLOOR = {ENERGY_FLOOR:g}) rather than by physics.")
        fired = [s for s in live if s["alpha_p_handover"] >= 2.0]
        print(f"      -> falsifier {'FIRED on ' + ', '.join(s['config'] for s in fired) if fired else 'did not fire'}")

    ramp = sum(s["rows_ramp_flagged"] for s in summaries)
    print(f"\nG-A3  dt_switchon ramp rows flagged invalid and excluded: {ramp} across "
          f"{len(summaries)} configs.")
    print("      Descriptive, no pass/fail. 0 on a config with energy rows means the run"
          "\n      never sampled the window; that is a sampling fact, not a physics one.")

    mrows = sum(s["momentum_rows"] for s in summaries)
    mbad = sum(s["momentum_rows_R1_ne_R2"] for s in summaries)
    if mrows:
        print(f"\nG-A4  R1 == R2 exactly on momentum rows: {mrows - mbad}/{mrows}.")
        print("      FALSIFIER: any row with R1 != R2 breaks the reading that phase 2 asserts"
              "\n      Rf = Rw (run_momentum_phase.py:587-588), and with it the claim that"
              "\n      alpha_p = 1 there is the phase definition rather than an omission.")
    else:
        print("\nG-A4  VOID -- no momentum rows in this run set.")

    # ---------------------------------------------------------------- write --
    for path, rows in ((args.out, summaries), (args.per_row, all_rows)):
        if not path:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            for run in args.runs:
                fh.write(f"# run {run}\n")
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
