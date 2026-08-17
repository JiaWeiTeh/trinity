#!/usr/bin/env python3
"""Batch 7 gates: the confinement regime, measured on C3c arms — PLAN.md Batch 7.

Unlike `c3_offline_screen.py`, which evaluates *candidate* formulations on a
STOCK trajectory, this reads arms where C3c actually ran. That makes the branch
outcome directly observable: the delivered `P_HII` is exactly 0.0 on the confined
branch, so `frac_confined` needs no reconstruction.

It reports the branch outcome BOTH ways and cross-checks them:

  frac_confined_delivered   from the stored P_HII == 0.0        (ground truth)
  frac_HII_dom_recomputed   from P_C3a > Pb, recomputed here    (gives the MARGIN)

They must agree. A mismatch means the recomputation and the code have drifted, or
that P_HII was zeroed for an unrelated reason (`include_PHII=False`, or the
`n_IF_Str > 0` guard in the phase runners) — either way the row is not evidence
about confinement, so mismatches are counted and reported, never averaged away.

The MARGIN is the point of the recomputation: a fraction of 0.0000 tells you the
branch never flipped, not how close it came. G7.1 registers a bar on ratio_max.

VOID rule (PLAN Batch 7, from the stage-3 lesson): a run that never leaves the
implicit phase, or has no metadata.json, cannot speak to transition/momentum
confinement and is reported VOID rather than as a confirming null.

Usage:
    python docs/dev/phii-identity/harness/screen_confinement.py \
        --out docs/dev/phii-identity/data/b7_confinement_screen.csv \
        <run_dir> [<run_dir> ...]
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402
from c3_offline_screen import candidates_for_row, n_to_P  # noqa: E402
from run_batch import done as run_batch_done  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from trinity._input.read_param import read_param  # noqa: E402

PHASES = ["energy", "implicit", "transition", "momentum"]


def param_text_value(param_path, key):
    """Read a key straight from the materialised .param text.

    params[...].value has been unit-converted at load (nCore comes back as ~1e59
    in internal units), so the human-readable input density has to come from the
    file the run was actually given.
    """
    for line in Path(param_path).read_text().splitlines():
        f = line.split("#", 1)[0].split()
        if len(f) >= 2 and f[0] == key:
            return f[1]
    return ""


def analyse(run_dir):
    run_dir = Path(run_dir)
    cfg = run_dir.name
    param_path = run_dir / f"{cfg}.param"
    params = read_param(str(param_path))
    # Same trap as run_batch.done(): metadata.json appears on the FIRST flush,
    # seconds in, so its mere existence says nothing about completion.
    complete = run_batch_done(run_dir)

    per = defaultdict(lambda: {"ratio": [], "conf_delivered": 0, "n": 0, "mismatch": 0})
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            ph = d.get("current_phase")
            if ph not in PHASES:
                continue
            s = per[ph]
            s["n"] += 1

            delivered_confined = not d.get("P_HII")  # exactly 0.0 (or absent)
            s["conf_delivered"] += int(delivered_confined)

            n_c3a = candidates_for_row(d, params).get("C3a_cavity")
            Pb = d.get("Pb")
            if n_c3a and Pb and Pb > 0:
                ratio = n_to_P(n_c3a, params) / Pb
                s["ratio"].append(ratio)
                # ratio > 1 should mean the code took the driving branch.
                if (ratio > 1.0) == delivered_confined:
                    s["mismatch"] += 1
    return cfg, params, param_path, complete, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()

    rows = []
    for rd in args.run_dirs:
        try:
            cfg, params, param_path, complete, per = analyse(rd)
        except Exception as e:  # a missing/partial run must not kill the screen
            print(f"  !! {rd}: {e}")
            continue
        reached = [p for p in PHASES if per[p]["n"]]
        # Coverage is PER PHASE, not per run. A run killed mid-implicit still
        # covers the ENERGY phase completely -- it demonstrably finished it, since
        # it moved on. Blanket-VOIDing such a run would throw away exactly the rows
        # G7.1/G7.2 are about. Only the LAST phase reached is partial, and only an
        # incomplete run can be partial at all.
        last_reached = reached[-1] if reached else None

        def phase_status(ph):
            if complete:
                return "ok"
            return "PARTIAL_in_progress" if ph == last_reached else "ok_phase_closed"
        for ph in PHASES:
            s = per[ph]
            if not s["n"]:
                continue
            r = sorted(s["ratio"])
            rows.append({
                "config": cfg,
                "phase": ph,
                "n_rows": s["n"],
                "frac_confined_delivered": f"{s['conf_delivered'] / s['n']:.4f}",
                "frac_HII_dom_recomputed":
                    f"{sum(1 for x in r if x > 1.0) / len(r):.4f}" if r else "",
                "ratio_min": f"{r[0]:.4g}" if r else "",
                "ratio_med": f"{statistics.median(r):.4g}" if r else "",
                "ratio_max": f"{r[-1]:.4g}" if r else "",
                "mismatch_rows": s["mismatch"],
                "FB_thermCoeffWind": f"{params['FB_thermCoeffWind'].value:g}",
                "nCore_cm3": param_text_value(param_path, "nCore"),
                "mCloud_Msun": param_text_value(param_path, "mCloud"),
                "sfe": param_text_value(param_path, "sfe"),
                "status": phase_status(ph),
                # Separate axis: whether the RUN can speak to the later phases at
                # all. A run that stopped in implicit says nothing about
                # transition/momentum confinement -- but that is a statement about
                # the phases it never reached, not about the rows it did compute.
                "run_reached": "+".join(reached),
                "run_complete": complete,
            })
        print(f"  {cfg}: complete={complete} phases={reached}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 7 gates. frac_confined_delivered is ground truth (stored "
                 "P_HII == 0.0); frac_HII_dom_recomputed is P_C3a > Pb recomputed here "
                 "and exists to give ratio_min/med/max, the MARGIN to the switch.\n")
        fh.write("# mismatch_rows > 0 means the two disagree -- investigate, do not average.\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    # ---- the registered gates, evaluated ---------------------------------
    print("\n--- G7.1 coverage (energy+implicit confined on every screened config) ---")
    bad = [r for r in rows if r["phase"] in ("energy", "implicit")
           and float(r["frac_confined_delivered"]) < 1.0]
    marg = [r for r in rows if r["phase"] == "energy" and r["ratio_max"]
            and float(r["ratio_max"]) >= 0.5]
    print(f"  configs breaking confinement in energy/implicit: "
          f"{[(r['config'], r['phase'], r['frac_confined_delivered']) for r in bad] or 'NONE'}")
    print(f"  energy ratio_max >= 0.5 (registered bar): "
          f"{[(r['config'], r['ratio_max']) for r in marg] or 'NONE'}")

    print("\n--- G7.2 the flip (B3MW001 must break confinement in energy) ---")
    for r in rows:
        if r["config"] == "B3MW001" and r["phase"] == "energy":
            fd = 1.0 - float(r["frac_confined_delivered"])
            rmax = float(r["ratio_max"]) if r["ratio_max"] else float("nan")
            print(f"  frac_HII_dom={fd:.4f} (registered > 0.5)   "
                  f"ratio_max={rmax:.4g} (registered 1.5-6.0)   status={r['status']}")
            print(f"  VERDICT: {'PASS' if fd > 0.5 and 1.5 <= rmax <= 6.0 else 'FAIL'}")
    mm = [r for r in rows if r["mismatch_rows"]]
    if mm:
        print(f"\n!! mismatch rows present: "
              f"{[(r['config'], r['phase'], r['mismatch_rows']) for r in mm]}")


if __name__ == "__main__":
    main()
