#!/usr/bin/env python3
"""Batch 1 (docs/dev/switchon-successor/PLAN.md §5): where does the energy go when
the `dt_switchon` ramp is absent — cooling, or PdV work?

The phase-1a energy equation (`energy_phase_ODEs.get_ODE_Edot_pure`) is

    dEb/dt = (Lmech_total - L_cool) - 4*pi*R2**2 * Pb * v2 - L_leak

and the ramp acts on exactly one term: it suppresses `R1`, which lowers `Pb`,
which lowers the PdV work. Cooling responds only *indirectly* (through the
bubble structure solve). So decomposing an existing pair of runs settles which
term actually drains the bubble — and therefore whether a geometric successor
(PLAN §3 S1/S2) is even addressing the cause, or would merely be masking a
cooling problem.

No new simulations: this reads the snapshots of runs already committed/held
from earlier batches, so it costs seconds rather than an hour.

    python docs/dev/switchon-successor/harness/drain_budget.py \\
        --ramp-on <dir>/dictionary.jsonl --ramp-off <dir>/dictionary.jsonl \\
        [--out docs/dev/switchon-successor/data/drain_budget.csv]

Also reports, per snapshot, the ramp's geometric leverage `(R1/R2)**3` and the
Weaver+77 Eq. 20 tracking ratio `(Eb/t) / ((5/11) L_w)`, which is bar N1.
"""
import argparse
import csv
import json
import math
import os

FOURPI = 4.0 * math.pi


def rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


DT_SWITCHON = 1e-3  # Myr — the constant under study


def ramped_Pb(r, ramp_active):
    """The pressure the ENERGY EQUATION actually used.

    Load-bearing subtlety, found by validating this budget against the observed
    dEb/dt and getting a sign flip: the snapshot's ``Pb`` is the **unramped**
    pressure from the bubble-structure solve, not the ramped value the RHS used
    (measured ~3x larger inside the ramp window). Diagnostics therefore do not
    show the pressure that drove the trajectory -- the same class of masking the
    phase1a-init work found for the old ``vd`` override. Reconstruct it by
    scaling with the shell-volume ratio, which needs no unit conversion.
    """
    if not ramp_active:
        return r["Pb"]
    frac = min(1.0, r["t_now"] / DT_SWITCHON)          # tSF = 0 on these configs
    vol_unramped = r["R2"] ** 3 - r["R1"] ** 3
    vol_ramped = r["R2"] ** 3 - (frac * r["R1"]) ** 3
    return r["Pb"] * vol_unramped / vol_ramped


def budget(r, ramp_active):
    """The RHS terms, in the code's own units [Msun pc^2 / Myr^3]."""
    gain = r["Lmech_total"]
    cool = r["bubble_LTotal"]
    pdv = FOURPI * r["R2"] ** 2 * ramped_Pb(r, ramp_active) * r["v2"]
    leak = r.get("bubble_Leak") or 0.0
    return gain, cool, pdv, leak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ramp-on", required=True)
    p.add_argument("--ramp-off", required=True)
    p.add_argument("--label", default="simple_cluster")
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "drain_budget.csv"))
    a = p.parse_args()

    out = []
    for arm, path in (("ramp_on", a.ramp_on), ("ramp_off", a.ramp_off)):
        rs = rows(path)
        ramp_active = arm == "ramp_on"
        for i, r in enumerate(rs):
            gain, cool, pdv, leak = budget(r, ramp_active)
            net = gain - cool - pdv - leak
            lw = gain  # Lmech_total is L_w in the Weaver Eq.20 sense here
            weaver = (r["Eb"] / r["t_now"]) / ((5.0 / 11.0) * lw) if lw else float("nan")
            out.append({
                "arm": arm, "config": a.label, "snapshot": i,
                "t_now_Myr": repr(r["t_now"]), "Eb_au": repr(r["Eb"]),
                "R1_over_R2": repr(r["R1"] / r["R2"]),
                "ramp_leverage_R1overR2_cubed": repr((r["R1"] / r["R2"]) ** 3),
                "Pb_snapshot_unramped": repr(r["Pb"]),
                "Pb_used_by_energy_eq": repr(ramped_Pb(r, ramp_active)),
                "gain_Lmech": repr(gain), "loss_cooling": repr(cool),
                "loss_PdV": repr(pdv), "loss_leak": repr(leak),
                "net_dEb_dt": repr(net),
                "cooling_over_gain": repr(cool / gain) if gain else "",
                "PdV_over_gain": repr(pdv / gain) if gain else "",
                "weaver_Eb_over_t_ratio": repr(weaver),
                # Weaver partition: 5/11 of L_w stays thermal, so PdV/L_w -> 6/11 = 0.545
                "PdV_over_gain_vs_weaver_6_11": repr((pdv / gain) / (6.0 / 11.0)) if gain else "",
                "obs_dEb_dt": repr((rs[i + 1]["Eb"] - r["Eb"]) / (rs[i + 1]["t_now"] - r["t_now"]))
                              if i + 1 < len(rs) else "",
            })
        if len(rs) > 1:
            g, c, d, lk = budget(rs[1], ramp_active)
            print(f"{arm:9} n={len(rs):4}  at snapshot 1: cooling/gain={c/g:8.4f}  "
                  f"PdV/gain={d/g:8.4f}  net/gain={(g-c-d-lk)/g:+8.4f}  "
                  f"(R1/R2)^3={(rs[1]['R1']/rs[1]['R2'])**3:.4f}")

    path = os.path.normpath(a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("# switchon-successor Batch 1: phase-1a energy budget, ramp-on vs ramp-off.\n")
        fh.write("# dEb/dt = (Lmech_total - L_cool) - 4*pi*R2^2*Pb*v2 - L_leak, decomposed per\n")
        fh.write("# snapshot from committed runs (no new simulations). Units are TRINITY AU.\n")
        fh.write("# weaver_Eb_over_t_ratio = (Eb/t) / ((5/11) L_w); 1.0 = on the Weaver Eq.20\n")
        fh.write("# analytic track (bar N1). ramp_leverage = (R1/R2)^3, what the ramp suppresses.\n")
        fh.write(f"# ramp_on={a.ramp_on}\n# ramp_off={a.ramp_off}\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {path}  ({len(out)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
