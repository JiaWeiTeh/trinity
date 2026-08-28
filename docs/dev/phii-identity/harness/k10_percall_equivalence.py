#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 18 G18.0 — does the IMPLEMENTED helper equal the closure Batches 16/17 screened?

G18.0 is BLOCKING and pre-registered in PLAN.md (§Batch 18). It exists because
Batches 16 and 17 validated an offline *model* of K10; the arm runs *production code*.
If the two differ, the arm measures something nobody gated. CLAUDE.md rule 5: a
per-call equivalence is necessary but NOT sufficient — G18.3's full-run arm is the
sufficient half.

Run this from a worktree with the Batch 18 arm patch applied. It drives the real
`get_bubbleParams.get_phii_k10` on the same committed rows Batch 17 screened, rebuilds
each phase's real `P_drive` composition around the returned value, and compares against
`data/b17_dust_closure.csv`'s `drive_selfconsistent` / `composed_selfconsistent`.

Bar: 1e-10 relative on every row. Falsifier: any row worse ⇒ the arm is VOID.

    python docs/dev/phii-identity/harness/k10_percall_equivalence.py \
        --out docs/dev/phii-identity/data/b18_percall.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402
from trinity.bubble_structure import get_bubbleParams  # noqa: E402

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"
BAR = 1e-10

# The state get_phii_k10 reads. Set on the params object, one row at a time.
STATE = ("R2", "Qi", "Eb", "Lmech_total", "v_mech_total", "t_now", "current_phase")

FIELDS = [
    "config", "phase", "t", "R2", "screened_drive", "implemented_drive", "drive_relerr",
    "screened_composed", "implemented_composed", "composed_relerr", "status",
]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def compose(phase, P_conf, P_HII, P_ram):
    """The real P_drive expressions (verified at cce8c924, Batch 16)."""
    if phase == "momentum":
        return P_HII + P_ram
    if phase == "transition":
        return max(P_conf, P_HII + P_ram)
    return max(P_conf, P_HII)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b18_percall.csv")
    args = ap.parse_args()

    if get_bubbleParams.get_phii_c3c is not getattr(get_bubbleParams, "get_phii_k10", None):
        sys.exit("this worktree does not have the Batch 18 arm applied — nothing to check")

    screened = [r for r in csv.DictReader(
        l for l in open(DATA / "b17_dust_closure.csv") if not l.startswith("#"))
        if r.get("status") == "ok"]
    # b17 rows carry the trajectory state we need to reconstruct the call.
    traj = {}
    for name in ("b7_regime_trajectory.csv", "b12_lowwind_trajectory.csv"):
        for r in csv.DictReader(l for l in open(DATA / name) if not l.startswith("#")):
            if r.get("arm") == "c3c":
                traj[(name, r["t_now"])] = r
    by_t = {}
    for (name, t), r in traj.items():
        by_t.setdefault(name, {})[round(float(t), 12)] = r

    src = {"B3M": "b7_regime_trajectory.csv", "B3MW01": "b12_lowwind_trajectory.csv"}
    params = read_param(str(
        REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/"
        "bench3_m1e5_r5__none_diag.param"))

    rows, missing = [], 0
    for sr in screened:
        cfg, t = sr["config"], fnum(sr, "t")
        tr = by_t.get(src[cfg], {}).get(round(t, 12))
        if tr is None:
            missing += 1
            continue
        # Drive the REAL helper on this row's state.
        params["R2"].value = fnum(tr, "R2")
        params["Qi"].value = fnum(tr, "Qi")
        params["Eb"].value = fnum(tr, "Eb")
        params["Lmech_total"].value = fnum(tr, "Lmech_total")
        params["v_mech_total"].value = fnum(tr, "v_mech_total")
        params["t_now"].value = t
        params["current_phase"].value = tr["current_phase"]
        try:
            ret = get_bubbleParams.get_phii_k10(params, None)
        except Exception as exc:
            rows.append(dict(config=cfg, phase=sr["phase"], t=t, R2=fnum(tr, "R2"),
                             status=f"raised:{type(exc).__name__}"))
            continue

        P_conf, P_ram = fnum(sr, "P_conf"), fnum(sr, "P_ram") or 0.0
        # The implemented drive: undo the mapping to recover P_conf*rho.
        phase = sr["phase"]
        if phase == "momentum":
            impl_drive = ret + P_ram
        elif phase == "transition":
            impl_drive = ret + P_ram
        else:
            impl_drive = ret
        impl_comp = compose(phase, P_conf, ret, P_ram)
        s_drive, s_comp = fnum(sr, "drive_selfconsistent"), fnum(sr, "composed_selfconsistent")
        rows.append(dict(
            config=cfg, phase=phase, t=t, R2=fnum(tr, "R2"),
            screened_drive=s_drive, implemented_drive=impl_drive,
            drive_relerr=abs(impl_drive / s_drive - 1.0) if s_drive else None,
            screened_composed=s_comp, implemented_composed=impl_comp,
            composed_relerr=abs(impl_comp / s_comp - 1.0) if s_comp else None,
            status="ok",
        ))

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)} rows compared ({missing} screened rows had no trajectory match, "
          f"{len(rows)-len(ok)} raised)")
    if not ok:
        sys.exit("no comparable rows")

    for key in ("drive_relerr", "composed_relerr"):
        e = [r[key] for r in ok if r[key] is not None]
        worst = max(e)
        print(f"G18.0 {key:18} worst {worst:.3e} vs {BAR:.0e} -> "
              f"{'PASS' if worst <= BAR else 'FAIL'}")
        for ph in ("energy", "implicit", "transition", "momentum"):
            sub = [r[key] for r in ok if r["phase"] == ph and r[key] is not None]
            if sub:
                print(f"      {ph:11} n={len(sub):3d} worst {max(sub):.3e}")

    worst_all = max([r[k] for r in ok for k in ("drive_relerr", "composed_relerr")
                     if r[k] is not None])
    print(f"\nG18.0 {'PASS' if worst_all <= BAR else 'FAIL'} overall "
          f"(worst {worst_all:.3e})")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 18 G18.0: implemented get_phii_k10 vs the Batch 17 screened closure.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    sys.exit(0 if worst_all <= BAR else 1)


if __name__ == "__main__":
    main()
