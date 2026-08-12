#!/usr/bin/env python3
"""Is the momentum-phase `P_HII` actually the Strömgren pressure, or is it the wind ram pressure?

Found 2026-08-08 while measuring the shell force budget for a different question. In every
momentum-phase row of every run checked, the recorded `P_HII` equals `pRam(R2, Lmech, v_mech)`
= `L/(2πR₂²v)` to all printed digits — even though `P_HII` is documented as coming from the
Strömgren ionization balance and `n_IF_Str` is populated and varying.

That matters because the momentum-phase ODE right-hand side
(`trinity/phase2_momentum/run_momentum_phase.py`) does:

    P_drive   = snapshot.P_HII + P_ram
    F_pressure = 4π R₂² (P_drive − P_ext)

so if `P_HII == P_ram` the shell is accelerated by **twice the wind ram pressure**, and the
photoionized-gas channel contributes nothing of its own physics. This is in the integrator, not
just the diagnostics.

⚠️ This harness REPORTS; it does not judge. Whether the equality is a bug or an intended
consequence of the `n_IF_Str = min(n_IF_Str, shell_n0)` "pressure equilibrium for thin skins" cap
(`trinity/shell_structure/shell_structure.py:239-251`) is a question about model intent — see the
workstream README. The one thing the harness does assert is the *measurement*: how close the two
are, over how many rows, across how much dynamic range.

Usage (from the repo root):
    python docs/dev/momentum-pdrive/harness/check_phii_pram.py <run_dir> [<run_dir> ...]

    # the three arms this was found on:
    python docs/dev/momentum-pdrive/harness/check_phii_pram.py \
        outputs/bench5/bench1_m5e4_r20__none_diag \
        outputs/bench5/bench2_m1e5_r10__none_diag \
        outputs/bench5/bench3_m1e5_r5__none_diag

Each <run_dir> holds a `dictionary.jsonl`. Regenerate those with, e.g.:
    python run.py docs/dev/transition/pdv-trigger/runs/params/bench5/bench3_m1e5_r5__none_diag.param

Writes docs/dev/momentum-pdrive/data/phii_pram_evidence.csv (one row per run + per-run extremes).
"""

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _stamp import stamp  # noqa: E402

OUT_CSV = HERE.parent / "data" / "phii_pram_evidence.csv"
COLS = [
    "run",
    "n_momentum_rows",
    "t_first",
    "t_last",
    "P_ram_max_over_min",
    "max_abs_rel_diff_PHII_vs_PRAM",
    "max_abs_rel_diff_PHII_vs_recomputed_pRam",
    "n_rows_bitequal",
    "F_HII_equals_F_ram_all_rows",
    "P_drive_equals_2x_P_ram",
]


def _fin(v):
    return (
        v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
    )


def rows_of(run_dir):
    out = []
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("current_phase") != "momentum":
                continue
            out.append(d)
    out.sort(key=lambda d: d.get("t_now", 0.0))
    return out


def analyse(run_dir):
    rows = rows_of(run_dir)
    if not rows:
        return None
    rel_direct = rel_calc = 0.0
    n_bit = 0
    f_equal = drive_2x = True
    prams = []
    for d in rows:
        P_HII, P_ram = _fin(d.get("P_HII")), _fin(d.get("P_ram"))
        R2, L, v = _fin(d.get("R2")), _fin(d.get("Lmech_total")), _fin(d.get("v_mech_total"))
        if None in (P_HII, P_ram) or not P_ram:
            continue
        prams.append(P_ram)
        rel_direct = max(rel_direct, abs(P_HII - P_ram) / abs(P_ram))
        if P_HII == P_ram:
            n_bit += 1
        # pRam = L / (2 pi r^2 v)  — get_bubbleParams.pRam
        if None not in (R2, L, v) and R2 and v:
            calc = L / (2.0 * math.pi * R2**2 * v)
            if calc:
                rel_calc = max(rel_calc, abs(P_HII - calc) / abs(calc))
        F_HII, F_ram = _fin(d.get("F_HII")), _fin(d.get("F_ram"))
        if F_HII is not None and F_ram is not None and F_HII != F_ram:
            f_equal = False
        P_drive = _fin(d.get("P_drive"))
        if P_drive is not None and abs(P_drive - 2.0 * P_ram) > 1e-12 * max(1.0, abs(P_drive)):
            drive_2x = False
    if not prams:
        return None
    return {
        "run": run_dir.name,
        "n_momentum_rows": len(rows),
        "t_first": f"{rows[0].get('t_now'):.6g}",
        "t_last": f"{rows[-1].get('t_now'):.6g}",
        "P_ram_max_over_min": f"{max(prams) / min(prams):.6g}",
        "max_abs_rel_diff_PHII_vs_PRAM": f"{rel_direct:.3g}",
        "max_abs_rel_diff_PHII_vs_recomputed_pRam": f"{rel_calc:.3g}",
        "n_rows_bitequal": n_bit,
        "F_HII_equals_F_ram_all_rows": f_equal,
        "P_drive_equals_2x_P_ram": drive_2x,
    }


def main(argv):
    dirs = [Path(a) for a in argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    out = []
    for d in dirs:
        if not (d / "dictionary.jsonl").exists():
            print(f"skip {d} — no dictionary.jsonl")
            continue
        r = analyse(d)
        if r:
            out.append(r)
    if not out:
        print("no momentum-phase rows found in any run dir")
        return 1
    w = max(len(r["run"]) for r in out)
    print(
        f"{'run':{w}}  {'rows':>5} {'P_ram range':>12} {'relΔ vs P_ram':>14} "
        f"{'relΔ vs calc':>13} {'bit-equal':>10} {'P_drive=2·P_ram':>16}"
    )
    for r in out:
        print(
            f"{r['run']:{w}}  {r['n_momentum_rows']:>5} {r['P_ram_max_over_min']+'x':>12} "
            f"{r['max_abs_rel_diff_PHII_vs_PRAM']:>14} "
            f"{r['max_abs_rel_diff_PHII_vs_recomputed_pRam']:>13} "
            f"{str(r['n_rows_bitequal'])+'/'+str(r['n_momentum_rows']):>10} "
            f"{str(r['P_drive_equals_2x_P_ram']):>16}"
        )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write(
            "# Momentum-phase P_HII vs wind ram pressure. relΔ vs calc compares the recorded\n"
        )
        fh.write("# P_HII against pRam = L/(2*pi*R2^2*v) recomputed from the row's own columns.\n")
        fh.write(
            "# See docs/dev/momentum-pdrive/README.md — this REPORTS, it does not judge intent.\n"
        )
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(out)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
