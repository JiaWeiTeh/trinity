#!/usr/bin/env python3
"""Batch 4b scoring: N0/N1/N2 for the S4 arms against the ramp-active reference.

Reads finished run directories only — no simulations. The reference arm is the
ramp-active "after" run of `docs/dev/phase1a-stiffness` Batch 4 (current HEAD
behaviour), the same reference Batches 2 and 3 scored against.

* **N0** stopping fate, from `metadata.json[termination]` (`screen.fate`).
* **N1** mean `|1 - (Eb/t) / ((5/11) L_wind)|` over the first snapshots, candidate
  vs reference. Read it with PLAN.md §0.3: Weaver is a wind-only *reference*, so
  the bar is "no worse than the shipped ramp", never "must match".
* **N2** `|dR2|%` vs the reference at matched `t` (linear interpolation, never
  nearest-snapshot, never extrapolated).

    python docs/dev/switchon-successor/harness/s4_compare.py --out <csv>
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
SP = ("/tmp/claude-0/-home-user-trinity/bba6f6fe-c67c-5539-b2c1-ded81f52c5db"
      "/scratchpad")
CONFIGS = ["simple_cluster", "f1edge_hidens", "f1edge_lowdens", "gmc_control",
           "m43_probe"]
GRID_MYR = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2]
# Snapshots 1-5, matching the window D3's data/s2_state_trigger.csv scored so the
# two batches are directly comparable. Snapshot 0 is excluded on purpose: E0 is
# seeded from Weaver Eq.20, so its ratio is 1.000 by construction and including it
# just rescales every number by 5/6 (verified: reference 0.0827 over 0-5 vs the
# 0.0992 over 1-5 recorded in D3, and 0.0992*5/6 = 0.0827).
N1_FIRST, N1_LAST = 1, 5
WEAVER = 5.0 / 11.0


def load_screen():
    spec = importlib.util.spec_from_file_location(
        "screen", os.path.join(REPO, "docs", "dev", "screen", "screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_run(run_dir):
    jsonl = os.path.join(run_dir, "dictionary.jsonl")
    if not os.path.exists(jsonl):
        return None
    with open(jsonl) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def weaver_series(rows):
    """Signed (Eb/t)/((5/11) L_wind) over the N1 window (snapshots 1-5)."""
    out = []
    for r in rows[N1_FIRST:N1_LAST + 1]:
        t, Eb = r.get("t_now"), r.get("Eb")
        Lw = r.get("Lmech_W", r.get("Lmech_total"))
        if not t or Eb is None or not Lw:
            continue
        out.append((Eb / t) / (WEAVER * Lw))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "..", "data",
                                                 "s4_consistent_seed.csv"))
    a = p.parse_args()
    screen = load_screen()
    rows_out = []

    for cfg in CONFIGS:
        ref_dir = os.path.join(SP, "b4", cfg, "wt", f"{cfg}-after", "outputs", "screen")
        ref = read_run(ref_dir)
        if ref is None:
            print(f"!! missing reference for {cfg}: {ref_dir}")
            continue
        ref_fate = screen.fate(ref, ref_dir)
        ref_w = weaver_series(ref)
        ref_mean = sum(abs(1 - v) for v in ref_w) / len(ref_w) if ref_w else float("nan")
        tb = [r["t_now"] for r in ref]
        rb = [r["R2"] for r in ref]

        for var in ("sustain", "similarity"):
            cand_dir = os.path.join(SP, "s4", var, cfg, "outputs", "screen")
            cand = read_run(cand_dir)
            if cand is None:
                rows_out.append(dict(gate="N0", config=cfg, variant=var,
                                     quantity="stopping_fate", reference=ref_fate,
                                     measured="(no output)", verdict="FAIL"))
                print(f"{cfg:16s} {var:11s} NO OUTPUT")
                continue
            cf = screen.fate(cand, cand_dir)
            rows_out.append(dict(gate="N0", config=cfg, variant=var,
                                 quantity="stopping_fate", reference=ref_fate,
                                 measured=cf,
                                 verdict="PASS" if cf == ref_fate else "FAIL"))

            cw = weaver_series(cand)
            cm = sum(abs(1 - v) for v in cw) / len(cw) if cw else float("nan")
            rows_out.append(dict(
                gate="N1", config=cfg, variant=var,
                quantity=f"mean abs(1 - Weaver Eb/t ratio) over snaps {N1_FIRST}-{N1_LAST}",
                reference=f"{ref_mean:.4f}", measured=f"{cm:.4f}",
                verdict="PASS" if cm <= ref_mean else "FAIL"))
            rows_out.append(dict(
                gate="N1-direction", config=cfg, variant=var,
                quantity="signed Weaver ratio per snapshot",
                reference=" ".join(f"{v:.3f}" for v in ref_w),
                measured=" ".join(f"{v:.3f}" for v in cw), verdict="-"))

            ta = [r["t_now"] for r in cand]
            ra = [r["R2"] for r in cand]
            last = min(tb[-1], ta[-1])
            grid = sorted({t for t in GRID_MYR if t <= last} | {last})
            cells, worst = [], 0.0
            for t in grid:
                b, c = screen.interp(tb, rb, t), screen.interp(ta, ra, t)
                if b is None or c is None or b == 0:
                    cells.append("--")
                    continue
                pct = 100 * (c - b) / b
                worst = max(worst, abs(pct))
                cells.append(f"{pct:+.3f}")
            rows_out.append(dict(
                gate="N2", config=cfg, variant=var,
                quantity="dR2% at " + "/".join(f"{t:g}" for t in grid) + " Myr",
                reference="0", measured=" ".join(cells),
                verdict="PASS" if worst < 0.5 else f"FAIL (worst {worst:.3f}%)"))
            print(f"{cfg:16s} {var:11s} N0 {cf!r} vs {ref_fate!r} | "
                  f"N1 {cm:.4f} vs {ref_mean:.4f} | N2 worst {worst:.3f}%")

    out = os.path.normpath(a.out)
    with open(out, "w", newline="") as fh:
        fh.write(
            "# switchon-successor Batch 4b: candidate S4 -- fix the handover state, no ramp.\n"
            "#   similarity: v0 = (3/5) v_wind      sustain: v0 = (R1/R2)^2/2 * v_wind\n"
            "# r0, E0, T0 and dt_phase0 are left as phase 0 computed them.\n"
            "# Reference arm = the ramp-active Batch 4 'after' runs of docs/dev/phase1a-stiffness\n"
            "# (current HEAD with the shipped 1e-3 Myr ramp). stop_t = 0.02 Myr, separate\n"
            "# processes, matched t by linear interpolation.\n"
            "# Command: python docs/dev/switchon-successor/harness/s4_consistent_seed.py \\\n"
            "#            --config <name> --variant <v> --stop-t 0.02 --workdir <dir>\n"
            "#          python docs/dev/switchon-successor/harness/s4_compare.py   (2026-08-06)\n")
        w = csv.DictWriter(fh, fieldnames=["gate", "config", "variant", "quantity",
                                           "reference", "measured", "verdict"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
