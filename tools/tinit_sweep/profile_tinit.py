"""Per-T_init detail table for the sweep report (one pinned pass per state).

Shows where the T_init sensitivity lives: prints, per state, L_total relative
to the 3e4 baseline at each T_init, and the L3 (linear intermediate patch)
fraction of L_total -- the suspected source. Determinism is already proven by
run_sweep.py; this is a single descriptive pass, so run it pinned.

Usage:  OMP_NUM_THREADS=1 ... python profile_tinit.py <states_dir> [base.param]
"""
from __future__ import annotations

import os
import sys
import glob
import json
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SWEEP = os.path.join(_HERE, "sweep.py")
BASELINE = 3.0e4


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    target = argv[0]
    base = argv[1] if len(argv) > 1 else None
    states = sorted(glob.glob(os.path.join(target, "*.pkl"))) if os.path.isdir(target) else [target]

    grid = None
    print(f"{'state':18} {'Eb_idx':>6}  L_total(3e4)   "
          "rel-to-baseline at T_init [1.5e4 2e4 3e4 4e4 5e4]      L3/L_total at each")
    for s in states:
        out = json.loads(subprocess.run(
            [sys.executable, _SWEEP, s] + ([base] if base else []),
            capture_output=True, text=True).stdout.strip().splitlines()[-1])
        cells = {c["t_init"]: c for c in out["cells"]}
        grid = sorted(cells)
        b = cells[BASELINE]
        if b["status"] != "ok":
            print(f"{out['state'][:18]:18}  baseline crashed: {b['status']}")
            continue
        Lb = b["L_total"]
        rels = " ".join(f"{(cells[t]['L_total']-Lb)/Lb:+.3%}" if cells[t]["status"] == "ok"
                        else f"[{cells[t]['status']}]" for t in grid)
        l3f = " ".join(f"{cells[t]['L3']/cells[t]['L_total']:.2%}" if cells[t]["status"] == "ok"
                       else "--" for t in grid)
        idx = out["state"].split("_")[1]
        print(f"{out['state'][:18]:18} {idx:>6}  {Lb:.4e}   {rels}   {l3f}")
    print(f"\ngrid T_init = {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
