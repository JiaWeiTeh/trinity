#!/usr/bin/env python3
"""Cross-validate the in-code 1b R1 blowout epoch against an OFFLINE computation
from the run's dictionary.jsonl (production snapshot stream).

The snapshot stream carries R2 but NOT rCloud (run-const, blank in snapshots), so
rCloud is sourced from docs/dev/transition/pt4/h2_rcloud_edge.csv. Offline blowout
epoch = first t where R2 > rCloud. We compare it to the in-code shadow_<config>.csv
blowout_t and confirm agreement to within one 1b segment.

Read-only; no sims, no production edits.
Run: python docs/dev/transition/pt4/r1shadow/cross_validate.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PT4 = os.path.normpath(os.path.join(HERE, ".."))

# configs to cross-validate (per task: at least simple_cluster + pl2_steep).
# be_sphere + large_diffuse_lowsfe added as extra cross-checks (all blowing-out
# cleanroom configs that reached 1b blowout).
CONFIGS = ["simple_cluster", "pl2_steep", "be_sphere", "large_diffuse_lowsfe"]


def _rcloud_table():
    out = {}
    with open(os.path.join(PT4, "h2_rcloud_edge.csv")) as fh:
        for r in csv.DictReader(fh):
            out[r["config"]] = float(r["rCloud_pc"])
    return out


def _load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda d: d.get("t_now", 0.0))
    return rows


def _shadow_rows(cfg):
    p = os.path.join(HERE, f"shadow_{cfg}.csv")
    with open(p) as fh:
        return list(csv.DictReader(fh))


def main():
    rc = _rcloud_table()
    print(f"{'config':18s} {'rCloud_pc':>11s} {'offline_blowout_t':>18s} "
          f"{'incode_blowout_t':>18s} {'|dt|':>10s} {'seg_dt':>10s} {'within_1seg':>12s}")
    print("-" * 100)
    results = []
    for cfg in CONFIGS:
        rcloud = rc[cfg]
        jsonl = os.path.join(HERE, "runs", cfg, "dictionary.jsonl")
        rows = _load_jsonl(jsonl)
        # offline blowout: first snapshot with R2 > rCloud
        off_t = None
        for r in rows:
            R2 = r.get("R2")
            if R2 is not None and R2 > rcloud:
                off_t = r.get("t_now")
                break

        # in-code blowout from shadow CSV
        srows = _shadow_rows(cfg)
        inc_t = None
        for r in srows:
            if r.get("blowout_fired", "").strip() == "True":
                inc_t = float(r["t_now"])
                break

        # typical 1b segment dt (from the shadow stream — the cadence of the
        # transition-site evaluations, which is the resolution of inc_t)
        st = [float(r["t_now"]) for r in srows]
        seg_dt = max((st[i + 1] - st[i] for i in range(len(st) - 1)), default=float("nan"))

        if off_t is not None and inc_t is not None:
            dt = abs(off_t - inc_t)
            within = dt <= 1.5 * seg_dt  # within one segment (small margin)
        else:
            dt = float("nan")
            within = False
        print(f"{cfg:18s} {rcloud:11.4f} {str(off_t):>18s} {str(inc_t):>18s} "
              f"{dt:10.3e} {seg_dt:10.3e} {str(within):>12s}")
        results.append((cfg, rcloud, off_t, inc_t, dt, seg_dt, within))
    return results


if __name__ == "__main__":
    main()
