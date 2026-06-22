#!/usr/bin/env python3
"""Build the R1 firing-epoch summary from the per-config shadow CSVs.

Reads each docs/dev/transition/pt4/r1shadow/shadow_<config>.csv (the in-code 1b
shadow sideline: t_now,R2,rCloud,R2_over_rCloud,Eb,v2,Pb,Lgain,Lloss,
cooling_ratio,edot_balance,blowout_fired,ebpeak_fired), derives the first-firing
epoch of each criterion (blowout / Eb-peak), and writes one row per config to
r1_shadow_summary.csv.

Runtime + status come from runs/<config>_status.txt written by the run loop
(runtime_s + completed/timeout). final_t/R2/v2 come from the last shadow row.

Read-only over committed CSVs; no sims, no production edits.
Run: python docs/dev/transition/pt4/r1shadow/build_summary.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = ["simple_cluster", "small_dense_highsfe", "midrange_pl0", "pl2_steep",
           "be_sphere", "large_diffuse_lowsfe", "fail_repro", "fail_helix"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_shadow(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _read_status(config):
    """runtime_s + status from runs/<config>_status.txt (KEY=VALUE lines)."""
    p = os.path.join(HERE, "runs", f"{config}_status.txt")
    out = {"runtime_s": "", "status": ""}
    if os.path.exists(p):
        for line in open(p):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                if k in out:
                    out[k] = v
    return out


def _first_fire(rows, flag):
    for r in rows:
        if r.get(flag, "").strip() == "True":
            return r
    return None


def main():
    out_rows = []
    for cfg in CONFIGS:
        rows = _read_shadow(os.path.join(HERE, f"shadow_{cfg}.csv"))
        st = _read_status(cfg)
        bo = _first_fire(rows, "blowout_fired")
        ep = _first_fire(rows, "ebpeak_fired")
        last = rows[-1] if rows else {}

        bo_t = _f(bo["t_now"]) if bo else None
        ep_t = _f(ep["t_now"]) if ep else None
        if bo_t is not None and ep_t is not None:
            which = "blowout" if bo_t <= ep_t else "ebpeak"
        elif bo_t is not None:
            which = "blowout"
        elif ep_t is not None:
            which = "ebpeak"
        else:
            which = "none"

        out_rows.append({
            "config": cfg,
            "n_seg": len(rows),
            "blowout_t": bo_t,
            "blowout_R2": _f(bo["R2"]) if bo else None,
            "blowout_R2overRc": _f(bo["R2_over_rCloud"]) if bo else None,
            "ebpeak_t": ep_t,
            "ebpeak_R2": _f(ep["R2"]) if ep else None,
            "which_fired_first": which,
            "reached_rCloud": bool(bo),
            "final_t": _f(last.get("t_now")) if last else None,
            "final_R2": _f(last.get("R2")) if last else None,
            "final_v2": _f(last.get("v2")) if last else None,
            "runtime_s": st["runtime_s"],
            "status": st["status"],
        })

    out = os.path.join(HERE, "r1_shadow_summary.csv")
    cols = ["config", "n_seg", "blowout_t", "blowout_R2", "blowout_R2overRc",
            "ebpeak_t", "ebpeak_R2", "which_fired_first", "reached_rCloud",
            "final_t", "final_R2", "final_v2", "runtime_s", "status"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out}")
    for r in out_rows:
        print(f"  {r['config']:22s} n_seg={r['n_seg']:>4} "
              f"blowout_t={r['blowout_t']} ebpeak_t={r['ebpeak_t']} "
              f"first={r['which_fired_first']} status={r['status']}")


if __name__ == "__main__":
    main()
