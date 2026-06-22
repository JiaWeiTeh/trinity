#!/usr/bin/env python3
"""Tabulate the H3 Eb-floor matrix from h3_eval.csv: one compact line per
(config, variant), grouped by class, plus the no-op check (does EBFLOOR match V0
where the floor never activates?) and the trajectory evidence for collapse
configs (does R2 keep growing? does anything else break?).

Usage: python h3_analyze.py [h3_eval.csv]   (default: alongside this script)
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CLASS = {  # config -> class
    "simple_cluster": "stall", "large_diffuse_lowsfe": "stall",
    "small_dense_highsfe": "stall", "midrange_pl0": "stall",
    "pl2_steep": "stall", "be_sphere": "stall",
    "fail_repro": "collapse", "fail_helix": "collapse",
    "mass_5e8": "collapse", "mass_1e9": "collapse",
    "small_1e5": "healthy", "small_1e6": "healthy", "small_1e7": "healthy",
}
ORDER = ["stall", "collapse", "healthy"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(v, prec=4):
    f = _f(v)
    if f is None:
        return str(v) if v not in (None, "") else "-"
    return f"{f:.{prec}g}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "h3_eval.csv")
    rows = list(csv.DictReader(open(path)))
    by_key = {(r["config"], r["variant"]): r for r in rows}

    print(f"# H3 Eb-floor matrix  ({path})\n")
    hdr = (f"{'config':22s} {'var':8s} {'class':8s} {'outcome':9s} {'reached':4s} "
           f"{'end_code':9s} {'final_t':10s} {'final_R2':9s} {'final_v2':9s} "
           f"{'final_Eb':11s} {'floor_act':9s} {'trig':5s} {'rmin':7s} {'rt_s':6s}")
    print(hdr)
    print("-" * len(hdr))

    for klass in ORDER:
        cfgs = [c for c, k in CLASS.items() if k == klass]
        for cfg in cfgs:
            for var in ("V0", "EBFLOOR"):
                r = by_key.get((cfg, var))
                if not r:
                    print(f"{cfg:22s} {var:8s} {klass:8s} {'(missing)':9s}")
                    continue
                ec = str(r.get("end_code", ""))
                if str(r.get("crashed")).lower() == "true":
                    outcome = "CRASHED"
                elif ec == "timeout":
                    outcome = "TIMEOUT"
                else:
                    outcome = "completed"
                print(f"{cfg:22s} {var:8s} {klass:8s} {outcome:9s} "
                      f"{str(r.get('reached_phase','')):4s} {ec:9s} "
                      f"{fmt(r.get('final_t'),6):10s} {fmt(r.get('final_R2')):9s} "
                      f"{fmt(r.get('final_v2')):9s} {fmt(r.get('final_Eb'),4):11s} "
                      f"{str(r.get('floor_activated')):9s} "
                      f"{str(r.get('trigger_fired'))[:5]:5s} {fmt(r.get('ratio_min'),3):7s} "
                      f"{fmt(r.get('runtime_s'),4):6s}")
        print()

    # No-op check: where floor never activated, EBFLOOR final state should match V0
    print("## No-op check (floor never activated => EBFLOOR ?= V0)")
    for cfg in CLASS:
        v0 = by_key.get((cfg, "V0"))
        eb = by_key.get((cfg, "EBFLOOR"))
        if not v0 or not eb:
            continue
        fa = str(eb.get("floor_activated"))
        if fa == "False":
            # compare final state at matched stop (both ran same stop_t)
            same = True
            for k in ("reached_phase", "end_code"):
                if str(v0.get(k)) != str(eb.get(k)):
                    same = False
            dR = abs((_f(v0.get("final_R2")) or 0) - (_f(eb.get("final_R2")) or 0))
            dEb_rel = None
            e0, e1 = _f(v0.get("final_Eb")), _f(eb.get("final_Eb"))
            if e0 and e1:
                dEb_rel = abs(e0 - e1) / (abs(e0) + 1e-300)
            verdict = "NO-OP(match)" if (same and dR < 1e-6 and (dEb_rel or 0) < 1e-9) else "DIFFERS"
            print(f"  {cfg:22s} floor_act=False  dR2={dR:.2e} dEb_rel={dEb_rel}  -> {verdict}")
        else:
            print(f"  {cfg:22s} floor_act={fa}  (activated or killed -- see trajectory)")


if __name__ == "__main__":
    main()
