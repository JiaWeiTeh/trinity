#!/usr/bin/env python3
"""Turnkey reduction of the 📏 theta5 matrix — the multiplier-knob calibration (PLAN OPEN(1)).

Reads runs/data/theta5_summary.csv (written by harvest_theta_max.py after the HPC matrix runs)
and produces, per the BEST PATH FORWARD:
  1. per-config theta0 (= theta_max of the __none arm — the true >=5 Myr starting deficit),
     leverage p (fit of log theta_max vs log f_mix over NON-fired arms), and f_fire (smallest
     f_mix arm that fired cooling_balance);
  2. the theta1-collapse fit for the multiplier knob (FINDINGS §9's law, re-fit):
     log10 f_fire = c + s*log10(0.95/theta0);
  3. the single-physical-f_mix scorecard: for each candidate f, which configs fire (the route-a
     split), against the acceptance target "Lancaster-band GMCs (nCore >~ 48 cm^-3) fire with
     theta_max in 0.9-0.99".

Usage:
    python make_theta5_calibration.py [--csv runs/data/theta5_summary.csv]
    python make_theta5_calibration.py --selftest     # synthetic data; no inputs needed
Deliverable (committed): runs/data/theta5_calibration.csv + stdout scorecard.
"""

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _stamp import stamp  # noqa: E402  (workstream provenance stamp)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "data" / "theta5_summary.csv"
TRIGGER = 0.95
F_OF_MODE = {"none": 1.0, "mult2": 2.0, "mult4": 4.0, "mult8": 8.0}

# nCore per config (cm^-3), for the Lancaster-band acceptance line (n_fire ~ 48).
NCORE = {
    "simple_cluster": 1e5,
    "small_dense_highsfe": 1e6,
    "pl2_steep": 1e5,
    "midrange_pl0": 1e4,
    "be_sphere": 1e4,
    "large_diffuse_lowsfe": 1e2,
    "fail_repro": 1e2,
    "small_1e6": 1e2,
}


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def reduce_rows(rows):
    """rows: dicts with run_name, theta_max, fired_cooling_balance -> per-config calibration."""
    by_cfg = {}
    for r in rows:
        name = r["run_name"]
        if "__" not in name:
            continue
        cfg, mode = name.rsplit("__", 1)
        if mode not in F_OF_MODE or not r.get("theta_max"):
            continue
        by_cfg.setdefault(cfg, {})[F_OF_MODE[mode]] = {
            "theta_max": float(r["theta_max"]),
            "fired": str(r.get("fired_cooling_balance", "")).lower() == "true",
        }

    out = []
    for cfg, arms in sorted(by_cfg.items()):
        theta0 = arms.get(1.0, {}).get("theta_max")
        fired_fs = sorted(f for f, a in arms.items() if a["fired"])
        f_fire = fired_fs[0] if fired_fs else None
        unfired = [(f, a["theta_max"]) for f, a in sorted(arms.items()) if not a["fired"]]
        p = None
        if len(unfired) >= 2:
            _, p = linfit([math.log10(f) for f, _ in unfired], [math.log10(t) for _, t in unfired])
        out.append(
            {
                "config": cfg,
                "nCore": NCORE.get(cfg, ""),
                "theta0": theta0,
                "leverage_p": p,
                "f_fire": f_fire,
                "arms": ";".join(
                    f"{f:g}:{a['theta_max']:.3f}{'*' if a['fired'] else ''}"
                    for f, a in sorted(arms.items())
                ),
            }
        )

    # theta1-collapse fit (configs with a bracketed fire above 1)
    pts = [
        (math.log10(TRIGGER / r["theta0"]), math.log10(r["f_fire"]))
        for r in out
        if r["theta0"] and r["f_fire"] and r["f_fire"] > 1 and 0 < r["theta0"] < TRIGGER
    ]
    collapse = linfit(*zip(*pts)) if len(pts) >= 2 else (None, None)
    return out, collapse


def scorecard(out, collapse):
    lines = []
    c, s = collapse
    if s is not None:
        lines.append(
            f"theta1-collapse (multiplier): log10 f_fire = {c:.3f} + {s:.3f}*log10(0.95/theta0)  (n={sum(1 for r in out if r['f_fire'] and r['f_fire']>1)})"
        )
    for cand in (2.0, 4.0, 8.0):
        fire = [r["config"] for r in out if r["f_fire"] and r["f_fire"] <= cand]
        stay = [r["config"] for r in out if not r["f_fire"] or r["f_fire"] > cand]
        band = [r for r in out if float(r["nCore"] or 0) >= 48]
        band_ok = (
            all(r["f_fire"] and r["f_fire"] <= cand for r in band) if band else "n/a (no nCore)"
        )
        lines.append(
            f"f_mix={cand:g}: fires {fire} | route-a {stay} | Lancaster-band(n>=48) all fire: {band_ok}"
        )
    return "\n".join(lines)


def selftest():
    # synthetic: theta = theta0 * f^p with p=0.3; fires (trigger truncates theta) when >= 0.95
    p = 0.3
    rows = []
    for cfg, theta0 in [("dense", 0.7), ("mid", 0.6), ("diffuse", 0.25)]:
        for mode, f in F_OF_MODE.items():
            theta = theta0 * f**p
            fired = theta >= TRIGGER
            rows.append(
                {
                    "run_name": f"{cfg}__{mode}",
                    "theta_max": f"{min(theta, 1.05):.6f}",
                    "fired_cooling_balance": str(fired),
                }
            )
    out, (c, s) = reduce_rows(rows)
    by = {r["config"]: r for r in out}
    assert abs(by["dense"]["theta0"] - 0.7) < 1e-6
    # dense: 0.7*f^0.3 >= 0.95 at f ~ 2.77 -> first arm that fires is 4
    assert by["dense"]["f_fire"] == 4.0, by["dense"]
    # diffuse: 0.25*8^0.3 = 0.466 -> never fires
    assert by["diffuse"]["f_fire"] is None
    # leverage recovered from unfired arms
    assert abs(by["diffuse"]["leverage_p"] - p) < 1e-6
    # collapse law: for an exact power law slope = 1/p, but f_fire is quantized to the
    # 4-point arm grid (dense fires first at 4 vs exact 2.77), which biases the 2-point
    # slope upward — accept the right order of magnitude
    assert s is not None and 1 / p * 0.6 < s < 1 / p * 2.0, (c, s)
    print("selftest OK:", scorecard(out, (c, s)), sep="\n")


def main(argv):
    if "--selftest" in argv:
        return selftest()
    csv_in = Path(argv[argv.index("--csv") + 1]) if "--csv" in argv else DEFAULT_CSV
    with csv_in.open() as fh:
        # skip the leading '# generated ...' provenance stamp (_stamp.py)
        rows = list(csv.DictReader(line for line in fh if not line.lstrip().startswith("#")))
    out, collapse = reduce_rows(rows)
    out_path = csv_in.parent / "theta5_calibration.csv"
    with out_path.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows -> {out_path}")
    print(scorecard(out, collapse))


if __name__ == "__main__":
    main(sys.argv[1:])
