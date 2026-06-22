#!/usr/bin/env python3
"""H1 audit: does bubble_Lloss (=bubble_LTotal=Lcool) surge UP or COLLAPSE over a
run, and is the cooling-balance ratio's non-firing driven by Lloss falling, Lgain
rising, or both? Compares hybr (c0_*_h0.csv) vs legacy (c0_*_legacy.csv).

Reads the COMMITTED cleanroom CSVs (no sim re-run). Emits a summary CSV next to
this script. The transition fires (run_energy_implicit_phase.py:1095) when
ratio = (Lgain - Lloss)/Lgain < 0.05, with Lgain=Lmech_total, Lloss=bubble_LTotal.

Usage: python docs/dev/transition/pt4/analyze_lcool_direction.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "cleanroom", "data")
CONFIGS = ["simple_cluster", "large_diffuse_lowsfe", "small_dense_highsfe",
           "midrange_pl0", "pl2_steep", "be_sphere"]


def load(path):
    """Return list of dict rows; floats parsed, blanks/nan dropped per-key."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def fcol(rows, key):
    """Parse a column to float, keeping (t, value) only where finite."""
    out = []
    for r in rows:
        try:
            t = float(r["t_now"])
            v = float(r[key])
        except (ValueError, TypeError, KeyError):
            continue
        if v != v:  # nan
            continue
        out.append((t, v))
    return out


def trend(series):
    """Crude direction: compare median of first third vs last third."""
    if len(series) < 6:
        return "n/a", None, None
    vals = [v for _, v in series]
    n = len(vals)
    first = sorted(vals[: n // 3])
    last = sorted(vals[-(n // 3):])
    mfirst = first[len(first) // 2]
    mlast = last[len(last) // 2]
    if mfirst == 0:
        return ("rise" if mlast > 0 else "flat"), mfirst, mlast
    ratio = mlast / mfirst
    if ratio > 1.3:
        d = "RISE"
    elif ratio < 0.77:
        d = "FALL"
    else:
        d = "flat"
    return d, mfirst, mlast


def ratio_series(rows):
    """(t, R2, ratio) where ratio=(Lgain-Lloss)/Lgain, matching the trigger."""
    out = []
    for r in rows:
        try:
            t = float(r["t_now"])
            lg = float(r["bubble_Lgain"])
            ll = float(r["bubble_Lloss"])
            R2 = float(r["R2"])
        except (ValueError, TypeError, KeyError):
            continue
        if lg != lg or ll != ll or lg <= 0:
            continue
        out.append((t, R2, (lg - ll) / lg))
    return out


def analyze(tag):
    summary = []
    for cfg in CONFIGS:
        path = os.path.join(DATA, f"c0_{cfg}_{tag}.csv")
        if not os.path.exists(path):
            summary.append(dict(config=cfg, tag=tag, status="MISSING"))
            continue
        rows = load(path)
        lloss = fcol(rows, "bubble_Lloss")
        lgain = fcol(rows, "bubble_Lgain")
        rs = ratio_series(rows)

        d_lloss, ll0, ll1 = trend(lloss)
        d_lgain, lg0, lg1 = trend(lgain)

        if rs:
            # min ratio and where
            tmin, r2min, rmin = min(rs, key=lambda x: x[2])
            t_first, _, _ = rs[0]
            t_last, _, ratio_last = rs[-1]
            # did it ever cross 0.05?
            crossed = [x for x in rs if x[2] < 0.05]
            cross_t = crossed[0][0] if crossed else None
            cross_R2 = crossed[0][1] if crossed else None
            ratio_final = ratio_last
        else:
            tmin = r2min = rmin = cross_t = cross_R2 = ratio_final = None

        # peak Lloss / Lgain values and times for the surge question
        ll_peak = max(lloss, key=lambda x: x[1]) if lloss else (None, None)
        lg_peak = max(lgain, key=lambda x: x[1]) if lgain else (None, None)

        summary.append(dict(
            config=cfg, tag=tag, n_rows=len(rows), n_ratio=len(rs),
            t_span=f"{rs[0][0]:.3g}..{rs[-1][0]:.3g}" if rs else "n/a",
            Lloss_dir=d_lloss, Lloss_first=ll0, Lloss_last=ll1,
            Lloss_peak_val=ll_peak[1], Lloss_peak_t=ll_peak[0],
            Lgain_dir=d_lgain, Lgain_first=lg0, Lgain_last=lg1,
            ratio_min=rmin, ratio_min_t=tmin, ratio_min_R2=r2min,
            ratio_final=ratio_final,
            crossed_0p05=("YES" if cross_t is not None else "NO"),
            cross_t=cross_t, cross_R2=cross_R2,
        ))
    return summary


def main():
    all_rows = []
    for tag in ("h0", "legacy"):
        all_rows.extend(analyze(tag))

    # print
    print(f"{'config':22} {'solver':7} {'Lloss':5} {'Lgain':5} "
          f"{'ratio_min':>10} {'@t':>8} {'ratio_fin':>10} {'cross<0.05':>10} {'@t':>8}")
    print("-" * 100)
    for s in all_rows:
        if s.get("status") == "MISSING":
            print(f"{s['config']:22} {s['tag']:7} MISSING")
            continue
        def f(x, fmt="{:.3g}"):
            return fmt.format(x) if isinstance(x, float) else str(x)
        print(f"{s['config']:22} {s['tag']:7} {s['Lloss_dir']:5} {s['Lgain_dir']:5} "
              f"{f(s['ratio_min']):>10} {f(s['ratio_min_t']):>8} "
              f"{f(s['ratio_final']):>10} {s['crossed_0p05']:>10} "
              f"{f(s['cross_t']):>8}")

    # detail table for Lloss surge/collapse magnitude
    print("\n=== Lloss (=Lcool) magnitude: first-third -> last-third median, and peak ===")
    print(f"{'config':22} {'solver':7} {'dir':5} {'first':>11} {'last':>11} "
          f"{'peak':>11} {'peak@t':>9}")
    for s in all_rows:
        if s.get("status") == "MISSING":
            continue
        def f(x):
            return "{:.3e}".format(x) if isinstance(x, float) and x is not None else "n/a"
        print(f"{s['config']:22} {s['tag']:7} {s['Lloss_dir']:5} "
              f"{f(s['Lloss_first']):>11} {f(s['Lloss_last']):>11} "
              f"{f(s['Lloss_peak_val']):>11} {f(s['Lloss_peak_t']):>9}")

    # write CSV
    out = os.path.join(HERE, "H1_lcool_direction_summary.csv")
    keys = ["config", "tag", "n_rows", "n_ratio", "t_span", "Lloss_dir",
            "Lloss_first", "Lloss_last", "Lloss_peak_val", "Lloss_peak_t",
            "Lgain_dir", "Lgain_first", "Lgain_last", "ratio_min", "ratio_min_t",
            "ratio_min_R2", "ratio_final", "crossed_0p05", "cross_t", "cross_R2"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for s in all_rows:
            if s.get("status") == "MISSING":
                w.writerow({"config": s["config"], "tag": s["tag"], "n_rows": "MISSING"})
            else:
                w.writerow({k: s.get(k) for k in keys})
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
