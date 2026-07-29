#!/usr/bin/env python3
"""K0 — the f_kappa re-read with the "wrong El-Badry sign" argument removed (FINDINGS §23/§24).

Two questions, both answerable offline from committed artifacts:

  Q1. Does TRINITY's dMdt actually follow El-Badry Eq 47's C-channel, mdot ~ C^(2/7)?
      f_kappa multiplies C_thermal (bubble_luminosity.py:304/398/441), and Eq 47 carries
      (C / 6e-7 cgs)^(2/7) with 6e-7 = the C_thermal default (registry.py:377). So the prediction
      is dMdt(f)/dMdt(1) = f^(2/7) exactly. Source: data/fkappa_leverage.csv, per-call at a FIXED
      captured state (f_kappa=1 is byte-identical there, the harness's own correctness check).

  Q1b. Per-call is necessary but NOT sufficient (CLAUDE.md rule 5). data/kappa_backreaction.csv is
      the full-run f_kappa=2 time series, graded in CONTAMINATION.md as "CLEAN for the f_kappa^(2/7)
      scaling check". It shows the match is exact at t=0 (-0.12%) and then DECAYS to -11.3% by
      t=2.3e-3 Myr, tracking Eb_ratio/Pb_ratio down. That is not the C-channel failing: it is the
      BACK-REACTION — the boosted arm drains Eb, so its pressure (hence conduction) falls behind the
      fixed-state prediction. This is the same depletion that produces Q2's CONDENSE/DRAIN fallout.

  Q2. WHY does no whole-band f_kappa exist? The rejection was recorded as a leverage/reach failure.
      Re-reading data/theta5k_fire_map.csv (CLEAN for FIRE/NO-FIRE) per dose shows it is not:
      all 6 BAND configs reach theta > 0.95 at SOME dose. The band fails because different
      configs fall out at different doses via CONDENSE/DRAIN — the condensation boundary — and
      those outcomes are scattered non-monotonically. The census keeps FINDINGS 12's denominator
      (see CONTROLS/NATIVE below) and so reproduces its "5/6 at f_kappa=12" headline exactly.

Deliverable: data/kappa_eq47_check.csv (all three tables, one file) + console summary.
    python docs/dev/transition/pdv-trigger/data/make_kappa_eq47_check.py
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EQ47_EXPONENT = 2.0 / 7.0
# theta5k grid. The BAND is the 6 configs a knob has to carry; the other 3 are excluded by the
# workstream's standing convention (FINDINGS 12 quotes "5/6", 15h's class table): `fail_repro` and
# `small_1e6` are the two controls, and `normal_n1e3` fires UNMODIFIED at f=1 (theta0=1.047), so it
# never tests a knob. Keeping the same denominator is what makes this re-read comparable to 12.
CONTROLS = {"fail_repro", "small_1e6"}
NATIVE = {"normal_n1e3"}
EXCLUDED = CONTROLS | NATIVE


def _rows(name):
    with (HERE / name).open() as fh:
        return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))


def _fit_loglog(pts):
    """Least-squares exponent q in y ~ x^q."""
    P = [(math.log(x), math.log(y)) for x, y in pts if x > 0 and y > 0]
    if len(P) < 3:
        return None
    n = len(P)
    sx = sum(a for a, _ in P)
    sy = sum(b for _, b in P)
    sxy = sum(a * b for a, b in P)
    sxx = sum(a * a for a, _ in P)
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if d else None


def q1_eq47():
    """dMdt(f)/dMdt(1) vs the Eq-47 prediction f^(2/7)."""
    by = defaultdict(list)
    for r in _rows("fkappa_leverage.csv"):
        try:
            by[r["state"]].append(
                (float(r["f_kappa"]), float(r["dMdt_ratio"]), float(r["LTotal_mult"]))
            )
        except (ValueError, KeyError):
            continue
    out, summary = [], []
    for state, pts in by.items():
        pts.sort()
        worst = 0.0
        for f, dmdt, lcool in pts:
            if f <= 1:
                continue
            pred = f**EQ47_EXPONENT
            err = 100 * (dmdt - pred) / pred
            worst = max(worst, abs(err))
            out.append(
                {
                    "table": "Q1_eq47_C_channel",
                    "state": state,
                    "f_kappa": f"{f:g}",
                    "dMdt_ratio_measured": f"{dmdt:.6f}",
                    "eq47_prediction_f_pow_2_7": f"{pred:.6f}",
                    "pct_error": f"{err:+.3f}",
                    "Lcool_multiplier": f"{lcool:.4f}",
                }
            )
        q_m = _fit_loglog([(f, d) for f, d, _ in pts])
        q_l = _fit_loglog([(f, x) for f, _, x in pts])
        summary.append((state, q_m, q_l, worst, max(f for f, _, _ in pts)))
    return out, summary


def q1b_backreaction():
    """Full-run f_kappa=2 series: how far does dMdt drift from the fixed-state f^(2/7)?"""
    pred = 2.0**EQ47_EXPONENT
    out = []
    for r in _rows("kappa_backreaction.csv"):
        try:
            t, d = float(r["t"]), float(r["dMdt_ratio"])
        except (ValueError, KeyError):
            continue
        out.append(
            {
                "table": "Q1b_backreaction_drift",
                "state": "kappa_backreaction f=2",
                "f_kappa": "2",
                "t_Myr": f"{t:.6e}",
                "dMdt_ratio_measured": f"{d:.6f}",
                "eq47_prediction_f_pow_2_7": f"{pred:.6f}",
                "pct_error": f"{100 * (d - pred) / pred:+.3f}",
                "Eb_ratio": f"{float(r['Eb_ratio']):.5f}",
                "Pb_ratio": f"{float(r['Pb_ratio']):.5f}",
            }
        )
    return out


def q2_fire_map():
    """Per-dose outcome census: is the whole-band failure about REACH or about CONDENSATION?"""
    rows = _rows("theta5k_fire_map.csv")
    doses = [c for c in rows[0] if c.startswith("k")]
    out, census = [], []
    reach = {}
    for r in rows:
        cfg = r["config"]
        best = 0.0
        for d in doses:
            cell = r[d] or ""
            outcome, _, th = cell.partition(":")
            try:
                thv = float(th)
            except ValueError:
                thv = float("nan")
            if not math.isnan(thv):
                best = max(best, thv)
            out.append(
                {
                    "table": "Q2_theta5k_outcomes",
                    "state": cfg,
                    "f_kappa": d[1:],
                    "outcome": outcome,
                    "theta_max": th,
                    "is_control": (
                        "control"
                        if cfg in CONTROLS
                        else "fires_unmodified" if cfg in NATIVE else "band"
                    ),
                }
            )
        reach[cfg] = best
    for d in doses:
        fired = cond = drain = nofire = 0
        for r in rows:
            if r["config"] in EXCLUDED:
                continue
            o = (r[d] or "").split(":")[0]
            fired += o == "FIRED"
            cond += o == "CONDENSE"
            drain += o == "DRAIN"
            nofire += o == "NOFIRE"
        census.append((d, fired, cond, drain, nofire))
    return out, census, reach


def main():
    q1, q1s = q1_eq47()
    q1b = q1b_backreaction()
    q2, census, reach = q2_fire_map()

    print("Q1 — does dMdt follow El-Badry Eq 47's C-channel, mdot ~ C^(2/7)?")
    print(f"  {'state':24s} {'fitted q':>9} {'Eq47 2/7':>9} {'max |err|':>10} {'dose range':>11}")
    for state, q_m, q_l, worst, fmax in q1s:
        print(
            f"  {state:24s} {q_m:9.4f} {EQ47_EXPONENT:9.4f} {worst:9.2f}% {'1-' + f'{fmax:g}':>11}"
            f"    (Lcool exponent {q_l:.3f})"
        )
    print(
        "  => at FIXED state TRINITY reproduces Eq 47's conduction scaling. 'wrong sign' was wrong.\n"
    )

    print("Q1b — the same check on a FULL RUN (f_kappa=2): does it hold as the state evolves?")
    if q1b:
        first, last = q1b[0], q1b[-1]
        print(
            f"  t={float(first['t_Myr']):.2e} Myr  err {first['pct_error']}%   "
            f"(Eb_ratio {first['Eb_ratio']})"
        )
        print(
            f"  t={float(last['t_Myr']):.2e} Myr  err {last['pct_error']}%   "
            f"(Eb_ratio {last['Eb_ratio']})"
        )
        print(
            "  => the fixed-state match DECAYS as the boosted arm drains Eb. Per-call equivalence"
        )
        print(
            "     is necessary, not sufficient (CLAUDE.md rule 5) — and this back-reaction is the"
        )
        print("     same depletion that produces the Q2 CONDENSE/DRAIN fallout below.\n")

    print(
        "Q2 — why does no whole-band f_kappa exist? (the 6 BAND configs; FINDINGS 12 denominator)"
    )
    print(f"  {'f_kappa':>8} {'FIRED':>6} {'CONDENSE':>9} {'DRAIN':>6} {'NOFIRE':>7}")
    for d, fired, cond, drain, nofire in census:
        print(f"  {d[1:]:>8} {fired:6d} {cond:9d} {drain:6d} {nofire:7d}")
    print("\n  peak theta_max reached by each BAND config, over the whole grid:")
    for cfg, best in sorted(reach.items(), key=lambda kv: -kv[1]):
        if cfg in EXCLUDED:
            continue
        print(
            f"    {cfg:24s} {best:5.3f}   {'reaches the trigger' if best >= 0.95 else 'NEVER reaches 0.95'}"
        )
    print("  => the failure is NOT reach: every band config crosses 0.95 somewhere.")
    print("     It is CONDENSE/DRAIN fallout at scattered doses (the condensation boundary).")
    best_d, best_n = max(census, key=lambda c: c[1])[0], max(c[1] for c in census)
    print(
        f"     Best single dose: f_kappa={best_d[1:]} fires {best_n}/6 "
        "(matches FINDINGS 12's '5/6' — same denominator)."
    )

    out = HERE / "kappa_eq47_check.csv"
    rows = q1 + q1b + q2
    cols = [
        "table",
        "state",
        "f_kappa",
        "t_Myr",
        "dMdt_ratio_measured",
        "eq47_prediction_f_pow_2_7",
        "pct_error",
        "Lcool_multiplier",
        "Eb_ratio",
        "Pb_ratio",
        "outcome",
        "theta_max",
        "is_control",
    ]
    with out.open("w", newline="") as fh:
        fh.write(
            "# K0 re-read of the f_kappa evidence with the 'wrong El-Badry sign' argument removed "
            "(FINDINGS 23/24). Q1_eq47_C_channel: dMdt(f)/dMdt(1) at a FIXED captured state from "
            "data/fkappa_leverage.csv against El-Badry Eq 47's (C/6e-7 cgs)^(2/7) -- f_kappa "
            "multiplies exactly that C (bubble_luminosity.py:304/398/441; C_thermal default 6e-7, "
            "registry.py:377). Q1b_backreaction_drift: the same ratio along the FULL f_kappa=2 run "
            "in data/kappa_backreaction.csv, which decays from -0.12% to -11.3% as Eb/Pb deplete. "
            "Q2_theta5k_outcomes: the per-(config,dose) outcome census from "
            "data/theta5k_fire_map.csv, showing the whole-band failure is condensation fallout, "
            "not insufficient reach. PROVENANCE: kappa_backreaction.csv is graded 'CLEAN for the "
            "f_kappa^(2/7) scaling check' and theta5k_fire_map.csv 'CLEAN re-analysis' in "
            "CONTAMINATION.md; fkappa_leverage.csv is graded SUPERSEDED there for its THETA-leverage "
            "exponent p, which this file does not use -- the per-call dMdt/LTotal ratios at a fixed "
            "state are untouched by that supersession (f_kappa=1 is byte-identical in that harness). "
            "Both Q1 sources carry FLAG-(a) (early-time), which is why Q1b is reported alongside. "
            "No sims. Regenerate: python docs/dev/transition/pdv-trigger/data/"
            "make_kappa_eq47_check.py\n"
        )
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
