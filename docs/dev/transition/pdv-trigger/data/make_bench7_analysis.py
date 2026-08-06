#!/usr/bin/env python3
"""bench7 analysis — the THREE-WAY band-entry head-to-head (f_A vs f_mix vs f_kappa).

This is the deliverable the whole kappa-3way campaign exists for: f_kappa has never had an L21b
Theta_cum band-entry number, so the published comparison was two-way. It reads the 2026-07-30
ALL-FRESH harvests only (bench7 + bench5r + bench6r); it will refuse to fall back to a pre-cutoff
file, because mixing generations is what the campaign was ordered to stop.

Six tables, all written to data/bench7_analysis.csv with a `table` column:

  ARMS      per-arm Theta_cum, fate, and the TRUNCATED flag (see below)
  ENTRY     band-entry dose per (knob, bench) + the uniformity spread -> the three-way headline
  FIREMAP   f_kappa fate vs dose for the 6 band configs (K2) and the dense benches (K1b)
  DETERM    P4 - the K3 pairs' trajectory hashes
  G6        K4's f_mix ladder vs bench6r's, on the overlapping doses
  BACKREACT P2 - dMdt(f)/dMdt(1) along the run, from the bubble_dMdt column

TRUNCATION IS THE HEADLINE CAVEAT, not a footnote. An arm that stops with no `outcome` recorded
ran out of wall-clock mid-solve; its Theta_cum integrates a SHORTER window than a completed arm,
which biases the value without any physics changing. The 2026-07-19 vs 2026-07-30 comparison proved
this is not hypothetical: 116/120 baseline arms reproduced BIT-IDENTICALLY, and every arm that
"moved" was a truncated one that simply stopped at a different step count. That is what broke gate
G0 (`bench1 f_A` entry 74.8 -> 83.2). So every band-entry number here carries a
`truncated_arms_in_bracket` count, and a number derived from a truncated bracket is an ARTIFACT OF
WALL-CLOCK, not a measurement — G4 language: report it as "estimated", never as the answer.

    python docs/dev/transition/pdv-trigger/data/make_bench7_analysis.py
Deliverable: data/bench7_analysis.csv + bench7_{entry,firemap}.png + console tables.
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
RD = PDV / "runs" / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PDV))
from _stamp import stamp  # noqa: E402
from make_bench5_analysis import _read_csv, theta_cum_prefire  # noqa: E402
from make_bench6_analysis import band_entry, band_entry_extrapolated  # noqa: E402

L21B = (0.90, 0.99)
CLEAN = ["bench3_m1e5_r5", "bench2_m1e5_r10", "bench1_m5e4_r20"]  # clean-blowout, the decision set
DENSE = ["bench4_m1e5_r2p5", "bench5_m5e5_r2p5"]
BAND_CONFIGS = [
    "simple_cluster",
    "small_dense_highsfe",
    "pl2_steep",
    "midrange_pl0",
    "be_sphere",
    "large_diffuse_lowsfe",
]
NBAR = {
    "bench1_m5e4_r20": 43.1,
    "bench2_m1e5_r10": 690.0,
    "bench3_m1e5_r5": 5520.0,
    "bench4_m1e5_r2p5": 44200.0,
    "bench5_m5e5_r2p5": 228000.0,
}


def load(summary, trajdir, required=True):
    p = RD / summary
    if not p.exists():
        if required:
            sys.exit(f"ABORT: {summary} missing — run ./sync_bench.sh <campaign> reduce && down.")
        return {}
    rows = {r["run_name"]: r for r in _read_csv(p)}
    for name, r in rows.items():
        tp = RD / trajdir / f"{name}.csv"
        r["_tcum"] = r["_traw"] = r["_tend"] = None
        r["_truncated"] = not r.get("outcome")
        if tp.exists():
            traj = _read_csv(tp)
            if traj:
                r["_tcum"], r["_traw"], r["_tend"], _ = theta_cum_prefire(traj)
                r["_traj"] = traj
    return rows


def parse7(name):
    """'k1_bench1_m5e4_r20__fk12_diag' -> (phase, subject, knob, dose, arm)."""
    phase, rest = name.split("_", 1)
    subject, tag = rest.split("__", 1)
    arm = "diag" if tag.endswith("_diag") else "prod"
    tag = tag.replace("_diag", "")
    rep = ""
    if phase == "k3" and tag[-2] == "_":
        rep, tag = tag[-1], tag[:-2]
    knob = "fkappa" if tag.startswith("fk") else "fmix" if tag.startswith("fm") else "?"
    return phase, subject, knob, float(tag[2:]), arm, rep


def fate(r):
    """The fire-map label, matching theta5k's four-way vocabulary."""
    if r["_truncated"]:
        return "TRUNC"
    if str(r.get("fired_cooling_balance", "")).lower() == "true":
        return "FIRED"
    out = r.get("outcome", "")
    if out == "shell_collapsed":
        return "CONDENSE"
    if out == "shell_dissolved":
        return "DRAIN"
    return "NOFIRE"


def series_with_flags(pts):
    """pts = [(dose, tcum, truncated)] -> (sorted (dose,tcum), n_truncated)."""
    s = sorted((d, t) for d, t, _ in pts if t is not None)
    return s, sum(1 for _, t, tr in pts if t is not None and tr)


def entry_row(knob, bench, pts):
    s, ntr = series_with_flags(pts)
    if not s:
        return None
    e = band_entry(s)
    ex = None if e else band_entry_extrapolated(s)
    return {
        "knob": knob,
        "bench": bench,
        "n_bar_H": f"{NBAR.get(bench, float('nan')):g}",
        "entry_dose": f"{(e or ex):.4g}" if (e or ex) else "",
        "measured_in_grid": "yes" if e else ("no — EXTRAPOLATED" if ex else "no — UNREACHED"),
        "grid_max": f"{max(d for d, _ in s):g}",
        "n_doses": str(len(s)),
        "truncated_arms": str(ntr),
        "track": "  ".join(f"{d:g}:{t:.3f}" for d, t in s),
    }


def main():
    b7 = load("bench7_summary.csv", "bench7_traj")
    b5 = load("bench5r_summary.csv", "bench5r_traj")
    b6 = load("bench6r_summary.csv", "bench6r_traj")
    rows = []

    # ---------------------------------------------------------------- ARMS
    for name, r in sorted(b7.items()):
        ph, subj, knob, dose, arm, rep = parse7(name)
        rows.append(
            {
                "table": "ARMS",
                "run_name": name,
                "phase": ph.upper(),
                "subject": subj,
                "knob": knob,
                "dose": f"{dose:g}",
                "arm": arm + (f"_{rep}" if rep else ""),
                "theta_max": r.get("theta_max", ""),
                "theta_cum": f"{r['_tcum']:.4f}" if r["_tcum"] else "",
                "t_final": r.get("t_final", ""),
                "n_impl": r.get("n_impl", ""),
                "fate": fate(r),
                "truncated": "YES" if r["_truncated"] else "",
            }
        )

    # ---------------------------------------------------------------- ENTRY (the headline)
    entries = []
    for bench in CLEAN:
        # f_kappa: K1 diag arms. Theta_0 (dose 1) comes from the FRESH bench5r __none arm.
        t0 = b5.get(f"{bench}__none_diag")
        base = [(1.0, t0["_tcum"], t0["_truncated"])] if t0 and t0["_tcum"] else []
        fk = base + [
            (parse7(n)[3], r["_tcum"], r["_truncated"])
            for n, r in b7.items()
            if n.startswith("k1_")
            and parse7(n)[1] == bench
            and parse7(n)[2] == "fkappa"
            and parse7(n)[4] == "diag"
            and r["_tcum"]
        ]
        # f_A: bench5r (<=16) + bench6r (24-128)
        fa = [
            (
                1.0 if n.endswith("__none_diag") else float(n.split("__fa")[1].split("_")[0]),
                r["_tcum"],
                r["_truncated"],
            )
            for src in (b5, b6)
            for n, r in src.items()
            if n.startswith(bench + "__")
            and n.endswith("_diag")
            and r["_tcum"]
            and ("__fa" in n or n.endswith("__none_diag"))
        ]
        # f_mix: K4 (fresh, in-grid) where present, else bench6r's ladder
        k4 = [
            (parse7(n)[3], r["_tcum"], r["_truncated"])
            for n, r in b7.items()
            if n.startswith("k4_")
            and parse7(n)[1] == bench
            and parse7(n)[4] == "diag"
            and r["_tcum"]
        ]
        fm6 = [
            (float(n.split("__fm")[1].split("_")[0]), r["_tcum"], r["_truncated"])
            for n, r in b6.items()
            if n.startswith(bench + "__fm") and n.endswith("_diag") and r["_tcum"]
        ]
        fm = base + (k4 if k4 else fm6)
        for knob, pts in (("fA", fa), ("fmix", fm), ("fkappa", fk)):
            row = entry_row(knob, bench, pts)
            if row:
                row["table"] = "ENTRY"
                rows.append(row)
                entries.append(row)

    for knob in ("fA", "fmix", "fkappa"):
        vals = [
            (r["bench"], float(r["entry_dose"]), r["measured_in_grid"], int(r["truncated_arms"]))
            for r in entries
            if r["knob"] == knob and r["entry_dose"]
        ]
        if len(vals) > 1:
            lo, hi = min(v[1] for v in vals), max(v[1] for v in vals)
            rows.append(
                {
                    "table": "ENTRY",
                    "knob": knob,
                    "bench": "SPREAD(max/min)",
                    "entry_dose": f"{hi / lo:.3f}",
                    "measured_in_grid": (
                        "all in-grid"
                        if all(v[2] == "yes" for v in vals)
                        else "PARTLY EXTRAPOLATED/UNREACHED"
                    ),
                    "n_doses": str(len(vals)),
                    "truncated_arms": str(sum(v[3] for v in vals)),
                    "track": ", ".join(f"{b.split('_')[0]}:{d:.3g}" for b, d, _, _ in vals),
                }
            )

    # ---------------------------------------------------------------- EXPONENT (P1's assumption)
    # P1 assumed Theta_cum ~ f^q with q in [0.55,0.70], carried over from the §24 FIXED-STATE L_cool
    # exponents. Fitting q on the integrated metric is the direct test of that carry-over. Points
    # from truncated or early-dissolving arms are excluded: a shrinking window lowers Theta_cum
    # without any dose-response change, which would bias q downward for the wrong reason.
    EXCLUDE = {
        ("fA", "bench1_m5e4_r20"): [128.0],
        ("fmix", "bench3_m1e5_r5"): [8.0],
        ("fmix", "bench2_m1e5_r10"): [12.0, 16.0],
        ("fmix", "bench1_m5e4_r20"): [16.0],
        ("fkappa", "bench3_m1e5_r5"): [16.0, 24.0, 32.0],
    }
    for r in entries:
        pts = [tuple(map(float, p.split(":"))) for p in r["track"].split("  ") if ":" in p]
        bad = EXCLUDE.get((r["knob"], r["bench"]), [])
        pts = [p for p in pts if p[0] not in bad and p[1] > 0]
        if len(pts) < 3:
            continue
        n = len(pts)
        sx = sum(math.log(p[0]) for p in pts)
        sy = sum(math.log(p[1]) for p in pts)
        sxx = sum(math.log(p[0]) ** 2 for p in pts)
        sxy = sum(math.log(p[0]) * math.log(p[1]) for p in pts)
        q = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        rows.append(
            {
                "table": "EXPONENT",
                "knob": r["knob"],
                "bench": r["bench"],
                "entry_dose": f"{q:.4f}",
                "n_doses": str(n),
                "measured_in_grid": (
                    "within P1 [0.55,0.70]"
                    if 0.55 <= q <= 0.70
                    else "BELOW P1 [0.55,0.70] — prediction missed"
                ),
                "truncated_arms": str(len(bad)),
                "track": "excluded doses: " + (",".join(f"{d:g}" for d in bad) or "none"),
            }
        )

    # ---------------------------------------------------------------- TRIGGER (instantaneous)
    # Theta_cum is an INTEGRATED, L_mech-weighted mean over the blowout window. It is the right
    # metric for the L21b comparison, because Lancaster measures a cumulative radiated fraction.
    # It is NOT what TRINITY's trigger uses: run_energy_implicit_phase.py:1250 fires on
    # (Lgain - Lloss)/Lgain <= phaseSwitch_LlossLgain, i.e. theta >= 0.95, evaluated per step with
    # no memory. So the dose that makes a cloud FIRE is set by max_t theta, not by the integral --
    # and the standard protocol's own rule 3 already says theta is reported as theta_max.
    # This table recomputes band entry on that instantaneous criterion, on the PROD arms (the ones
    # running the live trigger). theta_max >= 0.95 is exactly "the trigger fires at some point".
    TRIG = 0.95

    def _tmax(r):
        try:
            v = float(r.get("theta_max"))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    def _trigger_track(bench, knob):
        pts = {}
        base = b5.get(f"{bench}__none")
        if base and _tmax(base) is not None:
            pts[1.0] = (_tmax(base), bool(base.get("outcome")))
        srcs = {"fA": [(b5, "__fa"), (b6, "__fa")], "fmix": [(b6, "__fm"), (b7, "__fm")],
                "fkappa": [(b7, "__fk")]}[knob]
        for src, tag in srcs:
            for n, r in src.items():
                if n.endswith("_diag"):
                    continue
                stem = n[3:] if n[:3] in ("k1_", "k4_") else n
                if not stem.startswith(bench + tag):
                    continue
                try:
                    d = float(stem.split(tag)[1])
                except ValueError:
                    continue
                v = _tmax(r)
                if v is not None:
                    pts[d] = (v, bool(r.get("outcome")))
        return pts

    trig_entries = {}
    for bench in CLEAN:
        for knob in ("fmix", "fA", "fkappa"):
            pts = _trigger_track(bench, knob)
            if len(pts) < 2:
                continue
            p = sorted((d, v) for d, (v, _) in pts.items())
            e = None
            for (d0, t0), (d1, t1) in zip(p, p[1:]):
                if t0 < TRIG <= t1:
                    f = (TRIG - t0) / (t1 - t0)
                    e = math.exp(math.log(d0) + f * (math.log(d1) - math.log(d0)))
                    break
            if e is None and p and p[0][1] >= TRIG:
                e = p[0][0]
            trig_entries.setdefault(knob, {})[bench] = e
            rows.append({
                "table": "TRIGGER", "knob": knob, "bench": bench,
                "n_bar_H": f"{NBAR[bench]:g}",
                "entry_dose": f"{e:.4g}" if e else "",
                "measured_in_grid": "yes" if e else f"NEVER reaches {TRIG} within the grid",
                "grid_max": f"{max(p)[0]:g}", "n_doses": str(len(p)),
                "truncated_arms": str(sum(1 for v, ok in pts.values() if not ok)),
                "track": "  ".join(f"{d:g}:{v:.3f}" for d, v in p),
            })
    for knob in ("fmix", "fA", "fkappa"):
        v = trig_entries.get(knob, {})
        vals = [x for x in v.values() if x]
        rows.append({
            "table": "TRIGGER", "knob": knob, "bench": "SPREAD(max/min)",
            "entry_dose": f"{max(vals) / min(vals):.3f}" if len(vals) == len(CLEAN) else "",
            "measured_in_grid": "all benches fire" if len(vals) == len(CLEAN)
            else "UNREACHED on " + ", ".join(b for b in CLEAN if not v.get(b)),
            "n_doses": str(len(vals)),
            "track": ", ".join(f"{b.split('_')[0]}:{v[b]:.3g}" for b in CLEAN if v.get(b)),
        })

    # ---------------------------------------------------------------- FIREMAP
    def firemap(subjects, prefix):
        doses = sorted({parse7(n)[3] for n in b7 if n.startswith(prefix)})
        for s in subjects:
            cells = {}
            for n, r in b7.items():
                if n.startswith(prefix) and parse7(n)[1] == s and parse7(n)[4] == "prod":
                    cells[parse7(n)[3]] = fate(r)
            if cells:
                rows.append(
                    {
                        "table": "FIREMAP",
                        "subject": s,
                        "knob": "fkappa",
                        "track": "  ".join(f"{d:g}:{cells.get(d, '-')}" for d in doses),
                        "n_doses": str(len(cells)),
                        "truncated_arms": str(sum(1 for v in cells.values() if v == "TRUNC")),
                        "entry_dose": ",".join(f"{d:g}" for d in doses if cells.get(d) == "FIRED"),
                    }
                )
        return doses

    firemap(BAND_CONFIGS, "k2_")
    firemap(DENSE, "k1b_")

    # ---------------------------------------------------------------- DETERM (P4)
    hp = RD / "bench7_hashes.csv"
    if hp.exists():
        H = {r["run_name"]: r for r in _read_csv(hp)}
        for k in sorted(H):
            if k.startswith("k3_") and k.endswith("_a"):
                b = k[:-2] + "_b"
                same = b in H and H[k]["sha256"] == H[b]["sha256"]
                rows.append(
                    {
                        "table": "DETERM",
                        "run_name": k[:-2],
                        "subject": parse7(k)[1],
                        "dose": f"{parse7(k)[3]:g}",
                        "fate": fate(b7[k]) if k in b7 else "",
                        "measured_in_grid": "IDENTICAL" if same else "DIFFERS",
                        "track": H[k]["sha256"][:16],
                    }
                )

    # ---------------------------------------------------------------- G6
    for bench in ("bench1_m5e4_r20", "bench2_m1e5_r10"):
        for n, r in sorted(b7.items()):
            if not (n.startswith("k4_") and parse7(n)[1] == bench and parse7(n)[4] == "diag"):
                continue
            dose = parse7(n)[3]
            ref = b6.get(f"{bench}__fm{dose:g}_diag")
            if ref and ref["_tcum"] and r["_tcum"]:
                d = (r["_tcum"] - ref["_tcum"]) / ref["_tcum"] * 100
                rows.append(
                    {
                        "table": "G6",
                        "run_name": n,
                        "subject": bench,
                        "dose": f"{dose:g}",
                        "theta_cum": f"{r['_tcum']:.4f}",
                        "entry_dose": f"{ref['_tcum']:.4f}",
                        "track": f"{d:+.2f}%",
                        "measured_in_grid": "PASS" if abs(d) <= 2 else "FAIL (>2%)",
                        "truncated": "YES" if (r["_truncated"] or ref["_truncated"]) else "",
                    }
                )

    # ---------------------------------------------------------------- BACKREACT (P2)
    for bench in CLEAN:
        ref = b5.get(f"{bench}__none_diag")
        if not (ref and ref.get("_traj")):
            continue
        r1 = {
            float(x["t_now"]): float(x["bubble_dMdt"])
            for x in ref["_traj"]
            if x.get("bubble_dMdt") not in (None, "", "None")
        }
        for n, r in sorted(b7.items()):
            if not (n.startswith("k1_") and parse7(n)[1] == bench and parse7(n)[4] == "diag"):
                continue
            f = parse7(n)[3]
            traj = r.get("_traj") or []
            pts = [
                (float(x["t_now"]), float(x["bubble_dMdt"]))
                for x in traj
                if x.get("bubble_dMdt") not in (None, "", "None")
            ]
            if not pts or not r1:
                continue
            ts = sorted(r1)
            ratios = []
            for t, v in pts:
                near = min(
                    ts, key=lambda u: abs(math.log10(max(u, 1e-12)) - math.log10(max(t, 1e-12)))
                )
                if (
                    r1[near]
                    and abs(math.log10(max(near, 1e-12)) - math.log10(max(t, 1e-12))) < 0.05
                ):
                    ratios.append((t, v / r1[near]))
            if len(ratios) < 3:
                continue
            pred = f ** (2 / 7)
            first, last = ratios[0], ratios[-1]
            rows.append(
                {
                    "table": "BACKREACT",
                    "run_name": n,
                    "subject": bench,
                    "dose": f"{f:g}",
                    "entry_dose": f"{pred:.4f}",
                    "theta_cum": f"{first[1]:.4f}",
                    "track": f"{last[1]:.4f} @ t={last[0]:.3g}",
                    "measured_in_grid": f"{(first[1] / pred - 1) * 100:+.2f}% -> "
                    f"{(last[1] / pred - 1) * 100:+.2f}%",
                    "n_doses": str(len(ratios)),
                }
            )

    # ---------------------------------------------------------------- write
    cols = [
        "table",
        "run_name",
        "phase",
        "subject",
        "knob",
        "dose",
        "arm",
        "bench",
        "n_bar_H",
        "theta_max",
        "theta_cum",
        "t_final",
        "n_impl",
        "fate",
        "entry_dose",
        "measured_in_grid",
        "grid_max",
        "n_doses",
        "truncated_arms",
        "truncated",
        "track",
    ]
    out = HERE / "bench7_analysis.csv"
    with out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write(
            "# SOURCES READ: bench7_summary.csv, bench5r_summary.csv, bench6r_summary.csv "
            "(2026-07-30 ALL-FRESH harvests only — no pre-cutoff fallback)\n"
        )
        fh.write(
            "# TRUNCATED = the arm stopped with no `outcome`: it ran out of wall-clock "
            "mid-solve, so its Theta_cum integrates a SHORTER window and is biased LOW with no "
            "physics change. A band-entry dose whose bracketing arms include a truncated arm "
            "is an artifact of wall-clock, NOT a measurement — see the G0 failure "
            "(bench1 f_A 74.8 -> 83.2 from one truncated arm).\n"
        )
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}\n")

    # ---------------------------------------------------------------- console + plots
    print("=" * 100)
    print("THE THREE-WAY BAND-ENTRY TABLE  (L21b band [0.90,0.99]; clean-blowout benches)")
    print("=" * 100)
    print(
        f"{'knob':<8s} {'bench':<20s} {'n_H':>9s} {'entry':>9s} {'in-grid?':<22s} {'trunc':>6s}  track"
    )
    for r in rows:
        if r["table"] == "ENTRY" and r.get("bench") != "SPREAD(max/min)":
            print(
                f"{r['knob']:<8s} {r['bench']:<20s} {r['n_bar_H']:>9s} {r['entry_dose'] or '—':>9s} "
                f"{r['measured_in_grid']:<22s} {r['truncated_arms']:>6s}  {r['track'][:60]}"
            )
    print()
    for r in rows:
        if r["table"] == "ENTRY" and r.get("bench") == "SPREAD(max/min)":
            print(
                f"  SPREAD {r['knob']:<8s} = {r['entry_dose']:>7s}x   [{r['measured_in_grid']}] "
                f"truncated arms in brackets: {r['truncated_arms']}   ({r['track']})"
            )

    print("\n" + "=" * 100)
    print("f_kappa FIRE MAP")
    print("=" * 100)
    for r in rows:
        if r["table"] == "FIREMAP":
            print(f"  {r['subject']:<22s} {r['track']}")
            print(f"  {'':<22s} FIRED at: {r['entry_dose'] or '— none —'}")

    print("\n" + "=" * 100)
    print("DOSE-RESPONSE EXPONENT q  (Theta_cum ~ f^q) — P1 assumed q in [0.55,0.70]")
    print("=" * 100)
    for r in rows:
        if r["table"] == "EXPONENT":
            print(
                f"  {r['knob']:<8s} {r['bench']:<20s} q = {r['entry_dose']:>7s}  "
                f"({r['n_doses']} pts)  {r['measured_in_grid']}"
            )

    print("\n" + "=" * 100)
    print("P4 DETERMINISM (K3 pairs) · G6 (K4 vs bench6r) · P2 (dMdt back-reaction)")
    print("=" * 100)
    dt = [r for r in rows if r["table"] == "DETERM"]
    print(
        f"  P4: {sum(1 for r in dt if r['measured_in_grid'] == 'IDENTICAL')}/{len(dt)} pairs bit-identical"
    )
    g6 = [r for r in rows if r["table"] == "G6"]
    if g6:
        print(
            f"  G6: {sum(1 for r in g6 if r['measured_in_grid'].startswith('PASS'))}/{len(g6)} "
            f"overlapping f_mix doses reproduce bench6r within 2%"
        )
        for r in g6:
            if not r["measured_in_grid"].startswith("PASS"):
                print(
                    f"      FAIL {r['run_name']:<38s} {r['theta_cum']} vs {r['entry_dose']} ({r['track']})"
                    + ("  [TRUNCATED]" if r.get("truncated") else "")
                )
    br = [r for r in rows if r["table"] == "BACKREACT"]
    if br:
        print(
            f"  P2: {len(br)} f_kappa arms with dMdt ratios; "
            f"Eq-47 error start->end (a NEGATIVE drift = back-reaction as E_b drains):"
        )
        for r in br[:8]:
            print(f"      {r['run_name']:<40s} f^2/7={r['entry_dose']}  {r['measured_in_grid']}")

    trunc = [r for r in rows if r["table"] == "ARMS" and r["truncated"]]
    print(f"\n  ⚠️  G3: {len(trunc)}/174 bench7 arms TRUNCATED (wall-clock, no outcome recorded).")
    for r in trunc:
        print(f"      {r['run_name']:<44s} t={float(r['t_final']):.4f} n_impl={r['n_impl']}")

    # ---------------------------------------------------------------- plot 2: the fire map
    fm = [r for r in rows if r["table"] == "FIREMAP"]
    if fm:
        COL = {
            "FIRED": "#1a9850",
            "CONDENSE": "#4575b4",
            "DRAIN": "#fdae61",
            "NOFIRE": "#d9d9d9",
            "TRUNC": "#d73027",
            "-": "#ffffff",
        }
        doses = sorted({float(c.split(":")[0]) for r in fm for c in r["track"].split("  ")})
        figf, axf = plt.subplots(figsize=(1.05 * len(doses) + 3.4, 0.52 * len(fm) + 1.9))
        for y, r in enumerate(reversed(fm)):
            cells = dict(c.split(":") for c in r["track"].split("  "))
            for x, d in enumerate(doses):
                v = cells.get(f"{d:g}", "-")
                axf.add_patch(
                    plt.Rectangle(
                        (x, y), 1, 1, facecolor=COL.get(v, "#fff"), edgecolor="white", lw=1.5
                    )
                )
                if v not in ("-",):
                    axf.text(
                        x + 0.5,
                        y + 0.5,
                        v[0],
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if v != "NOFIRE" else "#555",
                        weight="bold",
                    )
        axf.set_xlim(0, len(doses))
        axf.set_ylim(0, len(fm))
        axf.set_xticks([i + 0.5 for i in range(len(doses))])
        axf.set_xticklabels([f"{d:g}" for d in doses], fontsize=9)
        axf.set_yticks([i + 0.5 for i in range(len(fm))])
        axf.set_yticklabels([r["subject"] for r in reversed(fm)], fontsize=9)
        axf.set_xlabel(r"$f_\kappa$", fontsize=11)
        axf.set_title(
            "f_kappa fire map — F=FIRED  C=CONDENSE  D=DRAIN  N=NOFIRE  T=truncated\n"
            "no single dose fires all 6 band configs (best 5/6 at 8, 9, 12)",
            fontsize=10,
        )
        for sp in axf.spines.values():
            sp.set_visible(False)
        axf.tick_params(length=0)
        figf.tight_layout()
        figf.savefig(PDV / "bench7_firemap.png", dpi=135)
        print(f"wrote {PDV / 'bench7_firemap.png'}")

    # ---------------------------------------------------------------- plot 1: the three-way tracks
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, bench in zip(axes, CLEAN):
        for knob, style in (("fA", "o-"), ("fmix", "s-"), ("fkappa", "^-")):
            r = next((x for x in entries if x["knob"] == knob and x["bench"] == bench), None)
            if not r or not r["track"]:
                continue
            pts = [tuple(map(float, p.split(":"))) for p in r["track"].split("  ") if ":" in p]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, ms=4, label=knob)
        ax.axhspan(*L21B, color="0.85", zorder=0)
        ax.set_xscale("log")
        ax.set_title(f"{bench}  (n̄={NBAR[bench]:g})", fontsize=10)
        ax.set_xlabel("dose")
        ax.set_ylim(0, 1.4)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\Theta_{\rm cum}$ over the blowout window")
    axes[0].legend(fontsize=8)
    fig.suptitle("Three-way band entry — L21b band shaded  (2026-07-30 ALL-FRESH)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PDV / "bench7_entry.png", dpi=135)
    print(f"\nwrote {PDV / 'bench7_entry.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
