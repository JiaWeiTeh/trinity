#!/usr/bin/env python3
"""Stale-segment (no-root) decomposition of Θ_cum — the shared caveat on the bench5/bench6 metric.

Found 2026-07-28 while executing the FINDINGS §17 X1 fix; recorded as §18's metric caveat.

WHAT A "STALE" ROW IS. `run_energy_implicit_phase.py` only refreshes the bubble state when the
beta-delta solve returns a root: `updateDict(params, bubble_props)` is guarded by
`if bubble_props is not None` (:893) and `params['bubble_Lloss'].value` by
`if betadelta_result.L_loss is not None` (:929-930). On a NO-ROOT segment both stay frozen at the
previous accepted values, but the row is still logged and still harvested. Offline signature: the
traj `Lcool` column (= raw `bubble_LTotal`) repeats BIT-IDENTICALLY from the previous row.

WHY IT MATTERS. Θ_cum = ∫θ·L_mech dt / ∫L_mech dt integrates θ across those frozen spans, so a
long no-root grind contributes a held-over θ times real elapsed time. This is NOT an f_mix
artifact — it hits BOTH knobs, and on the band-setting arms it is *larger* on the f_A side
(bench3 fa16 67% of Θ_cum from stale rows, vs bench3 fm4 33%). It is the reason bench3's fm8
Θ_cum = 4.635 and bench5's fm dose-response is non-monotone: at high dose the solver spends most
of the window with no root, holding θ frozen at values > 1.

Reads only committed trajectories; no sims.

    python docs/dev/transition/pdv-trigger/data/make_bench_stale_segments.py
Deliverable: data/bench_stale_segments.csv
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RDATA = HERE.parent / "runs" / "data"

import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from make_bench5_analysis import _fnum, _read_csv  # noqa: E402


def decompose(rows):
    """(n_rows, n_stale, stale_time_frac, Theta_stale, Theta_solved) for one trajectory."""
    ts = [_fnum(r["t_now"]) for r in rows]
    th = [_fnum(r["theta"]) for r in rows]
    lm = [_fnum(r["Lmech"]) for r in rows]
    stale = {i for i in range(1, len(rows)) if rows[i]["Lcool"] == rows[i - 1]["Lcool"]}
    num_s = num_f = den = stale_t = 0.0
    for i in range(1, len(rows)):
        if None in (ts[i], ts[i - 1], th[i], th[i - 1], lm[i], lm[i - 1]):
            continue
        dt = ts[i] - ts[i - 1]
        seg = 0.5 * (th[i - 1] * lm[i - 1] + th[i] * lm[i]) * dt
        den += 0.5 * (lm[i - 1] + lm[i]) * dt
        if i in stale:
            num_s += seg
            stale_t += dt
        else:
            num_f += seg
    span = (ts[-1] - ts[0]) if (ts[0] is not None and ts[-1] is not None) else None
    return (
        len(rows),
        len(stale),
        (stale_t / span) if span else None,
        (num_s / den) if den else None,
        (num_f / den) if den else None,
    )


def main():
    sources = [("bench5", RDATA / "bench5_traj_hpc"), ("bench6", RDATA / "bench6_traj")]
    out_rows = []
    for campaign, d in sources:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.csv")):
            rows = _read_csv(p)
            if len(rows) < 2:
                continue
            n, ns, tf, th_s, th_f = decompose(rows)
            tag = p.stem.split("__", 1)[1].replace("_diag", "")
            out_rows.append(
                {
                    "campaign": campaign,
                    "run_name": p.stem,
                    "bench": p.stem.split("__")[0],
                    "knob": "fmix" if tag.startswith("fm") else "fA",
                    "dose": 1 if tag == "none" else float(tag[2:]),
                    "arm": "diag" if p.stem.endswith("_diag") else "prod",
                    "n_rows": n,
                    "n_stale": ns,
                    "stale_row_frac": f"{ns / n:.4f}",
                    "stale_time_frac": f"{tf:.4f}" if tf is not None else "",
                    "theta_cum_from_stale": f"{th_s:.4f}" if th_s is not None else "",
                    "theta_cum_from_solved": f"{th_f:.4f}" if th_f is not None else "",
                    "theta_cum_total": f"{th_s + th_f:.4f}" if None not in (th_s, th_f) else "",
                }
            )

    out = HERE / "bench_stale_segments.csv"
    with out.open("w", newline="") as fh:
        fh.write(
            "# Stale (no-root) segment decomposition of Theta_cum, per bench5/bench6 arm "
            "(2026-07-28, FINDINGS 18). A row is STALE when the raw Lcool (=bubble_LTotal) repeats "
            "bit-identically from the previous row -- the offline signature of a beta-delta segment "
            "that found no root, so run_energy_implicit_phase.py:893/:929 left bubble_props and "
            "bubble_Lloss frozen. theta_cum_from_stale + theta_cum_from_solved = theta_cum_total "
            "(matches theta_cum in bench5_analysis.csv / bench6_analysis.csv). This is a SHARED "
            "caveat on the metric, NOT an f_mix artifact: on the band-setting arms the stale share "
            "is larger for f_A (bench3 fa16: 0.647/0.965) than for f_mix (bench3 fm4: 0.296/0.895). "
            "Regenerate: python docs/dev/transition/pdv-trigger/data/make_bench_stale_segments.py\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {out}\n")

    # X3 bound: the §16 fallback double-boost multiplies the LIVE trigger's Lloss by fmix on
    # no-root segments. It cannot have inflated a published fire threshold, because no arm's
    # label came from the live trigger — harvest_theta_max.py:95 sets
    # fired = meta_fired or (reached_momentum and theta_max >= 0.95), and meta_fired is False
    # everywhere (metadata carries only the FINAL termination, which is always stopping_time /
    # shell_collapsed / shell_dissolved). So every label is computed from the SINGLE-boosted
    # recorded theta. Assert that here so the bound fails loudly if a future harvest changes it.
    print("FIRE-LABEL PROVENANCE (X3 / §17 gap e) — how many arms recorded a real cooling_balance:")
    meta_hits = n_arms = 0
    for s in ("bench5_summary_hpc.csv", "bench6_summary.csv"):
        p = RDATA / s
        if not p.exists():
            continue
        for r in _read_csv(p):
            n_arms += 1
            if "cooling" in (str(r.get("outcome", "")) + str(r.get("detail", ""))).lower():
                meta_hits += 1
    print(
        f"  {meta_hits} / {n_arms} arms carry a cooling_balance termination in metadata "
        f"-> every 'fired' label is the INFERRED branch (reached_momentum AND theta_max>=0.95), "
        f"computed from the single-boosted theta. The §16 double-boost therefore cannot have "
        f"inflated any published fire threshold.\n"
    )

    print("Band-setting diagnostic arms — how much of Θ_cum comes from frozen no-root rows:")
    key = {
        ("bench3_m1e5_r5", "fA", 16.0),
        ("bench2_m1e5_r10", "fA", 64.0),
        ("bench1_m5e4_r20", "fA", 64.0),
        ("bench3_m1e5_r5", "fmix", 4.0),
        ("bench2_m1e5_r10", "fmix", 8.0),
        ("bench1_m5e4_r20", "fmix", 8.0),
        ("bench3_m1e5_r5", "fmix", 8.0),
    }
    print(f"  {'arm':34s} {'rows':>5} {'stale':>6} {'Θ_stale':>8} {'Θ_solved':>9} {'Θ_tot':>7}")
    for r in out_rows:
        if r["arm"] == "diag" and (r["bench"], r["knob"], float(r["dose"])) in key:
            print(
                f"  {r['run_name']:34s} {r['n_rows']:5d} {r['n_stale']:6d} "
                f"{r['theta_cum_from_stale']:>8} {r['theta_cum_from_solved']:>9} "
                f"{r['theta_cum_total']:>7}"
            )


if __name__ == "__main__":
    main()
