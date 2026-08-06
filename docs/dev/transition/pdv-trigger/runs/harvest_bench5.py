#!/usr/bin/env python3
"""bench5 harvest — fire-map summary + a compact, restart-durable θ(t) trajectory per arm.

Two outputs, both committed so the whole Phase-5 analysis (Θ_cum band, 1−θ dex metric, Lcool/Lleak
channel split, El-Badry overlay) runs OFFLINE from git — never re-running the sims (the theta5s
lesson: its raw arms were lost to a /tmp wipe and dMdt had to be salvaged in a scramble).

1. --csv <summary>: the fire map, via harvest_theta_max.harvest (θ_max, fired?, fate, t_final) —
   the same sanctioned θ = bubble_Lloss/Lmech_total on accepted rows.
2. --traj-dir <dir>: per arm, <arm>.csv with the accepted-implicit trajectory
   (t_now, theta, Lcool=bubble_LTotal, Lleak=bubble_Leak, Lmech=Lmech_total, R2). ALL accepted rows
   are kept (trapezoid Θ_cum needs them) up to a 4000-row cap; beyond that, log-t downsample keeping
   endpoints. θ numerator uses bubble_Lloss (the effective/boosted loss the trigger sees) =
   Lcool + Lleak; committing the split lets the Rogers & Pittard channel check run offline.

    python harvest_bench5.py <arm_dirs...> --csv runs/data/bench5_summary.csv \
        --traj-dir runs/data/bench5_traj
    # extra dictionary.jsonl columns (state-coupled f_A screen, FA_STATE_COUPLED.md SC-0):
    python harvest_bench5.py <arm_dirs...> --traj-dir <dir> \
        --extra-cols Pb,bubble_L2Conduction,bubble_L3Intermediate,bubble_dMdt

BIG CAMPAIGNS (500+ arms — bench8/f_area, F_AREA_PLAN.md §9a). Two flags keep what comes DOWN from
the cluster small and few, so the whole campaign is a couple of reviewable files in git rather than
one file per arm:
  --derived        append the DERIVED_COLS distilled scalars (Θ_cum, the solved/stale split,
                   θ_max_solved, leak fraction) to the summary — computed ON the cluster, so the
                   headline analysis reads ONE ~100 KB CSV and never needs a trajectory.
  --traj-bundle F  write every arm's trajectory into ONE long CSV keyed by a leading run_name
                   column, instead of (or alongside) N per-arm files. Same bytes, same precision,
                   one file; read it back with data/read_bundle.py.

    python harvest_bench5.py <arm_dirs...> --csv <summary> --derived \
        --traj-bundle <bundle.csv> --extra-cols Pb,bubble_dMdt
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from harvest_theta_max import COLUMNS, harvest  # noqa: E402
from _stamp import stamp  # noqa: E402

TRAJ_COLS = ["t_now", "theta", "Lcool", "Lleak", "Lmech", "R2"]
TRAJ_CAP = 4000

# Distilled per-arm scalars, appended to the summary CSV by --derived. These are what the headline
# analysis actually reads, so a 500+-arm campaign is answerable from ONE ~100 KB file instead of
# from N trajectory files (bench8 sizing: F_AREA_PLAN.md §9a).
DERIVED_COLS = ["n_rows", "n_stale", "stale_time_frac", "theta_cum", "theta_cum_raw",
                "theta_cum_solved", "theta_cum_stale", "t_window_end", "leak_frac",
                "theta_max_solved", "theta_max_is_stale"]
_T, _TH, _LCOOL, _LLEAK, _LMECH = 0, 1, 2, 3, 4


def derived(rows):
    """Per-arm distilled scalars from ``trajectory`` rows — computed ON the cluster during reduce.

    Deliberately duplicates two laptop-side functions rather than importing them, because both live
    in ``data/`` modules that import matplotlib at module level and the reduce step must stay
    dependency-light on the cluster. ``test/test_bench_derived.py`` pins this against the canonical
    pair, so the duplication cannot drift:

      • ``theta_cum``/``theta_cum_raw``/``t_window_end``/``leak_frac``  — the Θ_cum metric,
        data/make_bench5_analysis.py::theta_cum_prefire (effective-loss numerator, FINDINGS §18).
      • the ``*_stale``/``*_solved`` split — data/make_bench_stale_segments.py::decompose. Solved
        rows only is the STANDING convention for the trigger metric (kappa-3way FINDINGS §12): θ
        keeps climbing on β–δ rows the solver never solved (L_cool frozen, L_mech still evolving),
        and on 76/291 bench arms that stale drift is what set θ_max.

    The two families filter differently — theta_cum_prefire drops rows with a null t/θ/L_mech,
    decompose drops SEGMENTS touching a null — so ``theta_cum`` is not exactly
    ``theta_cum_solved + theta_cum_stale``. That asymmetry is inherited from the published record;
    both are reproduced faithfully rather than silently reconciled.
    """
    n = len(rows)
    if n < 2:
        return dict.fromkeys(DERIVED_COLS)
    # --- theta_cum_prefire: trapezoid over rows with a usable t, theta and L_mech
    pts = [(r[_T], r[_TH], (r[_LCOOL] or 0.0) + (r[_LLEAK] or 0.0), r[_LMECH]) for r in rows
           if r[_T] is not None and r[_TH] is not None and r[_LMECH]]
    num = raw_num = den = 0.0
    for (t0, h0, w0, m0), (t1, h1, w1, m1) in zip(pts, pts[1:]):
        dt = t1 - t0
        num += 0.5 * (h0 * m0 + h1 * m1) * dt
        raw_num += 0.5 * (w0 + w1) * dt
        den += 0.5 * (m0 + m1) * dt
    tot = sum((r[_LLEAK] or 0.0) + (r[_LCOOL] or 0.0) for r in rows)
    out = {
        "n_rows": n,
        "leak_frac": (sum(r[_LLEAK] or 0.0 for r in rows) / tot) if tot else 0.0,
        "t_window_end": pts[-1][0] if pts else None,
        "theta_cum": (num / den) if den else None,
        "theta_cum_raw": (raw_num / den) if den else None,
    }
    # --- decompose: a row is STALE when L_cool is unchanged from its predecessor (no solver root)
    stale = {i for i in range(1, n) if rows[i][_LCOOL] == rows[i - 1][_LCOOL]}
    num_s = num_f = den2 = stale_t = 0.0
    for i in range(1, n):
        a, b = rows[i - 1], rows[i]
        if None in (a[_T], b[_T], a[_TH], b[_TH], a[_LMECH], b[_LMECH]):
            continue
        dt = b[_T] - a[_T]
        seg = 0.5 * (a[_TH] * a[_LMECH] + b[_TH] * b[_LMECH]) * dt
        den2 += 0.5 * (a[_LMECH] + b[_LMECH]) * dt
        if i in stale:
            num_s += seg
            stale_t += dt
        else:
            num_f += seg
    span = (rows[-1][_T] - rows[0][_T]) if None not in (rows[0][_T], rows[-1][_T]) else None
    th = [r[_TH] for r in rows]
    finite = [v for v in th if v is not None]
    out.update({
        "n_stale": len(stale),
        "stale_time_frac": (stale_t / span) if span else None,
        "theta_cum_stale": (num_s / den2) if den2 else None,
        "theta_cum_solved": (num_f / den2) if den2 else None,
        "theta_max_is_stale": (th.index(max(finite)) in stale) if finite else None,
        "theta_max_solved": max((v for i, v in enumerate(th) if v is not None and i not in stale),
                                default=None),
    })
    return out


def _finite(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def trajectory(run_dir, extra=()):
    """Accepted implicit rows as [t_now, theta, Lcool, Lleak, Lmech, R2] + any ``extra`` keys.

    ``extra`` names dictionary.jsonl keys appended verbatim (finite-filtered) after R2 — used by
    the state-coupled-f_A screen, which needs Pb and the L2/L3 split that the six default columns
    do not carry (FA_STATE_COUPLED.md SC-0). Empty ``extra`` reproduces the original 6 columns.
    """
    out = []
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("current_phase") != "implicit":
                continue
            t = _finite(d.get("t_now"))
            Lloss = _finite(d.get("bubble_Lloss"))
            if Lloss is None:
                Lloss = _finite(d.get("bubble_LTotal"))
            Lmech = _finite(d.get("Lmech_total"))
            if t is None or Lloss is None or not Lmech:
                continue
            Lcool = _finite(d.get("bubble_LTotal"))
            Lleak = _finite(d.get("bubble_Leak")) or 0.0
            row = [t, Lloss / Lmech, Lcool, Lleak, Lmech, _finite(d.get("R2"))]
            row += [_finite(d.get(k)) for k in extra]
            out.append(row)
    out.sort(key=lambda r: r[0])
    if len(out) > TRAJ_CAP:                     # log-t downsample, keep endpoints
        import numpy as np
        ts = np.array([r[0] for r in out])
        lo = max(ts[0], 1e-9)
        grid = np.unique(np.geomspace(lo, ts[-1], TRAJ_CAP))
        idx = sorted({0, len(out) - 1} | {int(np.searchsorted(ts, g)) for g in grid})
        out = [out[min(i, len(out) - 1)] for i in idx]
    return out


def write_traj(run_dir, traj_dir, extra=(), rows=None):
    rows = trajectory(run_dir, extra) if rows is None else rows
    if not rows:
        return 0
    traj_dir.mkdir(parents=True, exist_ok=True)
    with (traj_dir / f"{run_dir.name}.csv").open("w", newline="") as fh:
        # Stamped like every other artifact (maintainer ALL-FRESH ruling 2026-07-29): the trajectory
        # CSVs ARE the data the Theta_cum metric reads, so "when was this measured" has to be legible
        # on the file itself, not inferred from the summary next to it. Readers already skip leading
        # '#' lines (make_bench5_analysis._read_csv), and the campaign hash is taken over the
        # non-comment lines (sync_bench.sh), so the stamp cannot make two identical runs differ.
        fh.write(stamp(__file__) + "\n")
        w = csv.writer(fh)
        w.writerow(TRAJ_COLS + list(extra))
        w.writerows(rows)
    return len(rows)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    csv_out = traj_dir = bundle = None
    if "--csv" in argv:
        csv_out = Path(argv[argv.index("--csv") + 1])
        args = [a for a in args if str(csv_out) != a]
    if "--traj-dir" in argv:
        traj_dir = Path(argv[argv.index("--traj-dir") + 1])
        args = [a for a in args if str(traj_dir) != a]
    if "--traj-bundle" in argv:
        bundle = Path(argv[argv.index("--traj-bundle") + 1])
        args = [a for a in args if str(bundle) != a]
    want_derived = "--derived" in argv
    extra = ()
    if "--extra-cols" in argv:
        spec = argv[argv.index("--extra-cols") + 1]
        extra = tuple(c.strip() for c in spec.split(",") if c.strip())
        args = [a for a in args if spec != a]

    fields = COLUMNS + DERIVED_COLS if want_derived else COLUMNS
    bundle_fh = bundle_w = None
    if bundle is not None:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle_fh = bundle.open("w", newline="")
        bundle_fh.write(stamp(str(HERE / "harvest_bench5.py")) + "\n")
        bundle_w = csv.writer(bundle_fh)
        bundle_w.writerow(["run_name"] + TRAJ_COLS + list(extra))

    rows = []
    for a in args:
        run_dir = Path(a)
        if not (run_dir / "dictionary.jsonl").exists():
            continue
        summary_row = harvest(run_dir)
        traj = trajectory(run_dir, extra) if (traj_dir or bundle or want_derived) else None
        if traj_dir is not None:
            write_traj(run_dir, traj_dir, extra, rows=traj)
        if bundle_w is not None:
            bundle_w.writerows([run_dir.name] + r for r in traj)
        if want_derived:
            summary_row.update(derived(traj or []))
        rows.append(summary_row)
    if bundle_fh is not None:
        bundle_fh.close()
        print(f"wrote bundled trajectories -> {bundle}")
    rows.sort(key=lambda r: r["run_name"])
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        stamp_line = stamp(str(HERE / "harvest_bench5.py"))
        with csv_out.open("w", newline="") as fh:
            fh.write(stamp_line + "\n")
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} summary rows -> {csv_out}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
