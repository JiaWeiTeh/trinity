#!/usr/bin/env python3
"""Fallback matcher for the Rosette Cf scan — radii-only chi^2 vs the Rosette observables.

⚠️ POLICY REIMPLEMENTATION, NOT THE FROZEN SOURCE. The frozen matching policy lives in
paper/rosette/matching/observables.py + match_runs.py, which are gitignored/local-only
(paper/CLAUDE.md) and NOT available in this container. The constants below transcribe that policy
as stated in the 2026-07-13 task brief: R2 <-> 7 +/- 1 pc (cavity), rShell <-> 19 +/- 2 pc,
radii-only chi^2 by default, flat age prior 1.5-2.5 Myr. If paper/rosette/matching/ is present
(maintainer's machine, or copied into the Phase-2 container), PREFER the frozen matcher and use
this one only as a cross-check; before quoting any number produced here, diff POLICY against the
frozen observables.py.

F-12 (7 pc vs 6.2 pc cavity-target conflict) is NOT silently resolved: every output carries both
bases (columns *_7 and *_62).

Reads the committed per-arm trajectory CSVs (from harvest_cf_scan.py) + the summary CSV (axes,
exit codes). Quotable = exit_code==0 (📏 a wall-killed 124 arm is never quoted); matchable
additionally needs t_final >= 1.5 Myr. Runs truncate at different t: per-run fixed-age columns are
empty beyond t_final, and each cell's overshoot collation is done at t_match = min(2.5 Myr, min
t_final over the cell's quotable runs) so Cf rows are compared at MATCHED simulation time
(the paper/rosette PLAN.md §0.3 sealed-baseline adjudication).

chi^2(Cf) resolution: 3 grid points {0.70, 0.85, 1.0} bracket the minimum by design; cf_star is
the vertex of the parabola through them — an interpolated estimate from 3 points, flagged when the
fit is non-convex or the vertex falls outside the grid. It is NOT a fitted minimum on a fine grid.

    python docs/dev/rosette-cf/harness/match_cf_scan.py \
        --summary docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv \
        --traj-dir docs/dev/rosette-cf/data/cf_scan_PISM1e5_traj \
        --out docs/dev/rosette-cf/data/match_cf_PISM1e5.csv \
        --cells-out docs/dev/rosette-cf/data/match_cf_PISM1e5_cells.csv
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _stamp import stamp  # noqa: E402

# ---- FROZEN POLICY (transcribed from the task brief — diff vs paper/rosette/matching/observables.py)
R2_TARGET, R2_ERR = 7.0, 1.0  # pc, cavity radius (default base)
R2_TARGET_ALT = 6.2  # pc, F-12 alternate cavity base — always reported alongside
RSHELL_TARGET, RSHELL_ERR = 19.0, 2.0  # pc
AGE_MIN, AGE_MAX = 1.5, 2.5  # Myr, flat age prior
FIXED_AGES = (1.5, 2.0, 2.5)  # Myr, matched-t collation grid

CELL_KEYS = ["mCloud", "sfe", "nCore", "cooling_boost_fmix", "include_PHII"]


def chi2(r2, rshell, base):
    return ((r2 - base) / R2_ERR) ** 2 + ((rshell - RSHELL_TARGET) / RSHELL_ERR) ** 2


def interp_at(rows, t):
    """Linear (R2, rShell) at time t from sorted (t, R2, rShell) rows; None outside range."""
    if not rows or t < rows[0][0] or t > rows[-1][0]:
        return None
    for i in range(1, len(rows)):
        t1, r1, s1 = rows[i - 1]
        t2, r2, s2 = rows[i]
        if t1 <= t <= t2:
            w = 0.0 if t2 == t1 else (t - t1) / (t2 - t1)
            return r1 + w * (r2 - r1), s1 + w * (s2 - s1)
    return None


def match_run(rows):
    """Best-age match of one trajectory under the flat age prior, on BOTH cavity bases.

    rows: sorted (t_now, R2, rShell). Returns a dict; empty-valued when the run never reaches
    AGE_MIN (age-censored out of the prior entirely).
    """
    out = {}
    t_final = rows[-1][0] if rows else 0.0
    out["t_final"] = t_final
    out["age_censored"] = t_final < AGE_MAX
    for age in FIXED_AGES:
        tag = str(age).replace(".", "p")
        v = interp_at(rows, age)
        out[f"R2_at_{tag}"] = v[0] if v else ""
        out[f"rShell_at_{tag}"] = v[1] if v else ""
    hi = min(AGE_MAX, t_final)
    if hi < AGE_MIN:
        for base_tag in ("7", "62"):
            out[f"t_best_{base_tag}"] = out[f"chi2_min_{base_tag}"] = ""
            out[f"R2_best_{base_tag}"] = out[f"rShell_best_{base_tag}"] = ""
        return out
    candidates = [(t, r, s) for t, r, s in rows if AGE_MIN <= t <= hi]
    for t in (AGE_MIN, hi):
        v = interp_at(rows, t)
        if v:
            candidates.append((t, v[0], v[1]))
    for base, base_tag in ((R2_TARGET, "7"), (R2_TARGET_ALT, "62")):
        best = min(candidates, key=lambda c: chi2(c[1], c[2], base))
        out[f"t_best_{base_tag}"] = best[0]
        out[f"chi2_min_{base_tag}"] = chi2(best[1], best[2], base)
        out[f"R2_best_{base_tag}"] = best[1]
        out[f"rShell_best_{base_tag}"] = best[2]
    return out


def parabola_vertex(pts):
    """Vertex of the parabola through 3 (cf, chi2) points -> (cf_star, chi2_star, flag).

    flag: 'ok' | 'non-convex' (min quoted at the best grid point instead) | 'outside-grid'
    (vertex extrapolates beyond the 3-point bracket — quote with the 3-point caveat).
    """
    (x1, y1), (x2, y2), (x3, y3) = sorted(pts)
    d = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
    b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / d
    c = y1 - a * x1**2 - b * x1
    if a <= 0:
        xb, yb = min(pts, key=lambda p: p[1])
        return xb, yb, "non-convex"
    xs = -b / (2 * a)
    ys = a * xs**2 + b * xs + c
    if not x1 <= xs <= x3:
        return xs, ys, "outside-grid"
    return xs, ys, "ok"


def read_traj(path):
    with open(path) as fh:
        rows = []
        for r in csv.DictReader(fh):
            try:
                rows.append((float(r["t_now"]), float(r["R2"]), float(r["rShell"])))
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda r: r[0])
    return rows


RUN_COLS = (
    ["run_name"]
    + CELL_KEYS
    + ["coverFraction", "quotable", "matchable", "t_final", "age_censored"]
    + [f"{q}_{b}" for b in ("7", "62") for q in ("t_best", "chi2_min", "R2_best", "rShell_best")]
    + [f"{q}_at_{str(a).replace('.', 'p')}" for q in ("R2", "rShell") for a in FIXED_AGES]
)
CELL_COLS = CELL_KEYS + [
    "n_quotable",
    "t_match",
    "cf_grid",
    "chi2_grid_7",
    "chi2_grid_62",
    "best_cf_7",
    "best_cf_62",
    "cf_star_7",
    "chi2_star_7",
    "fit_flag_7",
    "cf_star_62",
    "chi2_star_62",
    "fit_flag_62",
    "R2_at_tmatch",
    "over7_at_tmatch",
    "over62_at_tmatch",
    "note",
]


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else v


def _join(vals):
    return ";".join("" if v in ("", None) else f"{v:.4g}" for v in vals)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells-out")
    args = ap.parse_args(argv)

    with open(args.summary) as fh:
        summary = list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))

    run_rows, cells = [], {}
    for s in summary:
        name = s["run_name"]
        rec = {k: s.get(k, "") for k in ["run_name", "coverFraction"] + CELL_KEYS}
        rec["quotable"] = quotable = s.get("exit_code") == "0"
        traj_path = Path(args.traj_dir) / f"{name}.csv"
        rows = read_traj(traj_path) if traj_path.exists() else []
        m = match_run(rows) if rows else {"t_final": "", "age_censored": ""}
        rec.update(m)
        rec["matchable"] = bool(quotable and m.get("chi2_min_7") != "" and rows)
        run_rows.append(rec)
        if rec["matchable"]:
            cells.setdefault(tuple(s.get(k, "") for k in CELL_KEYS), []).append((rec, rows))

    header = stamp(str(HERE / "match_cf_scan.py"))
    policy = (
        "# POLICY (reimplemented from the task brief — diff vs the frozen "
        f"paper/rosette/matching/observables.py before quoting): R2<->{R2_TARGET}+/-{R2_ERR} pc "
        f"(alt base {R2_TARGET_ALT} pc, F-12), rShell<->{RSHELL_TARGET}+/-{RSHELL_ERR} pc, "
        f"radii-only chi^2, flat age prior {AGE_MIN}-{AGE_MAX} Myr. Non-quotable (exit!=0) runs "
        "are excluded from every minimum. ⚠️ PROVISIONAL / IN-CONTAINER — NOT HPC.\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        fh.write(header + "\n" + policy)
        w = csv.DictWriter(fh, fieldnames=RUN_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(run_rows, key=lambda r: r["run_name"]):
            w.writerow({k: _fmt(v) for k, v in r.items()})
    print(f"wrote {len(run_rows)} run rows -> {out}")

    if not args.cells_out:
        return
    cell_rows = []
    for key, members in sorted(cells.items()):
        members.sort(key=lambda m: float(m[0]["coverFraction"]))
        cfs = [float(m[0]["coverFraction"]) for m in members]
        rec = dict(zip(CELL_KEYS, key))
        rec.update(n_quotable=len(members), cf_grid=_join(cfs), note="")
        t_match = min(min(2.5, m[0]["t_final"]) for m in members)
        rec["t_match"] = _fmt(t_match) if t_match >= AGE_MIN else ""
        for base_tag in ("7", "62"):
            pts = [(cf, m[0][f"chi2_min_{base_tag}"]) for cf, m in zip(cfs, members)]
            rec[f"chi2_grid_{base_tag}"] = _join(p[1] for p in pts)
            rec[f"best_cf_{base_tag}"] = _fmt(min(pts, key=lambda p: p[1])[0])
            if len(pts) == 3:
                xs, ys, flag = parabola_vertex(pts)
                rec[f"cf_star_{base_tag}"] = _fmt(xs)
                rec[f"chi2_star_{base_tag}"] = _fmt(ys)
                rec[f"fit_flag_{base_tag}"] = flag
            else:
                rec[f"cf_star_{base_tag}"] = rec[f"chi2_star_{base_tag}"] = ""
                rec[f"fit_flag_{base_tag}"] = f"only-{len(pts)}-points"
                rec["note"] = "incomplete Cf grid (non-quotable arm?) — re-run before quoting"
        if rec["t_match"]:
            r2s = [(interp_at(rows, t_match) or ("",))[0] for _, rows in members]
            rec["R2_at_tmatch"] = _join(r2s)
            rec["over7_at_tmatch"] = _join((r - R2_TARGET) if r != "" else "" for r in r2s)
            rec["over62_at_tmatch"] = _join((r - R2_TARGET_ALT) if r != "" else "" for r in r2s)
        else:
            rec["R2_at_tmatch"] = rec["over7_at_tmatch"] = rec["over62_at_tmatch"] = ""
        cell_rows.append(rec)
    cells_out = Path(args.cells_out)
    with cells_out.open("w", newline="") as fh:
        fh.write(header + "\n" + policy)
        fh.write(
            "# One row per (mass-pair, nCore, fmix, P_HII) cell; cf_grid/chi2_grid_*/"
            "R2_at_tmatch/over*_at_tmatch are ';'-joined per ascending Cf. Overshoots are at "
            "MATCHED t (t_match = min over the cell's quotable runs of min(2.5, t_final)).\n"
        )
        w = csv.DictWriter(fh, fieldnames=CELL_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(cell_rows)
    print(f"wrote {len(cell_rows)} cell rows -> {cells_out}")


if __name__ == "__main__":
    main(sys.argv[1:])
