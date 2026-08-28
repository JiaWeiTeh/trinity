#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 15 G15.0 — what fraction of the SHELL MASS is the ionised layer?

K9 (PLAN.md §7.1) proposes debiting the ionised gas mass from the shell's inertia,
following Lancaster `eq:pr_spitzer_adj`. Before that can be posed, one number is
needed and it was missing: the ionised **mass** fraction of the shell.

Why it is not the thickness fraction. B11.0 measured `dR_ion/dR_full` = 0.9954 in
the momentum phase, which invites the reading "the shell is essentially all ionised".
The shell's density profile rises steeply outward, so most of the MASS sits in the
thin neutral outer part and the two fractions are nothing alike. Reasoning from
thickness to mass is wrong, and this harness exists so nobody has to.

Method: replay the shipped `shell_structure_pure` on each snapshot's own state (the
same pattern as `layer_density_check.py` / `mass_ledger_check.py`, so `shell_ion_idx`
and `shell_r_arr` share an index space), then integrate the returned profile

    m(r) = int 4 pi r^2 n(r) mu_convert dr

over the ionised part (up to `shell_ion_idx`) and over the whole profile. Reports
`m_ion/m_profile` and, as a trust check, `m_profile/shell_mass` — where that second
ratio is not ~1 the profile does not account for the run's own shell mass and the
fraction must NOT be used (it is ~2x off in energy/implicit, ~1.000-1.002 in
transition/momentum).

    python docs/dev/phii-identity/harness/ionised_mass_fraction.py <run_dir> [...] \
        --stride 6 --out docs/dev/phii-identity/data/b15_ionised_mass_fraction.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402
from trinity.shell_structure.shell_structure import shell_structure_pure  # noqa: E402

from _stamp import stamp  # noqa: E402

REPLAY_KEYS = ("bubble_mass", "Pb", "R2", "shell_mass", "Qi", "Li", "Ln", "rShell")
FIELDS = [
    "run",
    "row_idx",
    "phase",
    "t",
    "status",
    "R2",
    "shell_mass",
    "m_profile",
    "m_ion",
    "m_ion_over_profile",
    "m_profile_over_shell_mass",
    "dR_ion_over_dR_full",
    "profile_trustworthy",
]
_TZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def replay(run_dir, stride):
    run_dir = Path(run_dir)
    pfile = next(run_dir.glob("*.param"), None)
    if pfile is None:
        sys.exit(f"no .param in {run_dir} — need the run's own materialised config")
    params = read_param(str(pfile))
    mu = params["mu_convert"].value

    out = []
    for k, ln in enumerate(l for l in (run_dir / "dictionary.jsonl").open() if l.strip()):
        if k % stride:
            continue
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        if any(row.get(key) is None for key in REPLAY_KEYS):
            continue
        rec = dict(
            run=run_dir.name,
            row_idx=k,
            phase=row.get("current_phase"),
            t=row.get("t_now"),
            status="ok",
            R2=float(row["R2"]),
            shell_mass=float(row["shell_mass"]),
        )
        for key in REPLAY_KEYS:
            params[key].value = row[key]
        try:
            sp = shell_structure_pure(params)
        except Exception as exc:  # a replayed state the solver rejects is reported, not hidden
            rec["status"] = f"replay_failed:{type(exc).__name__}"
            out.append(rec)
            continue

        r = np.asarray(getattr(sp, "shell_r_arr", []), dtype=float)
        n = np.asarray(getattr(sp, "shell_n_arr", []), dtype=float)
        i = int(getattr(sp, "shell_ion_idx", -1))
        if r.size < 3 or i < 1:
            rec["status"] = "no_ionised_layer"
            out.append(rec)
            continue
        i = min(i, r.size - 1)
        w = 4.0 * np.pi * r**2 * mu
        m_prof = float(_TZ(n * w, r))
        m_ion = float(_TZ((n * w)[: i + 1], r[: i + 1]))
        if m_prof <= 0:
            rec["status"] = "zero_profile_mass"
            out.append(rec)
            continue
        ratio_sm = m_prof / rec["shell_mass"] if rec["shell_mass"] > 0 else None
        rec.update(
            m_profile=m_prof,
            m_ion=m_ion,
            m_ion_over_profile=m_ion / m_prof,
            m_profile_over_shell_mass=ratio_sm,
            dR_ion_over_dR_full=(r[i] - r[0]) / (r[-1] - r[0]) if r[-1] > r[0] else None,
            # the fraction is only usable where the profile accounts for the run's shell mass
            profile_trustworthy=bool(ratio_sm is not None and 0.98 <= ratio_sm <= 1.02),
        )
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [r for run in args.runs for r in replay(run, args.stride)]
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        sys.exit(f"no rows replayed successfully out of {len(rows)} — G15.0 is VOID")

    print("G15.0 — ionised MASS fraction of the shell (NOT the thickness fraction)\n")
    print(
        f"{'run':10s}{'phase':11s}{'n':>4}{'m_ion/m_prof':>26}{'m_prof/shell_mass':>21}"
        f"{'dR_ion/dR_full':>16}{'usable':>8}"
    )
    for run in sorted({r["run"] for r in ok}):
        for ph in ("energy", "implicit", "transition", "momentum"):
            v = [r for r in ok if r["run"] == run and r["phase"] == ph]
            if not v:
                continue
            f = sorted(r["m_ion_over_profile"] for r in v)
            g = [r["m_profile_over_shell_mass"] for r in v if r["m_profile_over_shell_mass"]]
            d = [r["dR_ion_over_dR_full"] for r in v if r["dR_ion_over_dR_full"] is not None]
            nt = sum(1 for r in v if r["profile_trustworthy"])
            print(
                f"{run:10s}{ph:11s}{len(v):>4d}"
                f"   {f[0]:.4f}..{f[-1]:.4f} (med {f[len(f)//2]:.4f})"
                f"{min(g):>10.3f}..{max(g):.3f}"
                f"{(sum(d)/len(d) if d else float('nan')):>16.4f}{nt}/{len(v):>3}"
            )

    danger = [r for r in ok if r["m_ion_over_profile"] >= 0.95]
    print(
        f"\nG15.2 admissibility: {len(danger)}/{len(ok)} rows have m_ion/m_prof >= 0.95 —"
        f" debiting there leaves a near-massless shell and vd = F/M diverges."
    )
    if danger:
        ph = sorted({r["phase"] for r in danger})
        print(
            f"   affected phases: {ph}  -> K9 is NOT admissible in those without a floor"
            f" or a different shell definition."
        )

    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print(f"\n{len(bad)} row(s) not replayed: {sorted({r['status'] for r in bad})}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(
                "# G15.0 pre-gate for K9. m_ion/m_profile is only usable where "
                "profile_trustworthy (m_profile/shell_mass in [0.98, 1.02]).\n"
            )
            for run in args.runs:
                fh.write(f"# run {run} (stride {args.stride})\n")
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
