#!/usr/bin/env python3
"""Q5a — is any rosette-cf fm4 fire dependent on the §16 trigger-fallback double-boost?

FINDINGS §19 bounded the §16 fm² double-boost out of the pdv-trigger bench fire maps because
0/120 bench arms ever fired the live trigger. That bound does NOT transfer to the sibling
campaigns that run `cooling_boost_mode=multiplier` with the DEFAULT `cooling_balance` trigger:
the rosette-cf PISM1e5 survey (36 fm4 arms; docs/dev/rosette-cf/) and the Paper II grid
(param/paperII_grid_sweep.param). This script closes the gap for rosette-cf by direct
measurement over the campaign's own dictionaries.

THE TEST. On a no-root segment the trigger reads the ALREADY-boosted `bubble_Lloss` and boosts
it again (run_energy_implicit_phase.py:1244-1247), so the trigger's theta is fmix x the true
effective theta. A fire is bug-DEPENDENT iff it happened on a stale row where the true effective
theta was still < 0.95 (the fixed code would not have fired there). Because the phase BREAKS at
the fire, the last implicit row of a fired arm IS the fire row — so scanning that row's
single-boost theta is exhaustive: any early-fire the bug caused would leave theta_eff < 0.95
imprinted on the last row.

DATA SOURCE. The 72 gzipped dictionaries were dropped from the working tree at 591e5e4
("superseded by HPC") but are intact in git history; this reads them from commit 5aa84723
("CAMPAIGN COMPLETE — 72/72 arms") via `git show`, read-only. Effective theta at Cf<1 includes
the leak term single-counted: theta_eff = (fmix*LTotal + Leak)/Lmech.

    python docs/dev/transition/pdv-trigger/data/make_rosette_fm4_doubleboost_check.py
Deliverable: data/rosette_fm4_doubleboost_check.csv
"""
import csv
import gzip
import io
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SUMMARY = REPO / "docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv"
COMMIT = "5aa84723"
DICTS = "docs/dev/rosette-cf/data/cf_scan_PISM1e5_dicts"
FMIX = 4.0
FIRE = 0.95


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def scan_arm(name):
    """(n_impl, stale_at_last, theta_eff_last, theta_doubleboost_last) from the historical dict."""
    blob = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{COMMIT}:{DICTS}/{name}.jsonl.gz"],
        capture_output=True, check=True,
    ).stdout
    impl = []
    with gzip.open(io.BytesIO(blob), "rt") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("current_phase") == "implicit":
                impl.append((d.get("bubble_LTotal"), d.get("bubble_Leak") or 0.0,
                             d.get("Lmech_total"), d.get("bubble_Lloss")))
    if len(impl) < 2:
        return len(impl), None, None, None
    lt, lk, lm, ll = impl[-1]
    stale = lt == impl[-2][0]
    th_eff = (FMIX * (lt or 0.0) + lk) / lm if lm else None       # the fixed-code trigger theta
    th_dbl = FMIX * (ll or 0.0) / lm if (lm and ll is not None) else None  # the buggy fallback theta
    return len(impl), stale, th_eff, th_dbl


def main():
    with SUMMARY.open() as fh:
        rows = list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))
    fm4 = [r for r in rows if r["cooling_boost_fmix"] == "4.0"]
    out_rows, n_dep, n_stale = [], 0, 0
    for r in fm4:
        name = r["run_name"]
        try:
            n_impl, stale, th_eff, th_dbl = scan_arm(name)
        except subprocess.CalledProcessError:
            out_rows.append({"run_name": name, "phase_final": r["phase_final"], "n_impl": "",
                             "stale_at_fire_row": "", "theta_eff_fire_row": "",
                             "theta_doubleboost_fire_row": "", "verdict": "DICT_NOT_IN_COMMIT"})
            continue
        fired_past_energy = r["phase_final"] in ("momentum", "transition")
        if stale:
            n_stale += 1
        dep = bool(fired_past_energy and stale and th_eff is not None and th_eff < FIRE)
        if dep:
            n_dep += 1
        verdict = ("DOUBLE_BOOST_DEPENDENT" if dep
                   else "clean_fire" if fired_past_energy
                   else "no_fire")
        out_rows.append({
            "run_name": name, "phase_final": r["phase_final"], "n_impl": n_impl,
            "stale_at_fire_row": stale,
            "theta_eff_fire_row": f"{th_eff:.4f}" if th_eff is not None else "",
            "theta_doubleboost_fire_row": f"{th_dbl:.4f}" if th_dbl is not None else "",
            "verdict": verdict,
        })

    out = HERE / "rosette_fm4_doubleboost_check.csv"
    with out.open("w", newline="") as fh:
        fh.write(
            "# Q5a (FINDINGS 21): does the run_energy_implicit_phase.py:1244-1247 fmix^2 "
            "double-boost move any rosette-cf fm4 fire? Scanned ALL 36 fm4 arms of the PISM1e5 "
            "survey from the historical dictionaries (git 5aa84723; dropped from the tree at "
            "591e5e4). A fired arm's LAST implicit row is its fire row (the loop breaks on fire); "
            "the bug can only fire EARLY, which would leave theta_eff = (4*LTotal+Leak)/Lmech "
            "< 0.95 imprinted there. verdict=DOUBLE_BOOST_DEPENDENT iff fired AND stale AND "
            "theta_eff < 0.95. Regenerate: python docs/dev/transition/pdv-trigger/data/"
            "make_rosette_fm4_doubleboost_check.py\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    fired = sum(1 for r in out_rows if r["verdict"] != "no_fire")
    print(f"wrote {len(out_rows)} rows -> {out}")
    print(f"fired arms: {fired}/36; fire-row stale (fallback active): {n_stale}; "
          f"DOUBLE_BOOST_DEPENDENT: {n_dep}")


if __name__ == "__main__":
    main()
