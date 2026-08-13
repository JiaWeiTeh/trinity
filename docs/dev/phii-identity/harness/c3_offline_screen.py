#!/usr/bin/env python3
"""Offline screen of the decoupled `P_HII` candidates — PLAN.md §3b / Batch 5 stage 1.

D2 asks for a `P_HII` that is a real, separate pressure rather than a relabelling
of the confining pressure. §3b showed the coupling is NOT the cap: it runs through
the ionised volume, because `shell_n0 = Pb/(kT)·μ` is the shell ODE's inner
boundary condition and therefore sets where the ionisation front lands.

Both candidate replacements are closed-form in quantities already stored in the
committed snapshots, so they can be screened WITHOUT running the solver:

  C3a  cavity Strömgren   n = sqrt(3·Q_abs / (4π χ_e α_B R2³))
                          depends on Qi and R2 only
  C3b  ambient/pre-shock  n = n_cloud(R2) from the density profile
                          depends on the cloud profile and R2 only

Both use the code's own density→pressure conversion, including the
`mu_convert/mu_ion_shell` factor that `P_HII` applies (omitting it understates the
pressure by 2.2× — an error this workstream has already made once).

The screen reports, per (run, phase):
  * `slope_vs_Pb` / `r_vs_Pb` — the decoupling test. Stock scores ~+1.0 (fully
    slaved); a decoupled candidate should score near 0, and the sign/þmagnitude
    matters more than r.
  * `ratio_Pram_*` and `crosses_Pram` — does the candidate ever trade places with
    the wind ram pressure? A candidate that is always above (or always below)
    `P_ram` cannot reproduce the coevolution behaviour D2 is asking for.
  * `n_cm3_*` — physical sanity. H II regions around young clusters run ~10–10³ cm⁻³.

Usage (from the repo root):
    python docs/dev/phii-identity/harness/c3_offline_screen.py \
        --out docs/dev/phii-identity/data/b5_c3_screen.csv \
        outputs/phii/b1__<sha>/<config> [...]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from trinity._input.read_param import read_param  # noqa: E402
from trinity.cloud_properties.density_profile import get_density_profile  # noqa: E402

PC_CM = 3.0856775814913673e18
PHASES = ["energy", "implicit", "transition", "momentum"]
CANDIDATES = ["stock", "uncapped", "C3a_cavity", "C3b_ambient"]

COLS = [
    "run",
    "phase",
    "candidate",
    "n_rows",
    "slope_vs_Pb",
    "r_vs_Pb",
    "ratio_Pb_min",
    "ratio_Pb_max",
    "ratio_Pram_min",
    "ratio_Pram_max",
    "crosses_Pram",
    "n_cm3_min",
    "n_cm3_max",
]


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def n_to_P(n, params):
    """The code's own density -> pressure conversion (run_*_phase.py)."""
    return (
        (params["mu_convert"].value / params["mu_ion_shell"].value)
        * n
        * params["k_B"].value
        * params["TShell_ion"].value
    )


def candidates_for_row(d, params):
    """Each candidate's ionised density for one snapshot, or None if unavailable."""
    R2, Qi = d.get("R2"), d.get("Qi")
    out = {"stock": d.get("n_IF_Str"), "uncapped": d.get("n_IF_Str_raw")}

    # C3a — Strömgren balance over the CAVITY, not the shell skin. Uses the
    # absorbed rate the code's own balance uses, not total Qi.
    f_abs = d.get("shell_fAbsorbedIon")
    if R2 and Qi and R2 > 0 and Qi > 0:
        q_abs = Qi * (f_abs if isinstance(f_abs, (int, float)) and 0 <= f_abs <= 1 else 1.0)
        denom = 4.0 * math.pi * params["chi_e_shell"].value * params["caseB_alpha"].value * R2**3
        out["C3a_cavity"] = math.sqrt(3.0 * q_abs / denom) if denom > 0 and q_abs > 0 else None
    else:
        out["C3a_cavity"] = None

    # C3b — the unperturbed cloud density at R2, via trinity's own profile.
    if R2 and R2 > 0:
        try:
            n_amb = float(get_density_profile(R2, params))
            out["C3b_ambient"] = n_amb if n_amb > 0 else None
        except Exception:
            out["C3b_ambient"] = None
    else:
        out["C3b_ambient"] = None
    return out


def analyse(run_dir):
    param_files = sorted(run_dir.glob("*.param"))
    if not param_files:
        return [], f"{run_dir.name}: no .param"
    params = read_param(str(param_files[0]))
    # read_param does NOT populate quantities derived during cloud init — rCloud
    # comes back 0, which makes get_density_profile() treat every radius as
    # outside the cloud and return nISM. metadata.json records what the run
    # actually used, so overlay it; without this C3b silently reports the ISM
    # density everywhere (caught in smoke-testing, 2026-08-13).
    meta_path = run_dir / "metadata.json"
    overlaid = []
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError:
            meta = {}
        for k, v in meta.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k in params:
                if params[k].value != v:
                    params[k].value = v
                    overlaid.append(k)
    if "rCloud" not in overlaid and float(params["rCloud"].value or 0) <= 0:
        return [], f"{run_dir.name}: rCloud unavailable — C3b would be meaningless"

    rows = []
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass

    out = []
    for phase in PHASES:
        sel = [d for d in rows if d.get("current_phase") == phase]
        if not sel:
            continue
        for cand in CANDIDATES:
            xs, ys, rPb, rRam, ns = [], [], [], [], []
            for d in sel:
                Pb, Pram = d.get("Pb"), d.get("P_ram")
                n = candidates_for_row(d, params).get(cand)
                if not n or n <= 0 or not Pb or Pb <= 0:
                    continue
                P = n_to_P(n, params)
                if P <= 0:
                    continue
                xs.append(math.log10(Pb))
                ys.append(math.log10(P))
                rPb.append(P / Pb)
                if Pram and Pram > 0:
                    rRam.append(P / Pram)
                ns.append(n / PC_CM**3)
            if len(xs) < 3:
                continue
            s, r = fit(xs, ys)
            crosses = "yes" if (rRam and min(rRam) < 1.0 < max(rRam)) else ("no" if rRam else "NA")
            out.append(
                {
                    "run": run_dir.name,
                    "phase": phase,
                    "candidate": cand,
                    "n_rows": len(xs),
                    "slope_vs_Pb": "NA" if s is None else f"{s:.4f}",
                    "r_vs_Pb": "NA" if r is None else f"{r:.4f}",
                    "ratio_Pb_min": f"{min(rPb):.4g}",
                    "ratio_Pb_max": f"{max(rPb):.4g}",
                    "ratio_Pram_min": f"{min(rRam):.4g}" if rRam else "NA",
                    "ratio_Pram_max": f"{max(rRam):.4g}" if rRam else "NA",
                    "crosses_Pram": crosses,
                    "n_cm3_min": f"{min(ns):.4g}",
                    "n_cm3_max": f"{max(ns):.4g}",
                }
            )
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    allrows, notes = [], []
    for run_dir in args.runs:
        if not (run_dir / "dictionary.jsonl").exists():
            notes.append(f"{run_dir}: no dictionary.jsonl")
            continue
        rows, err = analyse(run_dir)
        if err:
            notes.append(err)
        allrows.extend(rows)
    if not allrows:
        print("nothing to report:", "; ".join(notes))
        return 1

    w = max(len(r["run"]) for r in allrows)
    print(
        f"{'run':{w}} {'phase':>11} {'candidate':>13} {'slope vs Pb':>12} {'r':>8} "
        f"{'P/Pb range':>20} {'P/P_ram range':>20} {'cross':>6} {'n [cm^-3]':>20}"
    )
    for r in allrows:
        print(
            f"{r['run']:{w}} {r['phase']:>11} {r['candidate']:>13} {r['slope_vs_Pb']:>12} "
            f"{r['r_vs_Pb']:>8} {r['ratio_Pb_min']+'..'+r['ratio_Pb_max']:>20} "
            f"{r['ratio_Pram_min']+'..'+r['ratio_Pram_max']:>20} {r['crosses_Pram']:>6} "
            f"{r['n_cm3_min']+'..'+r['n_cm3_max']:>20}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Offline screen of decoupled P_HII candidates. NO SOLVER RUN: every candidate\n")
        fh.write(
            "# is closed-form in quantities already stored in the snapshots, evaluated ON the\n"
        )
        fh.write("# stock trajectory. It therefore answers 'what would this pressure have been',\n")
        fh.write(
            "# NOT 'what would the run have done' — a candidate that passes still needs an arm.\n"
        )
        fh.write("# slope_vs_Pb: stock scores ~+1 (fully slaved); decoupled should be near 0.\n")
        if notes:
            fh.write("# notes: " + "; ".join(notes) + "\n")
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(allrows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
