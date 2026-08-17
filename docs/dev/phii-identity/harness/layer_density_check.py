#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G9.4 — the ionised-layer density computed through the shell machinery, not scaled.

Batch 9 estimated the layer/cavity density ratio analytically as `sqrt(R2 / (3 dR))`,
reading `dR` from the SNAPSHOT's `shell_r_arr` indexed by `shell_ion_idx`. That has a
defect G9.4 exists to catch:

  * `shell_ion_idx` indexes the ORIGINAL `shell_r/n_arr` (`registry.py:499`), but the
    snapshot's `shell_r_arr` is the DOWNSAMPLED grid paired with `log_shell_n_arr`
    (`dictionary.py:688-700`). Different index spaces. On the B3M run `shell_ion_idx`
    reaches 26848 while the saved array is <= 100 long, so the index clamped to the last
    element on 100% of rows and `dR` came out as the FULL shell thickness
    (ionised + neutral), not the ionised layer.
  * `dR_full >= dR_ion`, and `ratio = sqrt(R2/(3 dR))` falls with `dR`, so Batch 9
    UNDERSTATED the ratio. Its momentum-phase falsification of G9.2 (ratio < 1) is
    therefore suspect in the direction that matters.

This harness replays the shipped `shell_structure_pure` on each snapshot's own state --
the same machinery, with the ORIGINAL un-downsampled arrays, so `shell_ion_idx` and
`shell_r_arr` share an index space -- and reports:

  dR_ion      true ionised-layer thickness, rShell_arr_ion[-1] - rShell_arr_ion[0]
  n_rms       recombination-weighted equivalent density of the REAL profile,
              sqrt(int n^2 dV / int dV) over the ionised layer. This is the density that,
              held uniform over that volume, recombines at the same total rate -- the
              honest counterpart to a one-number Stroemgren density.
  n_layer     analytic layer estimate, sqrt(Qi_abs / (4 pi R2^2 dR_ion chi_e alpha_B))
  n_cavity    what C3a actually uses, sqrt(3 Qi_abs / (4 pi R2^3 chi_e alpha_B))

    python docs/dev/phii-identity/harness/layer_density_check.py <run_dir> \
        --out docs/dev/phii-identity/data/b9_layer_density.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402
from trinity.shell_structure.shell_structure import shell_structure_pure  # noqa: E402

from _stamp import stamp  # noqa: E402

# The state shell_structure_pure reads (shell_structure.py:104-113). All are in the snapshot.
REPLAY_KEYS = ("bubble_mass", "Pb", "R2", "shell_mass", "Qi", "Li", "Ln", "rShell")


def stromgren(Qi_abs, volume, chi_e, alpha_B):
    """Uniform density that consumes Qi_abs by recombination in `volume`."""
    if not (Qi_abs > 0 and volume > 0):
        return None
    return math.sqrt(Qi_abs / (volume * chi_e * alpha_B))


def n_rms_over_layer(r, n):
    """sqrt( int n^2 dV / int dV ) with dV = 4 pi r^2 dr — the recombination-equivalent density."""
    if len(r) < 2:
        return None
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    w = 4.0 * np.pi * r**2
    num = np.trapezoid(n**2 * w, r) if hasattr(np, "trapezoid") else np.trapz(n**2 * w, r)
    den = np.trapezoid(w, r) if hasattr(np, "trapezoid") else np.trapz(w, r)
    if not (den > 0 and num >= 0):
        return None
    return math.sqrt(num / den)


def replay(run_dir, stride):
    """Replay the shell solve on every `stride`-th snapshot of a run."""
    run_dir = Path(run_dir)
    pfile = next(run_dir.glob("*.param"), None)
    if pfile is None:
        sys.exit(f"no .param in {run_dir} — need the run's own materialised config")
    params = read_param(str(pfile))
    chi_e = params["chi_e_shell"].value
    alpha_B = params["caseB_alpha"].value

    out = []
    lines = [ln for ln in (run_dir / "dictionary.jsonl").open() if ln.strip()]
    for k, ln in enumerate(lines):
        if k % stride:
            continue
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        if any(row.get(key) is None for key in REPLAY_KEYS):
            continue
        for key in REPLAY_KEYS:
            params[key].value = row[key]
        try:
            sp = shell_structure_pure(params)
        except Exception as exc:  # a replayed state the solver rejects is reported, not hidden
            out.append(
                dict(
                    row_idx=k,
                    phase=row.get("current_phase"),
                    t=row.get("t_now"),
                    status=f"replay_failed:{type(exc).__name__}",
                )
            )
            continue

        r_all = np.asarray(getattr(sp, "shell_r_arr", []), dtype=float)
        n_all = np.asarray(getattr(sp, "shell_n_arr", []), dtype=float)
        i = int(getattr(sp, "shell_ion_idx", -1))
        R2 = float(row["R2"])
        if r_all.size < 2 or i < 1:
            out.append(
                dict(
                    row_idx=k,
                    phase=row.get("current_phase"),
                    t=row.get("t_now"),
                    status="no_ionised_layer",
                )
            )
            continue

        # Index spaces match here: both come from THIS call, un-downsampled.
        i = min(i, r_all.size - 1)
        r_ion, n_ion = r_all[: i + 1], n_all[: i + 1]
        dR_ion = float(r_ion[-1] - r_ion[0])
        dR_full = float(r_all[-1] - r_all[0])
        f_abs = float(getattr(sp, "shell_fAbsorbedIon", 1.0))
        Qi_abs = float(row["Qi"]) * f_abs

        V_cav = 4.0 / 3.0 * math.pi * R2**3
        V_lay = 4.0 * math.pi * R2**2 * dR_ion if dR_ion > 0 else None
        n_cav = stromgren(Qi_abs, V_cav, chi_e, alpha_B)
        n_lay = stromgren(Qi_abs, V_lay, chi_e, alpha_B) if V_lay else None
        n_rms = n_rms_over_layer(r_ion, n_ion)

        # Mechanism check for a G9.4 gap: the analytic estimate assumes ALL of Qi_abs is
        # consumed by RECOMBINATION in the layer. The real shell also eats ionising photons
        # on dust, so the true recombination integral should fall SHORT of Qi_abs, and the
        # shortfall is exactly why a Stroemgren-balance density overestimates the profile.
        _tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        recomb = float(_tz(chi_e * alpha_B * n_ion**2 * 4.0 * np.pi * r_ion**2, r_ion))

        out.append(
            dict(
                row_idx=k,
                phase=row.get("current_phase"),
                t=row.get("t_now"),
                status="ok",
                R2=R2,
                dR_ion=dR_ion,
                dR_full=dR_full,
                ion_frac_of_shell=(dR_ion / dR_full) if dR_full > 0 else None,
                dR_ion_over_R2=dR_ion / R2,
                f_abs=f_abs,
                n_cavity=n_cav,
                n_layer_analytic=n_lay,
                n_rms_profile=n_rms,
                recomb_over_Qiabs=(recomb / Qi_abs) if Qi_abs > 0 else None,
                f_ionised_dust=getattr(sp, "shell_fIonisedDust", None),
                # G9.4: analytic layer estimate vs the real profile's equivalent density
                rms_over_analytic=(n_rms / n_lay) if (n_rms and n_lay) else None,
                # the Batch 9 quantity, now with the TRUE ionised thickness
                ratio_analytic=(n_lay / n_cav) if (n_lay and n_cav) else None,
                ratio_from_profile=(n_rms / n_cav) if (n_rms and n_cav) else None,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stride", type=int, default=4, help="replay every Nth snapshot")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [r for run in args.runs for r in replay(run, args.stride)]
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        sys.exit(f"no rows replayed successfully out of {len(rows)}")

    print(
        f"{'phase':11s}{'n':>4}{'dRion/R2':>10}{'ion/shell':>11}"
        f"{'ratio_an':>10}{'ratio_prof':>12}{'rms/analytic':>14}{'recomb/Qi':>12}"
    )
    for phase in ("energy", "implicit", "transition", "momentum"):
        v = [r for r in ok if r["phase"] == phase]
        if not v:
            continue

        def med(key):
            x = sorted(r[key] for r in v if r[key] is not None)
            return x[len(x) // 2] if x else float("nan")

        print(
            f"{phase:11s}{len(v):>4d}{med('dR_ion_over_R2'):>10.4f}{med('ion_frac_of_shell'):>11.4f}"
            f"{med('ratio_analytic'):>10.3f}{med('ratio_from_profile'):>12.3f}"
            f"{med('rms_over_analytic'):>14.3f}{med('recomb_over_Qiabs'):>12.3f}"
        )

    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        print(f"\n{len(bad)} row(s) not replayed: {sorted({r['status'] for r in bad})}")

    agree = [r["rms_over_analytic"] for r in ok if r["rms_over_analytic"]]
    if agree:
        worst = max(max(agree), 1.0 / min(agree))
        print(f"\nG9.4 profile vs analytic layer density: {min(agree):.3f}..{max(agree):.3f}")
        print(f"     worst disagreement factor = {worst:.3f}   (FALSIFIED IF > 2)")
    mom = [r["ratio_analytic"] for r in ok if r["phase"] == "momentum" and r["ratio_analytic"]]
    if mom:
        print(
            f"\nG9.2 recheck, momentum, TRUE ionised dR: ratio {min(mom):.3f}..{max(mom):.3f}, "
            f"frac>1 = {sum(1 for x in mom if x > 1) / len(mom):.4f}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            for run in args.runs:
                fh.write(f"# run {run} (stride {args.stride})\n")
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
