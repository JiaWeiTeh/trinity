#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B11.0 seam C — is the C3a cavity mass real, and where would it come from?

The 2026-08-18 self-consistency audit (PLAN.md §6b, seam C) claimed that on B3M's
momentum rows the C3a premise implies a cavity gas mass reaching `M_cav/M_shell`
= 0.564. That number was derived once, in one session, by INVERTING the shipped
`P_HII` back to a density. B11.0 exists to try to kill it, so this harness
computes the same quantity by TWO independent routes on every snapshot and also
asks the question the audit did not: is there any mass source in the model that
could supply `M_cav`?

  route P   n = P_HII / ((mu_convert/mu_ion_shell) * k_B * TShell_ion)
            — the audit's inversion. Zero on the confined branch by construction,
              so it only speaks on driving rows.
  route Q   n = sqrt(3 * Qi * f_abs / (4 pi chi_e alpha_B R2**3))
            — the forward map straight out of `get_phii_c3c`, with `f_abs` taken
              from a REPLAY of the shipped `shell_structure_pure` on the
              snapshot's own state (the snapshot does not persist
              `shell_fAbsorbedIon`). Independent of `P_HII` entirely.

Routes P and Q must agree to ~machine precision on driving rows; a disagreement
falsifies the audit's inversion. Both feed

  M_cav = (4/3) pi R2**3 * n * mu_convert          [Msun]

which is compared against three things:

  shell_mass    the swept material the dynamics actually carries (the double-book)
  bubble_mass   the cavity's own gas. CAVEAT: the momentum phase never recomputes it,
                so on those rows it is FROZEN at its transition-exit value and is only
                indicative. The honest wind-mass budget is the integral of
                `2*Lmech_total/v_mech_total**2` over the run's own snapshots — see the
                PLAN's B11.0 seam C entry for that number.
  M_avail       `params['mCloud']` (already post-star-formation) plus the ambient swept
                out to R2 — the total gas the simulation has to play with.
                `M_cav + shell_mass > M_avail` means the model asserts more gas than
                exists, which is stronger than a double-book.

    python docs/dev/phii-identity/harness/mass_ledger_check.py <run_dir> [...] \
        --stride 2 --out docs/dev/phii-identity/data/b11_mass_ledger.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402
from trinity.shell_structure.shell_structure import shell_structure_pure  # noqa: E402

from _stamp import stamp  # noqa: E402

# Same state `shell_structure_pure` reads as layer_density_check.py (shell_structure.py:104-113).
REPLAY_KEYS = ("bubble_mass", "Pb", "R2", "shell_mass", "Qi", "Li", "Ln", "rShell")

FIELDS = [
    "run",
    "row_idx",
    "phase",
    "t",
    "status",
    "R2",
    "Qi",
    "f_abs",
    "P_HII",
    "Pb",
    "n_from_PHII",
    "n_from_Qi",
    "route_ratio",
    "M_cav_P",
    "M_cav_Q",
    "shell_mass",
    "bubble_mass",
    "M_cav_over_shell",
    "M_cav_over_bubble",
    "M_avail",
    "M_cav_plus_shell_over_avail",
]


def available_gas(params, rCloud, R2):
    """Gas the run has: the cloud's gas mass plus the ambient swept beyond rCloud.

    Both terms in Msun. `params['mCloud']` is ALREADY the post-star-formation gas
    mass — read_param splits the input mCloud into mCloud + mCluster — so do not
    apply (1-sfe) again here. The ambient term uses the run's own nISM and the
    `rho = mu_convert * n_H` convention the registry documents.
    """
    m_cloud_gas = params["mCloud"].value
    if R2 <= rCloud:
        return m_cloud_gas
    rho_ism = params["nISM"].value * params["mu_convert"].value  # Msun/pc^3
    return m_cloud_gas + 4.0 / 3.0 * math.pi * (R2**3 - rCloud**3) * rho_ism


def replay(run_dir, stride):
    run_dir = Path(run_dir)
    pfile = next(run_dir.glob("*.param"), None)
    if pfile is None:
        sys.exit(f"no .param in {run_dir} — need the run's own materialised config")
    params = read_param(str(pfile))
    # rCloud is derived at runtime (phase0_init), so read_param leaves it 0 — take the
    # run's own materialised value from metadata rather than re-deriving the profile.
    rCloud = float(json.loads((run_dir / "metadata.json").read_text())["rCloud"])

    chi_e = params["chi_e_shell"].value
    alpha_B = params["caseB_alpha"].value
    mu_c = params["mu_convert"].value
    mu_i = params["mu_ion_shell"].value
    kB = params["k_B"].value
    T_ion = params["TShell_ion"].value
    # get_bubbleParams.get_phii_c3c: P = (mu_c/mu_i) * n * k_B * T, so this is the
    # exact factor that turns a pressure back into a hydrogen-nuclei density.
    p_per_n = (mu_c / mu_i) * kB * T_ion

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
        )
        for key in REPLAY_KEYS:
            params[key].value = row[key]
        try:
            sp = shell_structure_pure(params)
        except Exception as exc:  # a replayed state the solver rejects is reported, not hidden
            rec["status"] = f"replay_failed:{type(exc).__name__}"
            out.append(rec)
            continue

        R2 = float(row["R2"])
        Qi = float(row["Qi"])
        f_abs = float(getattr(sp, "shell_fAbsorbedIon", 1.0))
        P_HII = float(row.get("P_HII") or 0.0)
        Pb = float(row.get("Pb") or 0.0)
        m_shell = float(row["shell_mass"])
        m_bub = float(row["bubble_mass"])

        n_P = P_HII / p_per_n if P_HII > 0 else 0.0
        denom = 4.0 * math.pi * chi_e * alpha_B * R2**3
        n_Q = math.sqrt(3.0 * Qi * f_abs / denom) if (denom > 0 and Qi * f_abs > 0) else 0.0

        vol = 4.0 / 3.0 * math.pi * R2**3
        M_P, M_Q = vol * n_P * mu_c, vol * n_Q * mu_c
        M_avail = available_gas(params, rCloud, R2)

        rec.update(
            R2=R2,
            Qi=Qi,
            f_abs=f_abs,
            P_HII=P_HII,
            Pb=Pb,
            n_from_PHII=n_P,
            n_from_Qi=n_Q,
            route_ratio=(n_P / n_Q) if n_Q > 0 and n_P > 0 else None,
            M_cav_P=M_P,
            M_cav_Q=M_Q,
            shell_mass=m_shell,
            bubble_mass=m_bub,
            M_cav_over_shell=(M_P / m_shell) if m_shell > 0 else None,
            M_cav_over_bubble=(M_P / m_bub) if m_bub > 0 else None,
            M_avail=M_avail,
            M_cav_plus_shell_over_avail=((M_P + m_shell) / M_avail) if M_avail > 0 else None,
        )
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stride", type=int, default=2, help="replay every Nth snapshot")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [r for run in args.runs for r in replay(run, args.stride)]
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        sys.exit(f"no rows replayed successfully out of {len(rows)}")

    drive = [r for r in ok if r["P_HII"] > 0]
    print(f"{len(ok)} rows replayed, {len(drive)} on the driving branch (P_HII > 0)\n")

    # Route agreement — the check that decides whether the audit's inversion is sound.
    rr = [r["route_ratio"] for r in drive if r["route_ratio"]]
    if rr:
        print(
            f"route P / route Q on driving rows: {min(rr):.12f} .. {max(rr):.12f}"
            f"   (FALSIFIES the inversion if it strays from 1)"
        )
    else:
        print("no driving rows — route comparison VOID, not a confirming null")

    print(
        f"\n{'phase':11s}{'n':>4}{'t_first':>10}{'t_last':>9}"
        f"{'Mcav/Mshell':>13}{'Mcav/Mbub':>12}{'(Mcav+Msh)/Mavail':>19}"
    )
    for phase in ("energy", "implicit", "transition", "momentum"):
        v = [r for r in drive if r["phase"] == phase]
        if not v:
            continue
        last = v[-1]
        print(
            f"{phase:11s}{len(v):>4d}{v[0]['t']:>10.4f}{last['t']:>9.4f}"
            f"{last['M_cav_over_shell']:>13.4f}{last['M_cav_over_bubble']:>12.1f}"
            f"{last['M_cav_plus_shell_over_avail']:>19.4f}"
        )

    for run in sorted({r["run"] for r in drive}):
        v = [r for r in drive if r["run"] == run]
        lo, hi = v[0], v[-1]
        print(
            f"\n{run}: M_cav/M_shell {lo['M_cav_over_shell']:.4f} (t={lo['t']:.4f})"
            f" -> {hi['M_cav_over_shell']:.4f} (t={hi['t']:.4f})"
            f"   [{hi['M_cav_P']:.0f} vs {hi['shell_mass']:.0f} Msun]"
        )
        print(
            f"{'':>{len(run)}}  wind mass available in the cavity at t={hi['t']:.4f}:"
            f" {hi['bubble_mass']:.1f} Msun = {1.0 / hi['M_cav_over_bubble']:.2e} of M_cav"
        )

    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print(f"\n{len(bad)} row(s) not replayed: {sorted({r['status'] for r in bad})}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            for run in args.runs:
                fh.write(f"# run {run} (stride {args.stride})\n")
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
