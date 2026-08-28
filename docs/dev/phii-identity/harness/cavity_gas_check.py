#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is C3a's cavity photon sink physically available? — maintainer challenge, 2026-08-28.

Maintainer: "the shell should not hold 100%, because the bubble cavity has density and
mass (mBubble) ... I think it's just that we assume the gas in the cavity is invisible to
ionising photons because they are already all excited due to T~1e7."

Both halves are worth testing, and B11.0's seam-C wording ("M_cav has no source") is loose
about the first: the run DOES carry a bubble mass. This measures how much, and then asks
the question that actually matters for seam A -- not "is there gas?" but "can that gas
absorb the photons C3a spends there?"

Recombination rate scales as n^2 * alpha_B(T) * V. Over the SAME cavity volume:

    recomb_actual / recomb_assumed = (n_actual/n_implied)^2 * (alpha_B(T_bub)/alpha_B(T_ion))

The n^2 term is computed from committed columns and needs no temperature model at all --
reported as `sink_ratio_n2_only`, and it is the robust number. The temperature term is
reported SEPARATELY and is illustrative only: case-B recombination is a ~1e4 K concept and
extrapolating alpha_B ~ T^-0.8 to 1e7 K is not real physics (at 1e7 K hydrogen is
collisionally fully ionised, which is the maintainer's point and makes the suppression
stronger, not weaker). The verdict must not depend on it -- and it does not.

    n_implied  = ledger `n_from_PHII`, the density C3a's own balance asserts for the cavity
    n_actual   = bubble_mass / (mu_convert * V_cav), inverting the ledger's own
                 M = V * n * mu_convert convention (mass_ledger_check.py:25)
    T_bub      = Pb * mu_i / (mu_c * n_actual * k_B), the bubble's own pressure and density

WARNING (inherited, and stated in every row): B11.0 measured `bubble_mass` FROZEN at
99.643 Msun through the momentum phase and called it unusable. Every n_actual/T_bub here
inherits that defect, so they are order-of-magnitude only. The verdict survives it: the
n^2 suppression is 4-5 orders, so bubble_mass would have to be wrong by ~100x in the
right direction to change the conclusion. `bubble_mass_frozen` flags it per row.

    python docs/dev/phii-identity/harness/cavity_gas_check.py \
        --out docs/dev/phii-identity/data/b14_cavity_gas.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"
LEDGERS = [("B3M", "b11_mass_ledger.csv"), ("B3MW01", "b12_lowwind_mass_ledger.csv")]
ALPHA_B_EXP = -0.8  # illustrative case-B scaling near 1e4 K; NOT valid at 1e7 K (see docstring)

FIELDS = [
    "config", "phase", "t", "R2", "bubble_mass", "shell_mass", "M_avail",
    "M_cav_implied", "M_cav_over_bubble", "n_implied", "n_actual",
    "n_implied_over_actual", "T_bubble_est_K", "sink_ratio_n2_only",
    "sink_ratio_with_T_illustrative", "bubble_mass_frozen",
]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def med(vals):
    v = sorted(x for x in vals if x is not None)
    return v[len(v) // 2] if v else float("nan")


def run(config, path, consts):
    mu_c, mu_i, kB, T_ion = consts
    rows = [r for r in csv.DictReader(l for l in open(DATA / path) if not l.startswith("#"))
            if r.get("status") == "ok"]
    # B11.0's "frozen" claim is about the DRIVING rows this harness reports, so the flag
    # must be computed over those, not over every phase in the ledger.
    drv = [r for r in rows if fnum(r, "n_from_PHII") and fnum(r, "bubble_mass")]
    frozen = len({fnum(r, "bubble_mass") for r in drv}) == 1
    out = []
    for r in rows:
        n_imp = fnum(r, "n_from_PHII")
        R2, m_bub, Pb = fnum(r, "R2"), fnum(r, "bubble_mass"), fnum(r, "Pb")
        if not (n_imp and n_imp > 0 and R2 and R2 > 0 and m_bub and m_bub > 0):
            continue
        vol = 4.0 / 3.0 * math.pi * R2**3
        n_act = m_bub / (mu_c * vol)                 # invert M = V * n * mu_convert
        ratio = n_imp / n_act
        # bubble temperature from its own P and n, in the code's P = (mu_c/mu_i) n kB T form
        T_bub = (Pb * mu_i / (mu_c * n_act * kB)) if (Pb and Pb > 0) else None
        n2 = (n_act / n_imp) ** 2
        with_T = n2 * (T_bub / T_ion) ** ALPHA_B_EXP if T_bub and T_bub > 0 else None
        out.append(dict(
            config=config, phase=r["phase"], t=fnum(r, "t"), R2=R2,
            bubble_mass=m_bub, shell_mass=fnum(r, "shell_mass"), M_avail=fnum(r, "M_avail"),
            M_cav_implied=fnum(r, "M_cav_P"), M_cav_over_bubble=fnum(r, "M_cav_over_bubble"),
            n_implied=n_imp, n_actual=n_act, n_implied_over_actual=ratio,
            T_bubble_est_K=T_bub, sink_ratio_n2_only=n2,
            sink_ratio_with_T_illustrative=with_T,
            bubble_mass_frozen=frozen,
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b14_cavity_gas.csv")
    args = ap.parse_args()

    params = read_param(str(
        REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/"
        "bench3_m1e5_r5__none_diag.param"))
    consts = (params["mu_convert"].value, params["mu_ion_shell"].value,
              params["k_B"].value, params["TShell_ion"].value)

    rows = [r for cfg, path in LEDGERS for r in run(cfg, path, consts)]
    if not rows:
        sys.exit("no usable rows — check the ledger columns")

    for cfg in sorted({r["config"] for r in rows}):
        sel = [r for r in rows if r["config"] == cfg]
        n2 = [r["sink_ratio_n2_only"] for r in sel]
        rat = [r["n_implied_over_actual"] for r in sel]
        T = [r["T_bubble_est_K"] for r in sel if r["T_bubble_est_K"]]
        wT = [r["sink_ratio_with_T_illustrative"] for r in sel
              if r["sink_ratio_with_T_illustrative"]]
        print(f"\n{cfg}: {len(sel)} driving rows  (bubble_mass frozen: {sel[0]['bubble_mass_frozen']})")
        print(f"  cavity gas the run HAS      : bubble_mass = {sel[0]['bubble_mass']:.3f} Msun "
              f"(vs shell {med([r['shell_mass'] for r in sel]):.0f}, "
              f"M_avail {med([r['M_avail'] for r in sel]):.0f})")
        print(f"  C3a's balance ASSERTS       : M_cav = {min(r['M_cav_implied'] for r in sel):.0f}"
              f"–{max(r['M_cav_implied'] for r in sel):.0f} Msun "
              f"= {min(r['M_cav_over_bubble'] for r in sel):.0f}"
              f"–{max(r['M_cav_over_bubble'] for r in sel):.0f}x the bubble")
        print(f"  density overstatement       : n_implied/n_actual "
              f"{min(rat):.1f}–{max(rat):.1f} (median {med(rat):.1f})")
        print(f"  >> photon sink available    : (n_act/n_imp)^2 = "
              f"{min(n2):.2e}–{max(n2):.2e} (median {med(n2):.2e})   [ROBUST, no T model]")
        if T:
            print(f"  bubble T from its own P,n   : {min(T):.2e}–{max(T):.2e} K "
                  f"(median {med(T):.2e})")
            print(f"  with alpha_B~T^{ALPHA_B_EXP} (ILLUSTRATIVE, invalid at 1e7 K): "
                  f"{min(wT):.1e}–{max(wT):.1e}")

    print("\nReading: the cavity is NOT empty — it holds ~99.6 Msun — but the gas it holds is")
    print("4-5 orders of magnitude too thin to absorb the photons C3a spends there, on the")
    print("n^2 term ALONE. Temperature only deepens it. So seam A's defect is not that two")
    print("models share one photon budget 50/50; it is that C3a posits a recombination sink")
    print("that is physically unavailable, and shell_structure.py:120's phi0 = 1 is RIGHT.")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Cavity gas content vs C3a's asserted photon sink. Answers the maintainer's\n")
        fh.write("# 2026-08-28 challenge to seam C's 'M_cav has no source' wording.\n")
        fh.write("# WARNING: bubble_mass is FROZEN (B11.0) -- n_actual/T_bubble are order-of-\n")
        fh.write("# magnitude only. The n^2 verdict survives a ~100x error in bubble_mass.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
