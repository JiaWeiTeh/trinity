#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B11.G rung 0 — independently verify the CEM identities the K6 candidate leans on.

`LITERATURE_ASSESSMENT.md` §2.1/§4.2 make two claims that are central to the K6 row of
PLAN.md §7.1 and that its author verified with their own SymPy — i.e. nothing in THIS
workstream had re-derived them (C-0.5: external documents are never load-bearing):

  I1  The C3c momentum-phase switch point IS Lancaster's characteristic radius:
      solving P_C3a(R) = alpha_p * pdot / (4 pi R^2) gives R = R_eq^4 / R_St^3 = R_ch,
      and the direct eq:Rch_def form  alpha_B (alpha_p pdot)^2 / (12 pi (mu m_H c_i^2)^2 Q0)
      is the same number.
  I2  Trinity's two shipped branches are the CEM force's exact asymptotes, and the
      max/sum compositions err only near the crossover with opposite signs:
      F_CEM = alpha_p pdot (1 + R_w/R_ch)^{2/3} with  R_i = R_w (1 + R_w/R_ch)^{1/3};
      F_CEM -> alpha_p pdot         as R_w/R_ch -> 0    (the confined branch),
      F_CEM -> F_Sp(R_i)            as R_w/R_ch -> inf  (the unconfined branch),
      F_sum/F_CEM = 1.342 and F_max/F_CEM = 0.671 at R_i = R_ch.

No trinity run and no trinity import — the check is scale-free, so it works in units
where mu_H m_H = 1 (rho = n) and chi_e is folded into alpha_B; Batch 8's G8.2 already
established that trinity's (mu_convert/mu_ion_shell) prefactor makes P_C3a equal
rho c_i^2 (R_St/R)^{3/2} in exactly this sense. I1 is checked by a generic root-find
(brentq), NOT by the closed form, over random parameter draws spanning six decades —
a coincidence cannot track that.

    python docs/dev/phii-identity/harness/cem_closure_check.py \
        --out docs/dev/phii-identity/data/b11g_cem_closure_check.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402

FOUR_PI = 4.0 * math.pi


def draws(n, seed=20260818):
    rng = np.random.default_rng(seed)
    logu = lambda lo, hi, size: 10.0 ** rng.uniform(lo, hi, size)  # noqa: E731
    return zip(
        logu(2, 6, n),  # pdot
        logu(-2, 2, n),  # rho (= n, mu m_H = 1)
        logu(0, 2, n),  # c_i
        logu(4, 8, n),  # Q0
        logu(-6, -4, n),  # alpha_B
        rng.uniform(1.0, 8.0, n),  # alpha_p
    )


def scales(pdot, rho, ci, Q0, aB, ap):
    R_St = (3.0 * Q0 / (FOUR_PI * aB * rho**2)) ** (1.0 / 3.0)
    R_eq = math.sqrt(ap * pdot / (FOUR_PI * rho * ci**2))
    return R_St, R_eq, R_eq**4 / R_St**3


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--n", type=int, default=200, help="random parameter draws for I1")
    ap_.add_argument("--out", type=Path)
    args = ap_.parse_args()
    rows = []

    # ---- I1: generic root-find of P_C3a = P_w vs R_ch, plus the two R_ch forms ----
    # The root-find runs in log-R: R_ch spans ~12 decades over these draws, and brentq's
    # DEFAULT xtol is absolute (2e-12), which silently swallows any root below it — the
    # first version of this check "failed" at 2.4e4x for exactly that reason, with the
    # identity itself holding to 1.4e-16 at the same draw. Log-space makes the tolerance
    # relative. Still generic: f evaluates the two physical pressures, never the closed form.
    worst_root, worst_ratio, worst_form = 0.0, 0.0, 0.0
    for pdot, rho, ci, Q0, aB, ap in draws(args.n):
        R_St, R_eq, R_ch = scales(pdot, rho, ci, Q0, aB, ap)
        p_hii = lambda R: rho * ci**2 * (R_St / R) ** 1.5  # noqa: E731
        p_w = lambda R: ap * pdot / (FOUR_PI * R**2)  # noqa: E731
        g = lambda u: math.log(p_hii(math.exp(u)) / p_w(math.exp(u)))  # noqa: E731
        u = brentq(g, math.log(R_ch) - 20.0, math.log(R_ch) + 20.0, rtol=8.9e-16)
        worst_root = max(worst_root, abs(math.exp(u) / R_ch - 1.0))
        worst_ratio = max(worst_ratio, abs(p_hii(R_ch) / p_w(R_ch) - 1.0))
        direct = aB * (ap * pdot) ** 2 / (12.0 * math.pi * ci**4 * Q0)
        worst_form = max(worst_form, abs(direct / R_ch - 1.0))
    rows.append(("I1_crossover_over_Rch_worst_relerr", worst_root, 0.0))
    rows.append(("I1_PC3a_over_Pw_at_Rch_worst_relerr", worst_ratio, 0.0))
    rows.append(("I1_Rch_def_vs_Req4_RSt3_worst_relerr", worst_form, 0.0))
    print(
        f"I1  crossover/R_ch − 1 (log-space brentq): worst |rel| = {worst_root:.3e}"
        f" over {args.n} draws"
    )
    print(f"I1  P_C3a(R_ch)/P_w(R_ch) − 1:             worst |rel| = {worst_ratio:.3e}")
    print(f"I1  eq:Rch_def vs R_eq^4/R_St^3:           worst |rel| = {worst_form:.3e}")

    # ---- I2: asymptotes and the crossover table, on one representative draw ----
    pdot, rho, ci, Q0, aB, ap = next(iter(draws(1, seed=7)))
    R_St, R_eq, R_ch = scales(pdot, rho, ci, Q0, aB, ap)
    F_wind = ap * pdot
    F_Sp = lambda Ri: FOUR_PI * rho * ci**2 * R_St**1.5 * math.sqrt(Ri)  # noqa: E731
    Ri_of = lambda Rw: Rw * (1.0 + Rw / R_ch) ** (1.0 / 3.0)  # noqa: E731
    F_CEM = lambda Rw: F_wind * (1.0 + Rw / R_ch) ** (2.0 / 3.0)  # noqa: E731

    for x, ref, name in (
        (1e-3, F_wind, "confined"),
        (1e3, None, "unconfined"),
        (1e4, None, "unconfined"),
    ):
        Rw = x * R_ch
        ref_v = ref if ref is not None else F_Sp(Ri_of(Rw))
        r = F_CEM(Rw) / ref_v
        rows.append((f"I2_asymptote_{name}_x{x:g}_ratio", r, 1.0))
        print(f"I2  F_CEM/{'alpha_p*pdot' if ref else 'F_Sp':12s} at R_w/R_ch={x:g}: {r:.4f}")

    print(
        f"\nI2  {'Ri/Rch':>8}{'F_sum/F_CEM':>13}{'F_max/F_CEM':>13}   (assessment: +34%/−33% at 1)"
    )
    for target in (0.5, 1.0, 2.0, 10.0):
        Ri_t = target * R_ch
        Rw = brentq(lambda w: Ri_of(w) - Ri_t, 1e-9 * R_ch, 1e6 * R_ch, rtol=1e-14)
        fsum = (F_Sp(Ri_t) + F_wind) / F_CEM(Rw)
        fmax = max(F_Sp(Ri_t), F_wind) / F_CEM(Rw)
        rows.append((f"I2_Fsum_over_FCEM_at_{target:g}Rch", fsum, 1.342 if target == 1.0 else None))
        rows.append((f"I2_Fmax_over_FCEM_at_{target:g}Rch", fmax, 0.671 if target == 1.0 else None))
        print(f"    {target:>8g}{fsum:>13.4f}{fmax:>13.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(
                "# scale-free check (mu_H m_H = 1, chi_e folded into alpha_B); "
                "I1 root-found generically, not from the closed form\n"
            )
            w = csv.writer(fh)
            w.writerow(["quantity", "value", "expected"])
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
