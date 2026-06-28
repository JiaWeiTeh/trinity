#!/usr/bin/env python3
"""Monte-Carlo backing for the double-count-free claim of the max() closure.

The methods note states the max() closure removes only the missing part of the loss and so never
enters the "double-count" region where the input-rescale (theta*Lmech) AND the resolved integral
(Lcool) are both subtracted. That is true ANALYTICALLY by construction --

    Lloss_eff = max(Lcool + Lleak, theta_target * Lmech)
              <= (Lcool + Lleak) + theta_target * Lmech      (the double-count sink)
    with equality only in the degenerate case, and the counted fraction Lloss_eff/Lmech equals
    max(Lcool/Lmech + Lleak/Lmech, theta_target), i.e. it sits ON the single-count line, never 2*theta.

-- but the note quotes an empirical 5e5-draw check, so this script makes that number real and
committed (one-line CSV) rather than asserted. Seed fixed for reproducibility.

Run from the repo root:
  python docs/dev/transition/pdv-trigger/data/make_doublecount_mc.py
"""
import numpy as np
import pandas as pd

N = 500_000
DST = "docs/dev/transition/pdv-trigger/data/doublecount_mc.csv"


def main():
    rng = np.random.default_rng(20260624)
    Lmech = rng.uniform(0.1, 10.0, N)
    Lcool = rng.uniform(0.0, 1.0, N) * Lmech     # resolved cooling fraction in [0,1]
    Lleak = rng.uniform(0.0, 0.2, N) * Lmech     # small optional leak
    theta = rng.uniform(0.0, 1.0, N)             # target loss fraction

    eff = np.maximum(Lcool + Lleak, theta * Lmech)        # the max() closure
    double = (Lcool + Lleak) + theta * Lmech              # the forbidden double-count sink
    counted_frac = eff / Lmech                            # what Fig 1 plots on the y axis
    single_frac = np.maximum((Lcool + Lleak) / Lmech, theta)

    enters_double = eff > double + 1e-12                  # closure ever exceeds single-count?
    on_single = np.abs(counted_frac - single_frac) < 1e-12

    out = pd.DataFrame([dict(
        n_draws=N, seed=20260624,
        max_counted_over_single=round(float((counted_frac / single_frac).max()), 6),
        n_enter_double_count=int(enters_double.sum()),
        frac_on_single_line=round(float(on_single.mean()), 6),
    )])
    out.to_csv(DST, index=False)
    print(f"wrote {DST}\n")
    print(out.to_string(index=False))
    assert enters_double.sum() == 0, "max() closure entered the double-count region!"
    print("\nOK: over 5e5 draws the max() closure never enters the double-count region "
          "(single-count by construction).")


if __name__ == "__main__":
    main()
