#!/usr/bin/env python3
"""Emit the f_mix screening table (the methods-note Table 2) from committed data.

NO simulations. Pure read of pdv_combined_trigger.csv, which already logged, per config, the
resolved cooling ratio at blowout (`cool_at_blowout = (Lmech-Lcool)/Lmech`) and the same ratio with
PdV (`coolPdV_at_blowout = (Lmech-Lcool-PdV)/Lmech`). From those two numbers the multiplier f_mix
that brings the trigger ratio to the threshold EPS=0.05 AT BLOWOUT is pure algebra -- this script
makes that algebra reproducible instead of hand-computed.

Two trigger conventions, because they answer different questions:
  with PdV : solve (Lmech - f*Lcool - PdV)/Lmech = EPS  ->  f = (1-EPS - PdV/Lmech)/(Lcool/Lmech)
  no  PdV : solve (Lmech - f*Lcool      )/Lmech = EPS  ->  f = (1-EPS)          /(Lcool/Lmech)
where  Lcool/Lmech = 1 - cool_at_blowout  and  PdV/Lmech = cool_at_blowout - coolPdV_at_blowout.

The NO-PdV column is the one consistent with the note's recommended trigger (PdV in the ODE only,
NOT in the transition criterion -- the reversible/irreversible argument, note section "Implication
for the transition trigger"). The with-PdV column is kept only to show how folding PdV into the
*screening* criterion understates the boost by ~PdV/Lmech. Headline f_mix should quote NO-PdV.

FROZEN-TRAJECTORY SCREEN: these multipliers are reconstructed on the unboosted trajectory; a real
boost lowers Pb -> PdV -> moves blowout itself, so they BOUND the knob, they do not forecast it.

Run from the repo root:
  python docs/dev/transition/pdv-trigger/data/make_fmix_table.py
"""
import pandas as pd

EPS = 0.05
SRC = "docs/dev/transition/pdv-trigger/data/pdv_combined_trigger.csv"
DST = "docs/dev/transition/pdv-trigger/data/fmix_table.csv"
# density-ordered (densest first), normal clouds only -- heavy clouds never reach blowout
ORDER = ["small_dense_highsfe", "simple_cluster", "midrange_pl0",
         "be_sphere", "pl2_steep", "large_diffuse_lowsfe"]


def main():
    d = pd.read_csv(SRC).set_index("config")
    rows = []
    for c in ORDER:
        if c not in d.index or pd.isna(d.loc[c, "cool_at_blowout"]):
            continue
        cool = d.loc[c, "cool_at_blowout"]
        coolpdv = d.loc[c, "coolPdV_at_blowout"]
        lcool = 1.0 - cool            # Lcool/Lmech at blowout
        pdv = cool - coolpdv          # PdV/Lmech at blowout
        rows.append(dict(
            config=c,
            Lcool_over_Lmech_at_blowout=round(lcool, 3),
            PdV_over_Lmech_at_blowout=round(pdv, 3),
            fmix_with_pdv=round((1 - EPS - pdv) / lcool, 2),   # screening criterion (PdV in trigger)
            fmix_no_pdv=round((1 - EPS) / lcool, 2),           # note's RECOMMENDED trigger (PdV out)
        ))
    out = pd.DataFrame(rows)
    out.to_csv(DST, index=False)
    print(f"wrote {DST}\n")
    print(out.to_string(index=False))
    fn = out["fmix_no_pdv"]
    print(f"\nheadline (no-PdV, consistent with recommended trigger): "
          f"f_mix = {fn.min():.2f}-{fn.max():.2f}")
    return out


if __name__ == "__main__":
    main()
