# docs/dev/betadelta/velstruct — velocity-structure ("Problem 2") hunt & treatment tooling

Harness/tooling for **Phase 6** of the β–δ hybr study: does the transient inner
inflow (negative interior velocity during a feedback re-pressurisation) ever
contaminate anything? **Not source.** Canonical writeup:
`docs/dev/betadelta/stalling-energy-phase.md` ("Phase 6.0 contamination hunt" +
"Is the inflow physical?"); plan: `docs/dev/betadelta/HYBR_PLAN.md` Phase 6.

## Files

- `hunt.py` — runs a config and dumps one row per accepted energy-implicit
  segment (β, δ, β+δ, Pb, dMdt, Lmech, v_struct_min/nneg/npts, **v_neg_frac_thick**,
  …) → `docs/dev/data/hunt_*.csv`. `--hold-inflow` = the reject-and-hold
  counterfactual (treat interior-v<0 as a structure failure → hold last physical).
- `analyze_hunt.py` — the **Gate-G6** classifier (cosmetic vs contaminating) over
  the hunt CSVs.
- `compare_hold.py` — diff a baseline vs a `--hold-inflow` run (the Phase-6.1
  treatment-effect comparison).
- `h{1..6}_*.param` — the six hunt configs (all `betadelta_solver=hybr`).

## Hunt configs (h1–h6)

`cluster mass = sfe × mCloud`; all mCloud=1e6, profile densPL.

| config | sfe | cluster | α_ρ | nCore | stop_t | probe |
|---|---|---|---|---|---|---|
| **h1 base**  | 0.01 | 1e4 | −2 | 1e5 | 4 | reproduce the steep baseline |
| **h2 sfe10** | 0.10 | 1e5 | −2 | 1e5 | 6 | 10× stronger SN |
| **h3 sfe30** | 0.30 | 3e5 | −2 | 1e5 | 6 | strongest SN |
| **h4 dense** | 0.10 | 1e5 | −2 | 1e6 | 6 | dense halo (deep band = explicit→implicit handoff; excluded) |
| **h5 long**  | 0.03 | 3e4 | −2 | 1e5 | 8 | full WR→SN→decline |
| **h6 flat**  | 0.30 | 3e5 |  0 | 1e3 | 6 | flat control |

Plots from these CSVs: `docs/dev/betadelta/diagnostics/plot_hunt.py` (`hunt_trigger/massdep/dmdt_leads`).
See `docs/dev/betadelta/diagnostics/README.md` for the full run-name glossary and the other phases.
