# docs/dev/archive/betadelta/velstruct — velocity-structure ("Problem 2") hunt & treatment tooling

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🧊 **Frozen historical record — do not extend.** This workstream shipped or was
> superseded (see the Status line below); the doc is kept as evidence/history. Do
> not update or extend it — new work gets a new doc in an active workstream. The
> ⚠️ caveat above still applies: paths and line references reflect the code as it
> was when this was written.

Harness/tooling for **Phase 6** of the β–δ hybr study: does the transient inner
inflow (negative interior velocity during a feedback re-pressurisation) ever
contaminate anything? **Not source.** Canonical writeup:
`docs/dev/archive/betadelta/stalling-energy-phase.md` ("Phase 6.0 contamination hunt" +
"Is the inflow physical?"); plan: `docs/dev/archive/betadelta/HYBR_PLAN.md` Phase 6.

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

Plots from these CSVs: `docs/dev/archive/betadelta/diagnostics/plot_hunt.py` (`hunt_trigger/massdep/dmdt_leads`).
See `docs/dev/archive/betadelta/diagnostics/README.md` for the full run-name glossary and the other phases.
