# pt4 — transition-fix scoping: three routes to a *usable* energy→momentum handoff

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Date:** 2026-06-22. **Branch:** `fix/transition-trigger-problem-pt4`. **Status:** scoping only —
**no production change.** Decision deferred to the maintainer; this note exists so the route is
picked on evidence (rule 5: gate before touching production).

> **Update (2026-06-23):** **Route 1 has since partially shipped.** The blowout + PdV-inclusive
> **`ebpeak`** events are wired as an **opt-in, default-off** `transition_trigger` keyword
> (`run_energy_implicit_phase.py`) — an inert shadow plus an opt-in drive, **byte-identical** with the
> default `cooling_balance` (gate G1 passed). See `r1shadow/R1_FINDINGS.md`. So "no production change"
> describes the scoping snapshot, **not current state**. Still unbuilt: the default flip and the
> **heavy-cloud Eb-peak handoff into phase 1c** (the Eb-peak is a phase-1a event; the shadow lives in 1b).

## Why a fix is needed (not just documentation)

Under `hybr`, the cooling-balance trigger never fires (pt4 H1–H2: 0/6 cross 0.05), so every
normal-cloud run stays in the implicit **energy phase to the 15 Myr cap** — the bubble is treated as
energy-driven forever and never hands off to momentum. The early/in-cloud trajectory is fine, but the
**late-time/fate outputs (terminal momentum, final radius, stopping condition) are computed under the
wrong regime ⇒ untrustworthy.** "Always energy-driven" is physically wrong, so a transition that
actually fires is **required for usability**. Note also that legacy's transition was itself a
**β-clamp artifact** (H1), so reverting is not an option — the trigger must be re-grounded on an event
that actually occurs.

## Evidence the routes must respect (committed pt4 data; re-verify per banner)

Per config, from `../cleanroom/data/c0_*_h0.csv`, `h2_crossing_summary.csv`, `h4_eval.csv`, H1 summary:

| config | ratio_min (never <0.05) | Eb in-cloud peak? | blowout `R2>rCloud` @ | max PdV/Lmech |
|---|---|---|---|---|
| simple_cluster | 0.324 | **no (monotonic to cap)** | 0.090 Myr | <1 |
| small_dense_highsfe | 0.283 | no (monotonic) | 0.012 Myr | <1 |
| midrange_pl0 | 0.364 | no (monotonic) | 0.392 Myr | <1 |
| pl2_steep | 0.489 | no (monotonic) | 0.840 Myr | <1 |
| be_sphere | 0.471 | no (monotonic) | 0.856 Myr | <1 |
| large_diffuse_lowsfe | 0.465 | no (monotonic) | 3.660 Myr | <1 |
| fail_repro / fail_helix (5e9) | n/a (collapse) | **peaks at birth** | — | **2.65 / 2.36** |

**Two facts pin the design:** (1) for **normal clouds the only transition event that fires in-cloud
is blowout** — `Eb` never peaks, so an Eb-peak/net-energy criterion alone cannot fire for them; (2)
the **Eb-peak fires only for heavy clouds** (and is the principled crash handoff). So a regime-spanning
transition must combine **blowout (normal clouds) + Eb-peak (heavy clouds)**, *or* restore a real
cooling event (routes 2/3).

## The three routes

### Route 1 — Geometric blowout + Eb-peak handoff
- **Mechanism:** add transition events to the 1b terminator: `R2 > k·rCloud` (blowout) **OR** the
  PdV-inclusive net-energy zero-crossing `(Lgain − Lloss − 4πR2²·v2·Pb) ≤ 0` (Eb-peak). Keep the
  cooling-balance trigger as-is (so any future-cooled model / legacy still uses it). Code sites:
  `phase_general/phase_events.py` (blowout event already exists for 1a→1b, `is_simulation_ending=False`
  — needs a transition-firing sibling), `run_energy_implicit_phase.py` terminator (~:1095).
- **Where it fires:** blowout for **all** configs (0.012–3.66 Myr); Eb-peak for the 5e9 clouds (at
  birth). Additive ⇒ configs that already cross 0.05 are unchanged.
- **Effort:** medium (event machinery largely exists). **Solver risk:** low (a criterion, not a solve
  change). **Fixes over-retention:** **no** — interior stays too hot up to blowout.
- **Open questions:** value of `k` (1, or a few, or a sustained criterion); **continuity** of
  `Eb/R2/v2/P_drive` into phase 1c at the handoff — low-risk for normal clouds (energy-rich at
  blowout), but the **heavy-cloud Eb-peak handoff is the real risk** (Path-2 caveat: the reservoir
  barely grew; 1c may reject it).

### Route 2 — Leakage (`coverFraction < 1`)
- **Mechanism:** `Cf<1` adds the leak term to the loss (`Lloss = Lbubble + Lleak`,
  `get_bubbleParams.get_leak_luminosity`) ⇒ the **existing** ratio crosses 0.05. A param + existing
  machinery, no new physics path.
- **Where it fires:** prior cleanroom `data/leaktest/` — `Cf=0.95` (5% leak) crossed 0.05 @ t≈0.131,
  **solver-healthy** (unlike the bulk sink); `Cf=1` gap 0.145–0.292 dex. *(Re-verify + extend to all 6
  configs — only partially tested.)*
- **Effort:** low (param + calibration). **Solver risk:** low (demonstrated healthy).
  **Fixes over-retention:** **partially** — venting hot gas lowers `f_ret` toward the observed band,
  but it is **advective venting, not radiative cooling** (a different mechanism than mixing).
- **Open questions:** what `Cf` is physical (porosity/covering fraction — calibrated or free?); does it
  fire across **all** configs incl. the steep one (where `Lloss∝n²` collapses)?

### Route 3 — Mixing-layer cooling (root fix)
- **Mechanism:** integrate `θ≈0.25` mixing-layer cooling **into the betadelta structure solve** (so
  `β,δ` are solved *with* it, keeping `dMdt>0`). **Not** a bulk `dEb/dt` sink — that drives `dMdt<0`,
  no root, and the solver grinds (proven in `../cleanroom`, `mixcool_whatif.py`). Code: `get_betadelta.py`
  residual / bubble-structure ODE.
- **Where it fires:** offline `mixcool_whatif.py` brought `f_ret` into the observed 0.01–0.1 band for
  **all 6** ⇒ a genuine cooling event ⇒ the **existing** trigger fires at a physical time.
- **Effort:** **high** (a modelling change + solver integration). **Solver risk:** **high** (naive
  form already breaks it). **Fixes over-retention:** **yes — the point.**
- **Open questions:** the faithful `L_mix` form (`θ=const` is too naive — tie to interface
  area / mixing velocity / density); how it enters the residual without breaking `dMdt>0`; θ calibration.

## Comparison matrix

| axis | R1 blowout+Eb-peak | R2 leakage Cf<1 | R3 mixing-layer |
|---|---|---|---|
| makes code usable (transition fires) | **yes, all configs** | yes (if fires all configs) | yes |
| effort | medium | **low** | high |
| solver risk | **low** | **low** | high |
| fixes energy over-retention | no | partial | **yes** |
| physical correctness of transition | dynamical/geometric (defensible) | depends on true Cf | **highest** |
| regime-spanning (normal + 5e9) | **yes** (blowout+Eb-peak) | needs check on steep | yes |
| reversible / additive | **yes (additive)** | yes (param) | no (model change) |

## Recommendation (for the maintainer to confirm)

- **Immediate usability → Route 1 (additive blowout + Eb-peak).** Lowest risk, fires for every config,
  and is *additive* (keeps the cooling trigger, so it never perturbs a run that already transitions).
  It makes the code produce a defensible fate instead of a perpetual energy-driven bubble — but does
  **not** fix the energy over-retention.
- **Cheapest physical improvement → Route 2 (leakage), as a quick experiment first:** extend the
  `Cf<1` leaktest to all 6 configs; if a small physical `Cf` fires the existing trigger solver-healthy
  everywhere, it is a one-knob fix that *also* nudges `f_ret` toward observations.
- **Correctness root fix → Route 3 (mixing-layer):** the principled long-term answer; pursue behind a
  design for the faithful `L_mix` once R1 has restored usability.

Suggested ordering: **R1 now (usability) → R2 experiment (physical cross-check) → R3 (correctness
workstream).** R1 and R3 are complementary, not exclusive — R1 decides *when* to hand off; R3 fixes
*how much energy the interior should have* by then.

## Gates for whichever route is built (rule-5 ladder — all are risky/iterative/outward-facing)

1. **Gate first:** define "equivalent." For R1, the hard gate is **byte-identical on any config that
   already transitions via cooling** (the event is additive). For R2/R3, define the firing-epoch and
   `f_ret` targets up front.
2. **Baseline:** capture `git show HEAD` trajectories on the edge set (`simple_cluster` +
   `f1edge_{lowdens,hidens}` + a 5e9 point).
3. **Equivalence:** per-call first, then **full-run on the stiffest regimes in separate processes at
   matched `t`** (not just per-call).
4. **Apply** the smallest diff that passes; **re-verify** (gate + full `pytest` + ruff F-rules);
   **persist** diagnostics here.

Continuity is the make-or-break check for R1: at the handoff, `Eb/R2/v2/P_drive` must enter phase 1c
no more discontinuously than the cooling trigger does — verify especially the heavy-cloud Eb-peak case.
