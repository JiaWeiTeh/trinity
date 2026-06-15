# Plan: implicit→momentum transition trigger — characterize, then decide

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Promotes **Phase 5** of `docs/dev/BETADELTA_HYBR_PLAN.md` (deferred there) into
its own program now that the hybr solver has landed and exposed the behaviour.
Background data: `analysis/stalling-energy-phase.md`,
`analysis/BETADELTA_PHASE2_ARMS.md`, figures under `scratch/phase2/`
(`phase3_regime.png` is the headline). Work branch: `fix/transition-trigger`.

**Mandate (set with the maintainer 2026-06-15):** instrument *both* transition
clocks and *characterize first* — the shadow comparison decides the criterion;
do **not** pre-commit to a winner (single physical trigger vs. profile-aware
configurable) before the data is in.

## The three questions

1. **What causes the very long transition duration?** — but "duration" is two
   different clocks (see Current state); separate them before explaining.
2. **Is ε = 0.05 the right threshold?**
3. **Is the energy ratio the right *metric*?** Candidates: `t_cool < t_dyn`,
   a force/continuity ratio, blowout, others from the literature.

## Current state (verified 2026-06-15 against source — re-verify before trusting)

### There are two distinct "transition" clocks
- **(A) Time to *reach* transition** (`t_trans`): the implicit phase ends on the
  cooling-balance criterion `(Lgain − Lloss)/Lgain < ε`. Set by how fast the hot
  interior's radiative `Lloss` catches the mechanical input `Lgain`. This is the
  "2.5 Myr / stalls forever" number in `phase3_regime.png`.
- **(B) Length of the transition *phase* (1c)** — a *separate* ODE phase
  (`trinity/phase1c_transition/run_transition_phase.py`). `get_ODE_transition_pure`
  (`:195`) integrates `dEb/dt = min(Ed_energy_balance, Ed_soundcrossing)` with
  `Ed_soundcrossing = −Eb/(R2/c_sound)` (`:233–239`,
  `c_sound = operations.get_soundspeed`, `operations.py:191`). The phase ends when
  `Eb < ENERGY_FLOOR = 1e3` erg (`:94`, `:761` → momentum), or `reached_tmax`
  (`:601`), or `small_radius` (`:785`). So **(B) is set by the sound-crossing time
  `R2/c_sound`**, and the `min()` means a feedback surge can re-inject and *stall
  the drain* — a second, distinct stall mechanism from (A).

These have different physics and different fixes; conflating them is the first trap.

### ε is *already* a param — but two checks disagree
- **Live terminator** of the implicit phase is an **inline** check in the runner
  (`run_energy_implicit_phase.py:1069–1079`): it reads
  **`phaseSwitch_LlossLgain`** (registered, default `0.05`, `run_const=True`,
  `registry.py:346`, `default.param:279`) and breaks on
  `(Lgain − Lloss)/Lgain < threshold`. `Lgain = Lmech_total` (`:1056`),
  `Lloss = bubble_LTotal (+ bubble_Leak)` (`:1061–1064`).
- **Dead/inconsistent path:** `build_implicit_phase_events` builds
  `make_cooling_balance_event(threshold=0.05)` (`phase_events.py:495`,
  **hardcoded 0.05**); the runner unpacks the factory (`:650`) but **never uses
  it** — the inline check is what fires. So the hardcoded-0.05 event is the path
  the prior plan flagged, but it appears inert here; the *real* knob is the
  existing param. **Consequence: the ε sensitivity sweep needs no code change to
  start** (sweep `phaseSwitch_LlossLgain` across runs), and "Step 0" reduces to
  *reconciling the two checks*, not creating a param.

### No `t_cool`/`t_dyn` machinery exists
`grep` for `t_cool`/`t_dyn`/cooling-time/dynamical-time finds nothing in the
physics path — any time-ratio criterion is new diagnostics, not a re-wire.

### What the energy ratio means physically
`(Lgain − Lloss)/Lgain < 0.05` ⇔ `Lloss ≥ 0.95 Lgain` ⇔ "transition when cooling
radiates ≥95 % of the injected mechanical power." It marks *"Eb stops growing,"*
not *"the thermal-pressure drive stops pushing the shell."* The shell is driven
by `max(Pb, P_HII)` (`_analysis/check_yesno.py:11`; production force in
`compute_forces_pure` — re-verify the exact site), and the momentum phase deletes
exactly the `4πR²·Pb` thermal force — so the *dynamically* correct seam is a
force statement, not an energy-accumulation one. (Naming trap, carried from the
prior plan: the implicit output field `F_ram` *is* `4πR²·Pb`.)
`Lloss` uses `n²Λ(T)` / `dudt(n,T,φ)` only — **no velocity** (cooling integrals
`bubble_luminosity.py:612/659/677`, per `analysis/stalling-energy-phase.md` —
re-verify).

## Candidate criteria (literature menu — verify each before encoding)

The current criterion is the **WARPFIELD-style energy-retention** test
(TRINITY's ancestor, Rahner et al. 2017). Alternatives that fire at *different*
times:

| # | criterion | one-line rationale | provenance (verify) |
|---|---|---|---|
| C0 | `(Lgain−Lloss)/Lgain < ε` | energy retention drops below ε | current / Rahner+2017 |
| C1 | `t_cool < t_dyn` (or ratio < k) | interior becomes radiative faster than it expands | Mac Low & McCray 1988; Koo & McKee 1992 |
| C2 | `4πR²Pb / (surviving forces) < O(1)` | thermal drive subdominant to ram+radiation; continuity-preserving | BETADELTA plan §5 |
| C3 | `R2 > rCloud` (blowout) | shell escapes the cloud — the steep-halo fate | geometric |
| C4 | cooling-efficiency parameter | interface mixing ⇒ momentum-driven much earlier | Lancaster+2021; El-Badry+2019 |

These are **not** interchangeable: on a steep r⁻² halo C0 never fires (stall),
while C3 eventually does; on dense flat clouds C0 and C1 likely nearly coincide.
The likely outcome is that the right trigger is **profile-dependent** — which a
single hardcoded scalar cannot express — but that is a conclusion to *earn* from
P2, not assume.

## Methodology (reuse what worked for the hybr solver)

- **Shadow / diagnostic = zero production impact.** All candidate criteria are
  *computed and logged alongside* the live run, never allowed to drive it, until
  a decision is made (mirrors the Phase-2 shadow arms). Production snapshots stay
  byte-identical through P0–P3.
- **Report every criterion on every segment** — the value is *where they
  diverge*, so one-criterion conclusions are invalid.
- **Reuse the existing config set** — no new physics configs invented:
  dense-flat (1e6, n1e5), normal-flat / "typical" (1e6, n1e3), steep (1e6, α=−2),
  mock (4e3), simple (1e5, sfe 0.3), and the hunt sweep h1–h6
  (`scratch/phase6/`). These already bracket transition / long-delay / stall /
  feedback-surge.
- **Pre-registered gates**, staleness banner on every results doc.

## Phases

### P0 — Disambiguate the duration (cheap, decisive; no production change)
Instrument both clocks per run: `t_trans` (implicit end), the 1c-phase length
(1c start → `Eb<floor`/`stop_t`), and inside 1c log `R2/c_sound` vs the
`Ed_energy_balance` term (which branch of the `min()` is active, and whether a
feedback surge flips it). Output one row/segment to CSV under `analysis/data/`.
**Gate P0:** classify each "long" case as (A)-dominated, (B)-dominated, or both,
across the config set, *before* designing any criterion. Decide whether (B) (the
1c sound-crossing drain) even matters for macro outputs or is a thin tail.

### P1 — Reconcile the existing ε knob (small, safe)
Make the **one** live trigger read `phaseSwitch_LlossLgain` everywhere: either
have `make_cooling_balance_event` take the param (not hardcode 0.05) **or**
delete the unused factory path if it is confirmed dead in the implicit phase
(flag, don't assume — check every caller of `build_implicit_phase_events`).
Net: a single parameterized energy-ratio trigger, byte-identical at the default.
Test: param threads through; default 0.05 reproduces current `t_trans`.

### P2 — Shadow-compute the candidate criteria (the heart)
Alongside production, per accepted implicit segment, compute and log C0–C3 (and
C4 if a defensible form is found): the energy ratio, `t_cool` & `t_dyn` (define
both explicitly — interior cooling time `e_thermal/Lloss` vs expansion time
`R2/v2` and/or `R2/c_sound`), `4πR²Pb` vs the surviving forces
(`pdot_wind+SN`, `F_rad`, `F_HII`, respecting `max(Pb,P_HII)`), and `R2/rCloud`.
**Deliverable:** for each config, *the time each criterion would first fire* on
the **identical** trajectory, and the cross-criterion divergence. **Gate P2 /
pivot:** if all criteria coincide within the segment cadence everywhere, ε-tuning
is the whole story (→ P3 only). If they diverge by regime (expected), document
the divergence map; that *is* the characterization deliverable.

### P3 — ε sensitivity (no code change; uses the existing param)
Sweep `phaseSwitch_LlossLgain` over the flat configs that actually cross it;
macro response (`t_trans`, terminal momentum, R2 at fixed times, energy-budget
closure across the seam) vs ε. Report insensitive→robust (keep 0.05, document
range) or sensitive→must-pin-dynamically (feeds P4).

### P4 — Decide & promote (gated; deferred until P0–P3 land)
With the divergence map + sensitivity in hand, the maintainer chooses: keep C0
(ε retune), adopt one physical criterion across the board, or a selectable /
profile-aware trigger behind a param. Whatever wins ships **behind a switch,
default unchanged for one release**, with cross-config attribution and an
"implications for published tracks" note. No promotion is pre-authorized here.

## Decisions that belong to the maintainer
1. Does the 1c sound-crossing drain (clock B) need its own treatment, or is it a
   negligible tail? (P0 answers the *fact*; the *should-we-act* is a call.)
2. Single physical criterion vs. profile-aware configurable trigger — **held
   open by mandate** until P2's divergence map exists.
3. For C1/C2: the exact `t_dyn` definition / force set and the threshold constant
   are physics calls, made *after* P2 shows where they'd fire.

## Risks
| risk | mitigation |
|---|---|
| "duration" stays conflated → wrong fix | P0 separates A/B before any design |
| shadow criteria perturb production | pure/diagnostic only, byte-identical snapshot check |
| picking a criterion from theory before data | mandate: characterize first; P2 divergence map gates P4 |
| `t_cool`/`t_dyn` definitions are arbitrary | pin both explicitly in P2, report sensitivity to the choice |
| ε sweep changes published tracks | default held one release; P4 attribution |
